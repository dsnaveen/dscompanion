"""Tests for dscompanion.monitoring.MonitoringRunner — end-to-end monitoring."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import yaml

from dscompanion.monitoring import MonitoringRunner
from dscompanion.pipeline.config import DataConfig, ExplainConfig, ModelConfig, PipelineConfig
from dscompanion.pipeline.runner import PipelineRunner
from dscompanion.scoring import ScoringPipeline


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 300
    return pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "customer_id": np.arange(n),
            "target": rng.choice([0, 1], size=n),
        }
    )


@pytest.fixture(scope="module")
def train_result(tmp_path_factory, synthetic_df):
    train_path = tmp_path_factory.mktemp("monitoring_train") / "train.parquet"
    synthetic_df.to_parquet(train_path, index=False)

    cfg = PipelineConfig(
        name="monitoring_runner_test",
        data=DataConfig(path=str(train_path), target="target", ignore_columns=["customer_id"]),
        model=ModelConfig(task="classification", algorithm="xgboost"),
        explain=ExplainConfig(shap_enabled=False),
        reporting={"output_dir": str(tmp_path_factory.mktemp("monitoring_training_runs"))},
    )
    return PipelineRunner(cfg).run()


@pytest.fixture(scope="module")
def scored_parquet(tmp_path_factory, train_result, synthetic_df):
    """Realistic scored_data — produced via a real ScoringPipeline.predict() call."""
    scoring_pipeline = ScoringPipeline.load(train_result.scoring_pipeline_path)
    scored = scoring_pipeline.predict(synthetic_df, id_columns=["customer_id"])
    path = tmp_path_factory.mktemp("monitoring_scored") / "scored.parquet"
    scored.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def actuals_parquet(tmp_path_factory, synthetic_df):
    actuals = synthetic_df[["customer_id", "target"]].copy()
    path = tmp_path_factory.mktemp("monitoring_actuals") / "actuals.parquet"
    actuals.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def raw_parquet(tmp_path_factory, synthetic_df):
    path = tmp_path_factory.mktemp("monitoring_raw") / "raw.parquet"
    synthetic_df.drop(columns=["target"]).to_parquet(path, index=False)
    return path


def _write_monitoring_yaml(path, **overrides):
    doc = {
        "name": "monitoring_runner_test",
        "id_columns": ["customer_id"],
        "output": {"output_dir": str(path.parent / "monitoring_runs")},
    }
    doc.update(overrides)
    path.write_text(yaml.dump(doc))
    return path


def _base_config(train_result, scored_parquet, actuals_parquet):
    return {
        "scoring_pipeline_path": str(train_result.scoring_pipeline_path),
        "scored_data": {"path": str(scored_parquet), "format": "parquet"},
        "actuals_data": {
            "path": str(actuals_parquet),
            "format": "parquet",
            "target_column": "target",
        },
    }


class TestMonitoringRunnerEndToEnd:
    def test_run_writes_reports_and_config(
        self, tmp_path, train_result, scored_parquet, actuals_parquet
    ):
        yaml_path = _write_monitoring_yaml(
            tmp_path / "monitoring.yaml",
            **_base_config(train_result, scored_parquet, actuals_parquet),
        )
        result = MonitoringRunner.from_yaml(yaml_path).run()

        assert result.performance_report_path.is_file()
        assert (result.run_dir / "config.yaml").is_file()
        assert result.log_path.is_file()
        assert result.performance_report["split"].eq("monitoring").all()
        metrics = set(result.performance_report["metric"])
        assert {
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "gini",
            "ks_statistic",
            "log_loss",
        } <= metrics
        assert result.n_scored_rows == 300
        assert result.n_matched_rows == 300

    def test_run_id_is_name_prefixed_timestamp_format(
        self, tmp_path, train_result, scored_parquet, actuals_parquet
    ):
        yaml_path = _write_monitoring_yaml(
            tmp_path / "monitoring.yaml",
            **_base_config(train_result, scored_parquet, actuals_parquet),
        )
        result = MonitoringRunner.from_yaml(yaml_path).run()
        assert re.fullmatch(r"monitoring_runner_test_\d{8}_\d{6}(_[0-9a-f]{4})?", result.run_id)
        assert result.run_dir.name == result.run_id

    def test_feature_drift_report_present_when_raw_data_set(
        self, tmp_path, train_result, scored_parquet, actuals_parquet, raw_parquet
    ):
        cfg = _base_config(train_result, scored_parquet, actuals_parquet)
        cfg["raw_data"] = {"path": str(raw_parquet), "format": "parquet"}
        yaml_path = _write_monitoring_yaml(tmp_path / "monitoring.yaml", **cfg)
        result = MonitoringRunner.from_yaml(yaml_path).run()

        assert result.feature_drift_report is not None
        assert {"feature", "psi", "flag"} <= set(result.feature_drift_report.columns)
        assert result.feature_drift_report_path is not None
        assert result.feature_drift_report_path.is_file()

    def test_feature_drift_report_none_when_raw_data_omitted(
        self, tmp_path, train_result, scored_parquet, actuals_parquet
    ):
        yaml_path = _write_monitoring_yaml(
            tmp_path / "monitoring.yaml",
            **_base_config(train_result, scored_parquet, actuals_parquet),
        )
        result = MonitoringRunner.from_yaml(yaml_path).run()
        assert result.feature_drift_report is None
        assert result.feature_drift_report_path is None
        assert not (result.run_dir / "feature_drift_report.csv").exists()

    def test_partial_join_mismatch_logs_warning_and_proceeds(
        self, tmp_path, train_result, scored_parquet, actuals_parquet, caplog
    ):
        actuals_df = pd.read_parquet(actuals_parquet)
        dropped_path = tmp_path / "actuals_dropped.parquet"
        actuals_df.iloc[:-10].to_parquet(dropped_path, index=False)

        cfg = _base_config(train_result, scored_parquet, dropped_path)
        yaml_path = _write_monitoring_yaml(tmp_path / "monitoring.yaml", **cfg)

        with caplog.at_level("WARNING"):
            result = MonitoringRunner.from_yaml(yaml_path).run()

        assert result.n_matched_rows < result.n_scored_rows
        assert any("matched" in rec.message for rec in caplog.records)

    def test_zero_overlap_raises(self, tmp_path, train_result, scored_parquet):
        disjoint = pd.DataFrame({"customer_id": np.arange(10000, 10010), "target": [0] * 10})
        disjoint_path = tmp_path / "disjoint_actuals.parquet"
        disjoint.to_parquet(disjoint_path, index=False)

        cfg = _base_config(train_result, scored_parquet, disjoint_path)
        yaml_path = _write_monitoring_yaml(tmp_path / "monitoring.yaml", **cfg)

        with pytest.raises(ValueError, match="No rows matched"):
            MonitoringRunner.from_yaml(yaml_path).run()

    def test_output_format_parquet(self, tmp_path, train_result, scored_parquet, actuals_parquet):
        cfg = _base_config(train_result, scored_parquet, actuals_parquet)
        yaml_path = _write_monitoring_yaml(
            tmp_path / "monitoring.yaml",
            **cfg,
            output={"output_dir": str(tmp_path / "monitoring_runs"), "format": "parquet"},
        )
        result = MonitoringRunner.from_yaml(yaml_path).run()
        assert result.performance_report_path.suffix == ".parquet"
        reloaded = pd.read_parquet(result.performance_report_path)
        assert len(reloaded) == len(result.performance_report)

    def test_output_format_excel(self, tmp_path, train_result, scored_parquet, actuals_parquet):
        cfg = _base_config(train_result, scored_parquet, actuals_parquet)
        yaml_path = _write_monitoring_yaml(
            tmp_path / "monitoring.yaml",
            **cfg,
            output={"output_dir": str(tmp_path / "monitoring_runs"), "format": "excel"},
        )
        result = MonitoringRunner.from_yaml(yaml_path).run()
        assert result.performance_report_path.suffix == ".xlsx"

    def test_raw_data_set_but_no_feature_reference_raises(
        self, tmp_path, train_result, scored_parquet, actuals_parquet, raw_parquet
    ):
        loaded = ScoringPipeline.load(train_result.scoring_pipeline_path)
        loaded.feature_reference_ = None
        bad_bundle_path = tmp_path / "no_feature_ref_bundle.joblib"
        loaded.save(bad_bundle_path)

        cfg = _base_config(train_result, scored_parquet, actuals_parquet)
        cfg["scoring_pipeline_path"] = str(bad_bundle_path)
        cfg["raw_data"] = {"path": str(raw_parquet), "format": "parquet"}
        yaml_path = _write_monitoring_yaml(tmp_path / "monitoring.yaml", **cfg)

        with pytest.raises(ValueError, match="feature_reference_"):
            MonitoringRunner.from_yaml(yaml_path).run()


class TestMonitoringRunnerRegression:
    @pytest.fixture(scope="class")
    def synthetic_regression_df(self) -> pd.DataFrame:
        rng = np.random.RandomState(7)
        n = 500
        f1 = rng.randn(n)
        f2 = rng.randn(n)
        target = 3.0 * f1 - 2.0 * f2 + rng.randn(n) * 0.5
        return pd.DataFrame({"f1": f1, "f2": f2, "customer_id": np.arange(n), "target": target})

    @pytest.fixture(scope="class")
    def regression_train_result(self, tmp_path_factory, synthetic_regression_df):
        train_path = tmp_path_factory.mktemp("monitoring_reg_train") / "train.parquet"
        synthetic_regression_df.to_parquet(train_path, index=False)
        cfg = PipelineConfig(
            name="monitoring_regression_test",
            data=DataConfig(path=str(train_path), target="target", ignore_columns=["customer_id"]),
            model=ModelConfig(task="regression", algorithm="linear"),
            explain=ExplainConfig(shap_enabled=False),
            reporting={"output_dir": str(tmp_path_factory.mktemp("monitoring_reg_training_runs"))},
        )
        return PipelineRunner(cfg).run()

    def test_regression_task_metrics(
        self, tmp_path, regression_train_result, synthetic_regression_df, tmp_path_factory
    ):
        scoring_pipeline = ScoringPipeline.load(regression_train_result.scoring_pipeline_path)
        scored = scoring_pipeline.predict(synthetic_regression_df, id_columns=["customer_id"])
        scored_path = tmp_path_factory.mktemp("monitoring_reg_scored") / "scored.parquet"
        scored.to_parquet(scored_path, index=False)

        actuals_path = tmp_path_factory.mktemp("monitoring_reg_actuals") / "actuals.parquet"
        synthetic_regression_df[["customer_id", "target"]].to_parquet(actuals_path, index=False)

        yaml_path = _write_monitoring_yaml(
            tmp_path / "monitoring.yaml",
            scoring_pipeline_path=str(regression_train_result.scoring_pipeline_path),
            scored_data={"path": str(scored_path), "format": "parquet"},
            actuals_data={
                "path": str(actuals_path),
                "format": "parquet",
                "target_column": "target",
            },
        )
        result = MonitoringRunner.from_yaml(yaml_path).run()
        metrics = set(result.performance_report["metric"])
        assert {"rmse", "mae", "r2", "mape", "median_absolute_error"} <= metrics
