"""Tests for dscompanion.scoring.ScoringRunner — end-to-end YAML-driven batch scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from dscompanion.pipeline.config import DataConfig, ExplainConfig, ModelConfig, PipelineConfig
from dscompanion.pipeline.runner import PipelineRunner
from dscompanion.scoring import ScoringRunner


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
    """One real PipelineRunner.run(), producing a ScoringPipeline bundle to score against."""
    train_path = tmp_path_factory.mktemp("scoring_runner_train") / "train.parquet"
    synthetic_df.to_parquet(train_path, index=False)

    cfg = PipelineConfig(
        name="scoring_runner_test",
        data=DataConfig(path=str(train_path), target="target", ignore_columns=["customer_id"]),
        model=ModelConfig(task="classification", algorithm="xgboost"),
        explain=ExplainConfig(shap_enabled=False),
        reporting={"output_dir": str(tmp_path_factory.mktemp("training_runs"))},
    )
    return PipelineRunner(cfg).run()


@pytest.fixture(scope="module")
def new_data_parquet(tmp_path_factory, synthetic_df):
    path = tmp_path_factory.mktemp("scoring_runner_new") / "new_data.parquet"
    synthetic_df.drop(columns=["target"]).to_parquet(path, index=False)
    return path


def _write_scoring_yaml(path, **overrides):
    doc = {
        "name": "scoring_runner_test",
        "output": {"output_dir": str(path.parent / "scoring_runs")},
    }
    doc.update(overrides)
    path.write_text(yaml.dump(doc))
    return path


class TestScoringRunnerEndToEnd:
    def test_run_writes_scored_output_and_config(self, tmp_path, train_result, new_data_parquet):
        yaml_path = _write_scoring_yaml(
            tmp_path / "scoring.yaml",
            scoring_pipeline_path=str(train_result.scoring_pipeline_path),
            data={"path": str(new_data_parquet), "format": "parquet"},
            id_columns=["customer_id"],
        )
        result = ScoringRunner.from_yaml(yaml_path).run()

        assert result.output_path.is_file()
        assert result.output_path.parent == result.run_dir
        assert (result.run_dir / "config.yaml").is_file()
        assert result.log_path.is_file()
        assert "customer_id" in result.scored_df.columns
        assert "prediction" in result.scored_df.columns
        assert "probability" in result.scored_df.columns
        assert len(result.scored_df) == 300

    def test_run_id_is_ist_timestamp_format(self, tmp_path, train_result, new_data_parquet):
        import re

        yaml_path = _write_scoring_yaml(
            tmp_path / "scoring.yaml",
            scoring_pipeline_path=str(train_result.scoring_pipeline_path),
            data={"path": str(new_data_parquet), "format": "parquet"},
        )
        result = ScoringRunner.from_yaml(yaml_path).run()
        assert re.fullmatch(r"\d{8}_\d{6}(_[0-9a-f]{4})?", result.run_id)
        assert result.run_dir.name == result.run_id

    def test_check_drift_writes_drift_report(self, tmp_path, train_result, new_data_parquet):
        yaml_path = _write_scoring_yaml(
            tmp_path / "scoring.yaml",
            scoring_pipeline_path=str(train_result.scoring_pipeline_path),
            data={"path": str(new_data_parquet), "format": "parquet"},
            check_drift=True,
        )
        result = ScoringRunner.from_yaml(yaml_path).run()

        assert result.drift_report is not None
        assert list(result.drift_report["feature"]) == ["__score__"]
        assert (result.run_dir / "drift_report.csv").is_file()

    def test_check_drift_false_leaves_drift_report_none(
        self, tmp_path, train_result, new_data_parquet
    ):
        yaml_path = _write_scoring_yaml(
            tmp_path / "scoring.yaml",
            scoring_pipeline_path=str(train_result.scoring_pipeline_path),
            data={"path": str(new_data_parquet), "format": "parquet"},
        )
        result = ScoringRunner.from_yaml(yaml_path).run()
        assert result.drift_report is None
        assert not (result.run_dir / "drift_report.csv").exists()

    def test_output_format_csv(self, tmp_path, train_result, new_data_parquet):
        yaml_path = _write_scoring_yaml(
            tmp_path / "scoring.yaml",
            scoring_pipeline_path=str(train_result.scoring_pipeline_path),
            data={"path": str(new_data_parquet), "format": "parquet"},
            output={"output_dir": str(tmp_path / "scoring_runs"), "format": "csv"},
        )
        result = ScoringRunner.from_yaml(yaml_path).run()
        assert result.output_path.suffix == ".csv"
        reloaded = pd.read_csv(result.output_path)
        assert len(reloaded) == len(result.scored_df)

    def test_missing_column_in_new_data_raises(self, tmp_path, train_result, synthetic_df):
        bad_path = tmp_path / "missing_col.parquet"
        synthetic_df.drop(columns=["target", "f1"]).to_parquet(bad_path, index=False)

        yaml_path = _write_scoring_yaml(
            tmp_path / "scoring.yaml",
            scoring_pipeline_path=str(train_result.scoring_pipeline_path),
            data={"path": str(bad_path), "format": "parquet"},
        )
        with pytest.raises(ValueError, match="missing required column"):
            ScoringRunner.from_yaml(yaml_path).run()
