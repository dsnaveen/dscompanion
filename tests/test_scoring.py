"""Tests for dscompanion.scoring.ScoringPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscompanion.features.pipeline import FeatureProcessingPipeline
from dscompanion.models import ModelFactory
from dscompanion.pipeline.config import DataConfig, ExplainConfig, ModelConfig, PipelineConfig
from dscompanion.pipeline.runner import PipelineRunner
from dscompanion.scoring import ScoringPipeline
from dscompanion.selection.selection_pipeline import FeatureSelectionPipeline

IGNORE_COLUMNS = ["customer_id", "leakage_col", "account_open_date", "snapshot_date"]


@pytest.fixture(scope="module")
def synthetic_parquet(tmp_path_factory, synthetic_df):
    path = tmp_path_factory.mktemp("scoring_data") / "synthetic.parquet"
    synthetic_df.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def run_result(synthetic_parquet):
    """One real classification PipelineRunner.run(), shared across this module's tests."""
    cfg = PipelineConfig(
        name="scoring_test",
        data=DataConfig(
            path=str(synthetic_parquet),
            target="target",
            ignore_columns=IGNORE_COLUMNS,
        ),
        model=ModelConfig(task="classification", algorithm="xgboost"),
        explain=ExplainConfig(shap_enabled=False),
    )
    return PipelineRunner(cfg).run()


@pytest.fixture(scope="module")
def scoring_pipeline_path(run_result):
    assert run_result.scoring_pipeline_path is not None
    return run_result.scoring_pipeline_path


def _reference_scoring_model(run_result):
    """The exact scoring-time model object ScoringPipeline should be replaying."""
    if run_result.calibrator is not None:
        return run_result.calibrator.wrap(run_result.model)
    return run_result.model


class TestRoundTrip:
    def test_loaded_pipeline_is_a_genuinely_separate_object(
        self, scoring_pipeline_path, run_result
    ):
        # Proves this is a real deserialize-from-disk, not an accidental
        # reference to the same in-memory object from the training process.
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        assert loaded.model is not run_result.model

    def test_predictions_match_in_memory_model_on_raw_test_rows(
        self, scoring_pipeline_path, run_result, synthetic_df
    ):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index]

        scored = loaded.predict(new_df)

        reference_model = _reference_scoring_model(run_result)
        expected_labels = reference_model.predict(run_result.split.test_X)
        expected_proba = reference_model.predict_proba(run_result.split.test_X)[:, 1]

        np.testing.assert_array_equal(scored["prediction"].to_numpy(), expected_labels)
        np.testing.assert_allclose(scored["probability"].to_numpy(), expected_proba)

    def test_output_index_matches_input_index(
        self, scoring_pipeline_path, run_result, synthetic_df
    ):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index]
        scored = loaded.predict(new_df)
        pd.testing.assert_index_equal(scored.index, new_df.index)


class TestSchemaValidation:
    def test_missing_column_raises(self, scoring_pipeline_path, run_result, synthetic_df):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index].drop(columns=["f1"])
        with pytest.raises(ValueError, match="missing required column"):
            loaded.predict(new_df)

    def test_dtype_mismatch_raises(self, scoring_pipeline_path, run_result, synthetic_df):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index].copy()
        new_df["f1"] = new_df["f1"].astype(str)
        with pytest.raises(ValueError, match="incompatible dtype"):
            loaded.predict(new_df)

    def test_extra_column_is_tolerated(self, scoring_pipeline_path, run_result, synthetic_df):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index].copy()
        new_df["totally_unrelated_extra_col"] = 1
        scored = loaded.predict(new_df)
        assert len(scored) == len(new_df)

    def test_empty_dataframe_raises(self, scoring_pipeline_path, run_result, synthetic_df):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        empty_df = synthetic_df.loc[run_result.split.test_X.index].iloc[:0]
        with pytest.raises(ValueError, match="empty"):
            loaded.predict(empty_df)


class TestIdColumns:
    def test_id_column_passthrough_aligned_by_index(
        self, scoring_pipeline_path, run_result, synthetic_df
    ):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index].sample(frac=1.0, random_state=0)

        scored = loaded.predict(new_df, id_columns=["customer_id"])

        assert list(scored.columns) == ["customer_id", "prediction", "probability"]
        pd.testing.assert_series_equal(scored["customer_id"], new_df["customer_id"])

    def test_unknown_id_column_raises(self, scoring_pipeline_path, run_result, synthetic_df):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index]
        with pytest.raises(ValueError, match="id_columns not present"):
            loaded.predict(new_df, id_columns=["does_not_exist"])

    def test_id_column_candidates_captured_from_ignore_columns(self, scoring_pipeline_path):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        assert set(IGNORE_COLUMNS) <= set(loaded.id_column_candidates)


class TestPredictOne:
    def test_matches_predict_on_equivalent_single_row(
        self, scoring_pipeline_path, run_result, synthetic_df
    ):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        one_row = synthetic_df.loc[[run_result.split.test_X.index[0]]]

        one_result = loaded.predict_one(one_row.iloc[0].to_dict())
        batch_result = loaded.predict(one_row).iloc[0].to_dict()

        assert one_result["prediction"] == batch_result["prediction"]
        assert one_result["probability"] == pytest.approx(batch_result["probability"])


class TestComputeDrift:
    def test_unchanged_distribution_is_not_flagged(
        self, scoring_pipeline_path, run_result, synthetic_df
    ):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index]

        report = loaded.compute_drift(new_df)

        assert list(report["feature"]) == ["__score__"]
        assert not report["flag"].iloc[0]

    def test_shifted_distribution_is_flagged(self, scoring_pipeline_path, run_result, synthetic_df):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        shifted_df = synthetic_df.loc[run_result.split.test_X.index].copy()
        shifted_df["f1"] = shifted_df["f1"] + 8.0

        report = loaded.compute_drift(shifted_df)

        assert report["flag"].iloc[0]


class TestComputeFeatureDrift:
    def test_unchanged_distribution_is_not_flagged(
        self, scoring_pipeline_path, run_result, synthetic_df
    ):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index]

        report = loaded.compute_feature_drift(new_df)

        assert not report["flag"].any()

    def test_one_shifted_feature_is_flagged(self, scoring_pipeline_path, run_result, synthetic_df):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        shifted_df = synthetic_df.loc[run_result.split.test_X.index].copy()
        shifted_df["f1"] = shifted_df["f1"] + 8.0

        report = loaded.compute_feature_drift(shifted_df)

        assert report.loc[report["feature"] == "f1", "flag"].iloc[0]
        assert not report.loc[report["feature"] == "f2", "flag"].iloc[0]

    def test_raises_without_feature_reference(
        self, scoring_pipeline_path, run_result, synthetic_df
    ):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        loaded.feature_reference_ = None
        new_df = synthetic_df.loc[run_result.split.test_X.index]
        with pytest.raises(ValueError, match="feature_reference_"):
            loaded.compute_feature_drift(new_df)

    def test_does_not_require_transform_chain_to_succeed(
        self, scoring_pipeline_path, run_result, synthetic_df
    ):
        # A brand-new categorical value that a fitted encoder inside
        # predict()/compute_drift() might choke on should not break
        # compute_feature_drift(), since it bypasses the transform chain
        # entirely and compares raw columns directly.
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        new_df = synthetic_df.loc[run_result.split.test_X.index].copy()
        new_df["cat_low"] = "a_brand_new_category_never_seen_in_training"

        report = loaded.compute_feature_drift(new_df)
        assert "cat_low" in report["feature"].tolist()


class TestRegressionTask:
    @pytest.fixture(scope="class")
    def synthetic_regression_df(self) -> pd.DataFrame:
        """Small synthetic dataset with a continuous target — mirrors
        TestEndToEndRegressionPipeline's fixture in tests/test_pipeline.py
        (not shared via conftest.py, so re-created locally here).
        """
        rng = np.random.RandomState(7)
        n = 500
        f1 = rng.randn(n)
        f2 = rng.randn(n)
        target = 3.0 * f1 - 2.0 * f2 + rng.randn(n) * 0.5
        return pd.DataFrame({"f1": f1, "f2": f2, "target": target})

    @pytest.fixture(scope="class")
    def synthetic_regression_parquet(self, tmp_path_factory, synthetic_regression_df):
        path = tmp_path_factory.mktemp("scoring_regression_data") / "synthetic_regression.parquet"
        synthetic_regression_df.to_parquet(path, index=False)
        return path

    @pytest.fixture(scope="class")
    def regression_result(self, synthetic_regression_parquet):
        cfg = PipelineConfig(
            name="scoring_regression_test",
            data=DataConfig(path=str(synthetic_regression_parquet), target="target"),
            model=ModelConfig(task="regression", algorithm="linear"),
            explain=ExplainConfig(shap_enabled=False),
        )
        return PipelineRunner(cfg).run()

    def test_predict_has_no_probability_column(self, regression_result):
        loaded = ScoringPipeline.load(regression_result.scoring_pipeline_path)
        raw_df = pd.read_parquet(regression_result.config.data.path).loc[
            regression_result.split.test_X.index
        ]
        scored = loaded.predict(raw_df)
        assert "probability" not in scored.columns
        assert "prediction" in scored.columns

    def test_psi_reference_captured(self, regression_result):
        loaded = ScoringPipeline.load(regression_result.scoring_pipeline_path)
        assert loaded.psi_reference_ is not None


class TestClusteringTask:
    """Isolated unit tests — not routed through PipelineRunner, since PipelineRunner's
    clustering support is a separate, pre-existing concern outside this task's scope.
    Exercises ScoringPipeline.from_run()/predict() directly against a real, fitted
    ClusteringModel + feature/selection pipelines.
    """

    @pytest.fixture(scope="class")
    def clustering_scoring_pipeline(self):
        rng = np.random.RandomState(0)
        n = 200
        df = pd.DataFrame(
            {
                "f1": rng.randn(n),
                "f2": rng.randn(n),
            }
        )
        dummy_y = pd.Series(np.zeros(n))

        feature_pipeline = FeatureProcessingPipeline()
        train_X = feature_pipeline.fit_transform(df, dummy_y)
        selection_pipeline = FeatureSelectionPipeline()
        selected_X = selection_pipeline.fit_transform(train_X, dummy_y)

        model = ModelFactory.build("clustering", "kmeans")
        model.fit(selected_X, dummy_y)

        return ScoringPipeline.from_run(
            feature_pipeline=feature_pipeline,
            selection_pipeline=selection_pipeline,
            model=model,
            calibrator=None,
            schema_df=df,
            target_col="target",
            task="clustering",
            id_column_candidates=[],
            psi_reference=None,
        )

    def test_predict_has_no_probability_column(self, clustering_scoring_pipeline):
        rng = np.random.RandomState(1)
        new_df = pd.DataFrame({"f1": rng.randn(20), "f2": rng.randn(20)})
        scored = clustering_scoring_pipeline.predict(new_df)
        assert "probability" not in scored.columns
        assert "prediction" in scored.columns

    def test_compute_drift_raises_without_reference(self, clustering_scoring_pipeline):
        rng = np.random.RandomState(1)
        new_df = pd.DataFrame({"f1": rng.randn(20), "f2": rng.randn(20)})
        with pytest.raises(ValueError, match="psi_reference_"):
            clustering_scoring_pipeline.compute_drift(new_df)


class TestBundleVersioning:
    def test_mismatched_bundle_schema_version_raises_on_load(self, scoring_pipeline_path, tmp_path):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        loaded.bundle_schema_version = 999
        bad_path = tmp_path / "bad_bundle.joblib"
        loaded.save(bad_path)

        with pytest.raises(RuntimeError, match="bundle_schema_version"):
            ScoringPipeline.load(bad_path)

    def test_freshly_created_pipeline_has_current_bundle_schema_version(
        self, scoring_pipeline_path
    ):
        loaded = ScoringPipeline.load(scoring_pipeline_path)
        assert loaded.bundle_schema_version == 2
