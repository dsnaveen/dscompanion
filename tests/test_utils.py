"""Tests for dscompanion.utils — metrics, validators, and synthetic data generator."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dscompanion.utils.metrics import DECILE_TABLE_COLUMNS, decile_table, feature_psi_table
from dscompanion.utils.synthetic import SyntheticDataGenerator
from dscompanion.utils.validators import (
    _check_spark_conf,
    is_databricks,
    validate_binary_target,
    validate_dataframe,
)


class TestDecileTable:
    def test_perfect_ranking_top_decile_captures_all_events(self):
        # Scores perfectly rank-order the labels: highest score = positive.
        n = 1000
        y_true = np.array([1] * 100 + [0] * 900)
        y_prob = np.linspace(1.0, 0.0, n)
        df = decile_table(y_true, y_prob)
        assert len(df) == 10
        # All 100 events fall in the top decile (100 rows = top 10%).
        assert df.iloc[0]["events"] == 100
        assert df.iloc[0]["pct_of_total_events"] == pytest.approx(1.0)
        assert df.iloc[0]["cumulative_pct_of_total_events"] == pytest.approx(1.0)

    def test_random_scores_lift_near_one_in_every_decile(self):
        rng = np.random.RandomState(0)
        n = 5000
        y_true = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        y_prob = rng.uniform(size=n)  # uncorrelated with y_true
        df = decile_table(y_true, y_prob, n_bins=10)
        # No skill model: lift should hover near 1.0 in every decile.
        assert (df["lift"] - 1.0).abs().mean() < 0.25

    def test_decile_column_is_one_indexed_and_sequential(self):
        rng = np.random.RandomState(1)
        y_true = rng.choice([0, 1], size=500)
        y_prob = rng.uniform(size=500)
        df = decile_table(y_true, y_prob)
        assert df["decile"].tolist() == list(range(1, 11))

    def test_cumulative_count_sums_to_total(self):
        rng = np.random.RandomState(2)
        n = 777
        y_true = rng.choice([0, 1], size=n, p=[0.6, 0.4])
        y_prob = rng.uniform(size=n)
        df = decile_table(y_true, y_prob)
        assert df["cumulative_count"].iloc[-1] == n
        assert df["count"].sum() == n

    def test_cumulative_events_sums_to_total_events(self):
        rng = np.random.RandomState(3)
        n = 1200
        y_true = rng.choice([0, 1], size=n, p=[0.8, 0.2])
        y_prob = rng.uniform(size=n)
        df = decile_table(y_true, y_prob)
        assert df["cumulative_events"].iloc[-1] == y_true.sum()
        assert df["cumulative_pct_of_total_events"].iloc[-1] == pytest.approx(1.0)

    def test_empty_input_returns_correctly_columned_frame(self):
        df = decile_table(np.array([]), np.array([]))
        assert df.empty
        assert list(df.columns) == DECILE_TABLE_COLUMNS

    def test_no_events_returns_correctly_columned_frame(self):
        df = decile_table(np.zeros(100), np.random.uniform(size=100))
        assert df.empty
        assert list(df.columns) == DECILE_TABLE_COLUMNS

    def test_fewer_rows_than_bins_still_works(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
        df = decile_table(y_true, y_prob, n_bins=10)
        assert len(df) <= 5
        assert df["count"].sum() == 5
        assert df["events"].sum() == 3

    def test_custom_n_bins(self):
        rng = np.random.RandomState(4)
        y_true = rng.choice([0, 1], size=1000, p=[0.5, 0.5])
        y_prob = rng.uniform(size=1000)
        df = decile_table(y_true, y_prob, n_bins=5)
        assert len(df) == 5
        assert df["decile"].tolist() == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# feature_psi_table
# ---------------------------------------------------------------------------


class TestFeaturePsiTable:
    """Tests for feature_psi_table — per-column PSI between two DataFrames."""

    def _numeric_df(self, rng, n=500, shift=0.0):
        return pd.DataFrame(
            {
                "f1": rng.normal(shift, 1, n),
                "f2": rng.uniform(shift, 1 + shift, n),
            }
        )

    def test_schema(self):
        rng = np.random.RandomState(0)
        ref = self._numeric_df(rng)
        cmp = self._numeric_df(rng)
        result = feature_psi_table(ref, cmp)
        assert list(result.columns) == ["feature", "psi", "flag"]

    def test_identical_distributions_psi_near_zero(self):
        rng = np.random.RandomState(0)
        ref = self._numeric_df(rng, n=2000)
        result = feature_psi_table(ref, ref.copy())
        assert (result["psi"] < 0.05).all()
        assert not result["flag"].any()

    def test_large_shift_raises_flag(self):
        rng = np.random.RandomState(0)
        ref = pd.DataFrame({"f1": rng.normal(0, 1, 2000)})
        cmp = pd.DataFrame({"f1": rng.normal(5, 1, 2000)})
        result = feature_psi_table(ref, cmp)
        assert result.loc[result["feature"] == "f1", "flag"].values[0]

    def test_categorical_columns_included(self):
        rng = np.random.RandomState(0)
        ref = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], 500)})
        cmp = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], 500)})
        result = feature_psi_table(ref, cmp)
        assert "cat" in result["feature"].tolist()

    def test_empty_frames_return_empty_result(self):
        result = feature_psi_table(pd.DataFrame(), pd.DataFrame())
        assert result.empty
        assert list(result.columns) == ["feature", "psi", "flag"]

    def test_no_shared_columns_returns_empty(self):
        ref = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        cmp = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
        result = feature_psi_table(ref, cmp)
        assert result.empty

    def test_sorted_by_psi_descending(self):
        rng = np.random.RandomState(0)
        ref = pd.DataFrame(
            {
                "stable": rng.normal(0, 1, 1000),
                "shifted": rng.normal(0, 1, 1000),
            }
        )
        cmp = pd.DataFrame(
            {
                "stable": rng.normal(0, 1, 1000),
                "shifted": rng.normal(10, 1, 1000),  # large shift → high PSI
            }
        )
        result = feature_psi_table(ref, cmp)
        assert result.iloc[0]["feature"] == "shifted"

    def test_type_error_on_non_dataframe(self):
        with pytest.raises(TypeError):
            feature_psi_table(np.array([1, 2, 3]), pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(TypeError):
            feature_psi_table(pd.DataFrame({"a": [1, 2, 3]}), [1, 2, 3])


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------


class TestValidateDataframe:
    def test_passes_silently_when_all_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        validate_dataframe(df, ["a", "b"])  # must not raise

    def test_raises_type_error_for_non_dataframe(self):
        with pytest.raises(TypeError):
            validate_dataframe([1, 2, 3], [])

    def test_raises_value_error_for_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(pd.DataFrame({"a": []}), [])

    def test_raises_value_error_for_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df, ["a", "b"])

    def test_error_message_includes_custom_name(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="training set"):
            validate_dataframe(df, ["missing"], name="training set")

    def test_empty_required_cols_passes(self):
        df = pd.DataFrame({"a": [1]})
        validate_dataframe(df, [])  # must not raise


# ---------------------------------------------------------------------------
# validate_binary_target
# ---------------------------------------------------------------------------


class TestValidateBinaryTarget:
    def test_passes_for_binary_series(self):
        validate_binary_target(pd.Series([0, 1, 0, 1]))  # must not raise

    def test_passes_when_nan_present(self):
        validate_binary_target(pd.Series([0, 1, float("nan")]))  # NaN ignored

    def test_raises_type_error_for_non_series(self):
        with pytest.raises(TypeError):
            validate_binary_target([0, 1, 0])

    def test_raises_for_empty_series(self):
        with pytest.raises(ValueError, match="empty"):
            validate_binary_target(pd.Series([], dtype=float))

    def test_raises_for_single_class(self):
        with pytest.raises(ValueError, match="2 unique values"):
            validate_binary_target(pd.Series([0, 0, 0]))

    def test_raises_for_three_classes(self):
        with pytest.raises(ValueError, match="2 unique values"):
            validate_binary_target(pd.Series([0, 1, 2]))


# ---------------------------------------------------------------------------
# is_databricks
# ---------------------------------------------------------------------------


class TestEnvironmentHelpers:
    def test_is_databricks_returns_bool(self):
        result = is_databricks()
        assert isinstance(result, bool)

    def test_is_databricks_false_outside_cluster(self):
        assert is_databricks() is False

    def test_check_spark_conf_returns_false_when_no_active_session(self):
        mock_spark_cls = MagicMock()
        mock_spark_cls.getActiveSession.return_value = None
        mock_pyspark = MagicMock()
        mock_pyspark.sql.SparkSession = mock_spark_cls
        with patch.dict(sys.modules, {"pyspark": mock_pyspark, "pyspark.sql": mock_pyspark.sql}):
            assert _check_spark_conf() is False

    def test_check_spark_conf_returns_false_on_import_error(self):
        with patch.dict(sys.modules, {"pyspark": None, "pyspark.sql": None}):
            assert _check_spark_conf() is False


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------


class TestSyntheticDataGenerator:
    def test_generate_returns_dataframe_with_correct_shape(self):
        gen = SyntheticDataGenerator(n_rows=200, n_numeric=10, n_categorical=5, random_state=0)
        df = gen.generate()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (200, 15)

    def test_column_names_use_expected_prefixes(self):
        gen = SyntheticDataGenerator(
            n_rows=100,
            n_numeric=4,
            n_categorical=2,
            random_state=1,
            numeric_distribution={"high_null": 1, "low_variance": 1, "constant": 1, "normal": 1},
            categorical_distribution={
                "single_value": 0,
                "high_cardinality": 1,
                "dominant_category": 0,
                "high_null": 0,
                "normal": 1,
            },
        )
        df = gen.generate()
        col_names = df.columns.tolist()
        assert any(c.startswith("num_high_null") for c in col_names)
        assert any(c.startswith("num_low_var") for c in col_names)
        assert any(c.startswith("num_constant") for c in col_names)
        assert any(c.startswith("num_normal") for c in col_names)
        assert any(c.startswith("cat_high_card") for c in col_names)

    def test_high_null_numeric_columns_have_many_nans(self):
        gen = SyntheticDataGenerator(
            n_rows=1000,
            n_numeric=3,
            n_categorical=0,
            random_state=2,
            numeric_distribution={"high_null": 3, "low_variance": 0, "constant": 0, "normal": 0},
            categorical_distribution={
                "single_value": 0,
                "high_cardinality": 0,
                "dominant_category": 0,
                "high_null": 0,
                "normal": 0,
            },
        )
        df = gen.generate()
        for col in df.columns:
            assert df[col].isna().mean() >= 0.3

    def test_constant_columns_have_zero_variance(self):
        gen = SyntheticDataGenerator(
            n_rows=500,
            n_numeric=2,
            n_categorical=0,
            random_state=3,
            numeric_distribution={"high_null": 0, "low_variance": 0, "constant": 2, "normal": 0},
            categorical_distribution={
                "single_value": 0,
                "high_cardinality": 0,
                "dominant_category": 0,
                "high_null": 0,
                "normal": 0,
            },
        )
        df = gen.generate()
        for col in df.columns:
            assert df[col].std() == pytest.approx(0.0)

    def test_get_metadata_keys_and_types(self):
        gen = SyntheticDataGenerator(n_rows=100, n_numeric=5, n_categorical=3, random_state=4)
        meta = gen.get_metadata()
        assert meta["n_rows"] == 100
        assert meta["n_numeric"] == 5
        assert meta["n_categorical"] == 3
        assert meta["total_features"] == 8
        assert meta["random_state"] == 4
        assert isinstance(meta["numeric_distribution"], dict)
        assert isinstance(meta["categorical_distribution"], dict)

    def test_same_random_state_produces_identical_output(self):
        gen_a = SyntheticDataGenerator(n_rows=200, n_numeric=5, n_categorical=3, random_state=7)
        gen_b = SyntheticDataGenerator(n_rows=200, n_numeric=5, n_categorical=3, random_state=7)
        pd.testing.assert_frame_equal(gen_a.generate(), gen_b.generate())

    def test_explicit_numeric_distribution_must_match_n_numeric(self):
        with pytest.raises(ValueError, match="numeric_distribution"):
            SyntheticDataGenerator(
                n_rows=100,
                n_numeric=5,
                n_categorical=0,
                numeric_distribution={
                    "high_null": 1,
                    "low_variance": 0,
                    "constant": 0,
                    "normal": 3,
                },
                categorical_distribution={
                    "single_value": 0,
                    "high_cardinality": 0,
                    "dominant_category": 0,
                    "high_null": 0,
                    "normal": 0,
                },
            )

    def test_categorical_distribution_all_types(self):
        gen = SyntheticDataGenerator(
            n_rows=300,
            n_numeric=0,
            n_categorical=5,
            random_state=5,
            numeric_distribution={"high_null": 0, "low_variance": 0, "constant": 0, "normal": 0},
            categorical_distribution={
                "single_value": 1,
                "high_cardinality": 1,
                "dominant_category": 1,
                "high_null": 1,
                "normal": 1,
            },
        )
        df = gen.generate()
        assert df.shape == (300, 5)
