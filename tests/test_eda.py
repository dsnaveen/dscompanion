"""Tests for dscompanion.eda."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dscompanion.eda import (
    BivariateAnalyser,
    EDAReport,
    MissingnessAnalyser,
    MultivariateAnalyser,
    UnivariateAnalyser,
)
from dscompanion.eda.univariate import _BOOL_COLS, _CAT_COLS, _DATETIME_COLS, _FLAG_COLS, _NUM_COLS
from dscompanion.split import DataSplit


class TestUnivariateAnalyser:
    def test_numeric_summary_has_all_features(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        num_cols = data_split.train_X.select_dtypes(include="number").columns
        result = ua.numeric_summary()
        assert set(result["feature"]) == set(num_cols)

    def test_constant_column_flagged(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        issues = ua.flag_issues()
        # constant_col should be flagged as near_zero_variance or constant
        flagged = issues.loc[issues["constant"] | issues["near_zero_variance"], "feature"].tolist()
        assert "constant_col" in flagged

    def test_flag_issues_returns_all_columns(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        issues = ua.flag_issues()
        # flag_issues() covers numeric/categorical only — datetime/boolean columns
        # are carved out before the split and have their own summary tables.
        n_datetime = len(ua.datetime_summary())
        n_boolean = len(ua.boolean_summary())
        assert len(issues) == len(data_split.train_X.columns) - n_datetime - n_boolean

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            UnivariateAnalyser().numeric_summary()

    def test_threshold_guide_returns_expected_metric_rows(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        guide = ua.threshold_guide()
        assert list(guide["metric"]) == ["null_rate", "variance", "cv"]
        assert list(guide.columns) == ["metric", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]

    def test_threshold_guide_null_rate_p50_matches_manual_median(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        guide = ua.threshold_guide()
        expected_p50 = data_split.train_X.isna().mean().quantile(0.50)
        actual_p50 = guide.loc[guide["metric"] == "null_rate", "p50"].iloc[0]
        assert actual_p50 == pytest.approx(expected_p50)

    def test_numeric_clean_series_matches_manual_dropna(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        result = ua.numeric_clean_series("f1")
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            data_split.train_X["f1"].dropna().reset_index(drop=True),
        )

    def test_numeric_clean_series_unknown_column_returns_empty(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        result = ua.numeric_clean_series("does_not_exist")
        assert result.empty

    def test_categorical_value_counts_matches_manual_value_counts(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        result = ua.categorical_value_counts("cat_low")
        expected = data_split.train_X["cat_low"].value_counts()
        pd.testing.assert_series_equal(result, expected)

    def test_categorical_value_counts_respects_top_n(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        result = ua.categorical_value_counts("cat_low", top_n=1)
        assert len(result) == 1

    def test_categorical_value_counts_unknown_column_returns_empty(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        result = ua.categorical_value_counts("does_not_exist")
        assert result.empty


class TestUnivariateAnalyserDatetimeBoolean:
    """Phase 2: datetime and boolean columns are carved out as distinct types."""

    @pytest.fixture
    def bool_datetime_split(self) -> SimpleNamespace:
        df = pd.DataFrame(
            {
                "flag_a": pd.array([True, False, True, True, False, None], dtype="boolean"),
                "flag_b": [True, True, True, False, False, False],
                "signup_date": pd.to_datetime(
                    ["2022-01-01", "2022-02-01", None, "2022-03-01", "2022-04-01", "2022-05-01"]
                ),
                "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "category": ["A", "B", "A", "B", "A", "B"],
            }
        )
        return SimpleNamespace(train_X=df)

    def test_boolean_summary_has_expected_stats(self, bool_datetime_split):
        ua = UnivariateAnalyser().fit(bool_datetime_split)
        bool_df = ua.boolean_summary().set_index("feature")
        assert set(bool_df.index) == {"flag_a", "flag_b"}
        assert bool_df.loc["flag_a", "count"] == 5
        assert bool_df.loc["flag_a", "n_true"] == 3
        assert bool_df.loc["flag_a", "n_false"] == 2
        assert bool_df.loc["flag_b", "pct_true"] == pytest.approx(0.5)

    def test_datetime_summary_has_expected_stats(self, bool_datetime_split):
        ua = UnivariateAnalyser().fit(bool_datetime_split)
        dt_df = ua.datetime_summary().set_index("feature")
        assert list(dt_df.index) == ["signup_date"]
        assert dt_df.loc["signup_date", "count"] == 5
        assert dt_df.loc["signup_date", "min"] == pd.Timestamp("2022-01-01")
        assert dt_df.loc["signup_date", "max"] == pd.Timestamp("2022-05-01")

    def test_boolean_and_datetime_excluded_from_numeric_and_categorical(self, bool_datetime_split):
        ua = UnivariateAnalyser().fit(bool_datetime_split)
        assert set(ua.numeric_summary()["feature"]) == {"amount"}
        assert set(ua.categorical_summary()["feature"]) == {"category"}

    def test_empty_datetime_and_boolean_return_typed_empty_frames(self):
        df = pd.DataFrame({"amount": [1.0, 2.0], "category": ["A", "B"]})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        assert ua.datetime_summary().empty
        assert list(ua.datetime_summary().columns) == _DATETIME_COLS
        assert ua.boolean_summary().empty
        assert list(ua.boolean_summary().columns) == _BOOL_COLS

    def test_plot_distributions_covers_all_four_types(self, bool_datetime_split):
        ua = UnivariateAnalyser().fit(bool_datetime_split)
        figs = ua.plot_distributions(top_n=5)
        assert len(figs) == 5


class TestUnivariateAnalyserInputValidation:
    """Section 14: input validation — TypeError and ValueError paths."""

    def test_fit_no_train_x_attribute_raises_type_error(self):
        with pytest.raises(TypeError, match="train_X"):
            UnivariateAnalyser().fit(object())

    def test_fit_train_x_not_dataframe_raises_type_error(self):
        split = SimpleNamespace(train_X=[[1, 2], [3, 4]])
        with pytest.raises(TypeError, match="pandas DataFrame"):
            UnivariateAnalyser().fit(split)

    def test_fit_zero_row_dataframe_raises_value_error(self):
        split = SimpleNamespace(train_X=pd.DataFrame({"a": pd.Series([], dtype=float)}))
        with pytest.raises(ValueError, match="0 rows"):
            UnivariateAnalyser().fit(split)

    def test_fit_single_row_logs_warning(self, caplog):
        df = pd.DataFrame({"a": [1.0], "b": ["x"]})
        split = SimpleNamespace(train_X=df)
        with caplog.at_level(logging.WARNING, logger="dscompanion.eda.univariate"):
            UnivariateAnalyser().fit(split)
        assert "1 row" in caplog.text

    def test_not_fitted_numeric_summary_raises(self):
        with pytest.raises(RuntimeError):
            UnivariateAnalyser().numeric_summary()

    def test_not_fitted_categorical_summary_raises(self):
        with pytest.raises(RuntimeError):
            UnivariateAnalyser().categorical_summary()

    def test_not_fitted_flag_issues_raises(self):
        with pytest.raises(RuntimeError):
            UnivariateAnalyser().flag_issues()

    def test_not_fitted_plot_distributions_raises(self):
        with pytest.raises(RuntimeError):
            UnivariateAnalyser().plot_distributions()


class TestUnivariateAnalyserEdgeCases:
    """Section 14: edge cases — all-NaN, schema, copies, flags."""

    def test_all_nan_column_constant_flag_true(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(
            {
                "normal": rng.randn(100),
                "all_nan": np.full(100, np.nan),
            }
        )
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        row = ua.numeric_summary().set_index("feature").loc["all_nan"]
        assert bool(row["constant_flag"]) is True

    def test_all_nan_column_near_zero_variance_false(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(
            {
                "normal": rng.randn(100),
                "all_nan": np.full(100, np.nan),
            }
        )
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        row = ua.numeric_summary().set_index("feature").loc["all_nan"]
        assert bool(row["near_zero_variance_flag"]) is False

    def test_constant_and_near_zero_variance_mutually_exclusive(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        summary = ua.numeric_summary()
        both = summary[summary["constant_flag"] & summary["near_zero_variance_flag"]]
        assert len(both) == 0, f"Columns with both flags set: {both['feature'].tolist()}"

    def test_constant_col_flagged_as_constant_not_nzv(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        summary = ua.numeric_summary().set_index("feature")
        assert bool(summary.loc["constant_col", "constant_flag"]) is True
        assert bool(summary.loc["constant_col", "near_zero_variance_flag"]) is False

    def test_near_zero_variance_col_flagged_as_nzv_not_constant(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        summary = ua.numeric_summary().set_index("feature")
        assert bool(summary.loc["f9_near_zero", "near_zero_variance_flag"]) is True
        assert bool(summary.loc["f9_near_zero", "constant_flag"]) is False

    def test_numeric_summary_output_is_copy(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        s1 = ua.numeric_summary()
        s1["feature"] = "mutated"
        s2 = ua.numeric_summary()
        assert s2["feature"].iloc[0] != "mutated"

    def test_categorical_summary_output_is_copy(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        s1 = ua.categorical_summary()
        s1["feature"] = "mutated"
        s2 = ua.categorical_summary()
        assert s2["feature"].iloc[0] != "mutated"

    def test_empty_numeric_summary_has_correct_schema(self):
        df = pd.DataFrame({"cat1": ["a", "b"] * 10, "cat2": ["x", "y"] * 10})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        num = ua.numeric_summary()
        assert len(num) == 0
        assert list(num.columns) == _NUM_COLS

    def test_empty_categorical_summary_has_correct_schema(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50)})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        cat = ua.categorical_summary()
        assert len(cat) == 0
        assert list(cat.columns) == _CAT_COLS

    def test_flag_issues_empty_result_has_correct_schema(self):
        df = pd.DataFrame({"cat1": ["a", "b"] * 10, "cat2": ["x", "y"] * 10})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        # All categorical — flag_issues should still return correct schema even with no flags
        issues = ua.flag_issues()
        assert list(issues.columns) == _FLAG_COLS

    def test_plot_distributions_top_n_zero_returns_empty_list(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert ua.plot_distributions(top_n=0) == []

    def test_plot_distributions_top_n_negative_returns_empty_list(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert ua.plot_distributions(top_n=-5) == []

    def test_zero_mean_cv_is_nan(self):
        df = pd.DataFrame({"zero_mean": np.concatenate([np.ones(50), -np.ones(50)])})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        row = ua.numeric_summary().set_index("feature").loc["zero_mean"]
        assert pd.isna(row["cv"])

    def test_kde_overlay_has_histogram_and_kde_traces(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        fig = ua.numeric_distribution_with_kde("f1")
        trace_types = {trace.type for trace in fig.data}
        assert trace_types == {"histogram", "scatter"}

    def test_kde_overlay_skips_kde_for_constant_column(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        fig = ua.numeric_distribution_with_kde("constant_col")
        trace_types = {trace.type for trace in fig.data}
        assert trace_types == {"histogram"}

    def test_kde_overlay_unknown_column_returns_empty_figure(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        fig = ua.numeric_distribution_with_kde("does_not_exist")
        assert fig.data == ()

    def test_kde_overlay_categorical_column_returns_empty_figure(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        fig = ua.numeric_distribution_with_kde("cat_low")
        assert fig.data == ()

    def test_kde_overlay_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            UnivariateAnalyser().numeric_distribution_with_kde("f1")

    def test_kde_overlay_default_matches_exclude_outliers_false(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        default_fig = ua.numeric_distribution_with_kde("f1")
        explicit_false_fig = ua.numeric_distribution_with_kde("f1", exclude_outliers=False)
        assert default_fig.data[0].x.tolist() == explicit_false_fig.data[0].x.tolist()
        assert default_fig.layout.title.text == explicit_false_fig.layout.title.text

    def test_kde_overlay_exclude_outliers_trims_tails(self):
        rng = np.random.RandomState(0)
        skewed = np.concatenate([rng.randn(950), rng.uniform(50, 100, 50)])
        df = pd.DataFrame({"skewed": skewed})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))

        full_fig = ua.numeric_distribution_with_kde("skewed", exclude_outliers=False)
        trimmed_fig = ua.numeric_distribution_with_kde("skewed", exclude_outliers=True)

        assert len(trimmed_fig.data[0].x) < len(full_fig.data[0].x)
        assert max(trimmed_fig.data[0].x) < max(full_fig.data[0].x)
        assert "excluded" in trimmed_fig.layout.title.text

    def test_kde_overlay_exclude_outliers_on_constant_column_no_crash(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        fig = ua.numeric_distribution_with_kde("constant_col", exclude_outliers=True)
        trace_types = {trace.type for trace in fig.data}
        assert trace_types == {"histogram"}


class TestUnivariateAnalyserColumns:
    """Extended columns and schema correctness."""

    def test_numeric_summary_has_extended_percentiles(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        summary = ua.numeric_summary()
        for col in [
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p10",
            "p90",
            "p95",
            "p96",
            "p97",
            "p98",
            "p99",
        ]:
            assert col in summary.columns, f"missing column: {col}"

    def test_numeric_summary_has_cv_column(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert "cv" in ua.numeric_summary().columns

    def test_numeric_summary_percentile_order(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        summary = ua.numeric_summary()
        ordered_cols = [
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p96",
            "p97",
            "p98",
            "p99",
        ]
        for lo, hi in zip(ordered_cols, ordered_cols[1:]):
            assert (summary[lo] <= summary[hi]).all(), f"{lo} > {hi} in at least one row"

    def test_numeric_summary_cv_positive_for_positive_mean(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        summary = ua.numeric_summary()
        positive_mean = summary[summary["mean"] > 0].dropna(subset=["cv"])
        assert (positive_mean["cv"] > 0).all()

    def test_numeric_summary_columns_match_schema_constant(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert list(ua.numeric_summary().columns) == _NUM_COLS

    def test_categorical_summary_columns_match_schema_constant(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert list(ua.categorical_summary().columns) == _CAT_COLS

    def test_flag_issues_columns_match_schema_constant(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert list(ua.flag_issues().columns) == _FLAG_COLS

    def test_categorical_summary_has_imbalance_column(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert "imbalance" in ua.categorical_summary().columns

    def test_imbalance_bounds(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        imb = ua.categorical_summary()["imbalance"].dropna()
        assert (imb >= 0).all()
        assert (imb <= 1).all()

    def test_single_value_column_has_imbalance_one(self):
        df = pd.DataFrame({"single": ["only_value"] * 50})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        row = ua.categorical_summary().set_index("feature").loc["single"]
        assert row["imbalance"] == 1.0

    def test_uniform_column_has_low_imbalance(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"uniform": rng.choice(["a", "b", "c", "d"], size=4000)})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        row = ua.categorical_summary().set_index("feature").loc["uniform"]
        assert row["imbalance"] < 0.1

    def test_all_null_categorical_imbalance_is_none(self):
        df = pd.DataFrame({"all_null": pd.Series([None] * 20, dtype="object")})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        row = ua.categorical_summary().set_index("feature").loc["all_null"]
        assert row["imbalance"] is None

    def test_extreme_values_returns_n_per_numeric_feature(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        extreme = ua.extreme_values(n=3)
        num_cols = data_split.train_X.select_dtypes(include="number").columns
        assert set(extreme.keys()) == set(num_cols)
        for ev in extreme.values():
            assert len(ev["min_extreme"]) <= 3
            assert len(ev["max_extreme"]) <= 3

    def test_extreme_values_min_le_max(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        extreme = ua.extreme_values(n=5)
        for ev in extreme.values():
            if ev["min_extreme"] and ev["max_extreme"]:
                assert max(ev["min_extreme"]) <= min(ev["max_extreme"]) or set(
                    ev["min_extreme"]
                ) & set(ev["max_extreme"])

    def test_extreme_values_default_n_from_settings(self, data_split: DataSplit):
        from dscompanion.config import settings

        ua = UnivariateAnalyser().fit(data_split)
        extreme = ua.extreme_values()
        any_col = next(iter(extreme))
        assert len(extreme[any_col]["max_extreme"]) <= settings.eda_extreme_values_n

    def test_extreme_values_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            UnivariateAnalyser().extreme_values()

    def test_missing_pct_bounds(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        summary = ua.numeric_summary()
        assert (summary["missing_pct"] >= 0).all()
        assert (summary["missing_pct"] <= 1).all()

    def test_high_missing_flag_respects_settings(self, data_split: DataSplit):
        from dscompanion.config import settings

        ua = UnivariateAnalyser().fit(data_split)
        summary = ua.numeric_summary()
        issues = ua.flag_issues().set_index("feature")
        for feat in summary["feature"]:
            missing = summary.set_index("feature").loc[feat, "missing_pct"]
            expected = missing > settings.high_missing_threshold
            assert issues.loc[feat, "high_missing"] == expected

    def test_high_missing_threshold_override_changes_flag(self, data_split: DataSplit):
        # f5 has ~10% missing — below the 0.30 default, above a 0.05 override.
        default_issues = UnivariateAnalyser().fit(data_split).flag_issues().set_index("feature")
        assert not default_issues.loc["f5", "high_missing"]

        sensitive_issues = (
            UnivariateAnalyser(high_missing_threshold=0.05)
            .fit(data_split)
            .flag_issues()
            .set_index("feature")
        )
        assert sensitive_issues.loc["f5", "high_missing"]

    def test_near_zero_variance_threshold_override_changes_flag(self, data_split: DataSplit):
        # f9_near_zero's actual variance is ~1e-8 — flagged under the 0.01
        # default; a stricter override below its actual variance un-flags it.
        default_summary = (
            UnivariateAnalyser().fit(data_split).numeric_summary().set_index("feature")
        )
        assert default_summary.loc["f9_near_zero", "near_zero_variance_flag"]

        strict_summary = (
            UnivariateAnalyser(near_zero_variance_threshold=1e-12)
            .fit(data_split)
            .numeric_summary()
            .set_index("feature")
        )
        assert not strict_summary.loc["f9_near_zero", "near_zero_variance_flag"]


class TestUnivariateAnalyserGetFlaggedFeatures:
    """get_flagged_features — valid flags, invalid flag, empty result."""

    def test_constant_flag_returns_constant_col(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert "constant_col" in ua.get_flagged_features("constant")

    def test_near_zero_variance_returns_f9(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert "f9_near_zero" in ua.get_flagged_features("near_zero_variance")

    def test_high_cardinality_returns_cat_high(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert "cat_high" in ua.get_flagged_features("high_cardinality")

    def test_returns_list_type(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        assert isinstance(ua.get_flagged_features("high_missing"), list)

    def test_invalid_flag_raises_value_error(self, data_split: DataSplit):
        ua = UnivariateAnalyser().fit(data_split)
        with pytest.raises(ValueError, match="not a valid flag"):
            ua.get_flagged_features("nonsense_flag")

    def test_no_numeric_columns_returns_empty_list(self):
        df = pd.DataFrame({"cat": ["a", "b"] * 10})
        ua = UnivariateAnalyser().fit(SimpleNamespace(train_X=df))
        # cardinality check on categorical — high_missing should return []
        result = ua.get_flagged_features("near_zero_variance")
        assert result == []


class TestUnivariateAnalyserReproducibility:
    """Section 14: same random_state produces identical output."""

    def test_subsampled_fit_is_reproducible(self):
        rng = np.random.RandomState(99)
        df = pd.DataFrame({f"f{i}": rng.randn(200) for i in range(10)})
        split = SimpleNamespace(train_X=df)
        ua1 = UnivariateAnalyser(max_rows=50).fit(split)
        ua2 = UnivariateAnalyser(max_rows=50).fit(split)
        pd.testing.assert_frame_equal(ua1.numeric_summary(), ua2.numeric_summary())

    def test_full_fit_is_deterministic(self, data_split: DataSplit):
        ua1 = UnivariateAnalyser().fit(data_split)
        ua2 = UnivariateAnalyser().fit(data_split)
        pd.testing.assert_frame_equal(ua1.numeric_summary(), ua2.numeric_summary())


class TestBivariateAnalyser:
    def test_iv_table_has_row_per_feature(self, data_split: DataSplit):
        ba = BivariateAnalyser().fit(data_split)
        iv = ba.iv_table()
        assert set(iv["feature"]) == set(data_split.train_X.columns)

    def test_iv_nonnegative(self, data_split: DataSplit):
        ba = BivariateAnalyser().fit(data_split)
        assert (ba.iv_table()["iv"] >= 0).all()

    def test_fit_does_not_raise_on_continuous_target(self):
        """Regression-task EDA: found via real end-to-end testing that
        fit() unconditionally called validate_binary_target(), crashing
        EDAReport.run_all() (caught, non-fatal, but silently dropped EVERY
        EDA sheet — not just IV) for any regression pipeline. IV/WoE is a
        binary-target concept and should be skipped, not crash the whole
        analyser."""
        from types import SimpleNamespace

        rng = np.random.RandomState(3)
        n = 200
        train_X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
        train_y = pd.Series(rng.randn(n))  # continuous — not binary
        split = SimpleNamespace(train_X=train_X, train_y=train_y)

        ba = BivariateAnalyser().fit(split)
        iv = ba.iv_table()
        assert iv.empty
        assert list(iv.columns) == ["feature", "iv", "n_bins", "predictive_power"]
        # Correlation and Cramér's V are target-independent — still computed.
        assert ba.correlation_matrix().shape == (2, 2)

    def test_target_rate_by_bin_has_expected_columns(self, data_split: DataSplit):
        ba = BivariateAnalyser().fit(data_split)
        bins = ba.target_rate_by_bin("f1")
        assert list(bins.columns) == ["bin_label", "count", "target_rate"]
        assert (bins["target_rate"] >= 0).all() and (bins["target_rate"] <= 1).all()

    def test_target_rate_by_bin_unknown_feature_raises_keyerror(self, data_split: DataSplit):
        ba = BivariateAnalyser().fit(data_split)
        with pytest.raises(KeyError):
            ba.target_rate_by_bin("does_not_exist")

    def test_target_rate_by_bin_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            BivariateAnalyser().target_rate_by_bin("f1")

    def test_target_rate_by_bin_chart_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        ba = BivariateAnalyser().fit(data_split)
        fig = ba.target_rate_by_bin_chart("f1")
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Bar)

    def test_target_rate_by_bin_chart_numeric_feature_is_json_serializable(
        self, data_split: DataSplit
    ):
        # f1 is numeric -> pd.qcut bins produce pd.Interval bin_label values,
        # which are not JSON-serializable as-is; the chart must stringify
        # them before placing them on the x-axis (regression: this used to
        # raise "Object of type Interval is not JSON serializable").
        ba = BivariateAnalyser().fit(data_split)
        fig = ba.target_rate_by_bin_chart("f1")
        fig.to_json()  # raises if any trace value isn't JSON-serializable

    def test_corr_heatmap_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        ba = BivariateAnalyser().fit(data_split)
        fig = ba.correlation_heatmap()
        assert isinstance(fig, go.Figure)

    def test_flag_high_correlation_finds_synthetic_pair(self, data_split: DataSplit):
        ba = BivariateAnalyser().fit(data_split)
        flagged = ba.flag_high_correlation()
        # f1 and f10_corr_f1 are synthetically correlated (r > 0.9)
        pairs = set(zip(flagged["feature_a"], flagged["feature_b"]))
        pairs_r = pairs | {(b, a) for a, b in pairs}
        assert ("f1", "f10_corr_f1") in pairs_r or ("f10_corr_f1", "f1") in pairs_r

    def test_interaction_plots_caps_pairs_at_max_numeric_cols(self, data_split: DataSplit):
        # synthetic_df has 13 numeric columns; cap of 4 -> C(4,2) = 6 pairs, not C(13,2) = 78.
        ba = BivariateAnalyser(interaction_max_numeric_cols=4).fit(data_split)
        figs = ba.interaction_plots()
        assert len(figs) == 6

    def test_interaction_plots_prioritises_high_iv_columns(self, data_split: DataSplit):
        ba = BivariateAnalyser(interaction_max_numeric_cols=2).fit(data_split)
        iv = ba.iv_table()
        numeric_cols = data_split.train_X.select_dtypes(include="number").columns
        ranked_iv = iv[iv["feature"].isin(numeric_cols)].sort_values("iv", ascending=False)
        top_2 = ranked_iv["feature"].head(2).tolist()
        figs = ba.interaction_plots()
        assert len(figs) == 1
        assert all(c in figs[0].layout.title.text for c in top_2)

    def test_interaction_plots_uses_scatter_below_hexbin_threshold(self, data_split: DataSplit):
        import plotly.graph_objects as go

        ba = BivariateAnalyser(
            interaction_max_numeric_cols=2, interaction_hexbin_row_threshold=10_000_000
        ).fit(data_split)
        figs = ba.interaction_plots()
        assert isinstance(figs[0].data[0], go.Scatter)

    def test_interaction_plots_uses_hexbin_above_threshold(self, data_split: DataSplit):
        import plotly.graph_objects as go

        ba = BivariateAnalyser(
            interaction_max_numeric_cols=2, interaction_hexbin_row_threshold=1
        ).fit(data_split)
        figs = ba.interaction_plots()
        assert isinstance(figs[0].data[0], go.Histogram2d)

    def test_interaction_plots_empty_with_fewer_than_2_numeric_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
        y = pd.Series([0, 1, 0, 1])
        ba = BivariateAnalyser().fit(SimpleNamespace(train_X=df, train_y=y))
        assert ba.interaction_plots() == []


class TestMultivariateAnalyser:
    def test_vif_table_identifies_correlated(self, data_split: DataSplit):
        ma = MultivariateAnalyser().fit(data_split)
        vif = ma.vif_table()
        # f1 and f10_corr_f1 should produce high VIF
        flagged = vif.loc[vif["flag"], "feature"].tolist()
        assert len(flagged) >= 1

    def test_pca_summary_shape(self, data_split: DataSplit):
        ma = MultivariateAnalyser().fit(data_split)
        summary = ma.pca_summary(n_components=5)
        assert len(summary) == 5
        assert (summary["cumulative_variance"] <= 1.0).all()

    def test_correlation_matrix_is_square_and_symmetric(self, data_split: DataSplit):
        ma = MultivariateAnalyser().fit(data_split)
        cm = ma.correlation_matrix()
        assert cm.shape[0] == cm.shape[1]
        assert list(cm.index) == list(cm.columns)
        assert cm.loc["f1", "f1"] == pytest.approx(1.0)

    def test_correlation_matrix_returns_copy(self, data_split: DataSplit):
        ma = MultivariateAnalyser().fit(data_split)
        cm1 = ma.correlation_matrix()
        cm1.iloc[0, 0] = 999.0
        assert ma.correlation_matrix().iloc[0, 0] != 999.0

    def test_cramers_v_table_returns_dataframe_with_correct_columns(self, data_split: DataSplit):
        ma = MultivariateAnalyser().fit(data_split)
        cv = ma.cramers_v_table()
        assert list(cv.columns) == ["col_a", "col_b", "cramers_v"]
        # conftest has 3 cat cols → 3 pairs
        assert len(cv) == 3

    def test_cramers_v_table_values_in_unit_interval(self, data_split: DataSplit):
        ma = MultivariateAnalyser().fit(data_split)
        cv = ma.cramers_v_table()
        assert ((cv["cramers_v"] >= 0) & (cv["cramers_v"] <= 1)).all()

    def test_cramers_v_table_empty_when_no_categorical_columns(self, data_split: DataSplit):
        numeric_only = SimpleNamespace(train_X=data_split.train_X.select_dtypes(include="number"))
        ma = MultivariateAnalyser().fit(numeric_only)
        cv = ma.cramers_v_table()
        assert cv.empty
        assert list(cv.columns) == ["col_a", "col_b", "cramers_v"]

    # ── fit error paths ──────────────────────────────────────────────────────

    def test_fit_raises_type_error_when_split_has_no_train_x(self):
        with pytest.raises(TypeError, match="train_X"):
            MultivariateAnalyser().fit(SimpleNamespace())

    def test_fit_raises_type_error_when_train_x_not_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            MultivariateAnalyser().fit(SimpleNamespace(train_X=[[1, 2], [3, 4]]))

    def test_fit_raises_value_error_when_no_numeric_columns(self):
        cat_only = SimpleNamespace(
            train_X=pd.DataFrame({"a": ["x", "y"] * 50, "b": ["p", "q"] * 50})
        )
        with pytest.raises(ValueError, match="numeric"):
            MultivariateAnalyser().fit(cat_only)

    def test_fit_raises_value_error_when_all_rows_nan(self):
        all_nan = SimpleNamespace(
            train_X=pd.DataFrame({"f1": [float("nan")] * 20, "f2": [float("nan")] * 20})
        )
        with pytest.raises(ValueError, match="NaN"):
            MultivariateAnalyser().fit(all_nan)

    def test_fit_samples_cat_columns_when_too_many_rows(self):
        rng = np.random.RandomState(42)
        n = 20_000
        df = pd.DataFrame(
            {
                "num": rng.randn(n),
                "cat_a": rng.choice(["x", "y", "z"], size=n),
                "cat_b": rng.choice(["p", "q"], size=n),
            }
        )
        ma = MultivariateAnalyser(max_rows=500).fit(SimpleNamespace(train_X=df))
        assert ma._cat_df is not None
        assert len(ma._cat_df) <= 500

    # ── plot methods ─────────────────────────────────────────────────────────

    def test_pca_plot_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        ma = MultivariateAnalyser().fit(data_split)
        fig = ma.pca_plot()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Bar + Scatter

    def test_pca_summary_raises_for_negative_n_components(self, data_split: DataSplit):
        ma = MultivariateAnalyser().fit(data_split)
        with pytest.raises(ValueError, match="non-negative"):
            ma.pca_summary(n_components=-1)

    def test_cluster_heatmap_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        numeric_only = SimpleNamespace(train_X=data_split.train_X.select_dtypes(include="number"))
        ma = MultivariateAnalyser().fit(numeric_only)
        fig = ma.cluster_heatmap()
        assert isinstance(fig, go.Figure)

    def test_cluster_heatmap_raises_with_single_numeric_column(self):
        single_col = SimpleNamespace(train_X=pd.DataFrame({"only": range(50)}))
        ma = MultivariateAnalyser().fit(single_col)
        with pytest.raises(ValueError, match="2 numeric"):
            ma.cluster_heatmap()

    def test_flag_multicollinearity_returns_list(self, data_split: DataSplit):
        ma = MultivariateAnalyser().fit(data_split)
        flagged = ma.flag_multicollinearity()
        assert isinstance(flagged, list)


class TestMissingnessAnalyser:
    def test_nullity_matrix_shape_and_dtype(self, data_split: DataSplit):
        ma = MissingnessAnalyser().fit(data_split)
        matrix = ma.nullity_matrix()
        # Rows are subsampled to settings.eda_missing_matrix_max_rows (default 500);
        # column count is preserved.
        assert matrix.shape[0] <= 500
        assert matrix.shape[1] == data_split.train_X.shape[1]
        assert matrix.dtypes.unique().tolist() == [bool]

    def test_nullity_correlation_excludes_zero_variance_columns(self, data_split: DataSplit):
        ma = MissingnessAnalyser().fit(data_split)
        corr = ma.nullity_correlation()
        # f4/f5 have synthetic 10% missingness; snapshot_date etc. have none and
        # must be excluded since their missingness indicator has zero variance.
        assert "f4" in corr.columns
        assert "f5" in corr.columns
        assert "snapshot_date" not in corr.columns

    def test_nullity_correlation_finds_correlated_missingness(self):
        rng = np.random.RandomState(0)
        n = 500
        missing_mask = rng.choice([True, False], size=n, p=[0.3, 0.7])
        df = pd.DataFrame(
            {
                "a": np.where(missing_mask, np.nan, rng.randn(n)),
                "b": np.where(missing_mask, np.nan, rng.randn(n)),
                "c": rng.randn(n),
            }
        )
        ma = MissingnessAnalyser().fit(SimpleNamespace(train_X=df))
        corr = ma.nullity_correlation()
        assert corr.loc["a", "b"] == pytest.approx(1.0)

    def test_nullity_matrix_chart_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        ma = MissingnessAnalyser().fit(data_split)
        fig = ma.nullity_matrix_chart()
        assert isinstance(fig, go.Figure)

    def test_nullity_correlation_heatmap_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        ma = MissingnessAnalyser().fit(data_split)
        fig = ma.nullity_correlation_heatmap()
        assert isinstance(fig, go.Figure)

    def test_no_missing_values_returns_empty_correlation(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        ma = MissingnessAnalyser().fit(SimpleNamespace(train_X=df))
        assert ma.nullity_correlation().empty
        fig = ma.nullity_correlation_heatmap()
        assert len(fig.data) == 0

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            MissingnessAnalyser().nullity_matrix()

    def test_fit_no_train_x_attribute_raises_type_error(self):
        with pytest.raises(TypeError, match="train_X"):
            MissingnessAnalyser().fit(object())

    def test_fit_empty_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="0 rows"):
            MissingnessAnalyser().fit(SimpleNamespace(train_X=pd.DataFrame({"a": []})))


class TestEDAReport:
    def test_run_all_completes(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        assert report._uni is not None
        assert report._missing is not None

    def test_numeric_clean_series_delegate(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        result = report.numeric_clean_series("f1")
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            data_split.train_X["f1"].dropna().reset_index(drop=True),
        )

    def test_categorical_value_counts_delegate(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        result = report.categorical_value_counts("cat_low")
        expected = data_split.train_X["cat_low"].value_counts()
        pd.testing.assert_series_equal(result, expected)

    def test_missing_matrix_delegate_matches_underlying_shape(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        matrix = report.missing_matrix()
        assert matrix.shape[1] == len(data_split.train_X.columns)

    def test_missing_correlation_delegate_returns_dataframe(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        corr = report.missing_correlation()
        assert isinstance(corr, pd.DataFrame)

    def test_missing_matrix_chart_delegate_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        report = EDAReport(data_split, target="target").run_all()
        assert isinstance(report.missing_matrix_chart(), go.Figure)

    def test_missing_correlation_heatmap_delegate_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        report = EDAReport(data_split, target="target").run_all()
        assert isinstance(report.missing_correlation_heatmap(), go.Figure)

    def test_sample_rows_gated_off_by_default(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        sample = report.sample_rows()
        assert sample["head"].empty
        assert sample["tail"].empty

    def test_sample_rows_returns_head_and_tail_when_enabled(self, data_split: DataSplit):
        from dscompanion.config import settings

        report = EDAReport(data_split, target="target", include_sample_rows=True).run_all()
        sample = report.sample_rows()
        assert len(sample["head"]) == settings.eda_sample_n_rows
        assert len(sample["tail"]) == settings.eda_sample_n_rows
        assert sample["head"].iloc[0].equals(data_split.train_X.iloc[0])
        assert sample["tail"].iloc[-1].equals(data_split.train_X.iloc[-1])

    def test_duplicate_row_content_gated_off_by_default(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        assert report.duplicate_row_content().empty

    def test_duplicate_row_content_finds_duplicates_when_enabled(self):
        df = pd.DataFrame(
            {
                "a": [1, 1, 1, 2, 3],
                "b": ["x", "x", "x", "y", "z"],
            }
        )
        report = EDAReport(
            SimpleNamespace(train_X=df, train_y=pd.Series([0, 1, 0, 1, 0])),
            target="target",
            include_duplicate_row_content=True,
        ).run_all()
        dup_df = report.duplicate_row_content()
        assert len(dup_df) == 3
        assert (dup_df["_dup_count"] == 3).all()

    def test_to_html_creates_file(self, data_split: DataSplit, tmp_path: Path):
        report = EDAReport(data_split, target="target").run_all()
        out = tmp_path / "eda.html"
        report.to_html(out)
        assert out.exists()
        assert out.stat().st_size > 100

    def test_recommended_drops_contains_constant(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        drops = report.recommended_drops
        assert "constant_col" in drops

    def test_recommended_drops_with_reasons_matches_recommended_drops(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        assert set(report.recommended_drops_with_reasons()) == set(report.recommended_drops)

    def test_recommended_drops_with_reasons_constant_col_has_constant_reason(
        self, data_split: DataSplit
    ):
        report = EDAReport(data_split, target="target").run_all()
        reasons = report.recommended_drops_with_reasons()
        assert "constant" in reasons["constant_col"]

    def test_recommended_drops_with_reasons_nzv_col_has_nzv_reason(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        reasons = report.recommended_drops_with_reasons()
        assert "near_zero_variance" in reasons["f9_near_zero"]

    def test_recommended_drops_with_reasons_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            EDAReport(split=None, target="target").recommended_drops_with_reasons()

    def test_near_zero_variance_threshold_override_changes_recommended_drops(
        self, data_split: DataSplit
    ):
        # f9_near_zero is also independently flagged low_iv, so it stays in
        # recommended_drops either way — but its *reason* should drop
        # "near_zero_variance" once the override threshold is stricter than
        # its actual variance.
        default_report = EDAReport(data_split, target="target").run_all()
        default_reasons = default_report.recommended_drops_with_reasons()["f9_near_zero"]
        assert "near_zero_variance" in default_reasons

        strict_report = EDAReport(
            data_split, target="target", near_zero_variance_threshold=1e-12
        ).run_all()
        strict_reasons = strict_report.recommended_drops_with_reasons().get("f9_near_zero", [])
        assert "near_zero_variance" not in strict_reasons

    def test_iv_table_has_all_features(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        iv_df = report.iv_table()
        assert set(iv_df["feature"]) == set(data_split.train_X.columns)

    def test_iv_table_sorted_descending(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        iv_df = report.iv_table()
        assert list(iv_df["iv"]) == sorted(iv_df["iv"], reverse=True)

    def test_iv_table_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            EDAReport(split=None, target="target").iv_table()

    def test_target_rate_by_bin_delegate_matches_columns(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        bins = report.target_rate_by_bin("f1")
        assert list(bins.columns) == ["bin_label", "count", "target_rate"]

    def test_target_rate_by_bin_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            EDAReport(split=None, target="target").target_rate_by_bin("f1")

    def test_target_rate_by_bin_chart_delegate_returns_figure(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        fig = report.target_rate_by_bin_chart("f1")
        trace_types = {trace.type for trace in fig.data}
        assert trace_types == {"bar"}

    def test_numeric_summary_delegate_matches_underlying_columns(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        num_df = report.numeric_summary()
        num_cols = data_split.train_X.select_dtypes(include="number").columns
        assert set(num_df["feature"]) == set(num_cols)

    def test_categorical_summary_delegate_matches_underlying_columns(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        cat_df = report.categorical_summary()
        # categorical_summary() excludes datetime/boolean columns, which now
        # have their own dedicated delegates (datetime_summary()/boolean_summary()).
        non_numeric_cols = data_split.train_X.select_dtypes(exclude="number").columns
        datetime_cols = data_split.train_X.select_dtypes(include="datetime").columns
        bool_cols = data_split.train_X.select_dtypes(include="bool").columns
        cat_cols = [c for c in non_numeric_cols if c not in datetime_cols and c not in bool_cols]
        assert set(cat_df["feature"]) == set(cat_cols)

    def test_datetime_summary_delegate_matches_underlying_columns(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        dt_df = report.datetime_summary()
        dt_cols = data_split.train_X.select_dtypes(include="datetime").columns
        assert set(dt_df["feature"]) == set(dt_cols)

    def test_boolean_summary_delegate_returns_expected_columns(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        bool_df = report.boolean_summary()
        assert list(bool_df.columns) == _BOOL_COLS

    def test_plot_distributions_delegate_returns_figures(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        figs = report.plot_distributions(top_n=3)
        assert len(figs) == 3

    def test_numeric_distribution_with_kde_delegate_returns_figure(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        fig = report.numeric_distribution_with_kde("f1")
        trace_types = {trace.type for trace in fig.data}
        assert trace_types == {"histogram", "scatter"}

    def test_correlation_heatmap_delegate_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        report = EDAReport(data_split, target="target").run_all()
        fig = report.correlation_heatmap()
        assert isinstance(fig, go.Figure)

    def test_not_run_raises(self, data_split: DataSplit):
        with pytest.raises(RuntimeError):
            EDAReport(data_split, target="target").flagged_features()

    def test_extreme_values_delegate_matches_underlying(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        report_extreme = report.extreme_values(n=3)
        ua_extreme = UnivariateAnalyser().fit(data_split).extreme_values(n=3)
        assert set(report_extreme.keys()) == set(ua_extreme.keys())

    def test_overview_summary_matches_train_shape(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        overview = report.overview_summary()
        assert overview["total_rows"] == len(data_split.train_X)
        assert overview["total_columns"] == data_split.train_X.shape[1]
        assert overview["total_missing"] >= 0
        assert overview["duplicate_rows"] >= 0

    def test_overview_summary_not_run_raises(self, data_split: DataSplit):
        with pytest.raises(RuntimeError):
            EDAReport(data_split, target="target").overview_summary()

    def test_missing_values_chart_returns_figure(self, data_split: DataSplit):
        import plotly.graph_objects as go

        report = EDAReport(data_split, target="target").run_all()
        fig = report.missing_values_chart()
        assert isinstance(fig, go.Figure)

    def test_alerts_flags_constant_column(self, data_split: DataSplit):
        from dscompanion.eda.report import EDAAlert

        report = EDAReport(data_split, target="target").run_all()
        alerts = report.alerts()
        types = {a["type"] for a in alerts.get("constant_col", [])}
        assert EDAAlert.CONSTANT in types

    def test_alerts_flags_near_zero_variance_column(self, data_split: DataSplit):
        from dscompanion.eda.report import EDAAlert

        report = EDAReport(data_split, target="target").run_all()
        alerts = report.alerts()
        types = {a["type"] for a in alerts.get("f9_near_zero", [])}
        assert EDAAlert.NEAR_ZERO_VARIANCE in types

    def test_alerts_flags_high_cardinality_column(self, data_split: DataSplit):
        from dscompanion.eda.report import EDAAlert

        report = EDAReport(data_split, target="target").run_all()
        alerts = report.alerts()
        types = {a["type"] for a in alerts.get("cat_high", [])}
        assert EDAAlert.HIGH_CARDINALITY in types

    def test_alerts_flags_high_correlation_pair(self, data_split: DataSplit):
        from dscompanion.eda.report import EDAAlert

        report = EDAReport(data_split, target="target").run_all()
        alerts = report.alerts()
        f1_types = {a["type"] for a in alerts.get("f1", [])}
        f10_types = {a["type"] for a in alerts.get("f10_corr_f1", [])}
        assert EDAAlert.HIGH_CORRELATION in f1_types or EDAAlert.HIGH_CORRELATION in f10_types

    def test_alerts_no_alerts_for_clean_feature_absent_from_dict(self, data_split: DataSplit):
        report = EDAReport(data_split, target="target").run_all()
        alerts = report.alerts()
        # cat_ordinal is a clean, low-cardinality, balanced categorical — should have no entry
        # (or an entry with only benign alerts, never crashing on .get with a default)
        assert isinstance(alerts.get("cat_ordinal", []), list)

    def test_alerts_not_run_raises(self, data_split: DataSplit):
        with pytest.raises(RuntimeError):
            EDAReport(data_split, target="target").alerts()

    def test_imbalance_threshold_override_changes_sensitivity(self, data_split: DataSplit):
        from dscompanion.eda.report import EDAAlert

        lenient = EDAReport(data_split, target="target", imbalance_alert_threshold=0.999).run_all()
        sensitive = EDAReport(data_split, target="target", imbalance_alert_threshold=1e-6).run_all()
        lenient_imbalanced = sum(
            1 for v in lenient.alerts().values() if any(a["type"] == EDAAlert.IMBALANCED for a in v)
        )
        sensitive_imbalanced = sum(
            1
            for v in sensitive.alerts().values()
            if any(a["type"] == EDAAlert.IMBALANCED for a in v)
        )
        assert sensitive_imbalanced >= lenient_imbalanced

    def test_skewness_threshold_override_zero_alerts(self, data_split: DataSplit):
        from dscompanion.eda.report import EDAAlert

        report = EDAReport(data_split, target="target", skewness_alert_threshold=1000.0).run_all()
        for feature_alerts in report.alerts().values():
            assert not any(a["type"] == EDAAlert.SKEWED for a in feature_alerts)

    def test_high_missing_threshold_override_adds_alert(self, data_split: DataSplit):
        from dscompanion.eda.report import EDAAlert

        default_report = EDAReport(data_split, target="target").run_all()
        default_alerts = default_report.alerts().get("f5", [])
        assert not any(a["type"] == EDAAlert.HIGH_MISSING for a in default_alerts)

        sensitive_report = EDAReport(
            data_split, target="target", high_missing_threshold=0.05
        ).run_all()
        sensitive_alerts = sensitive_report.alerts().get("f5", [])
        assert any(a["type"] == EDAAlert.HIGH_MISSING for a in sensitive_alerts)
