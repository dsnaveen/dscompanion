"""Tests for dscompanion.features — all transformers and FeatureProcessingPipeline."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from dscompanion.features import (
    AutoBinner,
    BinaryEncoder,
    CategoryCombinerTransformer,
    ColumnArithmeticTransformer,
    ColumnRecipe,
    DateFeatureExtractor,
    DistributionTransformer,
    FeatureProcessingPipeline,
    FeatureTransformChain,
    GroupRelativeTransformer,
    HashEncoder,
    HighCardinalityEncoder,
    LeakageError,
    LeakageGuard,
    MultivariateImputer,
    NoiseInjector,
    OneHotEncoder,
    OrdinalEncoder,
    PCATransformer,
    PercentileRankTransformer,
    PolynomialFeaturesTransformer,
    RareCategoryGrouper,
    SmartImputer,
    SmartScaler,
    SplineFeatureTransformer,
    StatisticalOutlierCapper,
    TransformStep,
    WinsorizationTransformer,
    WoEEncoder,
    recommend_column_recipe,
    registered_transformer_names,
)
from dscompanion.features.registry import build_transformer

# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_cols():
    return [
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9_near_zero",
        "f10_corr_f1",
        "leakage_col",
    ]


@pytest.fixture
def X_train(data_split):
    return data_split.X_train


@pytest.fixture
def X_oot(data_split):
    return data_split.X_oot


@pytest.fixture
def y_train(data_split):
    return data_split.y_train


# ---------------------------------------------------------------------------
# SmartImputer
# ---------------------------------------------------------------------------


class TestSmartImputer:
    def test_adds_indicator_for_high_missingness(self, X_train):
        imp = SmartImputer(missing_indicator_threshold=0.05)
        out = imp.fit_transform(X_train)
        # f4 and f5 have 10% missingness — indicators must appear
        assert "f4_was_missing" in out.columns
        assert "f5_was_missing" in out.columns

    def test_no_indicator_for_zero_missingness(self, X_train):
        imp = SmartImputer(missing_indicator_threshold=0.05)
        out = imp.fit_transform(X_train)
        # f1 has no missing values
        assert "f1_was_missing" not in out.columns

    def test_no_missing_values_after_transform(self, X_train):
        imp = SmartImputer()
        out = imp.fit_transform(X_train)
        num_cols = out.select_dtypes(include="number").columns
        assert out[num_cols].isna().sum().sum() == 0

    def test_joblib_serialisable(self, X_train, tmp_path):
        imp = SmartImputer()
        imp.fit(X_train)
        path = tmp_path / "imputer.joblib"
        joblib.dump(imp, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(imp.transform(X_train), loaded.transform(X_train))

    def test_column_strategy_override_changes_value(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, np.nan, 30.0, 40.0]})
        imp = SmartImputer(numeric_strategy="mean", column_strategies={"a": "median"})
        out = imp.fit_transform(df)
        assert out.loc[2, "a"] == df["a"].median()
        assert out.loc[1, "b"] == df["b"].mean()

    def test_column_fill_value_override_independent_of_global(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, np.nan, 30.0]})
        imp = SmartImputer(
            fill_value=-1,
            column_strategies={"a": "constant"},
            column_fill_values={"a": 999},
        )
        out = imp.fit_transform(df)
        assert out.loc[1, "a"] == 999
        # b has no override and no strategy="constant" either — untouched by fill_value
        assert out.loc[1, "b"] == df["b"].mean()

    def test_column_overrides_none_preserves_global_behavior(self, X_train):
        default_imp = SmartImputer()
        overridden_imp = SmartImputer(column_strategies=None, column_fill_values=None)
        pd.testing.assert_frame_equal(
            default_imp.fit_transform(X_train), overridden_imp.fit_transform(X_train)
        )

    def test_joblib_serialisable_with_column_overrides(self, X_train, tmp_path):
        imp = SmartImputer(column_strategies={"f4": "median"})
        imp.fit(X_train)
        path = tmp_path / "imputer_overrides.joblib"
        joblib.dump(imp, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(imp.transform(X_train), loaded.transform(X_train))


# ---------------------------------------------------------------------------
# SmartScaler
# ---------------------------------------------------------------------------


class TestSmartScaler:
    @pytest.mark.parametrize("strategy", ["robust", "standard", "minmax"])
    def test_inverse_transform_round_trips(self, X_train, strategy):
        num_cols = X_train.select_dtypes(include="number").columns.tolist()
        scaler = SmartScaler(strategy=strategy)
        scaler.fit(X_train)
        scaled = scaler.transform(X_train)
        restored = scaler.inverse_transform(scaled)
        pd.testing.assert_frame_equal(
            restored[num_cols].fillna(0),
            X_train[num_cols].fillna(0),
            atol=1e-8,
            check_dtype=False,
        )

    def test_inverse_transform_raises_when_not_fitted(self, X_train):
        from sklearn.exceptions import NotFittedError

        scaler = SmartScaler()
        with pytest.raises(NotFittedError):
            scaler.inverse_transform(X_train)

    def test_inverse_transform_strategy_none_is_passthrough(self, X_train):
        scaler = SmartScaler(strategy="none")
        scaler.fit(X_train)
        scaled = scaler.transform(X_train)
        restored = scaler.inverse_transform(scaled)
        pd.testing.assert_frame_equal(restored, scaled)


# ---------------------------------------------------------------------------
# WoEEncoder
# ---------------------------------------------------------------------------


class TestWoEEncoder:
    def test_iv_matches_manual_computation(self):
        """WoE IV for a perfectly separating binary column should be high."""
        rng = np.random.RandomState(0)
        n = 1000
        y = pd.Series(rng.choice([0, 1], size=n))
        X = pd.DataFrame({"perfect": y.astype(str)})  # cat = target value
        enc = WoEEncoder()
        enc.fit(X, y)
        assert enc.iv_["perfect"] > 1.0  # near-perfect IV

    def test_output_no_nans(self, X_train, y_train):
        cat_cols = ["cat_low", "cat_ordinal"]
        enc = WoEEncoder()
        enc.fit(X_train[cat_cols], y_train)
        out = enc.transform(X_train[cat_cols])
        assert out.isna().sum().sum() == 0

    def test_joblib_serialisable(self, X_train, y_train, tmp_path):
        cat_cols = ["cat_low", "cat_ordinal"]
        enc = WoEEncoder()
        enc.fit(X_train[cat_cols], y_train)
        path = tmp_path / "woe.joblib"
        joblib.dump(enc, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(
            enc.transform(X_train[cat_cols]),
            loaded.transform(X_train[cat_cols]),
        )


# ---------------------------------------------------------------------------
# HighCardinalityEncoder
# ---------------------------------------------------------------------------


class TestHighCardinalityEncoder:
    def test_no_data_leakage_from_oot(self, X_train, X_oot, y_train):
        """Encoding map fitted on train only; OOT unseen categories get fallback."""
        enc = HighCardinalityEncoder(strategy="target_encoding")
        enc.fit(X_train[["cat_high"]], y_train)
        # Introduce a category that wasn't in train
        X_test_mod = X_oot[["cat_high"]].copy()
        X_test_mod["cat_high"] = "__UNSEEN__"
        out = enc.transform(X_test_mod)
        # All values should be the global mean fallback (no NaN)
        assert out["cat_high"].isna().sum() == 0
        expected_fallback = enc.global_means_["cat_high"]
        assert (out["cat_high"] == expected_fallback).all()

    def test_frequency_encoding_sums_to_approx_one(self, X_train):
        enc = HighCardinalityEncoder(strategy="frequency_encoding")
        enc.fit(X_train[["cat_high"]])
        freqs = enc.encoding_maps_["cat_high"]
        assert abs(sum(freqs.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# DateFeatureExtractor
# ---------------------------------------------------------------------------


class TestDateFeatureExtractor:
    def test_sin_cos_columns_created(self, X_train):
        dte = DateFeatureExtractor(date_cols=["account_open_date"])
        out = dte.fit_transform(X_train)
        assert "account_open_date_month_sin" in out.columns
        assert "account_open_date_month_cos" in out.columns

    def test_original_col_dropped_by_default(self, X_train):
        dte = DateFeatureExtractor(date_cols=["account_open_date"])
        out = dte.fit_transform(X_train)
        assert "account_open_date" not in out.columns

    def test_age_days_non_negative(self, X_train):
        dte = DateFeatureExtractor(date_cols=["account_open_date"])
        out = dte.fit_transform(X_train)
        assert (out["account_open_date_age_days"] >= 0).all()

    def test_joblib_serialisable(self, X_train, tmp_path):
        dte = DateFeatureExtractor(date_cols=["account_open_date"])
        dte.fit(X_train)
        path = tmp_path / "dte.joblib"
        joblib.dump(dte, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(dte.transform(X_train), loaded.transform(X_train))


# ---------------------------------------------------------------------------
# LeakageGuard
# ---------------------------------------------------------------------------


class TestLeakageGuard:
    def test_raises_on_leakage_column(self, X_train, y_train):
        guard = LeakageGuard(raise_on_critical=True)
        with pytest.raises(LeakageError):
            guard.check(X_train[["leakage_col"]], y_train, target_col="target")

    def test_no_raise_when_disabled(self, X_train, y_train):
        guard = LeakageGuard(raise_on_critical=False)
        report = guard.check(X_train[["leakage_col"]], y_train, target_col="target")
        assert report.n_critical >= 1
        assert not report.is_clean

    def test_name_overlap_warning(self, X_train, y_train):
        X_tmp = X_train[["f1"]].copy()
        X_tmp["target_ratio"] = X_train["f1"]
        guard = LeakageGuard(raise_on_critical=False)
        report = guard.check(X_tmp, y_train, target_col="target")
        assert any("target_ratio" in w for w in report.warnings)

    def test_clean_features_pass(self, X_train, y_train):
        clean_cols = ["f2", "f3", "f6", "f7", "f8"]
        guard = LeakageGuard(raise_on_critical=True, correlation_threshold=0.99)
        report = guard.check(X_train[clean_cols], y_train, target_col="target")
        assert report.is_clean


# ---------------------------------------------------------------------------
# FeatureProcessingPipeline
# ---------------------------------------------------------------------------


class TestFeatureProcessingPipeline:
    def _build_safe_X(self, X_train):
        """Return X without the leakage column for safe pipeline tests."""
        return X_train.drop(columns=["leakage_col"], errors="ignore")

    def test_fit_transform_no_errors(self, X_train, X_oot, y_train):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(
            encoder="ordinal",
            run_leakage_check=False,
        )
        pipe.fit(X, y_train)
        out = pipe.transform(self._build_safe_X(X_oot))
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(X_oot)

    def test_oot_no_refit(self, X_train, X_oot, y_train):
        """Transform OOT with train-fitted pipeline — shape must match."""
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(
            encoder="ordinal",
            run_leakage_check=False,
        )
        pipe.fit(X, y_train)
        n_features = len(pipe.get_feature_names_out())
        out = pipe.transform(self._build_safe_X(X_oot))
        assert out.shape[1] == n_features

    def test_leakage_check_raises(self, X_train, y_train):
        pipe = FeatureProcessingPipeline(
            encoder="ordinal",
            run_leakage_check=True,
        )
        with pytest.raises(LeakageError):
            pipe.fit(X_train[["leakage_col", "f1", "f2"]], y_train)

    def test_winsorize_clips_extreme_values_when_enabled(self, X_train, y_train):
        X = self._build_safe_X(X_train).copy()
        X.loc[X.index[0], "f1"] = X["f1"].max() * 1000  # inject an extreme outlier
        pipe = FeatureProcessingPipeline(
            encoder="ordinal",
            run_leakage_check=False,
            winsorize=True,
            winsorize_lower=0.01,
            winsorize_upper=0.01,
        )
        pipe.fit(X, y_train)
        capped = pipe._winsorizer.transform(X)["f1"].iloc[0]
        assert capped < X["f1"].iloc[0]

    def test_winsorize_disabled_by_default(self, X_train, y_train):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(encoder="ordinal", run_leakage_check=False)
        pipe.fit(X, y_train)
        assert pipe._winsorizer is None

    def test_inverse_transform_reverses_scaling_only(self, X_train, y_train):
        """inverse_transform() must invert exactly the scaling step — re-scaling the
        restored values reproduces transform()'s original output, regardless of what
        date-extraction/encoding/imputation upstream of scaling produced."""
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(encoder="ordinal", run_leakage_check=False)
        pipe.fit(X, y_train)
        transformed = pipe.transform(X)
        restored = pipe.inverse_transform(transformed)
        re_scaled = pipe._scaler.transform(restored)
        pd.testing.assert_frame_equal(re_scaled, transformed, atol=1e-6)

    def test_inverse_transform_raises_when_not_fitted(self, X_train):
        from sklearn.exceptions import NotFittedError

        pipe = FeatureProcessingPipeline(encoder="ordinal", run_leakage_check=False)
        with pytest.raises(NotFittedError):
            pipe.inverse_transform(X_train)

    def test_audit_report_has_rows(self, X_train, y_train):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(
            encoder="ordinal",
            run_leakage_check=False,
        )
        pipe.fit(X, y_train)
        audit = pipe.audit_report()
        assert isinstance(audit, pd.DataFrame)
        assert len(audit) > 0
        assert "transformations_applied" in audit.columns

    def test_joblib_serialisable(self, X_train, X_oot, y_train, tmp_path):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(
            encoder="ordinal",
            run_leakage_check=False,
        )
        pipe.fit(X, y_train)
        path = tmp_path / "pipeline.joblib"
        joblib.dump(pipe, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(
            pipe.transform(self._build_safe_X(X_oot)),
            loaded.transform(self._build_safe_X(X_oot)),
        )

    def test_woe_encoder_path(self, X_train, y_train):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(
            encoder="woe",
            run_leakage_check=False,
        )
        pipe.fit(X, y_train)
        out = pipe.transform(self._build_safe_X(X_train))
        assert out.isna().sum().sum() == 0

    def test_onehot_encoder_path_expands_columns(self, X_train, X_oot, y_train):
        """One-hot changes column count/names — _apply_cat_transform's special-case
        branch (drop + concat instead of in-place assignment) must handle this."""
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(encoder="onehot", run_leakage_check=False)
        pipe.fit(X, y_train)
        out = pipe.transform(self._build_safe_X(X_oot))
        # cat_low (3 levels), cat_high (80 levels -> high-cardinality, not one-hot'd),
        # cat_ordinal (3 levels) are the categorical columns in the shared fixture.
        assert "cat_low" not in out.columns  # original replaced by expansion
        assert any(c.startswith("cat_low_") for c in out.columns)
        assert out.isna().sum().sum() == 0
        assert len(out) == len(X_oot)

    def test_onehot_audit_report_has_one_row_per_category(self, X_train, y_train):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(encoder="onehot", run_leakage_check=False)
        pipe.fit(X, y_train)
        audit = pipe.audit_report()
        cat_low_rows = audit[audit["original_col"] == "cat_low"]
        n_categories = X["cat_low"].nunique()
        assert len(cat_low_rows) == n_categories
        assert set(cat_low_rows["transformations_applied"]) == {"impute,onehot_encoding"}

    def test_distribution_transform_applied_before_scaling(self, X_train, y_train):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(
            encoder="ordinal", distribution="yeo_johnson", run_leakage_check=False
        )
        pipe.fit(X, y_train)
        assert pipe._distribution is not None
        out = pipe.transform(self._build_safe_X(X_train))
        assert out.isna().sum().sum() == 0

    def test_distribution_none_by_default(self, X_train, y_train):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(encoder="ordinal", run_leakage_check=False)
        pipe.fit(X, y_train)
        assert pipe._distribution is None

    def test_onehot_and_distribution_combined(self, X_train, X_oot, y_train):
        X = self._build_safe_X(X_train)
        pipe = FeatureProcessingPipeline(
            encoder="onehot", distribution="yeo_johnson", run_leakage_check=False
        )
        pipe.fit(X, y_train)
        out = pipe.transform(self._build_safe_X(X_oot))
        assert out.isna().sum().sum() == 0
        assert len(out) == len(X_oot)


# ---------------------------------------------------------------------------
# SmartScaler — additional coverage
# ---------------------------------------------------------------------------


class TestSmartScalerExtended:
    @pytest.mark.parametrize("strategy", ["robust", "standard", "minmax", "none"])
    def test_all_strategies_fit_transform_no_error(self, X_train, strategy):
        num_df = X_train.select_dtypes(include="number")
        scaler = SmartScaler(strategy=strategy)
        out = scaler.fit_transform(num_df)
        assert isinstance(out, pd.DataFrame)
        assert out.shape == num_df.shape

    def test_non_numeric_columns_passed_through(self, X_train):
        scaler = SmartScaler(strategy="standard")
        scaler.fit(X_train)
        out = scaler.transform(X_train)
        pd.testing.assert_series_equal(out["cat_low"], X_train["cat_low"])

    def test_get_feature_names_out_returns_numeric_cols(self, X_train):
        num_cols = X_train.select_dtypes(include="number").columns.tolist()
        scaler = SmartScaler()
        scaler.fit(X_train)
        assert scaler.get_feature_names_out() == num_cols

    def test_get_feature_names_out_raises_when_not_fitted(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            SmartScaler().get_feature_names_out()

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            SmartScaler().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_type_error_on_non_dataframe_transform(self, X_train):
        scaler = SmartScaler()
        scaler.fit(X_train)
        with pytest.raises(TypeError):
            scaler.transform(X_train.values)

    def test_type_error_on_non_dataframe_inverse_transform(self, X_train):
        scaler = SmartScaler()
        scaler.fit(X_train)
        with pytest.raises(TypeError):
            scaler.inverse_transform(X_train.values)

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            SmartScaler().fit(pd.DataFrame())

    def test_transform_returns_copy(self, X_train):
        num_df = X_train.select_dtypes(include="number")
        scaler = SmartScaler()
        scaler.fit(num_df)
        out = scaler.transform(num_df)
        orig_val = num_df.iloc[0, 0]
        out.iloc[0, 0] = 999_999.0
        assert num_df.iloc[0, 0] == orig_val

    def test_joblib_serialisable(self, X_train, tmp_path):
        scaler = SmartScaler(strategy="standard")
        scaler.fit(X_train)
        path = tmp_path / "scaler.joblib"
        joblib.dump(scaler, path)
        loaded = joblib.load(path)
        num_cols = X_train.select_dtypes(include="number").columns.tolist()
        pd.testing.assert_frame_equal(
            scaler.transform(X_train)[num_cols],
            loaded.transform(X_train)[num_cols],
        )

    def test_minmax_output_in_zero_one_range(self, X_train):
        num_df = X_train.select_dtypes(include="number").fillna(0)
        scaler = SmartScaler(strategy="minmax")
        out = scaler.fit_transform(num_df)
        assert out.min().min() >= -1e-9
        assert out.max().max() <= 1 + 1e-9


# ---------------------------------------------------------------------------
# SmartScaler — max_abs / l2_norm
# ---------------------------------------------------------------------------


class TestSmartScalerMaxAbsAndL2Norm:
    def test_max_abs_scales_by_column_max_absolute_value(self):
        df = pd.DataFrame({"a": [-10.0, 5.0, 10.0]})
        out = SmartScaler(strategy="max_abs").fit_transform(df)
        assert out["a"].abs().max() == pytest.approx(1.0)

    def test_l2_norm_normalizes_rows_to_unit_length(self):
        df = pd.DataFrame({"a": [3.0, 0.0], "b": [4.0, 0.0]})
        out = SmartScaler(strategy="l2_norm").fit_transform(df)
        row0_norm = np.sqrt(out.loc[0, "a"] ** 2 + out.loc[0, "b"] ** 2)
        assert row0_norm == pytest.approx(1.0)

    def test_l2_norm_inverse_transform_raises(self):
        df = pd.DataFrame({"a": [3.0, 4.0], "b": [4.0, 3.0]})
        scaler = SmartScaler(strategy="l2_norm").fit(df)
        with pytest.raises(NotImplementedError):
            scaler.inverse_transform(df)

    def test_max_abs_inverse_transform_round_trips(self):
        df = pd.DataFrame({"a": [-10.0, 5.0, 10.0]})
        scaler = SmartScaler(strategy="max_abs").fit(df)
        out = scaler.transform(df)
        restored = scaler.inverse_transform(out)
        pd.testing.assert_frame_equal(restored, df)


# ---------------------------------------------------------------------------
# WinsorizationTransformer
# ---------------------------------------------------------------------------


class TestWinsorizationTransformer:
    def _num_df(self, rng, n=500) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "b": rng.uniform(0, 1, n),
            }
        )

    def test_clips_extreme_values(self):
        rng = np.random.RandomState(0)
        df = self._num_df(rng)
        # Inject extreme outlier
        df.loc[0, "a"] = 1_000.0
        wt = WinsorizationTransformer(lower=0.01, upper=0.01)
        out = wt.fit_transform(df)
        assert out["a"].iloc[0] < 1_000.0

    def test_upper_cap_from_training_data(self):
        rng = np.random.RandomState(1)
        df = self._num_df(rng)
        wt = WinsorizationTransformer(lower=0.05, upper=0.05)
        wt.fit(df)
        cap = wt.clip_values_["a"][1]
        out = wt.transform(df)
        assert out["a"].max() <= cap + 1e-9

    def test_transform_does_not_recompute_bounds(self):
        rng = np.random.RandomState(2)
        df_train = self._num_df(rng, n=500)
        df_oot = pd.DataFrame({"a": [100.0, 200.0, 300.0], "b": [0.1, 0.2, 0.3]})
        wt = WinsorizationTransformer(lower=0.01, upper=0.01)
        wt.fit(df_train)
        train_cap = wt.clip_values_["a"][1]
        out = wt.transform(df_oot)
        # OOT values must be clipped to train-derived cap, not OOT percentile
        assert out["a"].max() <= train_cap + 1e-9

    def test_all_nan_column_skipped(self):
        df = pd.DataFrame(
            {
                "a": [np.nan] * 50,
                "b": np.arange(50, dtype=float),
            }
        )
        wt = WinsorizationTransformer(lower=0.05, upper=0.05)
        wt.fit(df)
        assert "a" not in wt.clip_values_
        assert "b" in wt.clip_values_

    def test_explicit_features_list_respects_subset(self):
        rng = np.random.RandomState(3)
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 300),
                "b": rng.normal(0, 1, 300),
            }
        )
        # Inject outlier only in 'a'
        df.loc[0, "b"] = 1_000.0
        wt = WinsorizationTransformer(lower=0.01, upper=0.01, features=["a"])
        out = wt.fit_transform(df)
        # 'b' not in features — must be unchanged
        assert out["b"].iloc[0] == 1_000.0

    def test_get_feature_names_out(self):
        rng = np.random.RandomState(4)
        df = self._num_df(rng)
        wt = WinsorizationTransformer(lower=0.05, upper=0.05)
        wt.fit(df)
        assert set(wt.get_feature_names_out()) == {"a", "b"}

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            WinsorizationTransformer().transform(pd.DataFrame({"a": [1.0]}))

    def test_not_fitted_error_on_get_feature_names_out(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            WinsorizationTransformer().get_feature_names_out()

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            WinsorizationTransformer().fit(np.array([[1.0, 2.0]]))

    def test_type_error_on_non_dataframe_transform(self):
        rng = np.random.RandomState(5)
        df = self._num_df(rng)
        wt = WinsorizationTransformer(lower=0.05, upper=0.05)
        wt.fit(df)
        with pytest.raises(TypeError):
            wt.transform(np.array([[1.0, 2.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            WinsorizationTransformer().fit(pd.DataFrame({"a": pd.Series([], dtype=float)}))

    def test_transform_returns_copy(self):
        rng = np.random.RandomState(6)
        df = self._num_df(rng)
        wt = WinsorizationTransformer(lower=0.05, upper=0.05)
        wt.fit(df)
        out = wt.transform(df)
        out.iloc[0, 0] = 999_999.0
        assert df.iloc[0, 0] != 999_999.0

    def test_joblib_serialisable(self, tmp_path):
        rng = np.random.RandomState(7)
        df = self._num_df(rng)
        wt = WinsorizationTransformer(lower=0.05, upper=0.05)
        wt.fit(df)
        path = tmp_path / "winsorizer.joblib"
        joblib.dump(wt, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(wt.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# AutoBinner
# ---------------------------------------------------------------------------


class TestAutoBinner:
    def _num_df(self, rng=None, n=500) -> pd.DataFrame:
        if rng is None:
            rng = np.random.RandomState(0)
        return pd.DataFrame(
            {
                "x1": rng.normal(0, 1, n),
                "x2": rng.uniform(0, 10, n),
            }
        )

    def test_quantile_strategy_fit_transform(self):
        rng = np.random.RandomState(0)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="quantile", n_bins=5, encode="ordinal")
        out = binner.fit_transform(df)
        assert out["x1"].between(-1, 4).all()  # ordinal codes 0-4

    def test_uniform_strategy_fit_transform(self):
        rng = np.random.RandomState(1)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="uniform", n_bins=4, encode="ordinal")
        out = binner.fit_transform(df)
        assert set(out["x1"].unique()).issubset({-1, 0, 1, 2, 3})

    def test_ordinal_encode_produces_integer_codes(self):
        rng = np.random.RandomState(2)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="quantile", n_bins=5, encode="ordinal")
        out = binner.fit_transform(df)
        assert out["x1"].dtype in (np.int32, np.int64, int)

    def test_labels_encode_produces_string_ranges(self):
        rng = np.random.RandomState(3)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="quantile", n_bins=4, encode="labels")
        out = binner.fit_transform(df)
        # All non-"unknown" labels must look like "[lo, hi)"
        non_unk = out["x1"][out["x1"] != "unknown"]
        assert non_unk.str.startswith("[").all()

    def test_out_of_range_values_get_minus_one(self):
        df_train = pd.DataFrame({"x": np.linspace(0, 10, 200)})
        df_oot = pd.DataFrame({"x": [100.0, -100.0]})  # way outside training range
        binner = AutoBinner(strategy="quantile", n_bins=5, encode="ordinal")
        binner.fit(df_train)
        out = binner.transform(df_oot)
        assert (out["x"] == -1).all()

    def test_out_of_range_values_labels_get_unknown(self):
        df_train = pd.DataFrame({"x": np.linspace(0, 10, 200)})
        df_oot = pd.DataFrame({"x": [100.0, -100.0]})
        binner = AutoBinner(strategy="quantile", n_bins=5, encode="labels")
        binner.fit(df_train)
        out = binner.transform(df_oot)
        assert (out["x"] == "unknown").all()

    def test_tree_strategy_binary_target_classifier_path(self):
        rng = np.random.RandomState(4)
        n = 500
        x = rng.normal(0, 1, n)
        y = pd.Series((x > 0).astype(int))
        df = pd.DataFrame({"x": x})
        binner = AutoBinner(strategy="tree", n_bins=5, encode="ordinal")
        out = binner.fit_transform(df, y)
        assert isinstance(out, pd.DataFrame)
        assert "x" in binner._bin_edges

    def test_tree_strategy_continuous_target_regressor_path(self):
        rng = np.random.RandomState(5)
        n = 500
        x = rng.normal(0, 1, n)
        # Continuous target → many unique values → regressor path
        y = pd.Series(x * 2.5 + rng.normal(0, 0.1, n))
        df = pd.DataFrame({"x": x})
        binner = AutoBinner(strategy="tree", n_bins=5, encode="ordinal")
        out = binner.fit_transform(df, y)
        assert isinstance(out, pd.DataFrame)

    def test_tree_strategy_fallback_to_quantile_when_y_none(self):
        rng = np.random.RandomState(6)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="tree", n_bins=5, encode="ordinal")
        out = binner.fit_transform(df, y=None)
        # Still produces bins via fallback
        assert isinstance(out, pd.DataFrame)
        assert len(binner._bin_edges) > 0

    def test_tree_all_nan_column_skipped(self):
        rng = np.random.RandomState(7)
        n = 200
        y = pd.Series(rng.choice([0, 1], n))
        df = pd.DataFrame(
            {
                "good": rng.normal(0, 1, n),
                "all_nan": [np.nan] * n,
            }
        )
        binner = AutoBinner(strategy="tree", n_bins=5, encode="ordinal")
        binner.fit(df, y)
        assert "all_nan" not in binner._bin_edges
        assert "good" in binner._bin_edges

    def test_get_feature_names_out(self):
        rng = np.random.RandomState(8)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="quantile", n_bins=4)
        binner.fit(df)
        assert binner.get_feature_names_out() == ["x1", "x2"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            AutoBinner().transform(pd.DataFrame({"x": [1.0, 2.0]}))

    def test_not_fitted_error_on_get_feature_names_out(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            AutoBinner().get_feature_names_out()

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            AutoBinner().fit(np.array([[1.0, 2.0]]))

    def test_type_error_on_non_dataframe_transform(self):
        rng = np.random.RandomState(9)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="quantile", n_bins=4)
        binner.fit(df)
        with pytest.raises(TypeError):
            binner.transform(df.values)

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            AutoBinner().fit(pd.DataFrame({"x": pd.Series([], dtype=float)}))

    def test_transform_absent_column_silently_skipped(self):
        rng = np.random.RandomState(10)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="quantile", n_bins=4, encode="ordinal")
        binner.fit(df)
        # Transform with only one of the two columns
        out = binner.transform(df[["x1"]])
        assert "x1" in out.columns
        assert "x2" not in out.columns

    def test_transform_returns_copy(self):
        rng = np.random.RandomState(11)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="quantile", n_bins=4, encode="ordinal")
        binner.fit(df)
        out = binner.transform(df)
        out.iloc[0, 0] = 9_999
        assert df.iloc[0, 0] != 9_999

    def test_joblib_serialisable(self, tmp_path):
        rng = np.random.RandomState(12)
        df = self._num_df(rng)
        binner = AutoBinner(strategy="quantile", n_bins=5, encode="ordinal")
        binner.fit(df)
        path = tmp_path / "binner.joblib"
        joblib.dump(binner, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(binner.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# DateFeatureExtractor — additional coverage
# ---------------------------------------------------------------------------


class TestDateFeatureExtractorExtended:
    def _date_df(self, n=200) -> pd.DataFrame:
        rng = np.random.RandomState(0)
        dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.randint(0, 730, n), unit="D")
        return pd.DataFrame({"account_date": dates, "val": rng.randn(n)})

    def test_all_nine_suffixes_produced(self, X_train):
        suffixes = [
            "age_days",
            "age_months",
            "year",
            "month",
            "day_of_week",
            "quarter",
            "month_sin",
            "month_cos",
            "is_weekend",
        ]
        dte = DateFeatureExtractor(date_cols=["account_open_date"])
        out = dte.fit_transform(X_train)
        for sfx in suffixes:
            assert f"account_open_date_{sfx}" in out.columns

    def test_drop_original_false_retains_column(self, X_train):
        dte = DateFeatureExtractor(date_cols=["account_open_date"], drop_original=False)
        out = dte.fit_transform(X_train)
        assert "account_open_date" in out.columns
        assert "account_open_date_year" in out.columns

    def test_reference_date_controls_age(self):
        df = pd.DataFrame({"dt": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-07-01")]})
        ref = "2021-01-01"
        dte = DateFeatureExtractor(date_cols=["dt"], reference_date=ref)
        out = dte.fit_transform(df)
        expected_days = [
            (pd.Timestamp(ref) - pd.Timestamp("2020-01-01")).days,
            (pd.Timestamp(ref) - pd.Timestamp("2020-07-01")).days,
        ]
        assert list(out["dt_age_days"]) == expected_days

    def test_nat_values_produce_nan_in_output(self):
        df = pd.DataFrame({"dt": [pd.Timestamp("2020-06-15"), pd.NaT]})
        dte = DateFeatureExtractor(date_cols=["dt"])
        out = dte.fit_transform(df)
        assert pd.isna(out["dt_year"].iloc[1])

    def test_autodetect_by_name_pattern(self):
        n = 100
        rng = np.random.RandomState(42)
        dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.randint(0, 365, n), unit="D")
        df = pd.DataFrame({"open_date": dates, "val": rng.randn(n)})
        dte = DateFeatureExtractor()  # no date_cols — auto-detect
        out = dte.fit_transform(df)
        assert "open_date_year" in out.columns

    def test_absent_col_at_transform_silently_skipped(self):
        df_train = pd.DataFrame(
            {"dt": pd.to_datetime(["2020-01-01", "2020-06-15"]), "x": [1.0, 2.0]}
        )
        df_oot = pd.DataFrame({"x": [3.0, 4.0]})  # dt column absent
        dte = DateFeatureExtractor(date_cols=["dt"])
        dte.fit(df_train)
        out = dte.transform(df_oot)
        assert "dt_year" not in out.columns
        assert list(out.columns) == ["x"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            DateFeatureExtractor().transform(pd.DataFrame({"dt": [pd.Timestamp("2020-01-01")]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            DateFeatureExtractor().fit(np.array([[1, 2]]))

    def test_type_error_on_non_dataframe_transform(self, X_train):
        dte = DateFeatureExtractor(date_cols=["account_open_date"])
        dte.fit(X_train)
        with pytest.raises(TypeError):
            dte.transform(X_train.values)

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            DateFeatureExtractor(date_cols=["dt"]).fit(
                pd.DataFrame({"dt": pd.Series([], dtype="datetime64[ns]")})
            )

    def test_get_feature_names_out_returns_all_combos(self, X_train):
        dte = DateFeatureExtractor(date_cols=["account_open_date"])
        dte.fit(X_train)
        names = dte.get_feature_names_out()
        assert len(names) == 11  # one col × eleven suffixes (incl. day-of-week cyclical)


# ---------------------------------------------------------------------------
# OrdinalEncoder
# ---------------------------------------------------------------------------


class TestOrdinalEncoder:
    def _cat_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "color": ["red", "blue", "green", "blue", "red"],
                "size": ["S", "M", "L", "XL", "S"],
            }
        )

    def test_fit_transform_produces_integer_codes(self):
        df = self._cat_df()
        enc = OrdinalEncoder()
        out = enc.fit_transform(df)
        assert out["color"].dtype in (np.int32, np.int64, int)
        assert out["size"].dtype in (np.int32, np.int64, int)

    def test_alphabetical_order_by_default(self):
        df = pd.DataFrame({"x": ["c", "a", "b"]})
        enc = OrdinalEncoder()
        out = enc.fit_transform(df)
        # Alphabetical: a=0, b=1, c=2
        assert out.loc[0, "x"] == 2
        assert out.loc[1, "x"] == 0
        assert out.loc[2, "x"] == 1

    def test_custom_category_order_respected(self):
        df = pd.DataFrame({"size": ["S", "M", "L", "XL"]})
        enc = OrdinalEncoder(category_order={"size": ["S", "M", "L", "XL"]})
        out = enc.fit_transform(df)
        assert out.loc[0, "size"] == 0
        assert out.loc[3, "size"] == 3

    def test_unseen_category_encoded_as_minus_one(self):
        df_train = pd.DataFrame({"c": ["a", "b", "c"]})
        df_oot = pd.DataFrame({"c": ["d"]})  # unseen
        enc = OrdinalEncoder()
        enc.fit(df_train)
        out = enc.transform(df_oot)
        assert out.loc[0, "c"] == -1

    def test_nan_encoded_as_minus_one(self):
        df = pd.DataFrame({"c": ["a", None, "b"]})
        enc = OrdinalEncoder()
        out = enc.fit_transform(df)
        assert out.loc[1, "c"] == -1

    def test_numeric_columns_pass_through(self):
        df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1.0, 2.0, 3.0]})
        enc = OrdinalEncoder()
        out = enc.fit_transform(df)
        pd.testing.assert_series_equal(out["num"], df["num"])

    def test_get_feature_names_out(self):
        df = self._cat_df()
        enc = OrdinalEncoder()
        enc.fit(df)
        names = enc.get_feature_names_out()
        assert set(names) == {"color", "size"}

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            OrdinalEncoder().transform(pd.DataFrame({"c": ["a"]}))

    def test_not_fitted_error_on_get_feature_names_out(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            OrdinalEncoder().get_feature_names_out()

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            OrdinalEncoder().fit(np.array([["a", "b"]]))

    def test_type_error_on_non_dataframe_transform(self):
        df = self._cat_df()
        enc = OrdinalEncoder()
        enc.fit(df)
        with pytest.raises(TypeError):
            enc.transform(df.values)

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            OrdinalEncoder().fit(pd.DataFrame({"c": pd.Series([], dtype=str)}))

    def test_transform_returns_copy(self):
        df = self._cat_df()
        enc = OrdinalEncoder()
        enc.fit(df)
        out = enc.transform(df)
        out.iloc[0, 0] = 999
        assert df.iloc[0, 0] != 999

    def test_joblib_serialisable(self, tmp_path):
        df = self._cat_df()
        enc = OrdinalEncoder()
        enc.fit(df)
        path = tmp_path / "ordinal_enc.joblib"
        joblib.dump(enc, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(enc.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------


class TestOneHotEncoder:
    def _cat_df(self) -> pd.DataFrame:
        return pd.DataFrame({"color": ["red", "blue", "green", "blue", "red"]})

    def test_fit_transform_produces_one_column_per_category(self):
        df = self._cat_df()
        enc = OneHotEncoder()
        out = enc.fit_transform(df)
        assert set(out.columns) == {"color_blue", "color_green", "color_red"}
        assert out["color_red"].tolist() == [1, 0, 0, 0, 1]

    def test_drop_first_removes_one_category_per_column(self):
        df = self._cat_df()
        enc = OneHotEncoder(drop_first=True)
        out = enc.fit_transform(df)
        # alphabetical: blue, green, red -> blue dropped
        assert set(out.columns) == {"color_green", "color_red"}

    def test_unseen_category_produces_all_zero_row(self):
        df_train = self._cat_df()
        df_oot = pd.DataFrame({"color": ["purple"]})  # unseen
        enc = OneHotEncoder()
        enc.fit(df_train)
        out = enc.transform(df_oot)
        assert out.iloc[0].sum() == 0
        assert set(out.columns) == {"color_blue", "color_green", "color_red"}

    def test_missing_source_column_at_transform_is_all_zero(self):
        df_train = self._cat_df()
        enc = OneHotEncoder()
        enc.fit(df_train)
        out = enc.transform(pd.DataFrame({"other": [1, 2]}))
        assert out.shape == (2, 3)
        assert out.sum().sum() == 0

    def test_numeric_columns_ignored_by_default(self):
        df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1.0, 2.0, 3.0]})
        enc = OneHotEncoder()
        out = enc.fit_transform(df)
        assert "num" not in out.columns
        assert set(out.columns) == {"cat_a", "cat_b"}

    def test_get_feature_names_out(self):
        df = self._cat_df()
        enc = OneHotEncoder()
        enc.fit(df)
        assert enc.get_feature_names_out() == ["color_blue", "color_green", "color_red"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            OneHotEncoder().transform(pd.DataFrame({"c": ["a"]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            OneHotEncoder().fit(np.array([["a", "b"]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            OneHotEncoder().fit(pd.DataFrame({"c": pd.Series([], dtype=str)}))

    def test_joblib_serialisable(self, tmp_path):
        df = self._cat_df()
        enc = OneHotEncoder()
        enc.fit(df)
        path = tmp_path / "onehot_enc.joblib"
        joblib.dump(enc, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(enc.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# RareCategoryGrouper
# ---------------------------------------------------------------------------


class TestRareCategoryGrouper:
    def _cat_df(self) -> pd.DataFrame:
        # 'x' x6, 'y' x3, 'w' x1 (rare), 'z' x1 (rare) — 10 rows total
        return pd.DataFrame({"c": ["x", "y", "x", "z", "x", "y", "w", "x", "x", "y"]})

    def test_min_frequency_groups_rare_categories(self):
        df = self._cat_df()
        grp = RareCategoryGrouper(strategy="min_frequency", min_frequency=0.15)
        out = grp.fit_transform(df)
        assert set(out["c"].unique()) == {"x", "y", "Other"}
        assert out["c"].tolist().count("Other") == 2  # 'w' and 'z'

    def test_top_n_keeps_only_n_most_frequent(self):
        df = self._cat_df()
        grp = RareCategoryGrouper(strategy="top_n", top_n=1)
        out = grp.fit_transform(df)
        assert set(out["c"].unique()) == {"x", "Other"}

    def test_unseen_category_at_transform_grouped_to_other(self):
        df_train = self._cat_df()
        grp = RareCategoryGrouper(strategy="min_frequency", min_frequency=0.15)
        grp.fit(df_train)
        out = grp.transform(pd.DataFrame({"c": ["brand_new_category"]}))
        assert out["c"].iloc[0] == "Other"

    def test_custom_other_label(self):
        df = self._cat_df()
        grp = RareCategoryGrouper(strategy="min_frequency", min_frequency=0.15, other_label="RARE")
        out = grp.fit_transform(df)
        assert "RARE" in out["c"].unique()
        assert "Other" not in out["c"].unique()

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            RareCategoryGrouper(strategy="bogus").fit(self._cat_df())

    def test_numeric_columns_ignored_by_default(self):
        df = pd.DataFrame({"cat": ["a", "a", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
        grp = RareCategoryGrouper(strategy="min_frequency", min_frequency=0.5)
        out = grp.fit_transform(df)
        pd.testing.assert_series_equal(out["num"], df["num"])

    def test_get_feature_names_out(self):
        df = self._cat_df()
        grp = RareCategoryGrouper().fit(df)
        assert grp.get_feature_names_out() == ["c"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            RareCategoryGrouper().transform(pd.DataFrame({"c": ["a"]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            RareCategoryGrouper().fit(np.array([["a", "b"]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            RareCategoryGrouper().fit(pd.DataFrame({"c": pd.Series([], dtype=str)}))

    def test_joblib_serialisable(self, tmp_path):
        df = self._cat_df()
        grp = RareCategoryGrouper(strategy="min_frequency", min_frequency=0.15)
        grp.fit(df)
        path = tmp_path / "rare_grouper.joblib"
        joblib.dump(grp, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(grp.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# DistributionTransformer
# ---------------------------------------------------------------------------


class TestDistributionTransformer:
    def test_none_strategy_is_passthrough(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        out = DistributionTransformer(strategy="none").fit_transform(df)
        pd.testing.assert_frame_equal(out, df)

    def test_log_transforms_positive_values(self):
        df = pd.DataFrame({"a": [1.0, np.e, np.e**2]})
        out = DistributionTransformer(strategy="log").fit_transform(df)
        np.testing.assert_allclose(out["a"].tolist(), [0.0, 1.0, 2.0], atol=1e-9)

    def test_log_raises_on_non_positive_value(self):
        df = pd.DataFrame({"a": [1.0, 0.0, 5.0]})
        with pytest.raises(ValueError, match="strictly positive"):
            DistributionTransformer(strategy="log").fit(df)

    def test_log1p_allows_zero(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 3.0]})
        out = DistributionTransformer(strategy="log1p").fit_transform(df)
        np.testing.assert_allclose(out["a"].iloc[0], 0.0, atol=1e-9)

    def test_log1p_raises_below_negative_one(self):
        df = pd.DataFrame({"a": [1.0, -1.0, 5.0]})
        with pytest.raises(ValueError, match="> -1"):
            DistributionTransformer(strategy="log1p").fit(df)

    def test_yeo_johnson_handles_negative_values(self):
        df = pd.DataFrame({"a": [-100.0, -5.0, 0.0, 5.0, 100.0]})
        out = DistributionTransformer(strategy="yeo_johnson").fit_transform(df)
        assert out["a"].isna().sum() == 0

    def test_yeo_johnson_reduces_skew_on_skewed_data(self):
        # Right-skewed with a negative offset — Box-Cox couldn't handle this
        # (values <= 0), which is exactly why yeo_johnson is the recommended
        # default over log/log1p for arbitrary-range banking numerics.
        skewed = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 500.0, 800.0]) - 10.0
        df = pd.DataFrame({"a": skewed})
        out = DistributionTransformer(strategy="yeo_johnson").fit_transform(df)
        assert out["a"].isna().sum() == 0
        assert abs(out["a"].skew()) < abs(df["a"].skew())

    def test_only_specified_features_transformed(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        out = DistributionTransformer(strategy="log", features=["a"]).fit_transform(df)
        pd.testing.assert_series_equal(out["b"], df["b"])
        assert not out["a"].equals(df["a"])

    def test_sqrt_transforms_correctly(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 4.0, 9.0]})
        out = DistributionTransformer(strategy="sqrt").fit_transform(df)
        np.testing.assert_allclose(out["a"].tolist(), [0.0, 1.0, 2.0, 3.0])

    def test_sqrt_raises_on_negative_value(self):
        df = pd.DataFrame({"a": [1.0, -1.0, 4.0]})
        with pytest.raises(ValueError, match="non-negative"):
            DistributionTransformer(strategy="sqrt").fit(df)

    def test_cbrt_handles_negative_values(self):
        df = pd.DataFrame({"a": [-8.0, -1.0, 0.0, 1.0, 8.0]})
        out = DistributionTransformer(strategy="cbrt").fit_transform(df)
        np.testing.assert_allclose(out["a"].tolist(), [-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_reciprocal_transforms_correctly(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 4.0]})
        out = DistributionTransformer(strategy="reciprocal").fit_transform(df)
        np.testing.assert_allclose(out["a"].tolist(), [1.0, 0.5, 0.25])

    def test_reciprocal_raises_on_zero_value(self):
        df = pd.DataFrame({"a": [1.0, 0.0, 4.0]})
        with pytest.raises(ValueError, match="nonzero"):
            DistributionTransformer(strategy="reciprocal").fit(df)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            DistributionTransformer(strategy="bogus").fit(pd.DataFrame({"a": [1.0, 2.0]}))

    def test_get_feature_names_out(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "cat": ["x", "y"]})
        dt = DistributionTransformer(strategy="log").fit(df)
        assert dt.get_feature_names_out() == ["a"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            DistributionTransformer().transform(pd.DataFrame({"a": [1.0]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            DistributionTransformer().fit(np.array([[1.0, 2.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            DistributionTransformer().fit(pd.DataFrame({"a": pd.Series([], dtype=float)}))

    def test_joblib_serialisable(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})
        dt = DistributionTransformer(strategy="yeo_johnson").fit(df)
        path = tmp_path / "dist_transformer.joblib"
        joblib.dump(dt, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(dt.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# PercentileRankTransformer
# ---------------------------------------------------------------------------


class TestPercentileRankTransformer:
    def test_fit_transform_produces_expected_percentiles(self):
        df = pd.DataFrame({"a": [1.0, 5.0, 10.0, 100.0]})
        out = PercentileRankTransformer().fit_transform(df)
        assert out["a"].tolist() == [0.0, 0.25, 0.5, 0.75]

    def test_nan_stays_nan(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan]})
        out = PercentileRankTransformer().fit_transform(df)
        assert pd.isna(out["a"].iloc[2])

    def test_value_below_training_range_maps_to_zero(self):
        df_train = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        prt = PercentileRankTransformer().fit(df_train)
        out = prt.transform(pd.DataFrame({"a": [-100.0]}))
        assert out["a"].iloc[0] == 0.0

    def test_value_above_training_range_maps_to_one(self):
        df_train = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        prt = PercentileRankTransformer().fit(df_train)
        out = prt.transform(pd.DataFrame({"a": [1000.0]}))
        assert out["a"].iloc[0] == 1.0

    def test_all_nan_column_maps_to_nan_at_transform(self):
        df_train = pd.DataFrame({"a": [np.nan, np.nan]})
        prt = PercentileRankTransformer().fit(df_train)
        out = prt.transform(pd.DataFrame({"a": [1.0]}))
        assert pd.isna(out["a"].iloc[0])

    def test_only_specified_features_transformed(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        out = PercentileRankTransformer(features=["a"]).fit_transform(df)
        pd.testing.assert_series_equal(out["b"], df["b"])
        assert not out["a"].equals(df["a"])

    def test_dense_rank_collapses_ties(self):
        df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0, 3.0]})
        out = PercentileRankTransformer(method="dense").fit_transform(df)
        assert out["a"].tolist() == [0.0, 0.0, 1.0, 2.0, 2.0]

    def test_global_rank_counts_all_rows_including_ties(self):
        df = pd.DataFrame({"a": [1.0, 1.0, 2.0]})
        out = PercentileRankTransformer(method="global").fit_transform(df)
        assert out["a"].tolist() == [2.0, 2.0, 3.0]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            PercentileRankTransformer(method="bogus").fit(pd.DataFrame({"a": [1.0, 2.0]}))

    def test_numeric_columns_only_by_default(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "cat": ["x", "y"]})
        out = PercentileRankTransformer().fit_transform(df)
        pd.testing.assert_series_equal(out["cat"], df["cat"])

    def test_get_feature_names_out(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "cat": ["x", "y"]})
        prt = PercentileRankTransformer().fit(df)
        assert prt.get_feature_names_out() == ["a"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            PercentileRankTransformer().transform(pd.DataFrame({"a": [1.0]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            PercentileRankTransformer().fit(np.array([[1.0, 2.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            PercentileRankTransformer().fit(pd.DataFrame({"a": pd.Series([], dtype=float)}))

    def test_joblib_serialisable(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})
        prt = PercentileRankTransformer().fit(df)
        path = tmp_path / "rank.joblib"
        joblib.dump(prt, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(prt.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# GroupRelativeTransformer
# ---------------------------------------------------------------------------


class TestGroupRelativeTransformer:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "balance": [100.0, 200.0, 300.0, 10.0, 20.0, 30.0],
                "segment": ["A", "A", "A", "B", "B", "B"],
            }
        )

    def test_ratio_to_group_mean(self):
        out = GroupRelativeTransformer(group_col="segment", strategy="ratio").fit_transform(
            self._df()
        )
        assert out["balance_ratio_to_segment_mean"].tolist() == [0.5, 1.0, 1.5, 0.5, 1.0, 1.5]

    def test_zscore_within_group(self):
        out = GroupRelativeTransformer(group_col="segment", strategy="zscore").fit_transform(
            self._df()
        )
        assert out["balance_zscore_within_segment"].tolist() == [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]

    def test_both_strategy_adds_both_columns(self):
        out = GroupRelativeTransformer(group_col="segment", strategy="both").fit_transform(
            self._df()
        )
        assert "balance_ratio_to_segment_mean" in out.columns
        assert "balance_zscore_within_segment" in out.columns

    def test_original_columns_preserved(self):
        df = self._df()
        out = GroupRelativeTransformer(group_col="segment").fit_transform(df)
        pd.testing.assert_series_equal(out["balance"], df["balance"])
        pd.testing.assert_series_equal(out["segment"], df["segment"])

    def test_unseen_group_falls_back_to_global_stats(self):
        grt = GroupRelativeTransformer(group_col="segment").fit(self._df())
        out = grt.transform(pd.DataFrame({"balance": [50.0], "segment": ["C"]}))
        assert pd.notna(out["balance_ratio_to_segment_mean"].iloc[0])
        assert pd.notna(out["balance_zscore_within_segment"].iloc[0])

    def test_zero_group_mean_produces_nan_ratio_not_crash(self):
        df = pd.DataFrame({"a": [0.0, 0.0, 5.0], "grp": ["X", "X", "Y"]})
        out = GroupRelativeTransformer(group_col="grp", strategy="ratio").fit_transform(df)
        assert pd.isna(out["a_ratio_to_grp_mean"].iloc[0])

    def test_single_row_group_has_zero_std_not_nan(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "grp": ["X", "Y", "Y"]})
        grt = GroupRelativeTransformer(group_col="grp", strategy="zscore").fit(df)
        assert grt.group_stds_["a"]["X"] == 0.0

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            GroupRelativeTransformer(group_col="segment", strategy="bogus").fit(self._df())

    def test_missing_group_col_raises(self):
        with pytest.raises(ValueError, match="group_col"):
            GroupRelativeTransformer(group_col="nonexistent").fit(self._df())

    def test_get_feature_names_out(self):
        grt = GroupRelativeTransformer(group_col="segment", strategy="both").fit(self._df())
        assert grt.get_feature_names_out() == [
            "balance_ratio_to_segment_mean",
            "balance_zscore_within_segment",
        ]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            GroupRelativeTransformer(group_col="segment").transform(self._df())

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            GroupRelativeTransformer(group_col="segment").fit(np.array([[1.0, 2.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            GroupRelativeTransformer(group_col="segment").fit(
                pd.DataFrame({"segment": pd.Series([], dtype=str)})
            )

    def test_joblib_serialisable(self, tmp_path):
        grt = GroupRelativeTransformer(group_col="segment").fit(self._df())
        path = tmp_path / "group_relative.joblib"
        joblib.dump(grt, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(grt.transform(self._df()), loaded.transform(self._df()))

    def test_group_col_none_uses_overall_stats(self):
        df = pd.DataFrame({"balance": [10.0, 20.0, 30.0, 40.0]})
        out = GroupRelativeTransformer(group_col=None, strategy="ratio").fit_transform(df)
        overall_mean = df["balance"].mean()
        np.testing.assert_allclose(
            out["balance_ratio_to_mean"].tolist(), (df["balance"] / overall_mean).tolist()
        )

    def test_difference_strategy(self):
        df = pd.DataFrame({"segment": ["a", "a", "b", "b"], "balance": [10.0, 20.0, 100.0, 300.0]})
        out = GroupRelativeTransformer(group_col="segment", strategy="difference").fit_transform(df)
        assert out["balance_diff_from_segment_mean"].tolist() == [-5.0, 5.0, -100.0, 100.0]

    def test_pct_of_total_strategy(self):
        df = pd.DataFrame({"segment": ["a", "a", "b", "b"], "balance": [10.0, 30.0, 100.0, 300.0]})
        out = GroupRelativeTransformer(group_col="segment", strategy="pct_of_total").fit_transform(
            df
        )
        np.testing.assert_allclose(
            out["balance_pct_of_segment_total"].tolist(), [25.0, 75.0, 25.0, 75.0]
        )

    def test_pct_of_total_no_group(self):
        df = pd.DataFrame({"balance": [10.0, 30.0, 60.0]})
        out = GroupRelativeTransformer(group_col=None, strategy="pct_of_total").fit_transform(df)
        np.testing.assert_allclose(out["balance_pct_of_total"].tolist(), [10.0, 30.0, 60.0])

    def test_unknown_strategy_still_raises(self):
        df = pd.DataFrame({"segment": ["a", "b"], "balance": [1.0, 2.0]})
        with pytest.raises(ValueError, match="strategy"):
            GroupRelativeTransformer(group_col="segment", strategy="bogus").fit(df)


# ---------------------------------------------------------------------------
# PCATransformer
# ---------------------------------------------------------------------------


class TestPCATransformer:
    def _df(self) -> pd.DataFrame:
        rng = np.random.RandomState(0)
        return pd.DataFrame(
            {
                "x1": rng.randn(50),
                "x2": rng.randn(50),
                "x3": rng.randn(50),
                "cat": ["a"] * 50,
            }
        )

    def test_reduces_to_requested_components(self):
        out = PCATransformer(n_components=2).fit_transform(self._df())
        assert set(out.columns) == {"cat", "pc1", "pc2"}

    def test_original_columns_dropped(self):
        out = PCATransformer(n_components=2).fit_transform(self._df())
        assert "x1" not in out.columns
        assert "x2" not in out.columns
        assert "x3" not in out.columns

    def test_non_numeric_columns_untouched(self):
        df = self._df()
        out = PCATransformer(n_components=2).fit_transform(df)
        pd.testing.assert_series_equal(out["cat"], df["cat"])

    def test_explained_variance_ratio_populated(self):
        pca = PCATransformer(n_components=2).fit(self._df())
        assert len(pca.explained_variance_ratio_) == 2
        assert 0.0 <= pca.explained_variance_ratio_.sum() <= 1.0

    def test_custom_prefix(self):
        out = PCATransformer(n_components=2, prefix="comp").fit_transform(self._df())
        assert {"comp1", "comp2"}.issubset(out.columns)

    def test_only_specified_features_reduced(self):
        df = self._df()
        out = PCATransformer(n_components=1, features=["x1", "x2"]).fit_transform(df)
        assert "x3" in out.columns  # not part of the reduction, left untouched
        assert "x1" not in out.columns

    def test_raises_with_fewer_than_two_columns(self):
        df = pd.DataFrame({"only_one": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 2"):
            PCATransformer().fit(df)

    def test_missing_fitted_column_at_transform_raises(self):
        pca = PCATransformer(n_components=2).fit(self._df())
        with pytest.raises(ValueError, match="missing fitted columns"):
            pca.transform(pd.DataFrame({"x1": [1.0], "x2": [2.0]}))  # x3 missing

    def test_default_n_components_capped_to_available_columns(self):
        # settings.pca_default_components=10 but only 3 numeric cols available.
        # sklearn's PCA raises rather than capping itself — regression test
        # for PCATransformer's own capping logic (with an INFO log) that
        # avoids that crash for the extremely common "fewer columns than the
        # global default" case.
        pca = PCATransformer().fit(self._df())
        assert pca._pca.n_components_ == 3

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            PCATransformer().transform(pd.DataFrame({"a": [1.0]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            PCATransformer().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            PCATransformer().fit(pd.DataFrame({"a": pd.Series([], dtype=float)}))

    def test_joblib_serialisable(self, tmp_path):
        df = self._df()
        pca = PCATransformer(n_components=2).fit(df)
        path = tmp_path / "pca.joblib"
        joblib.dump(pca, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(pca.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# Transformer registry (dscompanion.features.registry)
# ---------------------------------------------------------------------------

_NUMERIC_STEPS = [
    "impute",
    "clip_lower",
    "clip_upper",
    "clip_both",
    "clip_iqr",
    "clip_zscore",
    "clip_mad",
    "log",
    "log1p",
    "sqrt",
    "cbrt",
    "reciprocal",
    "yeo_johnson",
    "quantile_uniform",
    "quantile_normal",
    "bucket_quantile",
    "bucket_uniform",
    "bucket_tree",
    "percentile_rank",
    "dense_rank",
    "global_rank",
    "scale",
    "polynomial",
    "spline",
    "noise",
]
_CATEGORICAL_STEPS = [
    "ordinal_encode",
    "onehot_encode",
    "rare_group",
    "target_encode",
    "frequency_encode",
    "woe_encode",
    "binary_encode",
    "hash_encode",
]
_STEPS_NEEDING_Y = {"target_encode", "woe_encode"}


class TestTransformerRegistry:
    def test_registered_names_sorted_and_nonempty(self):
        names = registered_transformer_names()
        assert names == sorted(names)
        assert len(names) >= 25

    def test_all_registered_names_covered_by_test_lists(self):
        covered = set(_NUMERIC_STEPS) | set(_CATEGORICAL_STEPS)
        assert covered == set(registered_transformer_names())

    def test_build_transformer_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown transform step"):
            build_transformer("does_not_exist")

    @pytest.fixture
    def numeric_col(self):
        return pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]})

    @pytest.fixture
    def categorical_col(self):
        return pd.DataFrame({"x": ["a", "b", "a", "b", "c", "a", "b", "c", "a", "b"]})

    @pytest.fixture
    def y(self):
        return pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    @pytest.mark.parametrize("name", _NUMERIC_STEPS)
    def test_every_numeric_transformer_builds_and_fit_transforms(self, name, numeric_col, y):
        transformer = build_transformer(name)
        transformer.fit(numeric_col, y)
        out = transformer.transform(numeric_col)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(numeric_col)

    @pytest.mark.parametrize("name", _CATEGORICAL_STEPS)
    def test_every_categorical_transformer_builds_and_fit_transforms(
        self, name, categorical_col, y
    ):
        needs_y = name in _STEPS_NEEDING_Y
        transformer = build_transformer(name)
        transformer.fit(categorical_col, y if needs_y else None)
        out = transformer.transform(categorical_col)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(categorical_col)


# ---------------------------------------------------------------------------
# TransformStep / ColumnRecipe (dscompanion.features.transform_chain)
# ---------------------------------------------------------------------------


class TestTransformStep:
    def test_valid_transformer_name_accepted(self):
        step = TransformStep(transformer="log")
        assert step.params == {}

    def test_unknown_transformer_name_rejected(self):
        with pytest.raises(ValidationError):
            TransformStep(transformer="does_not_exist")

    def test_params_override_stored(self):
        step = TransformStep(transformer="clip_lower", params={"lower": 0.1})
        assert step.params == {"lower": 0.1}


class TestColumnRecipe:
    def test_defaults(self):
        recipe = ColumnRecipe(column="balance")
        assert recipe.steps == []
        assert recipe.source == "manual"
        assert recipe.rationale is None

    def test_source_must_be_manual_or_recommended(self):
        with pytest.raises(ValidationError):
            ColumnRecipe(column="balance", source="bogus")

    def test_json_round_trip(self):
        recipe = ColumnRecipe(
            column="balance",
            steps=[TransformStep(transformer="impute"), TransformStep(transformer="log1p")],
            source="recommended",
            rationale="skewed",
        )
        restored = ColumnRecipe.model_validate_json(recipe.model_dump_json())
        assert restored == recipe


# ---------------------------------------------------------------------------
# FeatureTransformChain (dscompanion.features.transform_chain)
# ---------------------------------------------------------------------------


class TestFeatureTransformChain:
    @pytest.fixture
    def df(self):
        rng = np.random.RandomState(0)
        skewed = np.concatenate([rng.uniform(0, 100, 90), rng.uniform(5000, 10000, 10)])
        return pd.DataFrame(
            {
                "balance": skewed,
                "age": rng.randint(18, 70, 100).astype(float),
                "job": rng.choice(["engineer", "doctor", "teacher"], 100),
            }
        )

    @pytest.fixture
    def y(self):
        rng = np.random.RandomState(1)
        return pd.Series(rng.choice([0, 1], 100))

    def test_passthrough_when_no_recipes_given(self, df, y):
        chain = FeatureTransformChain(recipes={}).fit(df, y)
        out = chain.transform(df)
        pd.testing.assert_frame_equal(out, df)

    def test_empty_steps_recipe_is_passthrough(self, df, y):
        recipes = {"balance": ColumnRecipe(column="balance", steps=[])}
        chain = FeatureTransformChain(recipes=recipes).fit(df, y)
        out = chain.transform(df)
        pd.testing.assert_series_equal(out["balance"], df["balance"])

    def test_worked_example_chain_balance_full_pipeline(self, df, y):
        """impute -> clip_lower -> clip_upper -> log1p -> bucket -> bucket-encode,
        a full worked example for a skewed 'balance' column."""
        recipe = ColumnRecipe(
            column="balance",
            steps=[
                TransformStep(transformer="impute"),
                TransformStep(transformer="clip_lower"),
                TransformStep(transformer="clip_upper"),
                TransformStep(transformer="log1p"),
                TransformStep(
                    transformer="bucket_quantile", params={"n_bins": 4, "encode": "labels"}
                ),
                TransformStep(transformer="target_encode", params={"cardinality_threshold": 0}),
            ],
        )
        chain = FeatureTransformChain(recipes={"balance": recipe}).fit(df, y)
        out = chain.transform(df)

        assert list(out.columns) == ["balance", "age", "job"]
        assert pd.api.types.is_float_dtype(out["balance"])
        assert out["balance"].between(0, 1).all()
        pd.testing.assert_series_equal(out["age"], df["age"])
        pd.testing.assert_series_equal(out["job"], df["job"])

    def test_column_expanding_step_mixed_with_passthrough(self, df, y):
        recipe = ColumnRecipe(column="job", steps=[TransformStep(transformer="onehot_encode")])
        chain = FeatureTransformChain(recipes={"job": recipe}).fit(df, y)
        out = chain.transform(df)

        expanded_cols = [c for c in out.columns if c.startswith("job_")]
        assert len(expanded_cols) == df["job"].nunique()
        assert "job" not in out.columns
        assert list(out.columns) == ["balance", "age"] + expanded_cols

    def test_get_feature_names_out_matches_transform_columns(self, df, y):
        recipe = ColumnRecipe(column="job", steps=[TransformStep(transformer="onehot_encode")])
        chain = FeatureTransformChain(recipes={"job": recipe}).fit(df, y)
        assert chain.get_feature_names_out() == list(chain.transform(df).columns)

    def test_unknown_column_in_recipe_raises(self, df, y):
        recipe = ColumnRecipe(column="does_not_exist", steps=[TransformStep(transformer="log1p")])
        with pytest.raises(ValueError, match="not present in X"):
            FeatureTransformChain(recipes={"does_not_exist": recipe}).fit(df, y)

    def test_not_fitted_error_on_transform(self, df):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            FeatureTransformChain(recipes={}).transform(df)

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            FeatureTransformChain(recipes={}).fit(np.array([[1.0]]))

    def test_value_error_on_empty_dataframe_fit(self):
        with pytest.raises(ValueError, match="empty"):
            FeatureTransformChain(recipes={}).fit(pd.DataFrame({"a": pd.Series([], dtype=float)}))

    def test_joblib_serialisable(self, df, y, tmp_path):
        recipe = ColumnRecipe(column="balance", steps=[TransformStep(transformer="log1p")])
        chain = FeatureTransformChain(recipes={"balance": recipe}).fit(df, y)
        path = tmp_path / "chain.joblib"
        joblib.dump(chain, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(chain.transform(df), loaded.transform(df))

    def test_recipe_json_serialisable_independent_of_fitted_chain(self, df, y):
        recipe = ColumnRecipe(
            column="balance", steps=[TransformStep(transformer="log1p")], source="manual"
        )
        FeatureTransformChain(recipes={"balance": recipe}).fit(df, y)
        restored = ColumnRecipe.model_validate_json(recipe.model_dump_json())
        assert restored == recipe

    def test_fit_transform_surfaces_noised_training_data(self, df, y):
        """fit_transform() must return the actual noised training block — fit()
        alone discards it (see FeatureTransformChain._fit_columns)."""
        recipe = ColumnRecipe(
            column="balance",
            steps=[TransformStep(transformer="noise", params={"affected_row_frac": 0.3})],
        )
        chain = FeatureTransformChain(recipes={"balance": recipe})
        out = chain.fit_transform(df, y)
        perturbed_frac = (out["balance"].to_numpy() != df["balance"].to_numpy()).mean()
        assert 0.1 < perturbed_frac < 0.5

    def test_fit_then_transform_on_training_data_stays_clean(self, df, y):
        """Plain fit().transform() must never inject noise, even on the same data
        used to fit — only the chain's own fit_transform() does."""
        recipe = ColumnRecipe(
            column="balance",
            steps=[TransformStep(transformer="noise", params={"affected_row_frac": 0.3})],
        )
        chain = FeatureTransformChain(recipes={"balance": recipe}).fit(df, y)
        out = chain.transform(df)
        pd.testing.assert_series_equal(out["balance"], df["balance"])

    def test_holdout_data_never_perturbed_by_noise_step(self, df, y):
        recipe = ColumnRecipe(
            column="balance",
            steps=[TransformStep(transformer="noise", params={"affected_row_frac": 0.5})],
        )
        chain = FeatureTransformChain(recipes={"balance": recipe})
        chain.fit_transform(df, y)
        holdout = df.copy()
        out = chain.transform(holdout)
        pd.testing.assert_series_equal(out["balance"], holdout["balance"])


# ---------------------------------------------------------------------------
# recommend_column_recipe (dscompanion.features.recommend)
# ---------------------------------------------------------------------------


class TestRecommendColumnRecipe:
    def test_numeric_no_missing_no_skew_recommends_no_steps(self):
        stats = pd.Series({"missing_pct": 0.0, "skewness": 0.1, "min": 1.0, "constant_flag": False})
        recipe = recommend_column_recipe("age", stats, dtype="numeric")
        assert recipe.steps == []
        assert recipe.source == "recommended"
        assert recipe.rationale

    def test_numeric_constant_column_no_steps(self):
        stats = pd.Series({"missing_pct": 0.0, "skewness": 0.0, "min": 5.0, "constant_flag": True})
        recipe = recommend_column_recipe("flag", stats, dtype="numeric")
        assert recipe.steps == []
        assert "onstant" in recipe.rationale

    def test_numeric_missing_recommends_impute(self):
        stats = pd.Series({"missing_pct": 0.1, "skewness": 0.0, "min": 1.0, "constant_flag": False})
        recipe = recommend_column_recipe("age", stats, dtype="numeric")
        assert [s.transformer for s in recipe.steps] == ["impute"]

    def test_numeric_skewed_positive_min_recommends_clip_then_log(self):
        stats = pd.Series(
            {"missing_pct": 0.0, "skewness": 5.0, "min": 10.0, "constant_flag": False}
        )
        recipe = recommend_column_recipe("balance", stats, dtype="numeric")
        assert [s.transformer for s in recipe.steps] == ["clip_lower", "clip_upper", "log"]

    def test_numeric_skewed_zero_min_recommends_log1p(self):
        stats = pd.Series({"missing_pct": 0.0, "skewness": 5.0, "min": 0.0, "constant_flag": False})
        recipe = recommend_column_recipe("balance", stats, dtype="numeric")
        assert [s.transformer for s in recipe.steps] == ["clip_lower", "clip_upper", "log1p"]

    def test_numeric_skewed_negative_min_recommends_yeo_johnson(self):
        stats = pd.Series(
            {"missing_pct": 0.0, "skewness": 5.0, "min": -50.0, "constant_flag": False}
        )
        recipe = recommend_column_recipe("balance", stats, dtype="numeric")
        assert [s.transformer for s in recipe.steps] == ["clip_lower", "clip_upper", "yeo_johnson"]

    def test_numeric_missing_and_skewed_combines_steps(self):
        stats = pd.Series({"missing_pct": 0.2, "skewness": 6.0, "min": 5.0, "constant_flag": False})
        recipe = recommend_column_recipe("balance", stats, dtype="numeric")
        assert [s.transformer for s in recipe.steps] == [
            "impute",
            "clip_lower",
            "clip_upper",
            "log",
        ]

    def test_categorical_low_cardinality_recommends_onehot(self):
        stats = pd.Series({"missing_pct": 0.0, "cardinality_flag": False})
        recipe = recommend_column_recipe("job", stats, dtype="categorical")
        assert [s.transformer for s in recipe.steps] == ["onehot_encode"]

    def test_categorical_high_cardinality_recommends_rare_group_then_target_encode(self):
        stats = pd.Series({"missing_pct": 0.0, "cardinality_flag": True})
        recipe = recommend_column_recipe("customer_id", stats, dtype="categorical")
        assert [s.transformer for s in recipe.steps] == ["rare_group", "target_encode"]

    def test_categorical_missing_recommends_impute_first(self):
        stats = pd.Series({"missing_pct": 0.05, "cardinality_flag": False})
        recipe = recommend_column_recipe("job", stats, dtype="categorical")
        assert [s.transformer for s in recipe.steps] == ["impute", "onehot_encode"]

    def test_invalid_dtype_raises(self):
        stats = pd.Series({"missing_pct": 0.0})
        with pytest.raises(ValueError, match="dtype must be"):
            recommend_column_recipe("x", stats, dtype="bogus")

    def test_recommended_steps_are_all_registered_names(self):
        stats = pd.Series({"missing_pct": 0.2, "skewness": 6.0, "min": 5.0, "constant_flag": False})
        recipe = recommend_column_recipe("balance", stats, dtype="numeric")
        valid = set(registered_transformer_names())
        assert all(s.transformer in valid for s in recipe.steps)

    def test_recommended_recipe_is_directly_usable_by_feature_transform_chain(self):
        rng = np.random.RandomState(0)
        skewed = np.concatenate([rng.uniform(0, 100, 90), rng.uniform(5000, 10000, 10)])
        df = pd.DataFrame({"balance": skewed})
        y = pd.Series(rng.choice([0, 1], 100))

        stats = pd.Series(
            {
                "missing_pct": 0.0,
                "skewness": float(df["balance"].skew()),
                "min": float(df["balance"].min()),
                "constant_flag": False,
            }
        )
        recipe = recommend_column_recipe("balance", stats, dtype="numeric")
        chain = FeatureTransformChain(recipes={"balance": recipe}).fit(df, y)
        out = chain.transform(df)
        assert "balance" in out.columns
        assert len(out) == len(df)


# ---------------------------------------------------------------------------
# FT4.6 — column_overrides on existing transformers
# ---------------------------------------------------------------------------


class TestWinsorizationColumnOverrides:
    def test_column_override_uses_own_lower_upper(self):
        df = pd.DataFrame(
            {
                "a": list(range(1, 101)),  # 1..100
                "b": list(range(1, 101)),
            }
        )
        wt = WinsorizationTransformer(
            lower=0.01, upper=0.01, column_overrides={"a": {"lower": 0.10, "upper": 0.10}}
        ).fit(df)
        # "a" uses the override (10th/90th percentile) — a much tighter clip than "b"'s 1%/99%.
        a_lo, a_hi = wt.clip_values_["a"]
        b_lo, b_hi = wt.clip_values_["b"]
        assert a_lo > b_lo
        assert a_hi < b_hi

    def test_column_without_override_uses_global_default(self):
        df = pd.DataFrame({"a": list(range(1, 101)), "b": list(range(1, 101))})
        wt = WinsorizationTransformer(
            lower=0.05, upper=0.05, column_overrides={"a": {"lower": 0.20}}
        ).fit(df)
        # "b" has no override entry at all — uses the global 5%/5%.
        expected_lo = df["b"].quantile(0.05)
        expected_hi = df["b"].quantile(0.95)
        assert wt.clip_values_["b"] == (expected_lo, expected_hi)

    def test_override_partial_key_falls_back_for_other_key(self):
        df = pd.DataFrame({"a": list(range(1, 101))})
        # Override only "lower" — "upper" should fall back to the constructor default.
        wt = WinsorizationTransformer(
            lower=0.01, upper=0.01, column_overrides={"a": {"lower": 0.5}}
        ).fit(df)
        expected_hi = df["a"].quantile(1 - 0.01)
        assert wt.clip_values_["a"][1] == expected_hi


class TestRareCategoryGrouperColumnOverrides:
    def test_column_override_uses_own_strategy(self):
        df = pd.DataFrame(
            {
                "a": ["x"] * 50 + ["y"] * 30 + ["z"] * 20,
                "b": ["x"] * 50 + ["y"] * 30 + ["z"] * 20,
            }
        )
        rcg = RareCategoryGrouper(
            strategy="min_frequency",
            min_frequency=0.01,
            column_overrides={"a": {"strategy": "top_n", "top_n": 1}},
        ).fit(df)
        assert rcg.kept_categories_["a"] == {"x"}
        assert rcg.kept_categories_["b"] == {"x", "y", "z"}

    def test_invalid_override_strategy_raises(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="strategy must be one of"):
            RareCategoryGrouper(column_overrides={"a": {"strategy": "bogus"}}).fit(df)


class TestOneHotEncoderColumnOverrides:
    def test_column_override_drop_first(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["x", "y", "z"]})
        enc = OneHotEncoder(drop_first=False, column_overrides={"a": {"drop_first": True}}).fit(df)
        assert len(enc.categories_["a"]) == 2  # dropped one
        assert len(enc.categories_["b"]) == 3  # kept all


class TestDistributionTransformerColumnOverrides:
    def test_different_columns_use_different_strategies(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0], "b": [-5.0, 0.0, 5.0, 100.0]})
        dt = DistributionTransformer(
            strategy="none",
            column_overrides={"a": {"strategy": "log"}, "b": {"strategy": "yeo_johnson"}},
        ).fit(df)
        assert dt.column_strategies_ == {"a": "log", "b": "yeo_johnson"}
        out = dt.transform(df)
        np.testing.assert_allclose(out["a"].tolist(), np.log(df["a"]).tolist())
        assert not out["b"].equals(df["b"])

    def test_quantile_uniform_output_in_unit_range(self):
        df = pd.DataFrame({"a": np.linspace(0, 1000, 50)})
        out = DistributionTransformer(strategy="quantile_uniform").fit_transform(df)
        assert out["a"].between(0.0, 1.0).all()

    def test_quantile_normal_output_is_roughly_gaussian(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"a": rng.exponential(scale=2.0, size=200)})
        out = DistributionTransformer(strategy="quantile_normal").fit_transform(df)
        assert abs(out["a"].skew()) < abs(df["a"].skew())

    def test_column_override_invalid_strategy_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="strategy must be one of"):
            DistributionTransformer(column_overrides={"a": {"strategy": "bogus"}}).fit(df)


# ---------------------------------------------------------------------------
# BinaryEncoder (FT4.1)
# ---------------------------------------------------------------------------


class TestBinaryEncoder:
    def test_bit_pattern_matches_ordinal_index(self):
        df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e"]})
        enc = BinaryEncoder().fit(df)
        out = enc.transform(df)
        # a=0->000, b=1->001, c=2->010, d=3->011, e=4->100 (bin0=LSB)
        assert out.loc[0].tolist() == [0, 0, 0]
        assert out.loc[1].tolist() == [1, 0, 0]
        assert out.loc[2].tolist() == [0, 1, 0]
        assert out.loc[3].tolist() == [1, 1, 0]
        assert out.loc[4].tolist() == [0, 0, 1]

    def test_n_bits_is_ceil_log2_cardinality(self):
        df = pd.DataFrame({"cat": [f"v{i}" for i in range(9)]})  # 9 categories -> ceil(log2(9))=4
        enc = BinaryEncoder().fit(df)
        assert enc.n_bits_["cat"] == 4

    def test_single_category_uses_one_bit(self):
        df = pd.DataFrame({"cat": ["only"] * 5})
        enc = BinaryEncoder().fit(df)
        assert enc.n_bits_["cat"] == 1

    def test_unseen_category_at_transform_is_all_zero(self):
        enc = BinaryEncoder().fit(pd.DataFrame({"cat": ["a", "b", "c"]}))
        out = enc.transform(pd.DataFrame({"cat": ["unseen"]}))
        assert out.iloc[0].tolist() == [0, 0]

    def test_missing_source_column_at_transform_is_all_zero(self):
        enc = BinaryEncoder().fit(pd.DataFrame({"cat": ["a", "b", "c"]}))
        out = enc.transform(pd.DataFrame({"other": [1, 2]}))
        assert (out == 0).all().all()
        assert list(out.columns) == enc.feature_names_out_

    def test_numeric_columns_ignored_by_default(self):
        df = pd.DataFrame({"cat": ["a", "b"], "num": [1.0, 2.0]})
        enc = BinaryEncoder().fit(df)
        assert "num" not in enc.categories_

    def test_get_feature_names_out(self):
        enc = BinaryEncoder().fit(pd.DataFrame({"cat": ["a", "b", "c"]}))
        assert enc.get_feature_names_out() == enc.feature_names_out_

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            BinaryEncoder().transform(pd.DataFrame({"cat": ["a"]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            BinaryEncoder().fit(np.array([["a"]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            BinaryEncoder().fit(pd.DataFrame({"cat": pd.Series([], dtype=object)}))

    def test_joblib_serialisable(self, tmp_path):
        df = pd.DataFrame({"cat": ["a", "b", "c", "d"]})
        enc = BinaryEncoder().fit(df)
        path = tmp_path / "binary_encoder.joblib"
        joblib.dump(enc, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(enc.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# HashEncoder (FT3.4)
# ---------------------------------------------------------------------------


class TestHashEncoder:
    def test_output_shape_is_n_components_wide(self):
        df = pd.DataFrame({"cat": [f"merchant_{i}" for i in range(500)]})
        enc = HashEncoder(n_components=8).fit(df)
        out = enc.transform(df)
        assert out.shape == (500, 8)
        assert list(out.columns) == [f"cat_hash{i}" for i in range(8)]

    def test_exactly_one_nonzero_bucket_per_row(self):
        df = pd.DataFrame({"cat": [f"merchant_{i}" for i in range(500)]})
        enc = HashEncoder(n_components=8).fit(df)
        out = enc.transform(df)
        nonzero_per_row = (out != 0).sum(axis=1)
        assert (nonzero_per_row == 1).all()
        assert out.isin([-1, 0, 1]).all().all()

    def test_deterministic_across_instances(self):
        df = pd.DataFrame({"cat": [f"merchant_{i}" for i in range(200)]})
        out_a = HashEncoder(n_components=8).fit(df).transform(df)
        out_b = HashEncoder(n_components=8).fit(df).transform(df)
        pd.testing.assert_frame_equal(out_a, out_b)

    def test_same_value_always_hashes_to_same_bucket(self):
        df = pd.DataFrame({"cat": ["repeat_me"] * 10 + [f"other_{i}" for i in range(10)]})
        enc = HashEncoder(n_components=8).fit(df)
        out = enc.transform(df)
        # first 10 rows are all the same category -> identical output rows
        first = out.iloc[0]
        for i in range(1, 10):
            assert (out.iloc[i] == first).all()

    def test_no_fit_time_category_inventory_needed(self):
        # Unlike every other encoder, a HashEncoder fit on one set of values transforms
        # a disjoint set of values just fine — nothing was "seen" or "unseen" at fit time.
        enc = HashEncoder(n_components=8).fit(pd.DataFrame({"cat": ["a", "b", "c"]}))
        out = enc.transform(pd.DataFrame({"cat": ["totally_different_value"]}))
        assert (out != 0).sum(axis=1).iloc[0] == 1

    def test_missing_values_hash_consistently(self):
        df = pd.DataFrame({"cat": ["a", None, np.nan, "a"]})
        enc = HashEncoder(n_components=4).fit(df)
        out = enc.transform(df)
        pd.testing.assert_series_equal(out.iloc[1], out.iloc[2], check_names=False)

    def test_missing_source_column_at_transform_is_all_zero(self):
        enc = HashEncoder(n_components=4).fit(pd.DataFrame({"cat": ["a", "b", "c"]}))
        out = enc.transform(pd.DataFrame({"other": [1, 2]}))
        assert (out == 0).all().all()
        assert list(out.columns) == enc.feature_names_out_

    def test_numeric_columns_ignored_by_default(self):
        df = pd.DataFrame({"cat": ["a", "b"], "num": [1.0, 2.0]})
        enc = HashEncoder(n_components=4).fit(df)
        assert "num" not in enc._cols

    def test_default_n_components_from_settings(self):
        from dscompanion.config import settings

        enc = HashEncoder().fit(pd.DataFrame({"cat": ["a", "b"]}))
        assert enc._n_components == settings.hash_encoder_n_components

    def test_get_feature_names_out(self):
        enc = HashEncoder(n_components=4).fit(pd.DataFrame({"cat": ["a", "b", "c"]}))
        assert enc.get_feature_names_out() == enc.feature_names_out_

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            HashEncoder().transform(pd.DataFrame({"cat": ["a"]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            HashEncoder().fit(np.array([["a"]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            HashEncoder().fit(pd.DataFrame({"cat": pd.Series([], dtype=object)}))

    def test_joblib_serialisable(self, tmp_path):
        df = pd.DataFrame({"cat": [f"merchant_{i}" for i in range(50)]})
        enc = HashEncoder(n_components=8).fit(df)
        path = tmp_path / "hash_encoder.joblib"
        joblib.dump(enc, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(enc.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# MultivariateImputer (FT4.2)
# ---------------------------------------------------------------------------


class TestMultivariateImputer:
    @pytest.fixture
    def df(self):
        rng = np.random.RandomState(0)
        a = rng.randn(50)
        b = a * 2 + rng.randn(50) * 0.1
        df = pd.DataFrame({"a": a, "b": b})
        df.loc[5, "a"] = np.nan
        df.loc[10, "b"] = np.nan
        return df

    def test_knn_strategy_fills_missing_values(self, df):
        out = MultivariateImputer(strategy="knn").fit_transform(df)
        assert out.isna().sum().sum() == 0

    def test_iterative_strategy_fills_missing_values(self, df):
        out = MultivariateImputer(strategy="iterative").fit_transform(df)
        assert out.isna().sum().sum() == 0

    def test_invalid_strategy_raises(self, df):
        with pytest.raises(ValueError, match="strategy must be one of"):
            MultivariateImputer(strategy="bogus").fit(df)

    def test_fewer_than_two_columns_raises(self, df):
        with pytest.raises(ValueError, match="at least 2"):
            MultivariateImputer(features=["a"]).fit(df)

    def test_custom_n_neighbors_respected(self, df):
        imp = MultivariateImputer(strategy="knn", n_neighbors=3).fit(df)
        assert imp._imputer.n_neighbors == 3

    def test_custom_max_iter_respected(self, df):
        imp = MultivariateImputer(strategy="iterative", max_iter=2).fit(df)
        assert imp._imputer.max_iter == 2

    def test_get_feature_names_out(self, df):
        imp = MultivariateImputer().fit(df)
        assert imp.get_feature_names_out() == ["a", "b"]

    def test_not_fitted_error_on_transform(self, df):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            MultivariateImputer().transform(df)

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            MultivariateImputer().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            MultivariateImputer().fit(pd.DataFrame({"a": pd.Series([], dtype=float)}))

    def test_missing_fitted_column_at_transform_raises(self, df):
        imp = MultivariateImputer().fit(df)
        with pytest.raises(ValueError, match="missing fitted columns"):
            imp.transform(pd.DataFrame({"a": [1.0]}))

    def test_joblib_serialisable(self, df, tmp_path):
        imp = MultivariateImputer().fit(df)
        path = tmp_path / "mv_imputer.joblib"
        joblib.dump(imp, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(imp.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# PolynomialFeaturesTransformer (FT4.4)
# ---------------------------------------------------------------------------


class TestPolynomialFeaturesTransformer:
    def _df(self):
        return pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [10.0, 20.0, 30.0, 40.0]})

    def test_degree_two_produces_expected_columns(self):
        out = PolynomialFeaturesTransformer(degree=2).fit_transform(self._df())
        assert set(out.columns) == {"x1", "x2", "x1_pow2", "x1_x_x2", "x2_pow2"}

    def test_values_are_correct(self):
        out = PolynomialFeaturesTransformer(degree=2).fit_transform(self._df())
        assert out["x1_pow2"].tolist() == [1.0, 4.0, 9.0, 16.0]
        assert out["x1_x_x2"].tolist() == [10.0, 40.0, 90.0, 160.0]

    def test_interaction_only_excludes_pure_powers(self):
        out = PolynomialFeaturesTransformer(degree=2, interaction_only=True).fit_transform(
            self._df()
        )
        assert "x1_pow2" not in out.columns
        assert "x1_x_x2" in out.columns

    def test_only_specified_features_used(self):
        df = self._df()
        df["untouched"] = [100.0, 200.0, 300.0, 400.0]
        out = PolynomialFeaturesTransformer(degree=2, features=["x1", "x2"]).fit_transform(df)
        assert "untouched" not in out.columns

    def test_missing_fitted_column_at_transform_raises(self):
        pf = PolynomialFeaturesTransformer(degree=2).fit(self._df())
        with pytest.raises(ValueError, match="missing fitted columns"):
            pf.transform(pd.DataFrame({"x1": [1.0]}))

    def test_get_feature_names_out(self):
        pf = PolynomialFeaturesTransformer(degree=2).fit(self._df())
        assert pf.get_feature_names_out() == pf.feature_names_out_

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            PolynomialFeaturesTransformer().transform(pd.DataFrame({"x1": [1.0]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            PolynomialFeaturesTransformer().fit(np.array([[1.0, 2.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            PolynomialFeaturesTransformer().fit(pd.DataFrame({"x1": pd.Series([], dtype=float)}))

    def test_joblib_serialisable(self, tmp_path):
        df = self._df()
        pf = PolynomialFeaturesTransformer(degree=2).fit(df)
        path = tmp_path / "poly.joblib"
        joblib.dump(pf, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(pf.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# SplineFeatureTransformer (FT4.5)
# ---------------------------------------------------------------------------


class TestSplineFeatureTransformer:
    def _df(self, n=30):
        rng = np.random.RandomState(0)
        return pd.DataFrame({"balance": rng.uniform(0, 1000, n)})

    def test_output_column_count_matches_sklearn(self):
        from sklearn.preprocessing import SplineTransformer

        df = self._df()
        ours = SplineFeatureTransformer(n_knots=5, degree=3).fit(df)
        reference = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit(df[["balance"]])
        assert len(ours.feature_names_out_) == reference.transform(df[["balance"]]).shape[1]

    def test_custom_n_knots_and_degree_respected(self):
        df = self._df()
        ours = SplineFeatureTransformer(n_knots=4, degree=2).fit(df)
        assert ours._spline.n_knots == 4
        assert ours._spline.degree == 2

    def test_transform_shape_matches_rows(self):
        df = self._df()
        spl = SplineFeatureTransformer().fit(df)
        out = spl.transform(df)
        assert len(out) == len(df)

    def test_missing_fitted_column_at_transform_raises(self):
        spl = SplineFeatureTransformer().fit(self._df())
        with pytest.raises(ValueError, match="missing fitted columns"):
            spl.transform(pd.DataFrame({"other": [1.0]}))

    def test_get_feature_names_out(self):
        spl = SplineFeatureTransformer().fit(self._df())
        assert spl.get_feature_names_out() == spl.feature_names_out_

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            SplineFeatureTransformer().transform(pd.DataFrame({"balance": [1.0]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            SplineFeatureTransformer().fit(np.array([[1.0], [2.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            SplineFeatureTransformer().fit(pd.DataFrame({"balance": pd.Series([], dtype=float)}))

    def test_joblib_serialisable(self, tmp_path):
        df = self._df()
        spl = SplineFeatureTransformer().fit(df)
        path = tmp_path / "spline.joblib"
        joblib.dump(spl, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(spl.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# NoiseInjector (NI.2)
# ---------------------------------------------------------------------------


class TestNoiseInjector:
    def _numeric_df(self, n=1000, std=1.0):
        rng = np.random.RandomState(0)
        return pd.DataFrame({"x": rng.normal(0, std, n)})

    def _categorical_df(self, n=1000):
        rng = np.random.RandomState(0)
        return pd.DataFrame({"x": rng.choice(["a", "b", "c"], n, p=[0.7, 0.2, 0.1])})

    def _mixed_df(self, n=1000):
        rng = np.random.RandomState(0)
        return pd.DataFrame(
            {
                "num": rng.normal(0, 1, n),
                "cat": rng.choice(["a", "b", "c"], n, p=[0.7, 0.2, 0.1]),
            }
        )

    def test_fit_alone_does_not_mutate_data(self):
        df = self._mixed_df()
        original = df.copy()
        NoiseInjector(random_state=0).fit(df)
        pd.testing.assert_frame_equal(df, original)

    def test_transform_after_fit_is_exact_passthrough(self):
        df = self._mixed_df()
        injector = NoiseInjector(random_state=0).fit(df)
        out = injector.transform(df)
        pd.testing.assert_frame_equal(out, df)

    def test_transform_returns_copy_not_view(self):
        df = self._mixed_df()
        injector = NoiseInjector(random_state=0).fit(df)
        out = injector.transform(df)
        out.iloc[0, 0] = 9999.0
        assert df.iloc[0, 0] != 9999.0

    def test_fit_transform_perturbs_approximately_affected_row_frac(self):
        df = self._numeric_df(n=2000)
        out = NoiseInjector(affected_row_frac=0.2, random_state=0).fit_transform(df)
        perturbed_frac = (out["x"].to_numpy() != df["x"].to_numpy()).mean()
        assert 0.15 < perturbed_frac < 0.25

    def test_numeric_noise_scales_with_column_std(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(
            {"low_std": rng.normal(0, 1, 5000), "high_std": rng.normal(0, 1000, 5000)}
        )
        out = NoiseInjector(
            affected_row_frac=1.0, numeric_noise_scale=0.1, random_state=0
        ).fit_transform(df)
        low_std_diff = (out["low_std"] - df["low_std"]).abs().mean()
        high_std_diff = (out["high_std"] - df["high_std"]).abs().mean()
        assert high_std_diff > low_std_diff * 100

    @pytest.mark.parametrize("noise_type", ["gaussian", "uniform", "laplace"])
    def test_all_numeric_noise_types_run(self, noise_type):
        df = self._numeric_df()
        out = NoiseInjector(
            numeric_noise_type=noise_type, affected_row_frac=0.3, random_state=0
        ).fit_transform(df)
        assert (out["x"].to_numpy() != df["x"].to_numpy()).any()

    def test_categorical_empirical_noise_stays_within_observed_categories(self):
        df = self._categorical_df()
        out = NoiseInjector(
            categorical_noise_type="empirical", affected_row_frac=0.5, random_state=0
        ).fit_transform(df)
        assert set(out["x"].unique()) <= set(df["x"].unique())

    def test_categorical_uniform_noise_stays_within_observed_categories(self):
        df = self._categorical_df()
        out = NoiseInjector(
            categorical_noise_type="uniform", affected_row_frac=0.5, random_state=0
        ).fit_transform(df)
        assert set(out["x"].unique()) <= set(df["x"].unique())

    def test_zero_variance_numeric_column_guarded(self):
        df = pd.DataFrame({"x": [5.0] * 100})
        injector = NoiseInjector(affected_row_frac=1.0, random_state=0).fit(df)
        assert injector.num_cols_ == []
        out = injector.fit_transform(df)
        pd.testing.assert_series_equal(out["x"], df["x"])

    def test_single_category_column_guarded(self):
        df = pd.DataFrame({"x": ["only_one"] * 100})
        injector = NoiseInjector(affected_row_frac=1.0, random_state=0).fit(df)
        assert injector.cat_cols_ == []
        out = injector.fit_transform(df)
        pd.testing.assert_series_equal(out["x"], df["x"])

    def test_determinism_same_random_state_same_output(self):
        df = self._mixed_df()
        out1 = NoiseInjector(affected_row_frac=0.3, random_state=42).fit_transform(df)
        out2 = NoiseInjector(affected_row_frac=0.3, random_state=42).fit_transform(df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_different_random_state_different_output(self):
        df = self._numeric_df()
        out1 = NoiseInjector(affected_row_frac=0.5, random_state=1).fit_transform(df)
        out2 = NoiseInjector(affected_row_frac=0.5, random_state=2).fit_transform(df)
        assert not out1["x"].equals(out2["x"])

    def test_features_param_restricts_columns_considered(self):
        df = self._mixed_df()
        injector = NoiseInjector(features=["num"], random_state=0).fit(df)
        assert injector.num_cols_ == ["num"]
        assert injector.cat_cols_ == []

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            NoiseInjector().transform(pd.DataFrame({"x": [1.0]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            NoiseInjector().fit(np.array([[1.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            NoiseInjector().fit(pd.DataFrame({"x": pd.Series([], dtype=float)}))

    def test_value_error_on_invalid_numeric_noise_type(self):
        with pytest.raises(ValueError, match="numeric_noise_type"):
            NoiseInjector(numeric_noise_type="bogus").fit(self._numeric_df())

    def test_value_error_on_invalid_categorical_noise_type(self):
        with pytest.raises(ValueError, match="categorical_noise_type"):
            NoiseInjector(categorical_noise_type="bogus").fit(self._categorical_df())

    def test_get_feature_names_out_matches_fitted_columns(self):
        df = self._mixed_df()
        injector = NoiseInjector(random_state=0).fit(df)
        assert injector.get_feature_names_out() == ["num", "cat"]

    def test_joblib_serialisable(self, tmp_path):
        df = self._mixed_df()
        injector = NoiseInjector(random_state=0).fit(df)
        path = tmp_path / "noise.joblib"
        joblib.dump(injector, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(injector.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# StatisticalOutlierCapper
# ---------------------------------------------------------------------------


class TestStatisticalOutlierCapper:
    def test_iqr_caps_extreme_values(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
        out = StatisticalOutlierCapper(method="iqr", multiplier=1.5).fit_transform(df)
        assert out["a"].max() < 100.0

    def test_zscore_caps_extreme_values(self):
        df = pd.DataFrame({"a": [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 500.0]})
        out = StatisticalOutlierCapper(method="zscore", multiplier=2.0).fit_transform(df)
        assert out["a"].max() < 500.0

    def test_mad_caps_extreme_values(self):
        df = pd.DataFrame({"a": [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 500.0]})
        out = StatisticalOutlierCapper(method="mad", multiplier=3.0).fit_transform(df)
        assert out["a"].max() < 500.0

    def test_unknown_method_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="method"):
            StatisticalOutlierCapper(method="bogus").fit(df)

    def test_all_nan_column_skipped(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        capper = StatisticalOutlierCapper().fit(df)
        assert "a" not in capper.clip_values_

    def test_zero_spread_column_skipped(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0, 5.0]})
        capper = StatisticalOutlierCapper(method="zscore").fit(df)
        assert "a" not in capper.clip_values_

    def test_column_overrides_apply_per_column_method(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0], "b": [1.0, 2.0, 3.0, 100.0]})
        capper = StatisticalOutlierCapper(
            method="iqr", column_overrides={"b": {"method": "zscore", "multiplier": 1.0}}
        ).fit(df)
        assert capper.clip_values_["a"] != capper.clip_values_["b"]

    def test_get_feature_names_out(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "cat": ["x", "y", "z"]})
        capper = StatisticalOutlierCapper().fit(df)
        assert capper.get_feature_names_out() == ["a"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            StatisticalOutlierCapper().transform(pd.DataFrame({"a": [1.0]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            StatisticalOutlierCapper().fit(np.array([[1.0, 2.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            StatisticalOutlierCapper().fit(pd.DataFrame({"a": pd.Series([], dtype=float)}))

    def test_joblib_serialisable(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})
        capper = StatisticalOutlierCapper(method="iqr").fit(df)
        path = tmp_path / "outlier_capper.joblib"
        joblib.dump(capper, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(capper.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# ColumnArithmeticTransformer
# ---------------------------------------------------------------------------


class TestColumnArithmeticTransformer:
    def test_difference_operation(self):
        df = pd.DataFrame({"limit": [100.0, 200.0], "balance": [30.0, 50.0]})
        out = ColumnArithmeticTransformer("limit", "balance", operation="difference").fit_transform(
            df
        )
        assert out["limit_difference_balance"].tolist() == [70.0, 150.0]

    def test_sum_operation(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = ColumnArithmeticTransformer("a", "b", operation="sum").fit_transform(df)
        assert out["a_sum_b"].tolist() == [4.0, 6.0]

    def test_product_operation(self):
        df = pd.DataFrame({"a": [2.0, 3.0], "b": [5.0, 5.0]})
        out = ColumnArithmeticTransformer("a", "b", operation="product").fit_transform(df)
        assert out["a_product_b"].tolist() == [10.0, 15.0]

    def test_ratio_operation_guards_zero_denominator(self):
        df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 0.0]})
        out = ColumnArithmeticTransformer("a", "b", operation="ratio").fit_transform(df)
        assert out["a_ratio_b"].iloc[0] == 5.0
        assert pd.isna(out["a_ratio_b"].iloc[1])

    def test_custom_output_name(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        out = ColumnArithmeticTransformer(
            "a", "b", operation="sum", output_name="total"
        ).fit_transform(df)
        assert "total" in out.columns

    def test_unknown_operation_raises(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError, match="operation"):
            ColumnArithmeticTransformer("a", "b", operation="bogus").fit(df)

    def test_non_numeric_column_raises(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0]})
        with pytest.raises(ValueError, match="numeric"):
            ColumnArithmeticTransformer("a", "b").fit(df)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found"):
            ColumnArithmeticTransformer("a", "missing").fit(df)

    def test_get_feature_names_out(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        combiner = ColumnArithmeticTransformer("a", "b", operation="sum").fit(df)
        assert combiner.get_feature_names_out() == ["a_sum_b"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            ColumnArithmeticTransformer("a", "b").transform(pd.DataFrame({"a": [1.0], "b": [2.0]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            ColumnArithmeticTransformer("a", "b").fit(np.array([[1.0, 2.0]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            ColumnArithmeticTransformer("a", "b").fit(
                pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=float)})
            )

    def test_joblib_serialisable(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        combiner = ColumnArithmeticTransformer("a", "b", operation="sum").fit(df)
        path = tmp_path / "arithmetic_combiner.joblib"
        joblib.dump(combiner, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(combiner.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# CategoryCombinerTransformer
# ---------------------------------------------------------------------------


class TestCategoryCombinerTransformer:
    def test_combines_two_columns(self):
        df = pd.DataFrame({"city": ["NYC", "LA"], "store": ["Grocery", "Clothing"]})
        out = CategoryCombinerTransformer("city", "store").fit_transform(df)
        assert out["city_store"].tolist() == ["NYC_Grocery", "LA_Clothing"]

    def test_custom_separator(self):
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        out = CategoryCombinerTransformer("a", "b", separator="|").fit_transform(df)
        assert out["a_b"].iloc[0] == "x|y"

    def test_custom_output_name(self):
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        out = CategoryCombinerTransformer("a", "b", output_name="combo").fit_transform(df)
        assert "combo" in out.columns

    def test_cardinality_warning_logged(self, caplog):
        import itertools

        cities = [f"city_{i}" for i in range(20)]
        stores = [f"store_{i}" for i in range(20)]
        df = pd.DataFrame(list(itertools.product(cities, stores))[:150], columns=["city", "store"])
        with caplog.at_level(logging.WARNING):
            CategoryCombinerTransformer("city", "store").fit(df)
        assert any("distinct values" in message for message in caplog.messages)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": ["x"]})
        with pytest.raises(ValueError, match="not found"):
            CategoryCombinerTransformer("a", "missing").fit(df)

    def test_get_feature_names_out(self):
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        combiner = CategoryCombinerTransformer("a", "b").fit(df)
        assert combiner.get_feature_names_out() == ["a_b"]

    def test_not_fitted_error_on_transform(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            CategoryCombinerTransformer("a", "b").transform(pd.DataFrame({"a": ["x"], "b": ["y"]}))

    def test_type_error_on_non_dataframe_fit(self):
        with pytest.raises(TypeError):
            CategoryCombinerTransformer("a", "b").fit(np.array([["x", "y"]]))

    def test_value_error_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            CategoryCombinerTransformer("a", "b").fit(
                pd.DataFrame({"a": pd.Series([], dtype=object), "b": pd.Series([], dtype=object)})
            )

    def test_joblib_serialisable(self, tmp_path):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        combiner = CategoryCombinerTransformer("a", "b").fit(df)
        path = tmp_path / "category_combiner.joblib"
        joblib.dump(combiner, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(combiner.transform(df), loaded.transform(df))


# ---------------------------------------------------------------------------
# DateFeatureExtractor — cyclical extensions
# ---------------------------------------------------------------------------


class TestDateFeatureExtractorCyclical:
    def test_day_of_week_cyclical_added_by_default(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06"])})
        out = DateFeatureExtractor().fit_transform(df)
        assert "date_dow_sin" in out.columns
        assert "date_dow_cos" in out.columns

    def test_day_of_week_cyclical_matches_known_values(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"])})  # Monday, dayofweek=0
        out = DateFeatureExtractor().fit_transform(df)
        assert out["date_dow_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert out["date_dow_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_hour_cyclical_absent_by_default(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-05 14:00:00"])})
        out = DateFeatureExtractor().fit_transform(df)
        assert "date_hour_sin" not in out.columns

    def test_hour_cyclical_added_when_enabled(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-05 06:00:00"])})
        out = DateFeatureExtractor(include_hour_cyclical=True).fit_transform(df)
        assert out["date_hour_sin"].iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert out["date_hour_cos"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_get_feature_names_out_includes_hour_when_enabled(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"])})
        dfe = DateFeatureExtractor(include_hour_cyclical=True).fit(df)
        names = dfe.get_feature_names_out()
        assert "date_hour_sin" in names
        assert "date_hour_cos" in names

    def test_get_feature_names_out_excludes_hour_by_default(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"])})
        dfe = DateFeatureExtractor().fit(df)
        names = dfe.get_feature_names_out()
        assert "date_hour_sin" not in names
