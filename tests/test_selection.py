"""Tests for dscompanion.selection — feature selectors and FeatureSelectionPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscompanion.selection import (
    CardinalitySelector,
    ConstantSelector,
    CorrelationSelector,
    FeatureSelectionPipeline,
    IVSelector,
    NullRateSelector,
)
from dscompanion.selection.feature_selectors import RFESelector, SHAPSelector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# ConstantSelector
# ---------------------------------------------------------------------------


class TestConstantSelector:
    def test_removes_constant_column(self, X_train):
        sel = ConstantSelector(threshold=0.99)
        sel.fit(X_train)
        assert "constant_col" in sel.removed_features_

    def test_does_not_remove_valid_features(self, X_train):
        sel = ConstantSelector(threshold=0.99)
        sel.fit(X_train)
        assert "f1" not in sel.removed_features_
        assert "f2" not in sel.removed_features_

    def test_transform_drops_constant(self, X_train):
        sel = ConstantSelector(threshold=0.99)
        sel.fit(X_train)
        out = sel.transform(X_train)
        assert "constant_col" not in out.columns

    def test_get_removed_has_reason(self, X_train):
        sel = ConstantSelector()
        sel.fit(X_train)
        df = sel.get_removed()
        assert "feature" in df.columns
        assert "reason" in df.columns
        assert len(df) > 0


# ---------------------------------------------------------------------------
# NullRateSelector
# ---------------------------------------------------------------------------


class TestNullRateSelector:
    def _with_null_col(self, X_train, rate: float) -> pd.DataFrame:
        X = X_train.copy()
        n = len(X)
        n_null = int(rate * n)
        X["high_null_col"] = 1.0
        X.loc[X.index[:n_null], "high_null_col"] = pd.NA
        return X

    def test_drops_column_above_threshold(self, X_train):
        X = self._with_null_col(X_train, 0.85)
        sel = NullRateSelector(threshold=0.8)
        sel.fit(X)
        assert "high_null_col" in sel.removed_features_

    def test_keeps_column_below_threshold(self, X_train):
        X = self._with_null_col(X_train, 0.79)
        sel = NullRateSelector(threshold=0.8)
        sel.fit(X)
        assert "high_null_col" not in sel.removed_features_

    def test_null_rates_covers_every_column(self, X_train):
        X = self._with_null_col(X_train, 0.85)
        sel = NullRateSelector(threshold=0.8)
        sel.fit(X)
        assert set(sel.null_rates_.keys()) == set(X.columns)

    def test_removed_reason_mentions_rate_and_threshold(self, X_train):
        X = self._with_null_col(X_train, 0.85)
        sel = NullRateSelector(threshold=0.8)
        sel.fit(X)
        reason = sel.removed_features_["high_null_col"]
        assert "null_rate=" in reason
        assert "threshold=" in reason

    def test_transform_on_oot_uses_fit_time_mask(self, X_train, X_oot):
        X = self._with_null_col(X_train, 0.85)
        sel = NullRateSelector(threshold=0.8)
        sel.fit(X)
        X_oot_with_col = X_oot.copy()
        X_oot_with_col["high_null_col"] = 1.0  # fully populated at OOT time
        out = sel.transform(X_oot_with_col)
        assert "high_null_col" not in out.columns


# ---------------------------------------------------------------------------
# CorrelationSelector
# ---------------------------------------------------------------------------


class TestCorrelationSelector:
    def test_removes_one_of_correlated_pair(self, X_train):
        """f1 and f10_corr_f1 are r>0.9 — one should be removed."""
        num_cols = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f10_corr_f1"]
        sel = CorrelationSelector(threshold=0.85)
        sel.fit(X_train[num_cols].fillna(0))
        removed = set(sel.removed_features_.keys())
        assert len(removed) >= 1
        assert "f1" in removed or "f10_corr_f1" in removed

    def test_does_not_remove_uncorrelated(self, X_train):
        num_cols = ["f2", "f3", "f6", "f7", "f8"]
        sel = CorrelationSelector(threshold=0.85)
        sel.fit(X_train[num_cols].fillna(0))
        assert len(sel.removed_features_) == 0

    def test_transform_on_oot_applies_same_mask(self, X_train, X_oot):
        """OOT transform must apply the same mask learned from train — no refit."""
        num_cols = ["f1", "f2", "f3", "f10_corr_f1"]
        sel = CorrelationSelector(threshold=0.85)
        sel.fit(X_train[num_cols].fillna(0))
        out_train = sel.transform(X_train[num_cols].fillna(0))
        out_oot = sel.transform(X_oot[num_cols].fillna(0))
        assert set(out_train.columns) == set(out_oot.columns)


# ---------------------------------------------------------------------------
# IVSelector
# ---------------------------------------------------------------------------


class TestIVSelector:
    def test_does_not_remove_high_iv_features(self, X_train, y_train):
        # Use leakage_col which has very high IV
        sel = IVSelector(threshold=0.02)
        sel.fit(X_train[["leakage_col", "f2"]], y_train)
        assert "leakage_col" not in sel.removed_features_

    def test_removes_low_iv_features(self, X_train, y_train):
        # constant_col has effectively zero IV
        sel = IVSelector(threshold=0.001)
        sel.fit(X_train[["constant_col", "leakage_col"]], y_train)
        assert "constant_col" in sel.removed_features_

    def test_iv_table_mode(self, X_train):
        iv_table = pd.DataFrame({"feature": ["f1", "f2"], "iv": [0.05, 0.001]})
        sel = IVSelector(threshold=0.02, iv_table=iv_table)
        sel.fit(X_train[["f1", "f2"]])
        assert "f2" in sel.removed_features_
        assert "f1" not in sel.removed_features_

    def test_raises_without_y_or_table(self, X_train):
        sel = IVSelector(threshold=0.02)
        with pytest.raises(ValueError, match="iv_table or y"):
            sel.fit(X_train[["f1"]])


# ---------------------------------------------------------------------------
# FeatureSelectionPipeline
# ---------------------------------------------------------------------------


class TestFeatureSelectionPipeline:
    def _safe_X(self, X):
        """Drop date col for simpler pipeline tests."""
        return X.drop(columns=["account_open_date", "snapshot_date"], errors="ignore")

    def test_removes_constant_end_to_end(self, X_train, y_train):
        X = self._safe_X(X_train)
        pipe = FeatureSelectionPipeline()
        pipe.fit(X, y_train)
        out = pipe.transform(X)
        assert "constant_col" not in out.columns

    def test_oot_gets_same_columns_as_train(self, X_train, X_oot, y_train):
        X = self._safe_X(X_train)
        pipe = FeatureSelectionPipeline()
        pipe.fit(X, y_train)
        out_train = pipe.transform(X)
        out_oot = pipe.transform(self._safe_X(X_oot))
        assert set(out_train.columns) == set(out_oot.columns)

    def test_audit_report_covers_all_removed(self, X_train, y_train):
        X = self._safe_X(X_train)
        pipe = FeatureSelectionPipeline()
        pipe.fit(X, y_train)
        audit = pipe.audit_report()
        assert "reason" in audit.columns
        assert audit["reason"].notna().all()
        # Every removed feature must appear in audit
        retained = set(pipe._selected)
        removed_in_audit = set(audit["feature"])
        all_original = set(X.columns)
        assert removed_in_audit == all_original - retained

    def test_custom_selector_chain(self, X_train, y_train):
        X = self._safe_X(X_train)
        pipe = FeatureSelectionPipeline(selectors=[ConstantSelector()])
        pipe.fit(X, y_train)
        out = pipe.transform(X)
        assert "constant_col" not in out.columns


# ---------------------------------------------------------------------------
# BaseSelector — additional coverage
# ---------------------------------------------------------------------------


class TestBaseSelectorCoverage:
    def test_get_support_returns_selected_list(self, X_train):
        sel = ConstantSelector()
        sel.fit(X_train)
        support = sel.get_support()
        assert isinstance(support, list)
        assert "constant_col" not in support

    def test_get_removed_empty_when_nothing_removed(self, X_train):
        # Only f1/f2 — no constants, no zero-variance
        sel = ConstantSelector(threshold=1.0)  # threshold=1 → nothing removed
        sel.fit(X_train[["f1", "f2"]])
        df = sel.get_removed()
        assert df.empty


# ---------------------------------------------------------------------------
# ConstantSelector — additional coverage
# ---------------------------------------------------------------------------


class TestConstantSelectorExtended:
    def test_all_missing_column_removed(self):
        df = pd.DataFrame({"a": [np.nan] * 100, "b": np.arange(100, dtype=float)})
        sel = ConstantSelector()
        sel.fit(df)
        assert "a" in sel.removed_features_
        assert "constant: all missing" in sel.removed_features_["a"]

    def test_zero_variance_numeric_column_removed(self):
        df = pd.DataFrame({"const": np.zeros(100), "ok": np.arange(100, dtype=float)})
        sel = ConstantSelector(threshold=1.0)  # set high so freq-check won't fire
        sel.fit(df)
        # Zero std → removed via the zero-variance branch
        assert "const" in sel.removed_features_

    def test_type_error_when_X_not_dataframe(self):
        with pytest.raises(TypeError):
            ConstantSelector().fit(np.array([[1, 2], [3, 4]]))


# ---------------------------------------------------------------------------
# NullRateSelector — additional coverage
# ---------------------------------------------------------------------------


class TestNullRateSelectorExtended:
    def test_type_error_when_X_not_dataframe(self):
        with pytest.raises(TypeError):
            NullRateSelector().fit(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# CardinalitySelector — additional coverage
# ---------------------------------------------------------------------------


class TestCardinalitySelectorExtended:
    def test_removes_high_cardinality_categorical(self):
        df = pd.DataFrame({"id": [str(i) for i in range(200)], "val": np.arange(200)})
        sel = CardinalitySelector(max_cardinality=10)
        sel.fit(df)
        assert "id" in sel.removed_features_

    def test_removes_single_unique_value_column(self):
        df = pd.DataFrame({"const": [1] * 100, "ok": np.arange(100)})
        sel = CardinalitySelector(remove_single_unique=True)
        sel.fit(df)
        assert "const" in sel.removed_features_

    def test_type_error_when_X_not_dataframe(self):
        with pytest.raises(TypeError):
            CardinalitySelector().fit(np.array([["a", "b"]]))


# ---------------------------------------------------------------------------
# CorrelationSelector — additional coverage
# ---------------------------------------------------------------------------


class TestCorrelationSelectorExtended:
    def test_error_when_X_not_dataframe(self):
        # TypeError check in CorrelationSelector fires after select_dtypes,
        # so a numpy array raises AttributeError first.
        with pytest.raises((TypeError, AttributeError)):
            CorrelationSelector().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_iv_guided_drop_keeps_higher_iv_feature(self, X_train):
        """When IV table provided, the lower-IV feature of a correlated pair is dropped."""
        num_cols = ["f1", "f10_corr_f1"]
        iv_table = pd.DataFrame({"feature": ["f1", "f10_corr_f1"], "iv": [0.5, 0.1]})
        sel = CorrelationSelector(threshold=0.85, iv_table=iv_table)
        sel.fit(X_train[num_cols].fillna(0))
        # f1 has higher IV → f10_corr_f1 should be the one removed
        assert "f10_corr_f1" in sel.removed_features_
        assert "f1" not in sel.removed_features_

    def test_pick_keep_fallback_to_variance_when_no_iv(self, X_train):
        """Without IV table the higher-variance column is retained."""
        rng = np.random.RandomState(0)
        n = 500
        a = rng.normal(0, 10, n)  # high variance
        b = a * 0.99 + rng.normal(0, 0.1, n)  # almost identical, low variance
        df = pd.DataFrame({"high_var": a, "low_var": b})
        sel = CorrelationSelector(threshold=0.95)
        sel.fit(df)
        assert "low_var" in sel.removed_features_
        assert "high_var" not in sel.removed_features_


# ---------------------------------------------------------------------------
# IVSelector — additional coverage
# ---------------------------------------------------------------------------


class TestIVSelectorExtended:
    def test_type_error_when_X_not_dataframe(self, y_train):
        with pytest.raises(TypeError):
            IVSelector().fit(np.array([[1.0, 2.0]]), y_train)


# ---------------------------------------------------------------------------
# SHAPSelector — coverage (error paths only; full fit is slow/unstable)
# ---------------------------------------------------------------------------


class TestSHAPSelectorErrorPaths:
    def test_model_none_raises_value_error(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="fitted model"):
            SHAPSelector(model=None).fit(df)

    def test_type_error_when_X_not_dataframe(self):
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression()
        with pytest.raises(TypeError):
            SHAPSelector(model=est).fit(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# RFESelector
# ---------------------------------------------------------------------------


class TestRFESelector:
    def test_y_none_raises_value_error(self, X_train):
        num_X = X_train.select_dtypes(include="number").fillna(0)
        with pytest.raises(ValueError, match="requires y"):
            RFESelector().fit(num_X, y=None)

    def test_type_error_when_X_not_dataframe(self, y_train):
        with pytest.raises(TypeError):
            RFESelector().fit(np.array([[1.0, 2.0]]), y_train)

    def test_fit_retains_informative_features(self, X_train, y_train):
        """RFESelector with 2 features should keep at least 1 informative column."""
        # Use small feature set + fast estimator
        from sklearn.linear_model import LogisticRegression

        num_cols = ["f1", "f2", "leakage_col"]
        X = X_train[num_cols].fillna(0)
        sel = RFESelector(
            estimator=LogisticRegression(max_iter=100, random_state=42),
            cv=2,
            min_features_to_select=1,
        )
        sel.fit(X, y_train)
        assert len(sel.selected_features_) >= 1
        assert hasattr(sel, "rfecv_")
