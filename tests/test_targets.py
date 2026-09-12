"""Tests for dscompanion.targets — ImbalanceHandler and TargetBinariser."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscompanion.targets import ImbalanceHandler, TargetBinariser

# ---------------------------------------------------------------------------
# ImbalanceHandler
# ---------------------------------------------------------------------------


class TestImbalanceHandler:
    def test_class_weight_returns_unchanged_data(self, data_split):
        X, y = data_split.X_train, data_split.y_train
        handler = ImbalanceHandler(strategy="class_weight", random_state=42)
        X_res, y_res = handler.fit_resample(X, y)
        assert len(X_res) == len(X)
        assert len(y_res) == len(y)
        pd.testing.assert_frame_equal(X_res, X)
        pd.testing.assert_series_equal(y_res, y)

    def test_class_weights_exist_for_class_weight_strategy(self, data_split):
        handler = ImbalanceHandler(strategy="class_weight", random_state=42)
        handler.fit(data_split.X_train, data_split.y_train)
        assert handler.class_weights_ is not None
        assert set(handler.class_weights_.keys()) == {0, 1}

    def test_class_weights_sum_approx_n_classes(self, data_split):
        handler = ImbalanceHandler(strategy="class_weight")
        handler.fit(data_split.X_train, data_split.y_train)
        weight_sum = sum(handler.class_weights_.values())
        # sklearn balanced weights sum to n_classes (2 here) approximately
        assert abs(weight_sum - 2.0) < 0.5

    def test_none_strategy_is_passthrough(self, data_split):
        X, y = data_split.X_train, data_split.y_train
        handler = ImbalanceHandler(strategy="none")
        X_res, y_res = handler.fit_resample(X, y)
        assert len(X_res) == len(X)

    def test_class_weights_none_for_non_weight_strategy(self, data_split):
        handler = ImbalanceHandler(strategy="none")
        handler.fit(data_split.X_train, data_split.y_train)
        assert handler.class_weights_ is None

    def test_audit_report_has_before_after(self, data_split):
        handler = ImbalanceHandler(strategy="class_weight")
        handler.fit_resample(data_split.X_train, data_split.y_train)
        audit = handler.resample_audit_
        assert "count_before" in audit.columns
        assert "count_after" in audit.columns
        assert len(audit) == 2  # two classes

    def test_smote_produces_balanced_distribution(self, data_split):
        X, y = data_split.X_train, data_split.y_train
        # Use only numeric cols (SMOTE needs numeric)
        num_cols = X.select_dtypes(include="number").columns.tolist()
        X_num = X[num_cols].fillna(0)
        handler = ImbalanceHandler(strategy="smote", random_state=42)
        _, y_res = handler.fit_resample(X_num, y)
        counts = y_res.value_counts()
        ratio = counts.min() / counts.max()
        assert ratio > 0.9  # near-balanced after SMOTE

    def test_smote_auto_produces_exact_double_majority_row_count(self, data_split):
        # Locks in the exact row-count contract two UI/API preview functions
        # independently hardcode (2 x majority_count for binary "auto").
        X, y = data_split.X_train, data_split.y_train
        num_cols = X.select_dtypes(include="number").columns.tolist()
        X_num = X[num_cols].fillna(0)
        majority_count = int(y.value_counts().max())
        handler = ImbalanceHandler(strategy="smote", random_state=42)
        _, y_res = handler.fit_resample(X_num, y)
        assert len(y_res) == 2 * majority_count
        counts = y_res.value_counts()
        assert counts.min() == counts.max() == majority_count

    def test_undersample_reduces_majority_class(self, data_split):
        X, y = data_split.X_train, data_split.y_train
        num_cols = X.select_dtypes(include="number").columns.tolist()
        X_num = X[num_cols].fillna(0)
        handler = ImbalanceHandler(strategy="undersample", random_state=42)
        _, y_res = handler.fit_resample(X_num, y)
        assert len(y_res) < len(y)
        counts = y_res.value_counts()
        assert counts.min() == counts.max()  # perfectly balanced

    def test_smote_is_deterministic_given_same_random_state(self, data_split):
        X, y = data_split.X_train, data_split.y_train
        num_cols = X.select_dtypes(include="number").columns.tolist()
        X_num = X[num_cols].fillna(0)
        X_res_a, y_res_a = ImbalanceHandler(strategy="smote", random_state=7).fit_resample(X_num, y)
        X_res_b, y_res_b = ImbalanceHandler(strategy="smote", random_state=7).fit_resample(X_num, y)
        pd.testing.assert_frame_equal(X_res_a, X_res_b)
        pd.testing.assert_series_equal(y_res_a, y_res_b)

    def test_undersample_is_deterministic_given_same_random_state(self, data_split):
        X, y = data_split.X_train, data_split.y_train
        num_cols = X.select_dtypes(include="number").columns.tolist()
        X_num = X[num_cols].fillna(0)
        X_res_a, y_res_a = ImbalanceHandler(strategy="undersample", random_state=7).fit_resample(
            X_num, y
        )
        X_res_b, y_res_b = ImbalanceHandler(strategy="undersample", random_state=7).fit_resample(
            X_num, y
        )
        pd.testing.assert_frame_equal(X_res_a, X_res_b)
        pd.testing.assert_series_equal(y_res_a, y_res_b)

    def _multiclass_data(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.randn(200), "f2": rng.randn(200)})
        y = pd.Series([0] * 100 + [1] * 60 + [2] * 40, name="target")
        return X, y

    def test_smote_multiclass_brings_every_class_to_majority_count(self):
        X, y = self._multiclass_data()
        handler = ImbalanceHandler(strategy="smote", random_state=0)
        _, y_res = handler.fit_resample(X, y)
        counts = y_res.value_counts()
        assert counts.nunique() == 1
        assert counts.iloc[0] == 100

    def test_undersample_multiclass_brings_every_class_to_minority_count(self):
        X, y = self._multiclass_data()
        handler = ImbalanceHandler(strategy="undersample", random_state=0)
        _, y_res = handler.fit_resample(X, y)
        counts = y_res.value_counts()
        assert counts.nunique() == 1
        assert counts.iloc[0] == 40

    def test_smote_float_sampling_strategy_binary(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.randn(150), "f2": rng.randn(150)})
        y = pd.Series([0] * 100 + [1] * 50, name="target")
        handler = ImbalanceHandler(strategy="smote", sampling_strategy=0.5, random_state=0)
        _, y_res = handler.fit_resample(X, y)
        counts = y_res.value_counts()
        assert counts[1] == 50  # 0.5 * 100 majority count
        assert counts[0] == 100

    def test_undersample_float_sampling_strategy_binary(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.randn(150), "f2": rng.randn(150)})
        y = pd.Series([0] * 100 + [1] * 50, name="target")
        handler = ImbalanceHandler(strategy="undersample", sampling_strategy=0.5, random_state=0)
        _, y_res = handler.fit_resample(X, y)
        counts = y_res.value_counts()
        assert counts[0] == 100  # 50 / 0.5 majority target
        assert counts[1] == 50

    def test_smote_raises_clear_error_when_class_too_small_for_k_neighbors(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.randn(103), "f2": rng.randn(103)})
        y = pd.Series([0] * 100 + [1] * 3, name="target")  # fewer than default k=5 + 1
        handler = ImbalanceHandler(strategy="smote", random_state=0)
        with pytest.raises(ValueError, match="k_neighbors"):
            handler.fit_resample(X, y)

    def test_smote_raises_on_non_numeric_columns(self, data_split):
        X, y = data_split.X_train, data_split.y_train
        X_with_cat = X.select_dtypes(include="number").fillna(0).copy()
        X_with_cat["cat_col"] = "a"
        handler = ImbalanceHandler(strategy="smote", random_state=0)
        with pytest.raises(ValueError, match="numeric"):
            handler.fit_resample(X_with_cat, y)

    def test_smote_raises_on_missing_values(self, data_split):
        X, y = data_split.X_train, data_split.y_train
        num_cols = X.select_dtypes(include="number").columns.tolist()
        X_num = X[num_cols].copy()  # deliberately not fillna'd
        handler = ImbalanceHandler(strategy="smote", random_state=0)
        with pytest.raises(ValueError, match="missing"):
            handler.fit_resample(X_num, y)

    def test_fit_raises_type_error_for_non_dataframe(self, data_split):
        handler = ImbalanceHandler(strategy="class_weight")
        with pytest.raises(TypeError):
            handler.fit(data_split.X_train.values, data_split.y_train)

    def test_fit_raises_type_error_for_non_series(self, data_split):
        handler = ImbalanceHandler(strategy="class_weight")
        with pytest.raises(TypeError):
            handler.fit(data_split.X_train, data_split.y_train.values)

    def test_resample_audit_columns(self, data_split):
        handler = ImbalanceHandler(strategy="none")
        handler.fit_resample(data_split.X_train, data_split.y_train)
        audit = handler.resample_audit_
        assert list(audit.columns) == [
            "class",
            "count_before",
            "pct_before",
            "count_after",
            "pct_after",
        ]


# ---------------------------------------------------------------------------
# TargetBinariser
# ---------------------------------------------------------------------------


class TestTargetBinariser:
    def test_threshold_strategy_splits_correctly(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        tb = TargetBinariser(strategy="threshold", threshold=3.0)
        tb.fit(y)
        result = tb.transform(y)
        assert result.tolist() == [0, 0, 0, 1, 1]

    def test_median_strategy(self):
        y = pd.Series(range(10), dtype=float)
        tb = TargetBinariser(strategy="median")
        tb.fit(y)
        assert tb.threshold_ == y.median()

    def test_quantile_strategy(self):
        y = pd.Series(range(100), dtype=float)
        tb = TargetBinariser(strategy="quantile", quantile=0.75)
        tb.fit(y)
        assert abs(tb.threshold_ - y.quantile(0.75)) < 1e-6

    def test_threshold_raises_without_value(self):
        tb = TargetBinariser(strategy="threshold")  # threshold=None
        with pytest.raises(ValueError, match="threshold must be set"):
            tb.fit(pd.Series([1, 2, 3]))

    def test_inverse_transform_round_trip(self):
        y_cont = pd.Series([0.5, 1.5, 2.5, 3.5])
        tb = TargetBinariser(strategy="threshold", threshold=2.0)
        tb.fit(y_cont)
        y_bin = tb.transform(y_cont)
        y_inv = tb.inverse_transform(y_bin)
        # Positive class maps back to threshold, negative to 0.0
        assert y_inv[y_bin == 1].unique().tolist() == [tb.threshold_]
        assert y_inv[y_bin == 0].unique().tolist() == [0.0]

    def test_fit_raises_type_error_for_non_series(self):
        tb = TargetBinariser(strategy="threshold", threshold=1.0)
        with pytest.raises(TypeError):
            tb.fit([1, 2, 3])

    def test_unknown_strategy_raises_value_error(self):
        tb = TargetBinariser(strategy="invalid")
        with pytest.raises(ValueError, match="Unknown strategy"):
            tb.fit(pd.Series([1, 2, 3]))

    def test_transform_before_fit_raises_runtime_error(self):
        tb = TargetBinariser(strategy="threshold", threshold=1.0)
        with pytest.raises(RuntimeError, match="before fit"):
            tb.transform(pd.Series([1, 2, 3]))

    def test_inverse_transform_before_fit_raises_runtime_error(self):
        tb = TargetBinariser(strategy="threshold", threshold=1.0)
        with pytest.raises(RuntimeError, match="before fit"):
            tb.inverse_transform(pd.Series([0, 1, 0]))

    def test_custom_positive_label(self):
        y = pd.Series([1.0, 2.0, 3.0])
        tb = TargetBinariser(strategy="threshold", threshold=1.5, positive_label=2)
        tb.fit(y)
        result = tb.transform(y)
        assert set(result.unique()).issubset({2, -1})

    def test_output_preserves_series_name(self):
        y = pd.Series([1.0, 2.0, 3.0], name="my_target")
        tb = TargetBinariser(strategy="median")
        tb.fit(y)
        assert tb.transform(y).name == "my_target"

    def test_output_is_binary(self):
        rng = np.random.RandomState(0)
        y = pd.Series(rng.randn(200))
        tb = TargetBinariser(strategy="median")
        tb.fit(y)
        result = tb.transform(y)
        assert set(result.unique()).issubset({0, 1})
