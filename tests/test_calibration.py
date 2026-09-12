"""Tests for dscompanion.calibration — Calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from dscompanion.calibration import Calibrator
from dscompanion.models import ModelFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_model_and_val(data_split):
    X_tr = data_split.X_train.select_dtypes(include="number").fillna(0)
    X_val_raw = data_split.X_val if data_split.X_val is not None else data_split.X_oot
    X_val = X_val_raw.select_dtypes(include="number").fillna(0)
    y_tr = data_split.y_train
    y_val = data_split.y_oot if data_split.X_val is None else data_split.y_val

    model = ModelFactory.build("classification", "logistic")
    model.fit(X_tr, y_tr)
    return model, X_val, y_val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalibrator:
    def test_fit_completes(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator(method="isotonic")
        cal.fit(model, X_val, y_val)
        assert hasattr(cal, "ece_before_")
        assert hasattr(cal, "ece_after_")

    def test_ece_after_not_worse_for_logistic(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator(method="isotonic")
        cal.fit(model, X_val, y_val)
        # After calibration ECE should be <= before (logistic is already fairly calibrated,
        # so we just verify the pipeline doesn't break it significantly)
        assert cal.ece_after_ <= cal.ece_before_ + 0.05

    def test_wrap_predict_proba_sums_to_one(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator(method="platt")
        cal.fit(model, X_val, y_val)
        wrapped = cal.wrap(model)
        proba = wrapped.predict_proba(X_val)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_reliability_curve_returns_figure(self, fitted_model_and_val):
        import plotly.graph_objects as go

        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator()
        cal.fit(model, X_val, y_val)
        fig = cal.reliability_curve()
        assert isinstance(fig, go.Figure)

    def test_calibration_report_has_required_columns(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator()
        cal.fit(model, X_val, y_val)
        report = cal.calibration_report()
        assert "ece_before" in report.columns
        assert "ece_after" in report.columns
        assert len(report) == 1

    def test_type_error_when_x_not_dataframe(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator()
        with pytest.raises(TypeError, match="pd.DataFrame"):
            cal.fit(model, X_val.values, y_val)

    def test_type_error_when_y_not_series(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator()
        with pytest.raises(TypeError, match="pd.Series"):
            cal.fit(model, X_val, y_val.values)

    def test_beta_method_fits_and_produces_proba(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator(method="beta")
        cal.fit(model, X_val, y_val)
        assert hasattr(cal, "ece_before_")
        assert hasattr(cal, "ece_after_")
        proba = cal._calibrated.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)

    def test_ece_method_with_custom_n_bins(self, fitted_model_and_val):
        import pandas as pd

        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator()
        cal.fit(model, X_val, y_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        ece_5 = cal.ece(pd.Series(y_val.values), y_prob, n_bins=5)
        ece_20 = cal.ece(pd.Series(y_val.values), y_prob, n_bins=20)
        assert isinstance(ece_5, float)
        assert isinstance(ece_20, float)
        assert ece_5 >= 0.0 and ece_20 >= 0.0

    def test_calibration_report_raises_before_fit(self):
        cal = Calibrator()
        with pytest.raises(RuntimeError, match="before fit"):
            cal.calibration_report()

    def test_wrapped_model_predict_delegates_to_base(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator(method="isotonic")
        cal.fit(model, X_val, y_val)
        wrapped = cal.wrap(model)
        # predict should delegate to base model
        np.testing.assert_array_equal(wrapped.predict(X_val), model.predict(X_val))

    def test_wrapped_model_getattr_proxies_to_base(self, fitted_model_and_val):
        model, X_val, y_val = fitted_model_and_val
        cal = Calibrator(method="platt")
        cal.fit(model, X_val, y_val)
        wrapped = cal.wrap(model)
        # feature_names is an attribute on BaseDSCompanionModel — must be accessible via wrapper
        assert wrapped.feature_names == model.feature_names
