"""Calibrator: post-hoc probability calibration with reliability diagnostics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from dscompanion.config import settings
from dscompanion.utils.metrics import expected_calibration_error

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["Calibrator"]


class _IsotonicCal:
    """Duck-typed calibrated classifier using isotonic regression on predicted probabilities."""

    def __init__(self, base_est: Any, isotonic_model: IsotonicRegression) -> None:
        self._base = base_est
        self._ir = isotonic_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self._base.predict_proba(X)[:, 1]
        cal = self._ir.predict(probs)
        return np.column_stack([1 - cal, cal])


class _PlattCal:
    """Duck-typed calibrated classifier using logistic regression on predicted probabilities."""

    def __init__(self, base_est: Any, lr_model: LogisticRegression) -> None:
        self._base = base_est
        self._lr = lr_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self._base.predict_proba(X)[:, 1]
        cal = self._lr.predict_proba(probs.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - cal, cal])


class _CalibratedWrapper:
    """Thin wrapper that replaces predict_proba with calibrated output."""

    def __init__(
        self, base_model: Any, calibrated_clf: Any
    ) -> None:  # Any: duck-typed sklearn-compatible objects
        self._base = base_model
        self._cal = calibrated_clf

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._base.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._cal.predict_proba(X)


class Calibrator:
    """Applies post-hoc probability calibration to a fitted classifier and tracks
    Expected Calibration Error (ECE) before and after the calibration step.

    Supports isotonic regression, Platt (sigmoid) scaling, and a custom beta
    calibration implemented via scipy optimisation.  After calling ``fit``,
    the attributes ``ece_before_`` and ``ece_after_`` are populated so callers
    can compare calibration quality.

    Args:
        method (str): Calibration algorithm to use.  One of ``"isotonic"``
            (default), ``"platt"`` (sigmoid scaling via logistic regression on
            raw predicted probabilities), or ``"beta"`` (log-odds +
            log-probability linear transform optimised with Nelder-Mead; falls
            back to isotonic on failure).

    Attributes:
        ece_before_ (float): ECE measured on the calibration set *before*
            calibration; populated after ``fit``.
        ece_after_ (float): ECE measured on the calibration set *after*
            calibration; populated after ``fit``.
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method

    def fit(
        self,
        model: Any,  # Any: fitted BaseDSCompanionModel — avoids circular import
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "Calibrator":
        """Fit the calibration mapping on a held-out calibration set and record
        ECE before and after calibration.

        Stores a reference to the internal sklearn calibrated classifier and
        populates ``ece_before_`` and ``ece_after_`` as side-effects.  Emits an
        INFO log line with the ECE delta.

        Args:
            model (Any): Fitted ``BaseDSCompanionModel`` instance whose ``estimator``
                attribute exposes the underlying sklearn-compatible estimator.
            X_cal (pd.DataFrame): Feature matrix for the calibration set.
                Using a dedicated validation split is recommended to avoid
                overfitting the calibration mapping.
            y_cal (pd.Series): Binary target labels (0/1) aligned with
                ``X_cal``.

        Returns:
            Calibrator: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X_cal`` is not a ``pd.DataFrame`` or ``y_cal`` is
                not a ``pd.Series``.
        """
        if not isinstance(X_cal, pd.DataFrame):
            raise TypeError("X_cal must be a pd.DataFrame, got %s" % type(X_cal).__name__)
        if not isinstance(y_cal, pd.Series):
            raise TypeError("y_cal must be a pd.Series, got %s" % type(y_cal).__name__)

        self._model = model
        y_prob_before = model.predict_proba(X_cal)[:, 1]
        self.ece_before_ = expected_calibration_error(y_cal.values, y_prob_before)

        if self.method == "beta":
            self._calibrated = self._fit_beta(model, X_cal, y_cal)
        elif self.method == "platt":
            probs_cal = model.estimator.predict_proba(X_cal)[:, 1]
            lr = LogisticRegression()
            lr.fit(probs_cal.reshape(-1, 1), y_cal)
            self._calibrated = _PlattCal(model.estimator, lr)
        else:  # isotonic (default)
            probs_cal = model.estimator.predict_proba(X_cal)[:, 1]
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs_cal, y_cal)
            self._calibrated = _IsotonicCal(model.estimator, ir)

        y_prob_after = self._calibrated.predict_proba(X_cal)[:, 1]
        self.ece_after_ = expected_calibration_error(y_cal.values, y_prob_after)

        logger.info(
            "Calibrator fitted — method=%s, ECE: %.4f → %.4f",
            self.method,
            self.ece_before_,
            self.ece_after_,
        )
        return self

    def wrap(
        self, model: Any
    ) -> _CalibratedWrapper:  # Any: fitted BaseDSCompanionModel — avoids circular import
        """Return a thin wrapper around ``model`` that replaces its
        ``predict_proba`` output with the calibrated probabilities while
        delegating all other attribute access to the original object.

        Args:
            model (Any): Fitted ``BaseDSCompanionModel`` instance to wrap.  Its
                ``predict`` method is preserved unchanged; only
                ``predict_proba`` is overridden.

        Returns:
            _CalibratedWrapper: A new object whose ``predict_proba`` returns
            calibrated probabilities and all other attributes/methods
            are forwarded to the underlying base model.
        """
        return _CalibratedWrapper(model, self._calibrated)

    def reliability_curve(self) -> "go.Figure":
        """Render a reliability (calibration) diagram as a Plotly figure,
        showing the perfect-calibration diagonal for visual reference.

        This method currently plots the reference diagonal only; callers who
        want the empirical curves should compute them via
        ``sklearn.calibration.calibration_curve`` and add traces manually.

        Returns:
            plotly.graph_objects.Figure: A Plotly Figure object with the
            perfect-calibration diagonal trace and dscompanion theme applied.

        Raises:
            ImportError: If ``plotly`` is not installed.
        """
        from dscompanion.utils.plotting import apply_dscompanion_theme

        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required.") from exc

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dash")))
        apply_dscompanion_theme(fig)
        fig.update_layout(
            title="Reliability Curve",
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of positives",
        )
        return fig

    def ece(self, y_true: pd.Series, y_prob: np.ndarray, n_bins: int | None = None) -> float:
        """Compute the Expected Calibration Error for an arbitrary
        ``(y_true, y_prob)`` pair using equal-width probability bins.

        Delegates directly to ``dscompanion.utils.metrics.expected_calibration_error``.

        Args:
            y_true (pd.Series): Binary ground-truth labels (0/1) of shape
                ``(n_samples,)``.
            y_prob (np.ndarray): Predicted positive-class probabilities of
                shape ``(n_samples,)``, each in ``[0, 1]``.
            n_bins (int, optional): Number of equal-width bins in ``[0, 1]``
                used to group predictions.  Defaults to
                ``settings.calibration_ece_bins``.

        Returns:
            float: ECE value in ``[0, 1]``.  Returns 0.0 when every bin is
            empty (e.g. empty arrays passed in).
        """
        if n_bins is None:
            n_bins = settings.calibration_ece_bins
        return expected_calibration_error(y_true.values, y_prob, n_bins=n_bins)

    def calibration_report(self) -> pd.DataFrame:
        """Summarise calibration quality before and after the fitted
        calibration step in a single-row DataFrame.

        Must be called after ``fit``; relies on the ``ece_before_`` and
        ``ece_after_`` attributes that are populated during fitting.

        Returns:
            pd.DataFrame: Single-row DataFrame with columns ``method``
            (str), ``ece_before`` (float), and ``ece_after`` (float).
            Values are rounded to ``settings.evaluate_round_precision``
            decimal places.  Always returns exactly one row.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if not hasattr(self, "ece_before_"):
            raise RuntimeError("Calibrator.calibration_report() called before fit()")
        return pd.DataFrame(
            [
                {
                    "method": self.method,
                    "ece_before": round(self.ece_before_, settings.evaluate_round_precision),
                    "ece_after": round(self.ece_after_, settings.evaluate_round_precision),
                }
            ]
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fit_beta(
        self,
        model: Any,  # Any: fitted BaseDSCompanionModel — avoids circular import
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> Any:  # Any: duck-typed _BetaCal or CalibratedClassifierCV
        """Fit beta calibration using scipy minimisation (fallback to isotonic on failure)."""
        try:
            from scipy.optimize import minimize

            probs = model.predict_proba(X_cal)[:, 1]
            eps = 1e-6
            log_odds = np.log((probs + eps) / (1 - probs + eps))

            def neg_log_lik(params):
                a, b, c = params
                cal = 1 / (1 + np.exp(-(a * log_odds + b * np.log(probs + eps) + c)))
                return -np.sum(y_cal * np.log(cal + eps) + (1 - y_cal) * np.log(1 - cal + eps))

            res = minimize(neg_log_lik, x0=[1.0, 0.0, 0.0], method="Nelder-Mead")
            self._beta_params = res.x

            # Wrap as a duck-typed object with predict_proba
            class _BetaCal:
                def __init__(self, params, base_est, train_probs):
                    self.params = params
                    self._base = base_est
                    self._train_probs = train_probs

                def predict_proba(self, X):
                    probs = self._base.predict_proba(X)[:, 1]
                    a, b, c = self.params
                    eps = 1e-6
                    log_odds = np.log((probs + eps) / (1 - probs + eps))
                    cal = 1 / (1 + np.exp(-(a * log_odds + b * np.log(probs + eps) + c)))
                    return np.column_stack([1 - cal, cal])

            return _BetaCal(res.x, model.estimator, probs)
        except Exception as exc:
            logger.warning("Beta calibration failed (%s), falling back to isotonic", exc)
            probs_cal = model.estimator.predict_proba(X_cal)[:, 1]
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs_cal, y_cal)
            return _IsotonicCal(model.estimator, ir)
