"""RegressionModel: wrapper for regression estimators."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from dscompanion.models.base import BaseDSCompanionModel

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["RegressionModel"]


class RegressionModel(BaseDSCompanionModel):
    """Model wrapper for supervised regression tasks.

    Extends ``BaseDSCompanionModel`` with a standard set of regression error
    metrics and a residual diagnostic plot.  No additional state is maintained
    beyond what the parent class stores.

    Metrics computed on each split: ``rmse`` (root mean squared error),
    ``mae`` (mean absolute error), ``r2`` (coefficient of determination),
    ``mape`` (mean absolute percentage error — computed only on non-zero
    target values; ``nan`` when all targets are zero), and
    ``median_absolute_error``.

    Args:
        estimator (Any): An instantiated sklearn-compatible regressor that
            implements ``fit`` and ``predict``.
        **kwargs: Additional keyword arguments forwarded to
            ``BaseDSCompanionModel.__init__`` (``feature_names``).
    """

    def __init__(
        self, estimator: Any, **kwargs: Any
    ) -> None:  # Any: sklearn-compatible regressor with fit/predict interface
        super().__init__(estimator, **kwargs)

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        split_name: str = "",
    ) -> dict[str, float]:
        """Compute a standard suite of regression evaluation metrics.

        A pure function of its arguments (``@staticmethod``) — no fitted
        estimator instance is involved, so callers outside of ``evaluate()``
        (e.g. a monitoring job re-measuring performance on historical
        predictions) can call this directly on raw arrays.

        Calculates RMSE, MAE, R², MAPE, and median absolute error from the
        supplied true and predicted arrays.  MAPE is computed only over
        samples where ``y_true != 0`` to avoid division by zero; if all
        ``y_true`` values are zero, ``mape`` is set to ``float('nan')``.
        No side-effects.

        Args:
            y_true (np.ndarray): Ground-truth continuous target values, shape
                ``(n_samples,)``.
            y_pred (np.ndarray): Predicted continuous values produced by
                ``predict``, shape ``(n_samples,)``.
            y_prob (np.ndarray, optional): Ignored; present only to satisfy the
                abstract method signature.

        Returns:
            dict: Always contains exactly five keys — ``rmse``, ``mae``,
            ``r2``, ``mape``, and ``median_absolute_error`` — each mapping to a
            scalar float.  ``mape`` is ``float('nan')`` when all ``y_true``
            values are zero.
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            median_absolute_error,
            r2_score,
        )

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        med_ae = float(median_absolute_error(y_true, y_pred))

        nz = y_true != 0
        mape = (
            float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])))
            if nz.any()
            else float("nan")
        )

        return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "median_absolute_error": med_ae}

    def residual_plot(self, X: pd.DataFrame, y: pd.Series) -> "go.Figure":
        """Build and return a Plotly scatter plot of residuals versus predicted values.

        Calls ``predict`` to obtain fitted values, computes residuals as
        ``y - y_pred``, and renders a semi-transparent scatter plot with a
        horizontal dashed zero-line for reference.  The dscompanion visual theme is
        applied before returning.  No side-effects beyond the predict call.

        Args:
            X (pd.DataFrame): Feature matrix, shape ``(n_samples, n_features)``.
            y (pd.Series): True target values, shape ``(n_samples,)``; used
                only to compute residuals and not mutated.

        Returns:
            plotly.graph_objects.Figure: Interactive scatter plot with
            predicted values on the x-axis and residuals on the y-axis.  A
            well-fitted model should show residuals randomly scattered around
            zero with no systematic pattern.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame`` or ``y`` is not a
                ``pd.Series``.
            ImportError: If ``plotly`` is not installed in the environment.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pd.Series, got %s" % type(y).__name__)

        from dscompanion.utils.plotting import apply_dscompanion_theme

        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required.") from exc

        y_pred = self.predict(X)
        residuals = y.values - y_pred

        fig = go.Figure(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(opacity=0.4, size=4),
            )
        )
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(
            title="Residuals vs Predicted",
            xaxis_title="Predicted",
            yaxis_title="Residual",
        )
        apply_dscompanion_theme(fig)
        return fig
