"""PDPAnalyser: Partial Dependence Plot analysis and threshold discovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings
from dscompanion.utils.plotting import apply_dscompanion_theme

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)

__all__ = ["PDPAnalyser"]

_SUMMARY_COLS = [
    "feature",
    "n_grid_points",
    "baseline_prob",
    "peak_prob",
    "trough_prob",
    "range_of_effect",
]


class PDPAnalyser(BaseEstimator):
    """Computes Partial Dependence Plots and discovers business-action
    thresholds for selected features using a fitted model.

    For each feature, ``fit()`` marginalises over the reference dataset by
    sweeping the feature through a grid of values while holding all other
    features at their observed values, then averages the resulting predicted
    probabilities.  This removes the confounding effect of inter-feature
    correlations, making each feature's marginal relationship with the model
    output visible in isolation (Friedman, 2001).

    PDP curves are precomputed in ``fit()`` and cached; ``plot()``,
    ``discover_thresholds()``, and ``summary_table()`` read from the cache
    without re-running inference.

    Args:
        model (Any): Fitted model with a ``predict_proba`` method that accepts
            a ``pd.DataFrame`` and returns an array of shape
            ``(n_samples, n_classes)``.  Accepts both raw sklearn estimators
            and dscompanion wrapper objects.
        grid_resolution (int | None): Number of evenly-spaced grid points used
            for each numeric feature's sweep.  Categorical features use their
            unique training values instead.  When ``None``, defaults to
            ``settings.pdp_grid_resolution``.

    Attributes:
        pdp_results_ (dict[str, dict]): Per-feature PDP data after ``fit()``.
            Each key is a feature name; each value is a dict with
            ``grid_values`` (np.ndarray) and ``average`` (np.ndarray of
            averaged predicted probabilities, same length as ``grid_values``).
        feature_names_ (list[str]): Feature names passed to ``fit()``, in the
            supplied order.
    """

    def __init__(
        self,
        model: Any,
        grid_resolution: int | None = None,
    ) -> None:
        self.model = model
        self.grid_resolution = grid_resolution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, features: list[str]) -> PDPAnalyser:
        """Compute partial dependence for each feature in ``features``.

        Marginalises over a random subsample of ``X`` (capped at
        ``settings.pdp_sample_size`` rows) for computational efficiency.
        Features whose values are entirely NaN in the subsample are skipped
        with a warning and omitted from ``pdp_results_``.

        Args:
            X (pd.DataFrame): Reference dataset used to marginalise over.
                All columns used at model-training time must be present.
            features (list[str]): Column names for which to compute PDPs.
                Every name must exist in ``X.columns``.

        Returns:
            PDPAnalyser: Fitted instance (``self``).

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows or any name in ``features``
                is absent from ``X.columns``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PDPAnalyser.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("PDPAnalyser.fit() received an empty DataFrame (0 rows)")

        missing_cols = [f for f in features if f not in X.columns]
        if missing_cols:
            raise ValueError(
                "PDPAnalyser.fit(): features not found in X.columns: %s" % missing_cols
            )

        grid_res = (
            self.grid_resolution
            if self.grid_resolution is not None
            else settings.pdp_grid_resolution
        )
        sample_size = settings.pdp_sample_size

        X_ref = (
            X.sample(sample_size, random_state=settings.random_state) if len(X) > sample_size else X
        )
        logger.info(
            "PDPAnalyser.fit(): %d features, reference=%d rows, grid_res=%d",
            len(features),
            len(X_ref),
            grid_res,
        )

        predict_fn = self._build_predict_fn()

        self.pdp_results_: dict[str, dict[str, np.ndarray]] = {}
        self.feature_names_: list[str] = list(features)

        for feat in features:
            col_vals = X_ref[feat].dropna()
            if len(col_vals) == 0:
                logger.warning("PDPAnalyser: skipping all-NaN feature '%s'", feat)
                continue

            if pd.api.types.is_numeric_dtype(col_vals):
                grid: np.ndarray = np.linspace(
                    float(col_vals.min()), float(col_vals.max()), grid_res
                )
            else:
                grid = np.array(sorted(col_vals.unique()))

            avg = self._compute_pdp(predict_fn, X_ref, feat, grid)
            self.pdp_results_[feat] = {"grid_values": grid, "average": avg}
            logger.debug(
                "PDPAnalyser: '%s' — %d grid pts, avg_prob in [%.4f, %.4f]",
                feat,
                len(grid),
                float(avg.min()),
                float(avg.max()),
            )

        logger.info(
            "PDPAnalyser fitted — %d/%d features computed",
            len(self.pdp_results_),
            len(features),
        )
        return self

    def plot(self, feature: str) -> go.Figure:
        """Render a Plotly PDP curve for a single feature.

        Numeric features produce a line chart with a dashed baseline marker;
        categorical features produce a bar chart.  The y-axis shows the
        average predicted probability across the marginalised reference
        dataset.

        Args:
            feature (str): Feature name to plot.  Must be a key in
                ``pdp_results_``.

        Returns:
            plotly.graph_objects.Figure: PDP chart figure.

        Raises:
            NotFittedError: If called before ``fit()``.
            ValueError: If ``feature`` is not in ``pdp_results_``.
            ImportError: If ``plotly`` is not installed.
        """
        check_is_fitted(self, attributes=["pdp_results_"])
        if feature not in self.pdp_results_:
            raise ValueError(
                "feature '%s' not found in pdp_results_ " "— was it passed to fit()?" % feature
            )

        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required: pip install plotly") from exc

        result = self.pdp_results_[feature]
        grid = result["grid_values"]
        avg = result["average"]

        if np.issubdtype(grid.dtype, np.number):
            fig = go.Figure(
                go.Scatter(
                    x=grid.tolist(),
                    y=avg.tolist(),
                    mode="lines",
                    name="PDP",
                    line=dict(width=2),
                )
            )
            fig.add_hline(
                y=float(avg[0]),
                line=dict(color="gray", dash="dot", width=1),
                annotation_text="baseline",
            )
        else:
            fig = go.Figure(
                go.Bar(
                    x=[str(v) for v in grid.tolist()],
                    y=avg.tolist(),
                    name="PDP",
                )
            )

        fig.update_layout(
            title="Partial Dependence — %s" % feature,
            xaxis_title=feature,
            yaxis_title="Average predicted probability",
        )
        apply_dscompanion_theme(fig)
        return fig

    def discover_thresholds(
        self,
        min_slope: float | None = None,
        min_prob_increase: float | None = None,
    ) -> dict[str, Any]:
        """Identify feature value thresholds where predicted probability rises
        sharply, indicating business-actionable leverage points.

        Applies first-order finite differences to each numeric PDP curve.
        A grid point is flagged when the local slope (Δprob / Δfeature_value)
        meets or exceeds ``min_slope`` AND the cumulative probability gain
        from the PDP baseline also meets or exceeds ``min_prob_increase``.
        Categorical features (non-numeric grid) return an empty threshold list.

        Args:
            min_slope (float | None): Minimum absolute first-derivative
                magnitude (Δprob / Δfeature_value) to flag a point.
                Defaults to ``settings.pdp_min_slope`` when ``None``.
            min_prob_increase (float | None): Minimum probability increase
                above the PDP baseline required to flag a point.  Defaults to
                ``settings.pdp_min_prob_increase`` when ``None``.

        Returns:
            dict[str, Any]: Keyed by feature name.  Each entry is a dict
            with keys:

                - ``thresholds`` (list[dict]): Each detected threshold has
                  ``threshold_value``, ``prob_at_threshold``, ``prob_after``,
                  and ``slope``.
                - ``baseline`` (float): Average predicted probability at the
                  first grid value.
                - ``peak_prob`` (float): Maximum average probability on grid.
                - ``range_of_effect`` (float): ``peak_prob − baseline``.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["pdp_results_"])

        min_s = min_slope if min_slope is not None else settings.pdp_min_slope
        min_p = (
            min_prob_increase if min_prob_increase is not None else settings.pdp_min_prob_increase
        )

        out: dict[str, Any] = {}
        for feat, res in self.pdp_results_.items():
            grid = res["grid_values"]
            avg = res["average"]
            baseline = float(avg[0]) if len(avg) > 0 else 0.0
            peak = float(avg.max()) if len(avg) > 0 else 0.0

            thresholds: list[dict[str, float]] = []
            if np.issubdtype(grid.dtype, np.number) and len(grid) >= 2:
                dx = np.diff(grid.astype(float))
                dy = np.diff(avg)
                slopes = dy / np.where(dx != 0.0, dx, 1e-12)
                for i, slope in enumerate(slopes):
                    prob_increase = float(avg[i + 1]) - baseline
                    if abs(slope) >= min_s and prob_increase >= min_p:
                        thresholds.append(
                            {
                                "threshold_value": float(grid[i]),
                                "prob_at_threshold": float(avg[i]),
                                "prob_after": float(avg[i + 1]),
                                "slope": float(slope),
                            }
                        )

            out[feat] = {
                "thresholds": thresholds,
                "baseline": baseline,
                "peak_prob": peak,
                "range_of_effect": peak - baseline,
            }
        return out

    def summary_table(self) -> pd.DataFrame:
        """Return a per-feature PDP statistics summary as a DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: One row per successfully fitted feature with columns
            ``feature`` (str), ``n_grid_points`` (int), ``baseline_prob``
            (float, avg prob at first grid value), ``peak_prob`` (float),
            ``trough_prob`` (float), ``range_of_effect`` (float,
            peak − trough).  Returns an empty DataFrame with the same column
            schema when ``pdp_results_`` is empty.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["pdp_results_"])
        rows = []
        for feat, res in self.pdp_results_.items():
            avg = res["average"]
            grid = res["grid_values"]
            rows.append(
                {
                    "feature": feat,
                    "n_grid_points": int(len(grid)),
                    "baseline_prob": float(avg[0]),
                    "peak_prob": float(avg.max()),
                    "trough_prob": float(avg.min()),
                    "range_of_effect": float(avg.max() - avg.min()),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_SUMMARY_COLS)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_predict_fn(self):
        """Return model.predict_proba after validating the model exposes it."""
        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "PDPAnalyser requires a model with predict_proba, "
                "got %s" % type(self.model).__name__
            )
        return self.model.predict_proba

    @staticmethod
    def _compute_pdp(
        predict_fn,
        X: pd.DataFrame,
        col: str,
        grid_values: np.ndarray,
    ) -> np.ndarray:
        """Sweep ``col`` over ``grid_values`` and return average predicted prob."""
        results = np.empty(len(grid_values))
        for i, val in enumerate(grid_values):
            X_mod = X.copy()
            X_mod[col] = val
            proba = np.asarray(predict_fn(X_mod))
            results[i] = float(proba[:, 1].mean() if proba.ndim == 2 else proba.mean())
        return results
