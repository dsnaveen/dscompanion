"""SHAPExplainer and BootstrapSHAPExplainer: SHAP-based model explainability."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings
from dscompanion.utils.plotting import apply_dscompanion_theme

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["SHAPExplainer", "BootstrapSHAPExplainer"]

_TREE_CLASSES = (
    "xgbclassifier",
    "xgbregressor",
    "randomforestclassifier",
    "randomforestregressor",
    "gradientboostingclassifier",
    "gradientboostingregressor",
    "lgbmclassifier",
    "lgbmregressor",
)
_LINEAR_CLASSES = ("logisticregression", "linearregression", "ridge", "lasso")


def _detect_explainer_type(estimator) -> str:
    cls = type(estimator).__name__.lower()
    if any(cls.startswith(t) for t in _TREE_CLASSES):
        return "tree"
    if any(cls.startswith(t) for t in _LINEAR_CLASSES):
        return "linear"
    return "kernel"


class SHAPExplainer(BaseEstimator):
    """Computes and visualises SHAP (SHapley Additive exPlanations) values for
    any dscompanion model, automatically selecting the most appropriate SHAP
    explainer backend for the underlying estimator type.

    Supports tree-based models (``TreeExplainer``), linear models
    (``LinearExplainer``), and arbitrary black-box models
    (``KernelExplainer``).  When ``explainer_type="auto"`` the backend is
    detected from the estimator class name.  SHAP values are stored as a
    2-D array in ``shap_values_`` after calling ``fit``.

    Args:
        model (Any): Fitted dscompanion model instance whose ``estimator`` attribute
            exposes the underlying sklearn-compatible estimator.
        background_data (pd.DataFrame | None): Background reference dataset
            used by ``KernelExplainer`` and ``LinearExplainer``. Ignored for
            tree-based explainers. When ``None``, a random sample of up to
            ``settings.shap_background_samples`` rows from the ``fit`` dataset
            is used as the background.
        explainer_type (str): SHAP backend to use. One of ``"auto"``
            (default, detected from estimator class name), ``"tree"``,
            ``"linear"``, or ``"kernel"``.
        max_display (int | None): Maximum number of features to show in
            summary and waterfall plots. When ``None``, defaults to
            ``settings.shap_top_n``.

    Attributes:
        shap_values_ (np.ndarray): 2-D array of shape
            ``(n_samples, n_features)`` containing raw SHAP values; populated
            after ``fit``.
        feature_names_ (list[str]): Ordered list of feature column names from
            the DataFrame passed to ``fit``; populated after ``fit``.
    """

    def __init__(
        self,
        model: Any,  # Fitted model wrapper; estimator attr must expose the sklearn estimator
        background_data: pd.DataFrame | None = None,
        explainer_type: str = "auto",
        max_display: int | None = None,
    ) -> None:
        self.model = model
        self.background_data = background_data
        self.explainer_type = explainer_type
        self.max_display = max_display

    def fit(self, X: pd.DataFrame) -> "SHAPExplainer":
        """Compute SHAP values for every row in ``X`` and store them in
        ``shap_values_``, along with ``feature_names_`` and an internal
        copy of the reset-indexed ``X`` used by plotting methods.

        The SHAP backend is chosen based on ``explainer_type`` (or
        auto-detected from the estimator class name).  For binary
        classifiers with list-typed SHAP output only the positive-class
        slice (index 1) is retained.

        Args:
            X (pd.DataFrame): Feature matrix of shape
                ``(n_samples, n_features)`` to compute SHAP values for.
                Column names must match those used during model training.

        Returns:
            SHAPExplainer: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
            ImportError: If the ``shap`` package is not installed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SHAPExplainer.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("SHAPExplainer.fit() received an empty DataFrame (0 rows)")

        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required: pip install shap") from exc

        self.feature_names_ = list(X.columns)
        etype = (
            _detect_explainer_type(self.model.estimator)
            if self.explainer_type == "auto"
            else self.explainer_type
        )
        logger.info("SHAPExplainer using %s explainer on %s", etype, X.shape)

        est = self.model.estimator
        n_bg = settings.shap_background_samples

        if etype == "tree":
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            elif np.asarray(sv).ndim == 3:
                sv = np.asarray(sv)[:, :, 1]
            self.shap_values_ = np.array(sv)
        elif etype == "linear":
            bg = (
                self.background_data
                if self.background_data is not None
                else X.sample(min(n_bg, len(X)), random_state=settings.random_state)
            )
            explainer = shap.LinearExplainer(est, bg)
            self.shap_values_ = explainer.shap_values(X)
        else:
            bg = (
                self.background_data.sample(
                    min(n_bg, len(self.background_data)),
                    random_state=settings.random_state,
                )
                if self.background_data is not None
                else X.sample(min(n_bg, len(X)), random_state=settings.random_state)
            )
            _cols = list(X.columns)
            _raw_predict = self.model.predict_proba

            def predict_fn(X_arr):
                return _raw_predict(pd.DataFrame(X_arr, columns=_cols))

            explainer = shap.KernelExplainer(predict_fn, bg)
            sv = explainer.shap_values(X, nsamples=settings.shap_kernel_nsamples)
            if isinstance(sv, list):
                sv = sv[1]
            elif np.asarray(sv).ndim == 3:
                sv = np.asarray(sv)[:, :, 1]
            self.shap_values_ = np.array(sv)

        self._X = X.reset_index(drop=True)
        logger.info("SHAP values computed — shape=%s", self.shap_values_.shape)
        return self

    def mean_abs_shap(self) -> pd.DataFrame:
        """Compute mean absolute SHAP importance for each feature across all
        samples in the dataset passed to ``fit``, sorted from highest to
        lowest importance.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame with columns ``feature`` (str),
            ``mean_abs_shap`` (float, mean of \\|SHAP\\| values across all
            rows), and ``rank`` (int, 1-based importance rank). Sorted
            descending by ``mean_abs_shap``. Contains exactly one row per
            feature.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        importance = np.abs(self.shap_values_).mean(axis=0)
        df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "mean_abs_shap": importance,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        df["rank"] = df.index + 1
        return df

    def summary_plot(self) -> "go.Figure":
        """Render a horizontal bar chart of mean absolute SHAP importance for
        the top ``max_display`` features, with the most important feature at
        the top.

        Args:
            None

        Returns:
            plotly.graph_objects.Figure: A Plotly Figure with one horizontal
            bar per feature (up to ``max_display`` or ``settings.shap_top_n``),
            x-axis showing mean \\|SHAP value\\|.

        Raises:
            NotFittedError: If called before ``fit()``.
            ImportError: If ``plotly`` is not installed.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required: pip install plotly") from exc

        top_n = self.max_display if self.max_display is not None else settings.shap_top_n
        df = self.mean_abs_shap().head(top_n)
        fig = go.Figure(
            go.Bar(
                x=df["mean_abs_shap"],
                y=df["feature"],
                orientation="h",
            )
        )
        fig.update_layout(
            title="SHAP Feature Importance (top %d)" % top_n,
            xaxis_title="Mean |SHAP value|",
            yaxis=dict(autorange="reversed"),
        )
        apply_dscompanion_theme(fig)
        return fig

    def waterfall_plot(self, idx: int) -> "go.Figure":
        """Render a waterfall-style horizontal bar chart showing how each
        feature's SHAP value drives the prediction for a single observation
        away from the base rate.

        Positive SHAP values are shown in crimson; negative values in
        steelblue. Features are ranked by absolute SHAP magnitude and the
        chart is capped at ``max_display`` features.

        Args:
            idx (int): Zero-based row index into the dataset that was passed
                to ``fit``. Must be in the range
                ``[0, len(shap_values_) - 1]``.

        Returns:
            plotly.graph_objects.Figure: A Plotly Figure with one horizontal
            bar per feature (up to ``max_display`` or ``settings.shap_top_n``).

        Raises:
            NotFittedError: If called before ``fit()``.
            ValueError: If ``idx`` is outside ``[0, n_samples - 1]``.
            ImportError: If ``plotly`` is not installed.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        n = len(self.shap_values_)
        if not (0 <= idx < n):
            raise ValueError(
                "idx %d is out of range — shap_values_ has %d rows [0, %d)" % (idx, n, n)
            )
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required: pip install plotly") from exc

        top_n = self.max_display if self.max_display is not None else settings.shap_top_n
        shap_vals = self.shap_values_[idx]
        order = np.argsort(np.abs(shap_vals))[::-1][:top_n]
        features = [self.feature_names_[i] for i in order]
        values = shap_vals[order]
        colors = ["crimson" if v > 0 else "steelblue" for v in values]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title="SHAP Waterfall — row %d" % idx,
            xaxis_title="SHAP value",
            yaxis=dict(autorange="reversed"),
        )
        apply_dscompanion_theme(fig)
        return fig

    def dependence_plot(self, feature: str, interaction_feature: str | None = None) -> "go.Figure":
        """Render a scatter plot of raw feature values versus their
        corresponding SHAP values to reveal non-linear relationships and
        interaction effects.

        When ``interaction_feature`` is provided and present in the fit
        dataset, the scatter points are colour-encoded by that feature's
        values using the RdBu diverging scale.

        Args:
            feature (str): Name of the primary feature to plot on the x-axis.
                Must be present in ``feature_names_``.
            interaction_feature (str | None): Name of a secondary feature
                used to colour-encode the scatter points. Silently ignored
                if it is not present in the fit dataset. Defaults to ``None``
                (uniform point colour, no colour scale).

        Returns:
            plotly.graph_objects.Figure: A Plotly scatter Figure with
            ``feature`` on the x-axis and SHAP value on the y-axis.

        Raises:
            NotFittedError: If called before ``fit()``.
            ValueError: If ``feature`` is not in ``feature_names_``.
            ImportError: If ``plotly`` is not installed.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        if feature not in self.feature_names_:
            raise ValueError("feature '%s' not found in fitted feature_names_" % feature)
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required: pip install plotly") from exc

        feat_idx = self.feature_names_.index(feature)
        x = self._X[feature].values
        y = self.shap_values_[:, feat_idx]

        marker = dict(size=4, opacity=0.5)
        if interaction_feature and interaction_feature in self._X.columns:
            marker["color"] = self._X[interaction_feature].values
            marker["colorscale"] = "RdBu"
            marker["showscale"] = True

        fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", marker=marker))
        fig.update_layout(
            title="SHAP Dependence — %s" % feature,
            xaxis_title=feature,
            yaxis_title="SHAP value",
        )
        apply_dscompanion_theme(fig)
        return fig


class BootstrapSHAPExplainer(BaseEstimator):
    """Computes SHAP values across multiple independent bootstrap batches to
    provide mean feature importance with confidence intervals (±std).

    Each batch draws a fresh stratified (if ``y`` is provided, else random)
    sample of ``batch_size`` rows from ``X``, computes SHAP values via the
    same auto-detected backend as ``SHAPExplainer``, and accumulates results
    in ``shap_values_`` (shape ``(n_batches, batch_size, n_features)``).
    Feature importance statistics are then aggregated across the batch axis.
    All plots are rendered with Plotly — no matplotlib dependency.

    Args:
        model (Any): Fitted model instance.  If it exposes an ``estimator``
            attribute (dscompanion wrapper convention) the underlying sklearn
            estimator is used for SHAP; otherwise ``model`` is used directly.
        n_batches (int | None): Number of bootstrap batches. Defaults to
            ``settings.shap_bootstrap_batches`` when ``None``.
        batch_size (int | None): Rows sampled per batch. Defaults to
            ``settings.shap_bootstrap_batch_size`` when ``None``.
        background_size (int | None): Max background rows forwarded to
            ``LinearExplainer`` / ``KernelExplainer``. Defaults to
            ``settings.shap_background_samples`` when ``None``.

    Attributes:
        shap_values_ (np.ndarray): Raw per-batch SHAP values — shape
            ``(n_batches, actual_batch_size, n_features)``.
        mean_shap_values_ (np.ndarray): Element-wise mean across the batch
            axis — shape ``(actual_batch_size, n_features)``.
        std_shap_values_ (np.ndarray): Element-wise std across the batch
            axis — same shape as ``mean_shap_values_``.
        feature_importance_ (pd.Series): Mean \\|SHAP\\| per feature, sorted
            descending.  Populated after ``fit()``.
        feature_importance_std_ (pd.Series): Std of \\|SHAP\\| per feature,
            aligned to ``feature_importance_`` index.
        feature_names_ (list[str]): Column names from the DataFrame passed to
            ``fit()``.
        expected_value_ (float): SHAP explainer base value.
        batch_data_ (np.ndarray): Raw feature values per batch — shape
            ``(n_batches, actual_batch_size, n_features)``; used for
            colour-coding in scatter plots.
    """

    def __init__(
        self,
        model: Any,  # Model wrapper or raw sklearn estimator; estimator attr used when present
        n_batches: int | None = None,
        batch_size: int | None = None,
        background_size: int | None = None,
    ) -> None:
        self.model = model
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.background_size = background_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BootstrapSHAPExplainer":
        """Sample ``n_batches`` bootstrap batches from ``X``, compute SHAP
        values per batch, and aggregate importance statistics.

        Args:
            X (pd.DataFrame): Feature matrix to draw batches from.
            y (pd.Series | None): Target series used for stratified batch
                sampling when not ``None``.  Falls back to random sampling
                if stratification fails (e.g. too few samples per class).

        Returns:
            BootstrapSHAPExplainer: Fitted instance (``self``).

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
            ImportError: If ``shap`` is not installed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "BootstrapSHAPExplainer.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("BootstrapSHAPExplainer.fit() received an empty DataFrame (0 rows)")

        try:
            import shap  # noqa: F401 — presence check; actual use inside _build_explainer
        except ImportError as exc:
            raise ImportError("shap is required: pip install shap") from exc

        n_batches = (
            self.n_batches if self.n_batches is not None else settings.shap_bootstrap_batches
        )
        batch_sz = (
            self.batch_size if self.batch_size is not None else settings.shap_bootstrap_batch_size
        )
        bg_size = (
            self.background_size
            if self.background_size is not None
            else settings.shap_background_samples
        )

        self.feature_names_: list[str] = list(X.columns)
        n_features = len(self.feature_names_)
        actual_batch = min(batch_sz, len(X))

        logger.info(
            "BootstrapSHAPExplainer: model=%s, %d batches × %d samples, %d features",
            type(self.model).__name__,
            n_batches,
            actual_batch,
            n_features,
        )

        X_bg = self._stratified_sample(X, y, min(bg_size, len(X)), settings.random_state)
        self._explainer = self._build_explainer(X_bg)
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev_arr = np.asarray(ev).ravel()
            self.expected_value_: float = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
        else:
            self.expected_value_: float = float(ev)

        self.shap_values_: np.ndarray = np.zeros((n_batches, actual_batch, n_features))
        self.batch_data_: np.ndarray = np.zeros((n_batches, actual_batch, n_features))

        for b in range(n_batches):
            X_b = self._stratified_sample(X, y, actual_batch, settings.random_state + b)
            self.batch_data_[b] = X_b.values
            try:
                sv = self._explainer.shap_values(X_b)
                if isinstance(sv, list):
                    sv = sv[1]
                elif np.asarray(sv).ndim == 3:
                    sv = np.asarray(sv)[:, :, 1]
                self.shap_values_[b] = np.asarray(sv)
            except (RuntimeError, ValueError, MemoryError) as exc:
                logger.warning("BootstrapSHAPExplainer: batch %d failed — %s", b, exc)
            logger.debug("BootstrapSHAPExplainer: batch %d/%d done", b + 1, n_batches)

        self._compute_statistics()
        logger.info(
            "BootstrapSHAPExplainer fitted — %d batches, expected_value=%.4f",
            n_batches,
            self.expected_value_,
        )
        return self

    def get_feature_importance(self, with_confidence: bool = True) -> pd.DataFrame:
        """Return per-feature importances with optional bootstrap confidence
        intervals.

        Args:
            with_confidence (bool): When ``True`` (default), appends
                ``importance_std``, ``importance_ci_lower``, and
                ``importance_ci_upper`` columns derived from the bootstrap
                standard deviation of absolute SHAP values.

        Returns:
            pd.DataFrame: DataFrame indexed by feature name with columns
            ``importance`` (float, mean \\|SHAP\\|), ``rank`` (int, 1-based),
            and optionally ``importance_std``, ``importance_ci_lower``,
            ``importance_ci_upper``.  Sorted descending by ``importance``.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["feature_importance_"])
        df = pd.DataFrame(
            {
                "importance": self.feature_importance_,
                "rank": range(1, len(self.feature_importance_) + 1),
            }
        )
        if with_confidence:
            df["importance_std"] = self.feature_importance_std_
            df["importance_ci_lower"] = (df["importance"] - df["importance_std"]).clip(lower=0)
            df["importance_ci_upper"] = df["importance"] + df["importance_std"]
        return df

    def beeswarm_plot(self, max_display: int | None = None) -> "go.Figure":
        """Render a Plotly beeswarm-style scatter showing the distribution of
        SHAP values across bootstrap samples, colour-coded by normalised
        feature value (blue = low, red = high).

        Points are drawn from ``mean_shap_values_`` (SHAP averaged across
        batches), subsampled to at most 2 000 points per display for
        rendering performance.

        Args:
            max_display (int | None): Maximum number of features to show
                (top N by importance). Defaults to ``settings.shap_top_n``
                when ``None``.

        Returns:
            plotly.graph_objects.Figure: Horizontal scatter with feature
            names on the y-axis (most important at the top) and SHAP values
            on the x-axis.

        Raises:
            NotFittedError: If called before ``fit()``.
            ImportError: If ``plotly`` is not installed.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required: pip install plotly") from exc

        top_n = max_display if max_display is not None else settings.shap_top_n
        n_batches = self.shap_values_.shape[0]
        top_features = self.get_feature_importance(with_confidence=False).head(top_n).index.tolist()

        shap_display = self.mean_shap_values_  # (batch_size, n_features)
        X_display = self.batch_data_.mean(axis=0)  # (batch_size, n_features)
        n_pts = shap_display.shape[0]

        _MAX_PTS = settings.shap_beeswarm_max_display_samples
        if n_pts > _MAX_PTS:
            rng = np.random.default_rng(settings.random_state)
            idx = rng.choice(n_pts, _MAX_PTS, replace=False)
            shap_display = shap_display[idx]
            X_display = X_display[idx]

        xs: list[float] = []
        ys: list[str] = []
        cs: list[float] = []

        for feat in reversed(top_features):  # reversed → most important at top of chart
            fi = self.feature_names_.index(feat)
            sv = shap_display[:, fi]
            fv = X_display[:, fi]
            fmin, fmax = fv.min(), fv.max()
            norm = (fv - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(fv)
            xs.extend(sv.tolist())
            ys.extend([feat] * len(sv))
            cs.extend(norm.tolist())

        fig = go.Figure(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    color=cs,
                    colorscale="RdBu_r",
                    size=3,
                    opacity=0.6,
                    colorbar=dict(title="Feature value<br>(normalised)", thickness=12),
                ),
            )
        )
        fig.update_layout(
            title="Bootstrap SHAP Beeswarm — top %d features (%d batches)" % (top_n, n_batches),
            xaxis_title="SHAP value",
            yaxis_title="Feature",
        )
        apply_dscompanion_theme(fig)
        return fig

    def importance_plot(self, max_display: int | None = None) -> "go.Figure":
        """Render a horizontal bar chart of mean \\|SHAP\\| importance with
        bootstrap confidence intervals shown as ±std error bars.

        Args:
            max_display (int | None): Maximum number of features to show.
                Defaults to ``settings.shap_top_n`` when ``None``.

        Returns:
            plotly.graph_objects.Figure: Horizontal bar chart; error bars
            encode the bootstrap std of absolute SHAP values.

        Raises:
            NotFittedError: If called before ``fit()``.
            ImportError: If ``plotly`` is not installed.
        """
        check_is_fitted(self, attributes=["feature_importance_"])
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required: pip install plotly") from exc

        top_n = max_display if max_display is not None else settings.shap_top_n
        n_batches = self.shap_values_.shape[0]
        df = self.get_feature_importance(with_confidence=True).head(top_n)

        fig = go.Figure(
            go.Bar(
                x=df["importance"],
                y=df.index,
                orientation="h",
                error_x=dict(type="data", array=df["importance_std"].tolist(), visible=True),
            )
        )
        fig.update_layout(
            title="Bootstrap SHAP Importance ± Std — top %d (%d batches)" % (top_n, n_batches),
            xaxis_title="Mean |SHAP value|",
            yaxis=dict(autorange="reversed"),
        )
        apply_dscompanion_theme(fig)
        return fig

    def dependence_plot(self, feature: str) -> "go.Figure":
        """Render a scatter of feature-value deviation from mean versus SHAP
        deviation from mean, with quadrant annotations showing the low/high
        feature × low/high impact combinations.

        Args:
            feature (str): Name of the feature to plot.  Must be present in
                ``feature_names_``.

        Returns:
            plotly.graph_objects.Figure: Scatter figure; Pearson correlation
            between the two deviation axes appears in the title.

        Raises:
            NotFittedError: If called before ``fit()``.
            ValueError: If ``feature`` is not in ``feature_names_``.
            ImportError: If ``plotly`` is not installed.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        if feature not in self.feature_names_:
            raise ValueError("feature '%s' not found in fitted feature_names_" % feature)
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required: pip install plotly") from exc

        n_batches = self.shap_values_.shape[0]
        fi = self.feature_names_.index(feature)
        feat_vals = self.batch_data_[:, :, fi].flatten()
        shap_vals = self.shap_values_[:, :, fi].flatten()
        feat_dev = feat_vals - feat_vals.mean()
        shap_dev = shap_vals - shap_vals.mean()
        corr = float(np.corrcoef(feat_dev, shap_dev)[0, 1])

        fig = go.Figure(
            go.Scatter(
                x=feat_dev,
                y=shap_dev,
                mode="markers",
                marker=dict(
                    color=shap_dev,
                    colorscale="RdBu_r",
                    size=3,
                    opacity=0.5,
                    colorbar=dict(title="SHAP<br>deviation", thickness=12),
                ),
            )
        )
        fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1))
        fig.add_vline(x=0, line=dict(color="black", dash="dash", width=1))

        for x_pct, y_pct, xanchor, yanchor, text in [
            (0.01, 0.99, "left", "top", "Low feature<br>High impact"),
            (0.99, 0.99, "right", "top", "High feature<br>High impact"),
            (0.01, 0.01, "left", "bottom", "Low feature<br>Low impact"),
            (0.99, 0.01, "right", "bottom", "High feature<br>Low impact"),
        ]:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=x_pct,
                y=y_pct,
                text=text,
                showarrow=False,
                xanchor=xanchor,
                yanchor=yanchor,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.75)",
                borderpad=3,
            )

        fig.update_layout(
            title="Bootstrap SHAP Dependence — %s (corr=%.3f, %d batches)"
            % (feature, corr, n_batches),
            xaxis_title="%s deviation from mean" % feature,
            yaxis_title="SHAP deviation from mean",
        )
        apply_dscompanion_theme(fig)
        return fig

    def export_shap_values(self, path: str | Path) -> None:
        """Export all bootstrap SHAP values to a Parquet file.

        The output has one row per (batch, sample) pair with columns
        ``shap_{feature}`` for every feature, plus ``batch_id``,
        ``sample_id``, and ``expected_value`` metadata columns.

        Args:
            path (str | Path): Destination Parquet file path.  Parent
                directories are created if they do not exist.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n_batches, batch_sz, _ = self.shap_values_.shape
        flat = self.shap_values_.reshape(n_batches * batch_sz, -1)
        df = pd.DataFrame(flat, columns=["shap_%s" % c for c in self.feature_names_])
        df["batch_id"] = np.repeat(np.arange(n_batches), batch_sz)
        df["sample_id"] = np.tile(np.arange(batch_sz), n_batches)
        df["expected_value"] = self.expected_value_
        df.to_parquet(path, index=False)
        logger.info(
            "BootstrapSHAPExplainer: SHAP values exported — path=%s, shape=%s",
            path,
            df.shape,
        )

    def save(self, path: str | Path) -> None:
        """Serialise the fitted explainer to disk with joblib.

        Args:
            path (str | Path): Destination file path.  Parent directories
                are created if they do not exist.  An existing file is
                overwritten.

        Raises:
            NotFittedError: If called before ``fit()``.
            ImportError: If ``joblib`` is not installed.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required: pip install joblib") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info("BootstrapSHAPExplainer saved — path=%s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BootstrapSHAPExplainer":
        """Load a serialised ``BootstrapSHAPExplainer`` from disk.

        Args:
            path (str | Path): Path to a joblib file produced by ``save()``.

        Returns:
            BootstrapSHAPExplainer: Fully fitted explainer with all attributes
            restored.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ImportError: If ``joblib`` is not installed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("Explainer file not found: %s" % path)
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("joblib is required: pip install joblib") from exc
        instance = joblib.load(path)
        logger.info("BootstrapSHAPExplainer loaded — path=%s", path)
        return instance

    def get_summary_stats(self) -> dict[str, Any]:
        """Return stability metrics aggregated across bootstrap batches.

        Returns:
            dict[str, Any]: Keys:

                - ``n_batches`` (int)
                - ``batch_size`` (int)
                - ``total_samples`` (int)
                - ``n_features`` (int)
                - ``expected_value`` (float)
                - ``top_5_features`` (dict, feature → mean \\|SHAP\\|)
                - ``feature_stability`` (dict, feature → dict with
                  ``mean_abs_shap``, ``std_abs_shap``,
                  ``coefficient_of_variation``)
                - ``most_stable_features`` (list[tuple], 5 features with
                  lowest coefficient of variation, sorted ascending)

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["shap_values_"])
        n_batches, batch_sz, _ = self.shap_values_.shape
        stability: dict[str, dict[str, float]] = {}
        for i, feat in enumerate(self.feature_names_):
            feat_shap = self.shap_values_[:, :, i]
            m = float(np.mean(np.abs(feat_shap)))
            s = float(np.std(np.abs(feat_shap)))
            stability[feat] = {
                "mean_abs_shap": m,
                "std_abs_shap": s,
                "coefficient_of_variation": s / m if m > 0 else 0.0,
            }
        most_stable = sorted(stability.items(), key=lambda kv: kv[1]["coefficient_of_variation"])[
            :5
        ]
        return {
            "n_batches": n_batches,
            "batch_size": batch_sz,
            "total_samples": n_batches * batch_sz,
            "n_features": len(self.feature_names_),
            "expected_value": self.expected_value_,
            "top_5_features": self.feature_importance_.head().to_dict(),
            "feature_stability": stability,
            "most_stable_features": most_stable,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stratified_sample(
        X: pd.DataFrame,
        y: pd.Series | None,
        n: int,
        seed: int,
    ) -> pd.DataFrame:
        n = min(n, len(X))
        if y is not None and n < len(X):
            try:
                _, X_s = train_test_split(X, test_size=n, stratify=y, random_state=seed)
                return X_s
            except ValueError:
                pass
        return X.sample(n, random_state=seed)

    def _build_explainer(self, background_data: pd.DataFrame) -> Any:
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required: pip install shap") from exc

        est = self.model.estimator if hasattr(self.model, "estimator") else self.model
        etype = _detect_explainer_type(est)
        logger.info("BootstrapSHAPExplainer: using %s explainer", etype)

        if etype == "tree":
            return shap.TreeExplainer(est)
        if etype == "linear":
            return shap.LinearExplainer(est, background_data)
        bg = background_data.sample(
            min(settings.shap_kernel_nsamples, len(background_data)),
            random_state=settings.random_state,
        )
        _cols = list(background_data.columns)
        _raw_predict = (
            self.model.predict_proba if hasattr(self.model, "predict_proba") else est.predict_proba
        )

        def predict_fn(X_arr):
            return _raw_predict(pd.DataFrame(X_arr, columns=_cols))

        return shap.KernelExplainer(predict_fn, bg)

    def _compute_statistics(self) -> None:
        self.mean_shap_values_: np.ndarray = np.mean(self.shap_values_, axis=0)
        self.std_shap_values_: np.ndarray = np.std(self.shap_values_, axis=0)
        mean_abs = np.mean(np.abs(self.mean_shap_values_), axis=0)
        self.feature_importance_: pd.Series = pd.Series(
            mean_abs, index=self.feature_names_
        ).sort_values(ascending=False)
        std_abs = np.std(np.abs(self.shap_values_), axis=(0, 1))
        self.feature_importance_std_: pd.Series = pd.Series(
            std_abs, index=self.feature_names_
        ).reindex(self.feature_importance_.index)
