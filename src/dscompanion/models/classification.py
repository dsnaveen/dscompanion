"""ClassificationModel: wrapper for binary/multiclass classification estimators."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from dscompanion.config import settings
from dscompanion.models.base import BaseDSCompanionModel
from dscompanion.utils.metrics import gini_coefficient, ks_statistic, psi_score

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["ClassificationModel"]


class ClassificationModel(BaseDSCompanionModel):
    """Model wrapper for supervised binary and multiclass classification tasks.

    Extends ``BaseDSCompanionModel`` with classification-specific metrics and
    diagnostic plots.  After ``fit``, the positive-class training scores are
    cached internally so that Population Stability Index (PSI) can be computed
    against hold-out or OOT splits during evaluation.

    Metrics computed on each split: ``f1``, ``precision``, ``recall``,
    ``roc_auc``, ``gini``, ``ks_statistic``, ``log_loss``, and ``psi``
    (only when the split size differs from training, indicating a non-train
    partition).

    Args:
        estimator (Any): An instantiated sklearn-compatible classifier that
            implements ``fit``, ``predict``, and optionally ``predict_proba``.
        **kwargs: Additional keyword arguments forwarded to
            ``BaseDSCompanionModel.__init__`` (``feature_names``).
    """

    def __init__(self, estimator: Any, **kwargs: Any) -> None:  # Any: sklearn-compatible classifier
        super().__init__(estimator, **kwargs)
        self._train_scores: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list | None = None,
    ) -> "ClassificationModel":
        """Fit the classifier and cache positive-class training scores for PSI.

        Calls the parent ``BaseDSCompanionModel.fit`` (which handles timing and
        stdlib logging), then — if the estimator exposes ``predict_proba`` —
        stores the positive-class scores from the training set in
        ``self._train_scores`` so that PSI can be computed in later evaluation
        calls.  Side-effects: all side-effects of the parent ``fit`` apply, and
        ``self._train_scores`` is set.

        Args:
            X (pd.DataFrame): Training feature matrix, shape
                ``(n_samples, n_features)``.
            y (pd.Series): Binary or multiclass target vector, shape
                ``(n_samples,)``.
            eval_set (list, optional): List of ``(X_eval, y_eval)`` tuples for
                early stopping with XGBoost-compatible estimators.  Ignored
                silently for estimators that do not accept this argument.
                Defaults to ``None``.

        Returns:
            ClassificationModel: ``self``, allowing method chaining.
        """
        super().fit(X, y, eval_set=eval_set)
        if hasattr(self.estimator, "predict_proba"):
            self._train_scores = self.estimator.predict_proba(X)[:, 1]
        return self

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        split_name: str = "",
    ) -> dict[str, float]:
        """Compute a standard suite of classification evaluation metrics.

        Always computes label-based metrics (``f1``, ``precision``,
        ``recall``).  Probability-based metrics (``roc_auc``, ``gini``,
        ``ks_statistic``, ``log_loss``, ``psi``) are added only when
        ``y_prob`` is supplied.  PSI is included only when the length of
        ``y_prob`` differs from the cached training scores, indicating that
        evaluation is occurring on a non-training split.  Each metric is
        computed independently — one failure does not suppress the rest.
        No side-effects.

        Args:
            y_true (np.ndarray): Ground-truth binary labels, shape
                ``(n_samples,)``.
            y_pred (np.ndarray): Hard predicted labels produced by
                ``predict``, shape ``(n_samples,)``.
            y_prob (np.ndarray, optional): Predicted positive-class
                probabilities, shape ``(n_samples,)``.  Pass ``None`` when
                probability output is not available.

        Returns:
            dict: Mapping of metric name (str) to scalar float value.  Always
            contains at minimum ``f1``, ``precision``, and ``recall``.
            Returns an empty dict only if all metric computations raise.
        """
        from sklearn.metrics import (
            f1_score,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics: dict[str, float] = {}
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

        if y_prob is not None:
            for name, fn in [
                ("roc_auc", lambda: float(roc_auc_score(y_true, y_prob))),
                ("gini", lambda: gini_coefficient(y_true, y_prob)),
                ("ks_statistic", lambda: ks_statistic(y_true, y_prob)),
                ("log_loss", lambda: float(log_loss(y_true, y_prob))),
            ]:
                try:
                    metrics[name] = fn()
                except (ValueError, ArithmeticError) as exc:
                    logger.debug("metric '%s' skipped: %s", name, exc)

            if split_name != "train" and self._train_scores is not None:
                try:
                    metrics["psi"] = psi_score(self._train_scores, y_prob)
                except (ValueError, ArithmeticError) as exc:
                    logger.debug("metric 'psi' skipped: %s", exc)

        return metrics

    def roc_curve_plot(self, X: pd.DataFrame, y: pd.Series) -> "go.Figure":
        """Build and return an interactive Plotly ROC curve figure.

        Calls ``predict_proba`` internally to obtain scores, computes the ROC
        curve via sklearn, and overlays a random-classifier diagonal for
        reference.  The dscompanion visual theme is applied before returning.  No
        side-effects beyond the predict call.

        Args:
            X (pd.DataFrame): Feature matrix, shape ``(n_samples, n_features)``.
            y (pd.Series): True binary labels, shape ``(n_samples,)``.

        Returns:
            plotly.graph_objects.Figure: Interactive ROC curve with AUC
            annotated in the trace legend.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame`` or ``y`` is not a
                ``pd.Series``.
            ImportError: If ``plotly`` is not installed in the environment.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pd.Series, got %s" % type(y).__name__)

        from sklearn.metrics import roc_curve

        from dscompanion.utils.plotting import apply_dscompanion_theme

        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required.") from exc

        y_prob = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = float(np.trapz(tpr, fpr))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC (AUC=%.3f)" % auc, mode="lines"))
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines", line=dict(dash="dash"))
        )
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        apply_dscompanion_theme(fig)
        return fig

    def score_distribution_plot(self, X: pd.DataFrame, y: pd.Series) -> "go.Figure":
        """Build and return an overlaid Plotly histogram of predicted scores by class.

        Calls ``predict_proba`` to obtain positive-class probabilities, then
        renders a semi-transparent overlaid histogram with separate traces for
        the negative (0) and positive (1) classes.  The dscompanion visual theme is
        applied before returning.  No side-effects beyond the predict call.

        Args:
            X (pd.DataFrame): Feature matrix, shape ``(n_samples, n_features)``.
            y (pd.Series): True binary labels, shape ``(n_samples,)``; used
                only for class-conditional subsetting and not mutated.

        Returns:
            plotly.graph_objects.Figure: Interactive overlaid histogram with
            one trace per class and ``settings.score_dist_bins`` bins per trace.

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

        y_prob = self.predict_proba(X)[:, 1]
        fig = go.Figure()
        for cls, name in [(0, "Negative"), (1, "Positive")]:
            fig.add_trace(
                go.Histogram(
                    x=y_prob[y.values == cls],
                    name=name,
                    opacity=0.6,
                    nbinsx=settings.score_dist_bins,
                )
            )
        fig.update_layout(
            barmode="overlay",
            title="Score Distribution by Class",
            xaxis_title="Predicted Probability",
        )
        apply_dscompanion_theme(fig)
        return fig
