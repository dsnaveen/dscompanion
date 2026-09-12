"""ClusteringModel: wrapper for unsupervised clustering estimators."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from dscompanion.config import settings
from dscompanion.models.base import BaseDSCompanionModel

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)

__all__ = ["ClusteringModel"]


class ClusteringModel(BaseDSCompanionModel):
    """Model wrapper for unsupervised clustering algorithms.

    Extends ``BaseDSCompanionModel`` to support estimators that do not use a
    supervised target.  Overrides ``predict``, ``predict_proba``, and
    ``evaluate`` to handle the unsupervised evaluation paradigm where the
    feature matrix itself is used to compute internal validity metrics.

    Metrics computed when more than one cluster is present: ``silhouette_score``
    (sampled to at most ``settings.silhouette_sample_size`` points),
    ``davies_bouldin_score``, and ``calinski_harabasz_score``.  ``inertia`` is
    added when the estimator exposes an ``inertia_`` attribute (e.g. KMeans).

    Args:
        estimator (Any): An instantiated sklearn-compatible clustering
            estimator (e.g. ``KMeans``, ``DBSCAN``,
            ``AgglomerativeClustering``).
        **kwargs: Additional keyword arguments forwarded to
            ``BaseDSCompanionModel.__init__`` (``feature_names``).
    """

    def __init__(
        self, estimator: Any, **kwargs: Any
    ) -> None:  # Any: sklearn-compatible clustering estimator
        super().__init__(estimator, **kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return cluster label assignments for the supplied feature matrix.

        When the estimator exposes a ``predict`` method (e.g. KMeans), that
        method is called with ``X`` to assign new samples to clusters.
        Otherwise, the labels learned during ``fit`` (``estimator.labels_``)
        are returned directly; in that case ``X`` is ignored.  No side-effects.

        Args:
            X (pd.DataFrame): Feature matrix, shape ``(n_samples, n_features)``.
                Ignored when the estimator does not support ``predict``
                (e.g. DBSCAN after fitting).

        Returns:
            np.ndarray: Integer cluster label array, shape ``(n_samples,)``.
            Label values depend on the algorithm; DBSCAN uses ``-1`` for noise
            points.
        """
        if hasattr(self.estimator, "predict"):
            return self.estimator.predict(X)
        return self.estimator.labels_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Not supported for clustering models — always raises ``AttributeError``.

        Clustering is unsupervised and produces hard label assignments; there
        is no probabilistic class membership in the general case.  This method
        is provided to satisfy the ``BaseDSCompanionModel`` interface contract.

        Args:
            X (pd.DataFrame): Ignored.

        Returns:
            This method never returns; it always raises.

        Raises:
            AttributeError: Unconditionally, because clustering models do not
                produce class probabilities.
        """
        raise AttributeError("ClusteringModel does not support predict_proba.")

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        split_name: str = "",
    ) -> dict[str, float]:
        """Compute internal clustering validity metrics from the feature matrix and labels.

        Because clustering is unsupervised, the ``y_true`` slot is repurposed
        to carry the raw feature array (``X.values``).  Silhouette score is
        computed on a random sample of up to ``settings.silhouette_sample_size``
        points to manage runtime.  Individual metric failures are caught and
        logged at DEBUG level.  No side-effects.

        Args:
            y_true (np.ndarray): The raw feature array used for cluster
                assignment, shape ``(n_samples, n_features)``.  Despite its
                name, this is **not** a label array; it is the ``X`` matrix
                passed through the unsupervised evaluation path.
            y_pred (np.ndarray): Integer cluster label assignments for each
                sample, shape ``(n_samples,)``.
            y_prob (np.ndarray, optional): Ignored; present only to match the
                abstract method signature.

        Returns:
            dict: Mapping of metric name (str) to scalar float value.  Contains
            up to four keys: ``silhouette_score``, ``davies_bouldin_score``,
            ``calinski_harabasz_score``, and ``inertia``.  Returns an empty
            dict when only one unique cluster label is present (metrics are
            undefined for a single cluster), or when all metric computations
            raise exceptions.
        """
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        X_arr = y_true  # repurpose y_true slot to carry X for unsupervised metrics
        n_labels = len(np.unique(y_pred))

        metrics: dict[str, float] = {}
        if n_labels > 1:
            try:
                metrics["silhouette_score"] = float(
                    silhouette_score(
                        X_arr,
                        y_pred,
                        sample_size=min(settings.silhouette_sample_size, len(X_arr)),
                    )
                )
                metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_arr, y_pred))
                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_arr, y_pred))
            except Exception as exc:
                logger.debug("Clustering metrics skipped: %s", exc)

        if hasattr(self.estimator, "inertia_"):
            metrics["inertia"] = float(self.estimator.inertia_)

        return metrics

    def evaluate(self, split: Any) -> pd.DataFrame:  # Any: DataSplit — avoids circular import
        """Compute internal clustering validity metrics on the training partition.

        Overrides the parent ``evaluate`` to handle the unsupervised case:
        only ``X_train`` is used (there is no ``y_train`` for scoring), and
        the feature values are passed directly to ``_compute_metrics`` via the
        ``y_true`` slot.  Validation and OOT splits are not evaluated.  No
        side-effects.

        Args:
            split (DataSplit): Object with at minimum a ``X_train`` attribute
                (pd.DataFrame).  ``y_train`` and other split attributes are
                ignored.

        Returns:
            pd.DataFrame: Tidy frame with columns ``split`` (always
            ``"train"``), ``metric`` (str), and ``value`` (float, rounded to
            ``settings.evaluate_round_precision`` decimal places).  Returns a
            DataFrame with zero rows if ``_compute_metrics`` produces no results
            (e.g. single-cluster solution).
        """
        labels = self.predict(split.X_train)
        metrics = self._compute_metrics(split.X_train.values, labels)
        return pd.DataFrame(
            [
                {
                    "split": "train",
                    "metric": k,
                    "value": round(v, settings.evaluate_round_precision),
                }
                for k, v in metrics.items()
            ]
        )

    def elbow_plot(self, X: pd.DataFrame, max_k: int = 12) -> "go.Figure":
        """Fit KMeans for each k from 2 to ``max_k`` and return an inertia elbow plot.

        Trains a fresh ``KMeans`` instance for every k value, collecting
        inertia at each step.  NaN values in ``X`` are filled with 0.0 before
        fitting.  Side-effect: fits ``max_k - 1`` temporary KMeans models;
        none are stored on ``self``.  The dscompanion visual theme is applied
        before returning.

        Args:
            X (pd.DataFrame): Feature matrix to cluster, shape
                ``(n_samples, n_features)``.  Missing values are zero-filled
                internally.
            max_k (int): Largest number of clusters to evaluate (inclusive).
                Must be at least 2.  Defaults to 12.

        Returns:
            plotly.graph_objects.Figure: Line+marker plot of inertia versus k,
            useful for identifying the "elbow" where adding clusters yields
            diminishing returns.

        Raises:
            ImportError: If ``plotly`` is not installed in the environment.
        """
        from sklearn.cluster import KMeans

        from dscompanion.utils.plotting import apply_dscompanion_theme

        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required.") from exc

        ks: list[int] = list(range(2, max_k + 1))
        inertias: list[float] = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=settings.random_state, n_init="auto")
            km.fit(X.fillna(0))
            inertias.append(km.inertia_)

        fig = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers"))
        fig.update_layout(
            title="KMeans Elbow Plot",
            xaxis_title="Number of clusters (k)",
            yaxis_title="Inertia",
        )
        apply_dscompanion_theme(fig)
        return fig
