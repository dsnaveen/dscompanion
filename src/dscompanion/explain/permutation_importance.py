"""PermutationImportanceAnalyser: model-agnostic feature importance via row-shuffling.

Unlike ``SHAPExplainer``, this needs no ``shap.TreeExplainer`` at all — pure
``sklearn.inspection.permutation_importance`` against an already-fitted estimator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings
from dscompanion.utils.plotting import apply_dscompanion_theme

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["PermutationImportanceAnalyser"]


class PermutationImportanceAnalyser(BaseEstimator):
    """Computes feature importance by measuring the score drop when each feature's
    values are randomly shuffled, using ``sklearn.inspection.permutation_importance``
    against an already-fitted dscompanion model's underlying estimator.

    Args:
        model (Any): Fitted dscompanion model instance whose ``estimator`` attribute
            exposes the underlying sklearn-compatible estimator.
        scoring (str): Scorer name recognised by ``sklearn.metrics.get_scorer``
            (e.g. ``"roc_auc"``, ``"r2"``). Caller-resolved — this class does no
            task detection of its own.
        n_repeats (int | None): Number of times each feature is shuffled and
            re-scored. When ``None``, defaults to
            ``settings.permutation_importance_n_repeats``.
        sample_size (int | None): Max rows sampled from the dataset passed to
            ``fit()`` before computing importances. When ``None``, defaults to
            ``settings.permutation_importance_sample_size``.
        top_n (int | None): Maximum number of features shown in ``summary_plot()``.
            When ``None``, defaults to ``settings.shap_top_n`` (shared display-count
            convention with ``SHAPExplainer``).

    Attributes:
        importances_ (sklearn.utils.Bunch): Raw result of
            ``sklearn.inspection.permutation_importance``; populated after ``fit``.
        feature_names_ (list[str]): Ordered list of feature column names from the
            DataFrame passed to ``fit``; populated after ``fit``.
    """

    def __init__(
        self,
        model: Any,  # Fitted model wrapper; estimator attr must expose the sklearn estimator
        scoring: str,
        n_repeats: int | None = None,
        sample_size: int | None = None,
        top_n: int | None = None,
    ) -> None:
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.sample_size = sample_size
        self.top_n = top_n

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PermutationImportanceAnalyser":
        """Compute permutation importances for every feature in ``X``.

        Args:
            X (pd.DataFrame): Feature matrix of shape ``(n_samples, n_features)``.
                Column names must match those used during model training.
            y (pd.Series): True labels/targets aligned with ``X``, used to score
                the fitted estimator before and after each feature is shuffled.

        Returns:
            PermutationImportanceAnalyser: The fitted instance (``self``), enabling
            method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "PermutationImportanceAnalyser.fit() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError(
                "PermutationImportanceAnalyser.fit() received an empty DataFrame (0 rows)"
            )

        n_repeats = (
            self.n_repeats
            if self.n_repeats is not None
            else settings.permutation_importance_n_repeats
        )
        sample_size = (
            self.sample_size
            if self.sample_size is not None
            else settings.permutation_importance_sample_size
        )

        if len(X) > sample_size:
            X = X.sample(sample_size, random_state=settings.random_state)
            y = y.loc[X.index]

        self.feature_names_ = list(X.columns)
        logger.info(
            "PermutationImportanceAnalyser.fit — %d rows, %d features, scoring=%s, n_repeats=%d",
            len(X),
            X.shape[1],
            self.scoring,
            n_repeats,
        )

        self.importances_ = permutation_importance(
            self.model.estimator,
            X,
            y,
            scoring=self.scoring,
            n_repeats=n_repeats,
            random_state=settings.random_state,
        )
        return self

    def importance_table(self) -> pd.DataFrame:
        """Return per-feature permutation importance, sorted from highest to lowest.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame with columns ``feature`` (str),
            ``importance_mean`` (float, mean score drop across ``n_repeats``
            shuffles), ``importance_std`` (float, std of that score drop), and
            ``rank`` (int, 1-based importance rank). Sorted descending by
            ``importance_mean``. Contains exactly one row per feature.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["importances_"])
        df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance_mean": self.importances_.importances_mean,
                    "importance_std": self.importances_.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        df["rank"] = df.index + 1
        return df

    def summary_plot(self) -> "go.Figure":
        """Render a horizontal bar chart of permutation importance for the top
        ``top_n`` features, with the most important feature at the top.

        Args:
            None

        Returns:
            plotly.graph_objects.Figure: A Plotly Figure with one horizontal bar
            per feature (up to ``top_n`` or ``settings.shap_top_n``), x-axis
            showing mean score drop under permutation.

        Raises:
            NotFittedError: If called before ``fit()``.
            ImportError: If ``plotly`` is not installed.
        """
        check_is_fitted(self, attributes=["importances_"])
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required: pip install plotly") from exc

        top_n = self.top_n if self.top_n is not None else settings.shap_top_n
        df = self.importance_table().head(top_n)
        fig = go.Figure(
            go.Bar(
                x=df["importance_mean"],
                y=df["feature"],
                orientation="h",
            )
        )
        fig.update_layout(
            title="Permutation Importance (top %d)" % top_n,
            xaxis_title="Mean score drop (%s)" % self.scoring,
            yaxis=dict(autorange="reversed"),
        )
        apply_dscompanion_theme(fig)
        return fig
