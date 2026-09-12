"""FeatureSelectionPipeline: chains selectors in recommended order."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

from dscompanion.selection.feature_selectors import (
    BaseSelector,
    CardinalitySelector,
    ConstantSelector,
    CorrelationSelector,
    IVSelector,
)
from dscompanion.utils.plotting import apply_dscompanion_theme

__all__ = ["FeatureSelectionPipeline"]


def _default_selectors() -> list[BaseSelector]:
    return [
        ConstantSelector(),
        CardinalitySelector(),
        CorrelationSelector(),
        IVSelector(),
    ]


class FeatureSelectionPipeline(BaseEstimator, TransformerMixin):
    """Chain multiple ``BaseSelector`` instances in sequence, passing each stage's
    output to the next.

    The default chain applies selectors in the recommended order:
    ``ConstantSelector → CardinalitySelector → CorrelationSelector → IVSelector``.
    A full audit of every removal decision is accumulated across stages and
    exposed via ``audit_``.  Side-effects: populates ``audit_``, ``_selected``,
    ``_stage_counts``, and ``_stage_names`` after ``fit``.

    Args:
        selectors: Ordered list of ``BaseSelector`` instances to apply.
            Each selector is fitted on the output of the previous one.
            When ``None`` (default), the recommended four-stage chain is used.
        verbose: When ``True`` (default), logs the feature count entering and
            leaving each stage at INFO level.

    Attributes:
        audit_: DataFrame with columns ``feature``, ``removed_by``, ``reason``,
            and ``original_position``, containing one row per removed feature
            across all stages.  Populated after ``fit``.
    """

    def __init__(
        self,
        selectors: list[BaseSelector] | None = None,
        verbose: bool = True,
    ) -> None:
        self.selectors = selectors
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureSelectionPipeline":
        """Fit each selector in sequence and accumulate the removal audit.

        Each selector is fitted on the DataFrame produced by all preceding
        selectors.  ``y`` is forwarded to every selector unchanged so that
        supervised selectors (``IVSelector``, ``RFESelector``) receive the
        target without the caller needing to be aware of which selectors
        require it.  Logs per-stage feature counts when ``verbose=True``, and
        a final summary at INFO level.

        Args:
            X: Training feature DataFrame.  Shape ``(n_samples, n_features)``.
            y: Optional target series forwarded to each selector.  Required
                by ``IVSelector`` and ``RFESelector``; ignored by others.

        Returns:
            The fitted ``FeatureSelectionPipeline`` instance (``self``).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        self._selectors: list[BaseSelector] = (
            self.selectors if self.selectors is not None else _default_selectors()
        )

        X_current = X
        audit_rows = []
        stage_counts = [len(X.columns)]

        for selector in self._selectors:
            selector.fit(X_current, y)
            for feat, reason in selector.removed_features_.items():
                original_pos = list(X.columns).index(feat) if feat in X.columns else -1
                audit_rows.append(
                    {
                        "feature": feat,
                        "removed_by": type(selector).__name__,
                        "reason": reason,
                        "original_position": original_pos,
                    }
                )
            X_current = selector.transform(X_current)
            stage_counts.append(len(X_current.columns))
            if self.verbose:
                logger.info(
                    "%s: %d → %d features",
                    type(selector).__name__,
                    stage_counts[-2],
                    stage_counts[-1],
                )

        self._stage_counts = stage_counts
        self._stage_names = ["input"] + [type(s).__name__ for s in self._selectors]
        self.audit_ = pd.DataFrame(audit_rows)
        self._selected = X_current.columns.tolist()

        logger.info(
            "FeatureSelectionPipeline: %d → %d features retained",
            len(X.columns),
            len(self._selected),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` restricted to the feature set selected during ``fit``.

        Columns that were selected during ``fit`` but are absent from ``X``
        are silently skipped, making it safe to call on held-out or OOT sets
        that may differ slightly from the training frame.

        Args:
            X: Feature DataFrame to filter.  Typically the validation, test,
                or OOT set.  Shape ``(n_samples, n_features)``.

        Returns:
            DataFrame containing only the columns that were selected during
            ``fit`` and are present in ``X``.  Returns an empty DataFrame
            (zero columns) if none of the selected features appear in ``X``.
        """
        cols = [c for c in self._selected if c in X.columns]
        return X[cols]

    def audit_report(self) -> pd.DataFrame:
        """Return a copy of the full removal audit accumulated across all selector stages.

        Args:
            None

        Returns:
            DataFrame with columns ``feature`` (str), ``removed_by`` (str —
            class name of the selector that removed it), ``reason`` (str), and
            ``original_position`` (int — 0-based column index in the original
            input DataFrame, or ``-1`` if the feature was not found).  Contains
            one row per removed feature.  Returns an empty DataFrame with those
            four columns if no features were removed.
        """
        return self.audit_.copy()

    def plot_removal_waterfall(self):
        """Build a horizontal bar chart showing the feature count remaining after each
        pipeline stage.

        The chart includes one bar per stage (starting from ``"input"``),
        making it easy to see which selector removed the most features.  The
        dscompanion Plotly theme is applied before returning.

        Args:
            None

        Returns:
            Plotly ``go.Figure`` with horizontal bars labelled by stage name
            and annotated with the feature count.

        Raises:
            ImportError: If the ``plotly`` package is not installed.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required. pip install plotly") from exc

        fig = go.Figure(
            go.Bar(
                x=self._stage_counts,
                y=self._stage_names,
                orientation="h",
                text=self._stage_counts,
                textposition="outside",
                marker_color="steelblue",
            )
        )
        fig.update_layout(
            title="Feature count after each selection stage",
            xaxis_title="Feature count",
            yaxis_title="Stage",
        )
        apply_dscompanion_theme(fig)
        return fig
