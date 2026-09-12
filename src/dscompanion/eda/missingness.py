"""Missing-value pattern analysis: nullity matrix and nullity correlation."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from dscompanion.config import settings
from dscompanion.utils.plotting import apply_dscompanion_theme, render_corr_heatmap

__all__ = ["MissingnessAnalyser"]


class MissingnessAnalyser:
    """Analyse per-row, per-column missing-value patterns and inter-column missingness correlation.

    This is column-pairwise like ``BivariateAnalyser``, but it measures
    correlation between columns' *missingness indicators* rather than
    target-vs-feature relationships, so it does not belong inside
    ``BivariateAnalyser`` (whose scope is target-vs-feature analysis) —
    kept as its own analyser, mirroring the existing univariate/bivariate/
    multivariate separation of concerns. Row-dense operations (the nullity
    matrix itself) are subsampled before computation to bound memory, the
    same rule already applied in ``UnivariateAnalyser``/``BivariateAnalyser``/
    ``MultivariateAnalyser``. No side-effects outside this instance; does
    not mutate the DataSplit.

    Args:
        max_rows (int): Maximum rows used for the nullity matrix and nullity
            correlation. A value of ``0`` delegates to
            ``settings.eda_missing_matrix_max_rows``.
    """

    def __init__(self, max_rows: int = 0) -> None:
        self._max_rows = max_rows or settings.eda_missing_matrix_max_rows
        self._null_df: pd.DataFrame | None = None
        self._corr_matrix: pd.DataFrame | None = None
        self._fitted = False

    def fit(
        self, split: Any
    ) -> "MissingnessAnalyser":  # Any avoids circular dscompanion.split import
        """Compute the nullity matrix and nullity correlation on a DataSplit's training portion.

        Subsamples ``split.train_X`` to ``max_rows`` rows, converts every
        column to its missingness indicator (``True`` where null), and
        computes the Pearson correlation between indicator columns — which,
        for 0/1-valued series, is exactly the phi coefficient. Columns that
        are either never-null or always-null (zero variance) are excluded
        from the correlation matrix since their correlation against any
        other column is undefined.

        Args:
            split (Any): A ``DataSplit`` instance that exposes a ``train_X``
                attribute containing a ``pandas.DataFrame`` of training
                features.

        Returns:
            MissingnessAnalyser: This instance, allowing method chaining.

        Raises:
            TypeError: If ``split`` has no ``train_X`` attribute or if
                ``train_X`` is not a ``pandas.DataFrame``.
            ValueError: If ``split.train_X`` contains zero rows.
        """
        if not hasattr(split, "train_X"):
            raise TypeError("split must expose a train_X attribute — expected a DataSplit instance")
        if not isinstance(split.train_X, pd.DataFrame):
            raise TypeError(
                f"split.train_X must be a pandas DataFrame, got {type(split.train_X).__name__}"
            )
        df = split.train_X
        if len(df) == 0:
            raise ValueError("split.train_X has 0 rows — cannot compute missingness statistics")

        df_sample = (
            df
            if len(df) <= self._max_rows
            else df.sample(n=self._max_rows, random_state=settings.random_state)
        )

        self._null_df = df_sample.isna()
        varying_cols = [c for c in self._null_df.columns if self._null_df[c].nunique() > 1]
        self._corr_matrix = self._null_df[varying_cols].astype(int).corr(method="pearson")
        self._fitted = True
        logger.info(
            "MissingnessAnalyser fitted — %d rows, %d columns, %d with partial missingness",
            len(df_sample),
            len(df_sample.columns),
            len(varying_cols),
        )
        return self

    def nullity_matrix(self) -> pd.DataFrame:
        """Return a copy of the boolean per-row, per-column missingness matrix.

        Returns:
            pandas.DataFrame: Same shape and column order as the
            (row-subsampled) training data, with ``True`` where the original
            value was null and ``False`` otherwise.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._null_df.copy()

    def nullity_correlation(self) -> pd.DataFrame:
        """Return a copy of the phi-coefficient correlation matrix between missingness indicators.

        Returns:
            pandas.DataFrame: Square matrix indexed and columned by feature
            name, restricted to columns with partial missingness (excludes
            always-null and never-null columns, whose indicator has zero
            variance). Returns an empty DataFrame when fewer than 2 columns
            qualify.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._corr_matrix.copy()

    def nullity_matrix_chart(self) -> go.Figure:
        """Render the boolean nullity matrix as a themed Plotly heatmap.

        Returns:
            plotly.graph_objects.Figure: Heatmap with rows as observations,
            columns as features, and cell colour indicating missing (1) vs
            present (0). Row order follows the (subsampled) training data;
            row labels are omitted (only the missingness pattern matters).

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        z = self._null_df.astype(int).values
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=self._null_df.columns.tolist(),
                colorscale=[[0, "#f8f9fb"], [1, "#c0392b"]],
                showscale=False,
            )
        )
        fig.update_layout(
            title="Missing Values Matrix", xaxis_title="feature", yaxis_title="row (subsampled)"
        )
        return apply_dscompanion_theme(fig)

    def nullity_correlation_heatmap(self) -> go.Figure:
        """Render the nullity correlation matrix as a themed Plotly heatmap.

        Returns:
            plotly.graph_objects.Figure: Annotated heatmap, empty (no traces)
            when fewer than 2 columns have partial missingness.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        if self._corr_matrix.empty or len(self._corr_matrix.columns) < 2:
            return apply_dscompanion_theme(go.Figure())
        return render_corr_heatmap(self._corr_matrix, title="Nullity Correlation (phi coefficient)")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing results.")
