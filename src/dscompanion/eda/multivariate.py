"""Multivariate analysis: VIF, PCA, hierarchical clustering."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

from dscompanion.config import settings
from dscompanion.utils.plotting import apply_dscompanion_theme

__all__ = ["MultivariateAnalyser"]

# Canonical column schemas
_VIF_COLS: list[str] = ["feature", "vif", "flag"]
_PCA_COLS: list[str] = ["component", "explained_variance_ratio", "cumulative_variance"]


class MultivariateAnalyser:
    """Run multivariate diagnostics — VIF, PCA scree analysis, and hierarchical
    cluster heatmap — on numeric training features.

    Detects multicollinearity by computing VIF for every numeric column, fits
    a full PCA on standardised features to reveal principal-component
    structure, and reorders the correlation matrix via average-linkage
    hierarchical clustering for visual inspection.  Only numeric columns are
    used; rows with any null value are dropped before fitting and the number
    of dropped rows is logged.  No side-effects outside this instance; does
    not mutate the DataSplit.

    Args:
        max_rows (int): Maximum rows used for VIF, PCA, and correlation
            computation.  A value of ``0`` delegates to
            ``settings.max_eda_rows``.
    """

    def __init__(self, max_rows: int = 0) -> None:
        self._max_rows = max_rows or settings.max_eda_rows
        self._vif_df: pd.DataFrame | None = None
        self._pca: PCA | None = None
        self._pca_cols: list[str] | None = None
        self._corr_matrix: pd.DataFrame | None = None
        self._cat_df: pd.DataFrame | None = None
        self._fitted = False

    def fit(
        self, split: Any
    ) -> "MultivariateAnalyser":  # Any avoids circular import with dscompanion.split
        """Compute VIF, PCA, and the Pearson correlation matrix on numeric training features.

        Selects numeric columns from ``split.train_X``, drops rows with any
        null value (logging the count), subsamples to ``max_rows`` for
        performance, then computes the correlation matrix, VIF table, and
        fits PCA.  Logs an INFO message with shape details.

        Args:
            split (Any): A ``DataSplit`` instance that exposes a ``train_X``
                attribute containing a ``pandas.DataFrame`` of training
                features.

        Returns:
            MultivariateAnalyser: This instance, allowing method chaining.

        Raises:
            TypeError: If ``split`` has no ``train_X`` attribute or if
                ``train_X`` is not a ``pandas.DataFrame``.
            ValueError: If ``train_X`` has 0 numeric columns, or if the
                DataFrame is empty after dropping null rows.
        """
        if not hasattr(split, "train_X"):
            raise TypeError("split must expose a train_X attribute — expected a DataSplit instance")
        if not isinstance(split.train_X, pd.DataFrame):
            raise TypeError(
                f"split.train_X must be a pandas DataFrame, got {type(split.train_X).__name__}"
            )

        num = split.train_X.select_dtypes(include="number")
        if num.shape[1] == 0:
            raise ValueError(
                "split.train_X has no numeric columns — MultivariateAnalyser requires at least one"
            )

        n_before = len(num)
        df = num.dropna()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                "MultivariateAnalyser.fit — dropped %d rows (%.1f%%) containing NaN "
                "before computation",
                n_dropped,
                100 * n_dropped / max(n_before, 1),
            )
        if len(df) == 0:
            raise ValueError(
                "All rows contain at least one NaN — no data remains after dropna(); "
                "consider imputing missing values before fitting MultivariateAnalyser"
            )

        df = (
            df
            if len(df) <= self._max_rows
            else df.sample(n=self._max_rows, random_state=settings.random_state)
        )
        logger.info("MultivariateAnalyser.fit — %d rows, %d numeric columns", len(df), df.shape[1])

        self._pca_cols = df.columns.tolist()
        self._corr_matrix = df.corr()
        self._vif_df = self._compute_vif(df)
        self._pca = self._fit_pca(df)

        cat = split.train_X.select_dtypes(include=["object", "category"])
        if cat.shape[1] >= 2:
            cat = cat.dropna()
            if len(cat) > self._max_rows:
                cat = cat.sample(n=self._max_rows, random_state=settings.random_state)
            self._cat_df = cat
        else:
            self._cat_df = None

        self._fitted = True
        return self

    # ── VIF ──────────────────────────────────────────────────────────────────

    def vif_table(self) -> pd.DataFrame:
        """Return a copy of the Variance Inflation Factor table sorted by VIF descending.

        Features for which the computation raises an exception are assigned
        ``vif=NaN`` and ``flag=False``.  The returned object is a defensive
        copy; mutating it does not affect internal state.

        Returns:
            pandas.DataFrame: One row per numeric feature with columns —
            ``feature``, ``vif`` (float, rounded to
            ``settings.vif_round_precision`` decimal places, or ``NaN`` on
            error), ``flag`` (bool, ``True`` when ``vif`` exceeds
            ``settings.vif_threshold``).  Returns a DataFrame with
            ``vif=NaN`` and ``flag=False`` when fewer than two numeric columns
            were present.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._vif_df.copy()

    def correlation_matrix(self) -> pd.DataFrame:
        """Return a copy of the Pearson correlation matrix computed during fit.

        Returns:
            pd.DataFrame: Square correlation matrix with numeric feature names
            as both index and columns.  Values are Pearson ``r`` in ``[-1, 1]``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._corr_matrix.copy()

    def cramers_v_table(self) -> pd.DataFrame:
        """Return Cramér's V association strength between every categorical column pair.

        Computes the chi-squared-based Cramér's V statistic for every unique
        pair of categorical columns stored at fit time.  Pairs are sorted by
        Cramér's V descending.  Returns an empty DataFrame when fewer than two
        categorical columns were present.

        Returns:
            pd.DataFrame: Columns ``col_a`` (str), ``col_b`` (str),
            ``cramers_v`` (float in ``[0, 1]``, ``NaN`` on computation error).
            One row per pair, sorted by ``cramers_v`` descending.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        _COLS = ["col_a", "col_b", "cramers_v"]
        if self._cat_df is None or self._cat_df.shape[1] < 2:
            return pd.DataFrame(columns=_COLS)

        from scipy.stats.contingency import association

        cols = self._cat_df.columns.tolist()
        rows = []
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                try:
                    ct = pd.crosstab(self._cat_df[a], self._cat_df[b])
                    v = float(association(ct.values, method="cramer"))
                except (ValueError, ZeroDivisionError) as exc:
                    logger.warning("Cramer's V skipped for (%r, %r): %s", a, b, exc)
                    v = float("nan")
                rows.append(
                    {
                        "col_a": a,
                        "col_b": b,
                        "cramers_v": round(v, 4) if not np.isnan(v) else v,
                    }
                )
        return (
            pd.DataFrame(rows, columns=_COLS)
            .sort_values("cramers_v", ascending=False)
            .reset_index(drop=True)
        )

    def flag_multicollinearity(self) -> list[str]:
        """Return names of numeric features whose VIF exceeds the configured threshold.

        Reads ``settings.vif_threshold`` at call time.  Features with
        ``vif=NaN`` are excluded.

        Returns:
            list[str]: Column names flagged as multicollinear, in VIF
            descending order.  Returns an empty list when no feature exceeds
            the threshold.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._vif_df.loc[self._vif_df["flag"], "feature"].tolist()

    # ── PCA ──────────────────────────────────────────────────────────────────

    def pca_summary(self, n_components: int = 0) -> pd.DataFrame:
        """Return explained variance ratio and cumulative variance for leading PCA components.

        Args:
            n_components (int): Number of leading components to include.
                A value of ``0`` delegates to ``settings.pca_default_components``.
                Capped to the total number of fitted components.

        Returns:
            pandas.DataFrame: One row per component with columns —
            ``component``, ``explained_variance_ratio``, ``cumulative_variance``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
            ValueError: If ``n_components`` is negative.
        """
        self._check_fitted()
        n_components = n_components or settings.pca_default_components
        if n_components < 0:
            raise ValueError(f"n_components must be non-negative, got {n_components}")
        evr = self._pca.explained_variance_ratio_[:n_components]
        return pd.DataFrame(
            {
                "component": range(1, len(evr) + 1),
                "explained_variance_ratio": evr,
                "cumulative_variance": np.cumsum(evr),
            }
        )

    def pca_plot(self) -> go.Figure:
        """Produce a themed Plotly scree plot of per-component and cumulative explained variance.

        Renders up to ``settings.pca_plot_max_components`` components.

        Returns:
            plotly.graph_objects.Figure: Dual-trace figure — ``Bar`` for
            per-component variance, ``Scatter`` for cumulative variance.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        n = min(settings.pca_plot_max_components, len(self._pca_cols))
        summary = self.pca_summary(n_components=n)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=summary["component"],
                y=summary["explained_variance_ratio"],
                name="per component",
                marker_color="#4f86c6",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=summary["component"],
                y=summary["cumulative_variance"],
                name="cumulative",
                line={"color": "#e05c5c", "width": 2},
                mode="lines+markers",
            )
        )
        fig.update_layout(
            title="PCA — Explained Variance",
            xaxis_title="Component",
            yaxis_title="Variance ratio",
        )
        return apply_dscompanion_theme(fig)

    # ── Cluster heatmap ──────────────────────────────────────────────────────

    def cluster_heatmap(self) -> go.Figure:
        """Produce a themed Plotly heatmap of the correlation matrix reordered by
        hierarchical clustering.

        Distance is defined as ``1 - |r|``; average-linkage is used.  Null
        values in the correlation matrix are filled with zero before distance
        computation.

        Returns:
            plotly.graph_objects.Figure: Square heatmap with ``RdBu`` colour
            scale, axes labelled by feature names in dendrogram leaf order.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
            ValueError: If fewer than two numeric columns were present at fit
                time — hierarchical clustering requires at least two items.
        """
        self._check_fitted()
        if len(self._pca_cols) < 2:
            raise ValueError(
                "cluster_heatmap requires at least 2 numeric columns; "
                f"found {len(self._pca_cols)}"
            )
        corr = self._corr_matrix.fillna(0)
        dist = 1 - corr.abs()
        try:
            Z = linkage(dist.values, method="average")
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(
                "cluster_heatmap: hierarchical clustering failed — "
                "check for near-singular correlation matrix: %s" % exc
            ) from exc
        order = dendrogram(Z, no_plot=True)["leaves"]
        reordered = corr.iloc[order, order]

        fig = go.Figure(
            go.Heatmap(
                z=reordered.values,
                x=reordered.columns.tolist(),
                y=reordered.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "r"},
            )
        )
        fig.update_layout(title="Clustered Correlation Heatmap")
        return apply_dscompanion_theme(fig)

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_vif(df: pd.DataFrame) -> pd.DataFrame:
        from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore

        if df.shape[1] < 2:
            return pd.DataFrame(
                {"feature": df.columns.tolist(), "vif": np.nan, "flag": False},
                columns=_VIF_COLS,
            )
        rows = []
        arr = df.values
        for i, col in enumerate(df.columns):
            try:
                vif = variance_inflation_factor(arr, i)
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning("VIF skipped for %r: %s", col, e)
                vif = np.nan
            rows.append(
                {
                    "feature": col,
                    "vif": (
                        round(vif, settings.vif_round_precision) if not np.isnan(vif) else np.nan
                    ),
                    "flag": vif > settings.vif_threshold if not np.isnan(vif) else False,
                }
            )
        return (
            pd.DataFrame(rows, columns=_VIF_COLS)
            .sort_values("vif", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def _fit_pca(df: pd.DataFrame) -> PCA:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.fillna(df.median()))
        n = min(df.shape[1], df.shape[0])
        pca = PCA(n_components=n, random_state=settings.random_state)
        pca.fit(scaled)
        return pca

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing results.")
