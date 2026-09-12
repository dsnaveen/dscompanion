"""Bivariate analysis: IV/WoE, correlation, Cramér's V."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)

from dscompanion.config import settings
from dscompanion.utils.metrics import iv_score, woe_bins
from dscompanion.utils.plotting import apply_dscompanion_theme, render_corr_heatmap

__all__ = ["BivariateAnalyser"]

# Canonical column schemas — guarantee consistent empty-DataFrame structure
_IV_COLS: list[str] = ["feature", "iv", "n_bins", "predictive_power"]
_CRAMERS_COLS: list[str] = ["feature_a", "feature_b", "cramers_v"]
_CORR_FLAG_COLS: list[str] = ["feature_a", "feature_b", "pearson_r"]

_IV_LABELS = {
    (0.0, 0.02): "Useless",
    (0.02, 0.1): "Weak",
    (0.1, 0.3): "Medium",
    (0.3, 0.5): "Strong",
    (0.5, float("inf")): "Very Strong",
}


def _iv_label(iv: float) -> str:
    for (lo, hi), label in _IV_LABELS.items():
        if lo <= iv < hi:
            return label
    return "Unknown"


class BivariateAnalyser:
    """Measure pairwise relationships between features and the binary target via
    IV/WoE, Pearson correlation, and Cramér's V.

    For numeric features the analyser computes Information Value (IV) and
    Weight of Evidence (WoE) bin tables using quantile-based binning — a
    binary-target concept, skipped (empty IV table) when ``train_y`` isn't
    binary, e.g. a continuous regression target.  For categorical features
    it computes pairwise Cramér's V via chi-squared contingency tables.
    Pearson correlation is computed across all numeric feature pairs.
    Pairwise operations (correlation, Cramér's V) are target-independent
    and always run, on a row-subsampled copy of the training data to
    prevent OOM on large datasets.  IV, when computed, uses the full
    training set for accuracy.  Results are stored internally after
    ``fit`` is called and exposed through dedicated accessor methods.  No
    side-effects outside this instance.

    Args:
        n_bins (int): Number of quantile bins for WoE and IV computation.
            A value of ``0`` delegates to ``settings.default_iv_bins``.
        max_rows (int): Maximum rows used for pairwise operations (correlation,
            Cramér's V). A value of ``0`` delegates to ``settings.max_eda_rows``.
        interaction_max_numeric_cols (int): Cap on numeric columns considered
            in ``interaction_plots()`` (bounds the O(n²) pair count). A value
            of ``0`` delegates to ``settings.eda_interaction_max_numeric_cols``.
        interaction_hexbin_row_threshold (int): Row count above which
            ``interaction_plots()`` switches from scatter to hexbin-style 2D
            density plots. A value of ``0`` delegates to
            ``settings.eda_interaction_hexbin_row_threshold``.
    """

    def __init__(
        self,
        n_bins: int = 0,
        max_rows: int = 0,
        interaction_max_numeric_cols: int = 0,
        interaction_hexbin_row_threshold: int = 0,
    ) -> None:
        self._n_bins = n_bins or settings.default_iv_bins
        self._max_rows = max_rows or settings.max_eda_rows
        self._interaction_max_numeric_cols = (
            interaction_max_numeric_cols or settings.eda_interaction_max_numeric_cols
        )
        self._interaction_hexbin_row_threshold = (
            interaction_hexbin_row_threshold or settings.eda_interaction_hexbin_row_threshold
        )
        self._iv_df: pd.DataFrame | None = None
        self._corr_matrix: pd.DataFrame | None = None
        self._cramers_df: pd.DataFrame | None = None
        self._train_X: pd.DataFrame | None = None
        self._train_y: pd.Series | None = None
        self._sample_X: pd.DataFrame | None = None
        self._fitted = False

    def fit(
        self, split: Any
    ) -> "BivariateAnalyser":  # Any avoids circular import with dscompanion.split
        """Compute IV/WoE, Pearson correlation, and Cramér's V on the training portion
        of a DataSplit.

        Validates inputs, subsamples to ``max_rows`` for pairwise operations,
        then runs IV computation (full data), Pearson correlation, and
        Cramér's V.  Sets the internal ``_fitted`` flag so that accessor
        methods become available. IV/WoE is a binary-target concept — when
        ``train_y`` doesn't have exactly 2 unique non-null values (e.g. a
        continuous regression target), IV computation is skipped
        (``iv_table()`` returns an empty DataFrame) rather than raising;
        correlation and Cramér's V are target-independent and always run.

        Args:
            split (Any): A ``DataSplit`` instance that exposes ``train_X``
                (``pandas.DataFrame`` of training features) and ``train_y``
                (``pandas.Series`` — binary 0/1 labels for IV/WoE to be
                computed, any dtype otherwise).

        Returns:
            BivariateAnalyser: This instance, allowing method chaining.

        Raises:
            TypeError: If ``split`` lacks ``train_X`` / ``train_y`` attributes,
                or if either is not the expected pandas type.
            ValueError: If ``train_X`` has 0 rows.
        """
        if not hasattr(split, "train_X") or not hasattr(split, "train_y"):
            raise TypeError(
                "split must expose train_X and train_y attributes — expected a DataSplit instance"
            )
        if not isinstance(split.train_X, pd.DataFrame):
            raise TypeError(
                f"split.train_X must be a pandas DataFrame, got {type(split.train_X).__name__}"
            )
        if not isinstance(split.train_y, pd.Series):
            raise TypeError(
                f"split.train_y must be a pandas Series, got {type(split.train_y).__name__}"
            )
        df = split.train_X
        y = split.train_y
        if len(df) == 0:
            raise ValueError("split.train_X has 0 rows — cannot compute bivariate statistics")
        # IV/WoE (_compute_iv) is a binary-target concept — skip it (empty
        # table, not a crash) for a regression target instead of validating
        # binary-ness unconditionally. Correlation and Cramér's V are
        # target-independent (feature-vs-feature) and still run either way.
        is_binary_target = y.dropna().nunique() == 2

        logger.info("BivariateAnalyser.fit — %d rows, %d features", len(df), len(df.columns))

        # subsample for pairwise operations only — .corr() and Cramér's V are O(n²) memory
        df_sample = (
            df
            if len(df) <= self._max_rows
            else df.sample(n=self._max_rows, random_state=settings.random_state)
        )

        self._train_X = df
        self._train_y = y
        self._sample_X = df_sample
        self._iv_df = self._compute_iv() if is_binary_target else pd.DataFrame(columns=_IV_COLS)
        self._corr_matrix = df_sample.select_dtypes(include="number").corr(method="pearson")
        self._cramers_df = self._compute_cramers_v(df_sample)
        self._fitted = True
        return self

    # ── IV / WoE ─────────────────────────────────────────────────────────────

    def iv_table(self) -> pd.DataFrame:
        """Return a copy of the Information Value table sorted by IV descending.

        Features for which IV computation raised an exception are included
        with ``iv=0.0``, ``n_bins=0``, and ``predictive_power="Error"``
        so that every input column is represented.  The returned object is a
        defensive copy; mutating it does not affect internal state.

        Returns:
            pandas.DataFrame: One row per feature with columns —
            ``feature``, ``iv``, ``n_bins``, ``predictive_power``
            (one of ``"Useless"``, ``"Weak"``, ``"Medium"``, ``"Strong"``,
            ``"Very Strong"``, or ``"Error"``).  Sorted by ``iv`` descending.
            Returns an empty DataFrame with those columns when the training
            set has no columns.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._iv_df.copy()

    def target_rate_by_bin(self, feature: str, n_bins: int = 0) -> pd.DataFrame:
        """Return the observed target rate for each WoE bin of a single feature.

        Args:
            feature (str): Name of a column in the training DataFrame.
            n_bins (int): Quantile bins for numeric features. A value of
                ``0`` delegates to ``settings.default_iv_bins``.

        Returns:
            pandas.DataFrame: One row per bin with columns — ``bin_label``,
            ``count``, ``target_rate``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
            KeyError: If ``feature`` is not a column in the training DataFrame.
        """
        self._check_fitted()
        if not isinstance(feature, str):
            raise TypeError("feature must be a str, got %s" % type(feature).__name__)
        if feature not in self._train_X.columns:
            raise KeyError("feature %r not found in training DataFrame columns" % feature)
        n_bins = n_bins or settings.default_iv_bins
        bins = woe_bins(self._train_X[feature], self._train_y, n_bins)
        bins = bins.rename(columns={"bin": "bin_label"})
        return bins[["bin_label", "count", "event_rate"]].rename(
            columns={"event_rate": "target_rate"}
        )

    def target_rate_by_bin_chart(self, feature: str, n_bins: int = 0) -> go.Figure:
        """Render a bar chart of the observed target rate across a single feature's WoE bins.

        Args:
            feature (str): Name of a column in the training DataFrame.
            n_bins (int): Quantile bins for numeric features. A value of
                ``0`` delegates to ``settings.default_iv_bins``.

        Returns:
            plotly.graph_objects.Figure: Themed bar chart with one bar per
            bin, target rate on the y-axis and each bar annotated with its
            row count — lets a viewer judge both the trend and how much
            data backs each bin.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
            KeyError: If ``feature`` is not a column in the training DataFrame.
        """
        bins = self.target_rate_by_bin(feature, n_bins)
        fig = go.Figure(
            go.Bar(
                # str(): numeric features' bin_label is a pd.Interval (from
                # pd.qcut) — not JSON-serializable as a Plotly axis value.
                x=bins["bin_label"].astype(str),
                y=bins["target_rate"],
                text=bins["count"],
                texttemplate="n=%{text}",
                textposition="outside",
                marker_color="#4f86c6",
            )
        )
        fig.update_layout(
            title=f"{feature} — target rate by bin",
            xaxis_title="bin",
            yaxis_title="target rate",
        )
        return apply_dscompanion_theme(fig)

    # ── Correlations ─────────────────────────────────────────────────────────

    def correlation_matrix(self) -> pd.DataFrame:
        """Return a copy of the raw Pearson correlation matrix for all numeric columns.

        No recomputation — exposes the matrix already computed at fit time
        and reused by ``correlation_heatmap()``/``flag_high_correlation()``.

        Args:
            None

        Returns:
            pd.DataFrame: Square correlation matrix, numeric columns only,
            index and columns both feature names. Empty DataFrame if fewer
            than 2 numeric columns were present at fit time.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._corr_matrix.copy()

    def correlation_heatmap(self) -> go.Figure:
        """Produce a themed Plotly heatmap of the Pearson correlation matrix for all
        numeric columns.

        Returns:
            plotly.graph_objects.Figure: Annotated heatmap with ``RdBu``
            colour scale centred at zero.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return render_corr_heatmap(self._corr_matrix, title="Pearson Correlation Matrix")

    def interaction_columns(self) -> list[str]:
        """Return the numeric columns considered for pairwise interaction plots.

        Numeric columns are capped at ``interaction_max_numeric_cols``; when
        more numeric columns exist, the top-N by IV (descending, via the
        already-computed ``iv_table()``) are kept rather than an arbitrary
        first-N slice, so the most predictive features are prioritised.
        Shared by ``interaction_plots()`` and any other consumer (e.g. an
        Excel exporter) that needs the same column selection without
        duplicating it.

        Args:
            None

        Returns:
            list[str]: Numeric column names, length
            ``min(n_numeric_cols, interaction_max_numeric_cols)``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        numeric_cols = self._sample_X.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) > self._interaction_max_numeric_cols:
            ranked = (
                self._iv_df[self._iv_df["feature"].isin(numeric_cols)]
                .sort_values("iv", ascending=False)["feature"]
                .tolist()
            )
            numeric_cols = ranked[: self._interaction_max_numeric_cols]
        return numeric_cols

    def interaction_plots(self) -> list[go.Figure]:
        """Produce one pairwise plot per numeric-column combination on row-subsampled training data.

        Below ``interaction_hexbin_row_threshold`` rows, pairs are rendered
        as scatter plots; above it, as 2D histogram (hexbin-style) density
        plots, since a scatter of that many points is unreadable and slow
        to render in a static HTML report.

        Args:
            None

        Returns:
            list[plotly.graph_objects.Figure]: One themed figure per
            ``C(k, 2)`` numeric column pair from ``interaction_columns()``.
            Returns an empty list when fewer than 2 numeric columns are
            present.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        numeric_cols = self.interaction_columns()
        if len(numeric_cols) < 2:
            return []

        use_hexbin = len(self._sample_X) > self._interaction_hexbin_row_threshold
        figures = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1 :]:
                x = self._sample_X[c1]
                y = self._sample_X[c2]
                if use_hexbin:
                    fig = go.Figure(go.Histogram2d(x=x, y=y, colorscale="Blues"))
                else:
                    fig = go.Figure(
                        go.Scatter(x=x, y=y, mode="markers", marker={"size": 4, "opacity": 0.5})
                    )
                fig.update_layout(title=f"{c1} vs {c2}", xaxis_title=c1, yaxis_title=c2)
                figures.append(apply_dscompanion_theme(fig))
        return figures

    def cramers_v_table(self) -> pd.DataFrame:
        """Return a copy of the pairwise Cramér's V association table for all
        categorical column pairs.

        Returns:
            pandas.DataFrame: One row per pair with columns —
            ``feature_a``, ``feature_b``, ``cramers_v``.  Returns an empty
            DataFrame with those columns when fewer than two categorical
            columns are present or all pairs fail.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._cramers_df.copy()

    def flag_high_correlation(self) -> pd.DataFrame:
        """Return numeric column pairs whose absolute Pearson correlation exceeds the
        configured threshold.

        The threshold is read from ``settings.correlation_threshold``.  Only
        the upper triangle of the correlation matrix is inspected to avoid
        duplicate pairs.

        Returns:
            pandas.DataFrame: One row per flagged pair with columns —
            ``feature_a``, ``feature_b``, ``pearson_r``.  Returns an empty
            DataFrame with those columns when no pair exceeds the threshold.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        corr = self._corr_matrix
        rows = []
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                r = corr.loc[c1, c2]
                if abs(r) > settings.correlation_threshold:
                    rows.append(
                        {
                            "feature_a": c1,
                            "feature_b": c2,
                            "pearson_r": round(r, settings.corr_round_precision),
                        }
                    )
        return (
            pd.DataFrame(rows, columns=_CORR_FLAG_COLS)
            if rows
            else pd.DataFrame(columns=_CORR_FLAG_COLS)
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _compute_iv(self) -> pd.DataFrame:
        if not self._train_X.columns.tolist():
            return pd.DataFrame(columns=_IV_COLS)
        rows = []
        for col in self._train_X.columns:
            try:
                iv = iv_score(self._train_X[col], self._train_y, self._n_bins)
                bins_df = woe_bins(self._train_X[col], self._train_y, self._n_bins)
                rows.append(
                    {
                        "feature": col,
                        "iv": round(iv, settings.iv_round_precision),
                        "n_bins": len(bins_df),
                        "predictive_power": _iv_label(iv),
                    }
                )
            except (ValueError, ArithmeticError) as e:
                logger.warning("IV skipped for %r: %s", col, e)
                rows.append({"feature": col, "iv": 0.0, "n_bins": 0, "predictive_power": "Error"})
        return (
            pd.DataFrame(rows, columns=_IV_COLS)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def _compute_cramers_v(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if len(cat_cols) < 2:
            return pd.DataFrame(columns=_CRAMERS_COLS)
        rows = []
        for i, c1 in enumerate(cat_cols):
            for c2 in cat_cols[i + 1 :]:
                try:
                    ct = pd.crosstab(df[c1], df[c2])
                    chi2, _, _, _ = chi2_contingency(ct)
                    n = ct.sum().sum()
                    r, k = ct.shape
                    v = np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0
                    rows.append(
                        {
                            "feature_a": c1,
                            "feature_b": c2,
                            "cramers_v": round(v, settings.corr_round_precision),
                        }
                    )
                except (ValueError, ArithmeticError) as e:
                    logger.warning("Cramers V skipped for (%r, %r): %s", c1, c2, e)
        return (
            pd.DataFrame(rows, columns=_CRAMERS_COLS)
            if rows
            else pd.DataFrame(columns=_CRAMERS_COLS)
        )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing results.")
