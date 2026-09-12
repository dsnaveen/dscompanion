"""Univariate analysis of a DataSplit."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

logger = logging.getLogger(__name__)

from dscompanion.config import settings
from dscompanion.utils.plotting import apply_dscompanion_theme

__all__ = ["UnivariateAnalyser"]

# Canonical column schemas — used to guarantee consistent empty-DataFrame structure
_NUM_COLS: list[str] = [
    "feature",
    "count",
    "missing_pct",
    "zero_pct",
    "mean",
    "std",
    "cv",
    "min",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p96",
    "p97",
    "p98",
    "p99",
    "max",
    "skewness",
    "kurtosis",
    "outlier_pct_iqr",
    "outlier_pct_zscore",
    "constant_flag",
    "near_zero_variance_flag",
]
_CAT_COLS: list[str] = [
    "feature",
    "count",
    "missing_pct",
    "n_unique",
    "top_value",
    "top_freq",
    "cardinality_flag",
    "imbalance",
]
_FLAG_COLS: list[str] = [
    "feature",
    "high_missing",
    "near_zero_variance",
    "high_cardinality",
    "constant",
]
_DATETIME_COLS: list[str] = ["feature", "count", "missing_pct", "n_unique", "min", "max"]
_BOOL_COLS: list[str] = ["feature", "count", "missing_pct", "n_true", "n_false", "pct_true"]


def _imbalance_score(counts: np.ndarray) -> float | None:
    """Compute an entropy-based imbalance score in [0, 1] from value-frequency counts.

    A score of ``0`` means values are uniformly distributed; ``1`` means a
    single value accounts for every observation. Uses the exact value-count
    distribution (not a top-N approximation), so the score is precise rather
    than a conservative estimate.

    Args:
        counts (np.ndarray): Non-null frequency count per unique value, in
            any order (e.g. ``series.value_counts().values``).

    Returns:
        float | None: Imbalance score rounded to 4 decimal places, or
        ``None`` when ``counts`` is empty or sums to zero.
    """
    total = counts.sum()
    if len(counts) == 0 or total <= 0:
        return None
    if len(counts) == 1:
        return 1.0
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    return float(round(1.0 - entropy / max_entropy, 4)) if max_entropy > 0 else None


class UnivariateAnalyser:
    """Compute per-column summary statistics and flag data quality issues for a DataSplit.

    Iterates over every column in the training feature matrix, separating
    numeric columns from categorical ones, and computes descriptive statistics,
    outlier rates, missing-value rates, and cardinality metrics.  Results are
    stored internally and exposed through dedicated accessor methods.  No side-
    effects outside this instance; does not mutate the DataSplit.

    Args:
        max_rows (int): Maximum number of rows sampled from the training
            DataFrame when computing statistics and building plots.  A value
            of ``0`` delegates to ``settings.max_eda_rows``.  Sampling uses
            ``settings.random_state`` for reproducibility.
        high_missing_threshold (float): Missing-value rate above which a
            column is flagged ``high_missing`` by ``flag_issues()``.  A
            value of ``0`` delegates to ``settings.high_missing_threshold``.
        near_zero_variance_threshold (float): Variance below which a numeric
            column is flagged ``near_zero_variance_flag`` in
            ``numeric_summary()``.  A value of ``0`` delegates to
            ``settings.near_zero_variance_threshold``.

    Example::

        analyser = UnivariateAnalyser()
        analyser.fit(split)
        print(analyser.numeric_summary())
    """

    def __init__(
        self,
        max_rows: int = 0,
        high_missing_threshold: float = 0.0,
        near_zero_variance_threshold: float = 0.0,
    ) -> None:
        self._max_rows = max_rows or settings.max_eda_rows
        self._high_missing_threshold = high_missing_threshold or settings.high_missing_threshold
        self._near_zero_variance_threshold = (
            near_zero_variance_threshold or settings.near_zero_variance_threshold
        )
        self._num_summary: pd.DataFrame | None = None
        self._cat_summary: pd.DataFrame | None = None
        self._datetime_summary: pd.DataFrame | None = None
        self._bool_summary: pd.DataFrame | None = None
        self._bool_cols: list[str] = []
        self._datetime_cols: list[str] = []
        self._train_X: pd.DataFrame | None = None
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self, split: Any
    ) -> "UnivariateAnalyser":  # Any avoids circular import with dscompanion.split
        """Compute summary statistics on the training portion of a DataSplit and store
        results internally.

        Selects the training feature matrix from ``split.train_X``, optionally
        down-samples it to ``max_rows`` rows, then runs numeric and categorical
        summary computations.  Sets the internal ``_fitted`` flag so that
        accessor methods become available.  Logs a summary line at INFO level.

        Args:
            split (Any): A ``DataSplit`` instance that exposes a ``train_X``
                attribute containing a ``pandas.DataFrame`` of training
                features.

        Returns:
            UnivariateAnalyser: This instance, allowing method chaining
            (e.g. ``UnivariateAnalyser().fit(split).numeric_summary()``).

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
            raise ValueError(
                "split.train_X has 0 rows — cannot compute statistics on an empty DataFrame"
            )
        if len(df) == 1:
            logger.warning(
                "split.train_X has only 1 row — std will be NaN for all columns; "
                "statistics are unreliable"
            )

        if len(df) > self._max_rows:
            df = df.sample(n=self._max_rows, random_state=settings.random_state)

        self._train_X = df
        bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in bool_cols]
        cat_cols = [
            c
            for c in df.select_dtypes(exclude="number").columns
            if c not in bool_cols and c not in datetime_cols
        ]

        self._bool_cols = bool_cols
        self._datetime_cols = datetime_cols
        self._num_summary = self._compute_numeric(df, num_cols)
        self._cat_summary = self._compute_categorical(df, cat_cols)
        self._datetime_summary = self._compute_datetime(df, datetime_cols)
        self._bool_summary = self._compute_boolean(df, bool_cols)
        self._fitted = True
        logger.info(
            "UnivariateAnalyser fitted — %d numeric, %d categorical, %d datetime, %d boolean",
            len(num_cols),
            len(cat_cols),
            len(datetime_cols),
            len(bool_cols),
        )
        return self

    # ── Outputs ──────────────────────────────────────────────────────────────

    def numeric_summary(self) -> pd.DataFrame:
        """Return a copy of the descriptive-statistics table for all numeric columns.

        Each row in the returned DataFrame corresponds to one numeric feature
        from the training set.  The returned object is a defensive copy; mutating
        it does not affect internal state.

        Returns:
            pandas.DataFrame: One row per numeric feature with columns —
            ``feature``, ``count``, ``missing_pct``, ``zero_pct``, ``mean``,
            ``std``, ``cv`` (std/mean; ``NaN`` when mean is zero or all values
            are null), ``min``, ``p1``, ``p2``, ``p3``, ``p4``, ``p5``, ``p10``,
            ``p25``, ``p50``, ``p75``, ``p90``, ``p95``, ``p96``, ``p97``,
            ``p98``, ``p99``, ``max``, ``skewness``, ``kurtosis``,
            ``outlier_pct_iqr``, ``outlier_pct_zscore``,
            ``constant_flag`` (bool, std == 0 or all values null),
            ``near_zero_variance_flag`` (bool, 0 < std < threshold — mutually
            exclusive with ``constant_flag``).  Returns an empty DataFrame
            with those columns when no numeric columns are present.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._num_summary.copy()

    def categorical_summary(self) -> pd.DataFrame:
        """Return a copy of the summary statistics table for all categorical (non-numeric) columns.

        Each row corresponds to one categorical feature from the training set.
        The returned object is a defensive copy; mutating it does not affect
        internal state.

        Returns:
            pandas.DataFrame: One row per categorical feature with columns —
            ``feature``, ``count``, ``missing_pct``, ``n_unique``,
            ``top_value`` (most frequent value or ``None`` when entirely null),
            ``top_freq`` (relative frequency of most common value, 0–1),
            ``cardinality_flag`` (bool, ``True`` when ``n_unique`` exceeds
            ``settings.high_cardinality_threshold``), ``imbalance``
            (entropy-based score in [0, 1]; 0=uniform, 1=dominated by a
            single value; ``None`` when entirely null).  Returns an empty
            DataFrame with those columns when no categorical columns are present.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._cat_summary.copy()

    def datetime_summary(self) -> pd.DataFrame:
        """Return a copy of the summary statistics table for all ``datetime64``-typed columns.

        Scope is explicitly limited to columns already typed ``datetime64``
        at fit time — detecting date-like strings inside object/string
        columns is out of scope (a much larger heuristic-parsing problem).

        Returns:
            pandas.DataFrame: One row per datetime feature with columns —
            ``feature``, ``count``, ``missing_pct``, ``n_unique``, ``min``,
            ``max``. Returns an empty DataFrame with those columns when no
            datetime columns are present.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._datetime_summary.copy()

    def boolean_summary(self) -> pd.DataFrame:
        """Return a copy of the summary statistics table for all ``bool``-typed columns.

        Returns:
            pandas.DataFrame: One row per boolean feature with columns —
            ``feature``, ``count``, ``missing_pct``, ``n_true``, ``n_false``,
            ``pct_true`` (fraction of non-null values that are ``True``).
            Returns an empty DataFrame with those columns when no boolean
            columns are present.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return self._bool_summary.copy()

    def threshold_guide(self) -> pd.DataFrame:
        """Return percentile distributions of null rate, variance, and CV across all columns.

        A meta-distribution over features (not over rows) — e.g. "what's the
        p90 null rate across every column in this dataset" — intended to
        help calibrate selector thresholds (such as
        ``NullRateSelector.threshold``) against the dataset actually in
        front of you, rather than guessing a global default. ``null_rate``
        is computed across every column regardless of dtype; ``variance``
        and ``cv`` are numeric-only and reuse the already-computed ``std``/
        ``cv`` columns from ``numeric_summary()`` rather than recomputing.

        Args:
            None

        Returns:
            pandas.DataFrame: One row per metric (``"null_rate"``,
            ``"variance"``, ``"cv"``) with columns ``metric``, ``p10``,
            ``p25``, ``p50``, ``p75``, ``p90``, ``p95``, ``p99``. A metric
            row is all ``NaN`` (besides ``metric``) when its source series
            is empty (e.g. ``variance``/``cv`` when there are no numeric
            columns).

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        pct_labels = ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]

        def _percentile_row(metric: str, series: pd.Series) -> dict[str, Any]:
            if series.empty:
                return {"metric": metric, **dict.fromkeys(pct_labels, np.nan)}
            values = series.quantile(percentiles).tolist()
            return {"metric": metric, **dict(zip(pct_labels, values))}

        null_rate = self._train_X.isna().mean()
        rows = [
            _percentile_row("null_rate", null_rate),
            _percentile_row("variance", self._num_summary["std"] ** 2),
            _percentile_row("cv", self._num_summary["cv"]),
        ]
        return pd.DataFrame(rows, columns=["metric", *pct_labels])

    def plot_distributions(self, top_n: int = 20) -> list[go.Figure]:
        """Generate one Plotly distribution figure per feature for the first ``top_n`` columns.

        Numeric columns receive a histogram (``settings.eda_histogram_bins``
        bins); categorical columns receive a bar chart of the top
        ``settings.eda_top_categorical_values`` most frequent values.  The
        dscompanion visual theme is applied to every figure via
        ``apply_dscompanion_theme``.  Column order follows ``train_X.columns``.

        Args:
            top_n (int): Maximum number of features to include.  Columns
                beyond this index are silently ignored.  Defaults to ``20``.

        Returns:
            list[plotly.graph_objects.Figure]: A list of themed Plotly figures,
            one per feature, in column order.  Returns an empty list when
            ``top_n`` is zero or negative (with a warning logged).

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        if top_n <= 0:
            logger.warning("plot_distributions called with top_n=%d — returning empty list", top_n)
            return []

        df = self._train_X
        figures = []
        for col in df.columns[:top_n]:
            if col in self._bool_cols:
                fig = self._boolean_plot(df[col], col)
            elif col in self._datetime_cols:
                fig = self._datetime_plot(df[col], col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                fig = self._numeric_plot(df[col], col)
            else:
                fig = self._categorical_plot(df[col], col)
            figures.append(apply_dscompanion_theme(fig))
        return figures

    def numeric_distribution_with_kde(self, col: str, exclude_outliers: bool = False) -> go.Figure:
        """Render one numeric column's histogram with a fitted KDE curve overlaid.

        Unlike ``plot_distributions()``'s plain count histogram (used by
        ``ModelCard`` reports and not changed here to avoid breaking that
        contract), this renders the histogram on a probability-density
        scale so a kernel density estimate curve sits on the same axis —
        intended for interactive, single-column inspection rather than a
        fixed report.

        Args:
            col (str): Numeric column name, as it appears in ``train_X``.
            exclude_outliers (bool): When ``True``, trims the series to the
                ``[settings.eda_chart_clip_lower_pct,
                1 - settings.eda_chart_clip_upper_pct]`` quantile range
                before building the histogram and KDE — chart rendering
                only, the underlying modelling data is never touched.
                Intended for highly skewed columns where a few extreme
                values otherwise dominate the x-axis and hide the bulk
                shape. Defaults to ``False`` (today's untrimmed behavior).

        Returns:
            plotly.graph_objects.Figure: A themed figure with a density
            histogram trace and (when the column has at least 2 unique
            non-null values) a KDE line trace on top. When
            ``exclude_outliers`` is set and rows were actually trimmed,
            the chart title reports how many. Returns an empty, themed
            ``go.Figure`` (no traces) when ``col`` is absent from
            ``train_X`` or is not numeric — callers should check
            ``fig.data`` before rendering, same convention as
            ``MissingnessAnalyser.missing_correlation_heatmap()``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        if col not in self._train_X.columns:
            return apply_dscompanion_theme(go.Figure())
        if not pd.api.types.is_numeric_dtype(self._train_X[col]):
            return apply_dscompanion_theme(go.Figure())

        clean = self._train_X[col].dropna()
        title = f"{col} — distribution with KDE"
        if exclude_outliers and len(clean) > 0:
            lower_pct = settings.eda_chart_clip_lower_pct
            upper_pct = settings.eda_chart_clip_upper_pct
            lower_bound = clean.quantile(lower_pct)
            upper_bound = clean.quantile(1 - upper_pct)
            trimmed = clean[(clean >= lower_bound) & (clean <= upper_bound)]
            n_excluded = len(clean) - len(trimmed)
            if n_excluded > 0:
                title += (
                    f" (P{lower_pct * 100:.0f}–P{(1 - upper_pct) * 100:.0f}, "
                    f"{n_excluded:,} of {len(clean):,} rows excluded)"
                )
            clean = trimmed

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=clean,
                nbinsx=settings.eda_histogram_bins,
                name="density",
                histnorm="probability density",
                marker_color="#4f86c6",
                opacity=0.6,
            )
        )
        if clean.nunique() >= 2 and clean.std() > 0:
            kde = stats.gaussian_kde(clean)
            x_grid = np.linspace(clean.min(), clean.max(), settings.eda_kde_points)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=kde(x_grid),
                    mode="lines",
                    name="KDE",
                    line={"color": "#e74c3c", "width": 2},
                )
            )
        else:
            logger.debug(
                "numeric_distribution_with_kde: column '%s' has < 2 unique values — KDE skipped",
                col,
            )
        fig.update_layout(title=title, xaxis_title=col, yaxis_title="density")
        return apply_dscompanion_theme(fig)

    def numeric_clean_series(self, col: str) -> pd.Series:
        """Return the dropna'd values for one numeric column from the fitted training data.

        Same raw values used internally by ``plot_distributions()``'s
        histogram trace for this column — exposed directly so callers that
        need the raw data (rather than a rendered Plotly figure) don't have
        to re-derive it.

        Args:
            col (str): Numeric column name, as it appears in ``train_X``.

        Returns:
            pd.Series: ``train_X[col]`` with missing values removed.
            Returns an empty ``pd.Series`` if ``col`` is not present in
            ``train_X``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        if col not in self._train_X.columns:
            return pd.Series(dtype=float)
        return self._train_X[col].dropna()

    def categorical_value_counts(self, col: str, top_n: int = 0) -> pd.Series:
        """Return the top-N most frequent values for one categorical column.

        Args:
            col (str): Categorical column name, as it appears in
                ``train_X``.
            top_n (int): Number of most frequent values to keep. A value
                of ``0`` delegates to ``settings.eda_top_categorical_values``.
                Defaults to ``0``.

        Returns:
            pd.Series: Value counts in descending order, indexed by
            category value, capped at ``top_n`` entries. Returns an empty
            ``pd.Series`` if ``col`` is not present in ``train_X``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        if col not in self._train_X.columns:
            return pd.Series(dtype=int)
        n = top_n if top_n > 0 else settings.eda_top_categorical_values
        return self._train_X[col].value_counts().head(n)

    def flag_issues(self) -> pd.DataFrame:
        """Identify columns that exhibit common data quality problems and return a flag table.

        Combines results from both the numeric and categorical summary tables.
        A column is flagged for ``high_missing`` when its missing-value rate
        exceeds the constructor's ``high_missing_threshold`` (which itself
        delegates to ``settings.high_missing_threshold`` when unset), for
        ``near_zero_variance``
        when the variance falls below the constructor's
        ``near_zero_variance_threshold`` (which delegates to
        ``settings.near_zero_variance_threshold`` when unset) (numeric only,
        mutually exclusive with ``constant``), for
        ``high_cardinality`` when unique-value count exceeds
        ``settings.high_cardinality_threshold`` (categorical only), and for
        ``constant`` when the column is entirely constant or all-null (numeric)
        or ``n_unique <= 1`` (categorical).

        Returns:
            pandas.DataFrame: One row per feature with boolean columns —
            ``feature``, ``high_missing``, ``near_zero_variance``,
            ``high_cardinality``, ``constant``.  Returns an empty DataFrame
            with those columns when the training DataFrame had no columns.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        rows = []
        if self._num_summary is not None and len(self._num_summary) > 0:
            for _, r in self._num_summary.iterrows():
                rows.append(
                    {
                        "feature": r["feature"],
                        "high_missing": r["missing_pct"] > self._high_missing_threshold,
                        "near_zero_variance": bool(r["near_zero_variance_flag"]),
                        "high_cardinality": False,
                        "constant": bool(r["constant_flag"]),
                    }
                )
        if self._cat_summary is not None and len(self._cat_summary) > 0:
            for _, r in self._cat_summary.iterrows():
                rows.append(
                    {
                        "feature": r["feature"],
                        "high_missing": r["missing_pct"] > self._high_missing_threshold,
                        "near_zero_variance": False,
                        "high_cardinality": bool(r["cardinality_flag"]),
                        "constant": r["n_unique"] <= 1,
                    }
                )
        return pd.DataFrame(rows, columns=_FLAG_COLS) if rows else pd.DataFrame(columns=_FLAG_COLS)

    def get_flagged_features(self, flag: str) -> list[str]:
        """Return names of features where a specific data quality flag is True.

        Args:
            flag (str): One of ``"high_missing"``, ``"near_zero_variance"``,
                ``"high_cardinality"``, ``"constant"``.

        Returns:
            list[str]: Feature names where the flag is ``True``, in DataFrame
            column order.  Returns an empty list when no columns were present
            at fit time.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
            ValueError: If ``flag`` is not one of the recognised flag columns.
        """
        issues = self.flag_issues()
        if issues.empty:
            return []
        valid_flags = _FLAG_COLS[1:]  # excludes "feature"
        if flag not in valid_flags:
            raise ValueError(f"'{flag}' is not a valid flag — choose from {valid_flags}")
        return issues.loc[issues[flag], "feature"].tolist()

    def extreme_values(self, n: int = 0) -> dict[str, dict[str, list]]:
        """Return the smallest and largest ``n`` raw values for every numeric feature.

        Useful for spotting data-entry errors (e.g. ``age=999``) that
        percentile-based summaries alone can hide.

        Args:
            n (int): Number of extreme values to capture per tail.  A value
                of ``0`` delegates to ``settings.eda_extreme_values_n``.

        Returns:
            dict[str, dict[str, list]]: Keyed by numeric feature name.  Each
            value is ``{"min_extreme": list, "max_extreme": list}`` — the
            ``n`` smallest / largest non-null values, sorted ascending and
            descending respectively.  A numeric feature with zero non-null
            values maps to ``{"min_extreme": [], "max_extreme": []}``.
            Returns an empty dict when there are no numeric columns.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        n = n or settings.eda_extreme_values_n
        df = self._train_X
        result: dict[str, dict[str, list]] = {}
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            clean = df[col].dropna()
            result[col] = {
                "min_extreme": clean.nsmallest(n).tolist() if len(clean) else [],
                "max_extreme": clean.nlargest(n).tolist() if len(clean) else [],
            }
        return result

    def value_counts(self, feature: str, top_n: int = 0) -> pd.DataFrame:
        """Return the full (or top-N) value-frequency table for a single feature.

        Reuses the raw column already held in ``self._train_X`` — no
        recomputation against ``numeric_summary()``/``categorical_summary()``.
        Unlike ``extreme_values()`` (numeric-only, min/max tails), this works
        for any column dtype and returns actual frequencies, not just the
        most extreme raw values.

        Args:
            feature (str): Column name to compute value counts for.
            top_n (int): Maximum number of distinct values returned, ordered
                by descending frequency.  A value of ``0`` returns every
                distinct value (no truncation).

        Returns:
            pd.DataFrame: Columns ``value``, ``count``, ``pct`` (frequency as
            a fraction of non-null rows). Empty (correctly-columned)
            DataFrame when ``feature`` has zero non-null values.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
            KeyError: If ``feature`` is not a column in the fitted data.
        """
        self._check_fitted()
        df = self._train_X
        if feature not in df.columns:
            raise KeyError(f"'{feature}' is not a column in the fitted data")
        s = df[feature]
        vc = s.value_counts()
        if top_n > 0:
            vc = vc.head(top_n)
        non_null = s.count()
        if len(vc) == 0:
            return pd.DataFrame(columns=["value", "count", "pct"])
        return pd.DataFrame(
            {
                "value": vc.index,
                "count": vc.values,
                "pct": vc.values / non_null if non_null > 0 else 0.0,
            }
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_numeric(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(columns=_NUM_COLS)
        rows = []
        for col in cols:
            s = df[col]
            q = s.quantile(
                [
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                    0.96,
                    0.97,
                    0.98,
                    0.99,
                ]
            )
            q1, p50, q3 = q[0.25], q[0.5], q[0.75]
            iqr = q3 - q1
            fence = settings.iqr_multiplier * iqr
            outlier_iqr = ((s < q1 - fence) | (s > q3 + fence)).mean()
            _clean = s.dropna()
            z = np.abs(stats.zscore(_clean)) if len(_clean) > 1 else np.array([])
            outlier_z = (z > settings.zscore_outlier_threshold).mean() if len(z) else 0.0
            mean = s.mean()
            std = s.std()
            # all-NaN guard: std=NaN makes (std == 0) False — use count() == 0 as additional check
            is_constant = (std == 0) or (s.count() == 0)
            rows.append(
                {
                    "feature": col,
                    "count": s.count(),
                    "missing_pct": s.isna().mean(),
                    "zero_pct": (s == 0).mean(),
                    "mean": mean,
                    "std": std,
                    "cv": std / mean if (mean != 0 and not pd.isna(mean)) else np.nan,
                    "min": s.min(),
                    "p1": q[0.01],
                    "p2": q[0.02],
                    "p3": q[0.03],
                    "p4": q[0.04],
                    "p5": q[0.05],
                    "p10": q[0.1],
                    "p25": q1,
                    "p50": p50,
                    "p75": q3,
                    "p90": q[0.9],
                    "p95": q[0.95],
                    "p96": q[0.96],
                    "p97": q[0.97],
                    "p98": q[0.98],
                    "p99": q[0.99],
                    "max": s.max(),
                    "skewness": s.skew(),
                    "kurtosis": s.kurtosis(),
                    "outlier_pct_iqr": outlier_iqr,
                    "outlier_pct_zscore": outlier_z,
                    "constant_flag": is_constant,
                    "near_zero_variance_flag": (
                        (not is_constant) and (s.var() < self._near_zero_variance_threshold)
                    ),
                }
            )
        return pd.DataFrame(rows, columns=_NUM_COLS)

    def _compute_categorical(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(columns=_CAT_COLS)
        rows = []
        for col in cols:
            s = df[col]
            vc = s.value_counts()
            n_unique = s.nunique()
            rows.append(
                {
                    "feature": col,
                    "count": s.count(),
                    "missing_pct": s.isna().mean(),
                    "n_unique": n_unique,
                    "top_value": vc.index[0] if len(vc) > 0 else None,
                    "top_freq": vc.iloc[0] / max(s.count(), 1) if len(vc) > 0 else 0,
                    "cardinality_flag": n_unique > settings.high_cardinality_threshold,
                    "imbalance": _imbalance_score(vc.values),
                }
            )
        return pd.DataFrame(rows, columns=_CAT_COLS)

    def _compute_datetime(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(columns=_DATETIME_COLS)
        rows = []
        for col in cols:
            s = df[col]
            clean = s.dropna()
            rows.append(
                {
                    "feature": col,
                    "count": s.count(),
                    "missing_pct": s.isna().mean(),
                    "n_unique": s.nunique(),
                    "min": clean.min() if len(clean) else None,
                    "max": clean.max() if len(clean) else None,
                }
            )
        return pd.DataFrame(rows, columns=_DATETIME_COLS)

    def _compute_boolean(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(columns=_BOOL_COLS)
        rows = []
        for col in cols:
            s = df[col]
            non_null = s.count()
            n_true = int(s.sum()) if non_null else 0
            rows.append(
                {
                    "feature": col,
                    "count": non_null,
                    "missing_pct": s.isna().mean(),
                    "n_true": n_true,
                    "n_false": int(non_null - n_true),
                    "pct_true": n_true / non_null if non_null > 0 else None,
                }
            )
        return pd.DataFrame(rows, columns=_BOOL_COLS)

    @staticmethod
    def _numeric_plot(s: pd.Series, col: str) -> go.Figure:
        clean = s.dropna()
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=clean,
                nbinsx=settings.eda_histogram_bins,
                name="count",
                marker_color="#4f86c6",
                opacity=0.75,
            )
        )
        fig.update_layout(title=f"{col} — distribution", xaxis_title=col, yaxis_title="count")
        return fig

    @staticmethod
    def _categorical_plot(s: pd.Series, col: str) -> go.Figure:
        vc = s.value_counts().head(settings.eda_top_categorical_values)
        fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color="#27ae60"))
        fig.update_layout(title=f"{col} — top values", xaxis_title="value", yaxis_title="count")
        return fig

    @staticmethod
    def _datetime_plot(s: pd.Series, col: str) -> go.Figure:
        clean = s.dropna()
        fig = go.Figure(
            go.Histogram(x=clean, nbinsx=settings.eda_histogram_bins, marker_color="#8e44ad")
        )
        fig.update_layout(title=f"{col} — over time", xaxis_title=col, yaxis_title="count")
        return fig

    @staticmethod
    def _boolean_plot(s: pd.Series, col: str) -> go.Figure:
        vc = s.value_counts()
        fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color="#16a085"))
        fig.update_layout(title=f"{col} — counts", xaxis_title="value", yaxis_title="count")
        return fig

    def text_analysis(
        self,
        top_n_words: int = 0,
        top_n_chars: int = 0,
        include_sample_values: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Compute word/char frequency and Unicode breakdown for all categorical columns.

        Uses stdlib only (``unicodedata``, ``str.split()``) — no additional
        dependencies. Aggregate statistics (word/character frequency, Unicode
        category distribution, average string length) are always computed.
        Raw sample values are only included when ``include_sample_values=True``
        (compliance opt-in — raw cell content may contain customer data).

        Args:
            top_n_words (int): Number of top word-frequency entries to return
                per column. A value of ``0`` delegates to
                ``settings.eda_text_analysis_top_n_words``.
            top_n_chars (int): Number of top character-frequency entries to
                return per column. A value of ``0`` delegates to
                ``settings.eda_text_analysis_top_n_chars``.
            include_sample_values (bool): When ``True``, includes the first 5
                non-null raw values as strings in the result for each column.
                Defaults to ``False`` — compliance gating, since raw cell
                values may contain customer identifiers or PII.

        Returns:
            dict[str, dict[str, Any]]: Keyed by categorical column name. Each
            value is a dict with keys:
                - ``"word_freq"``: ``dict[str, int]`` — top-N word-to-count
                  pairs, whitespace-split and lowercased, sorted descending.
                - ``"char_freq"``: ``dict[str, int]`` — top-N character-to-count
                  pairs, sorted descending by count.
                - ``"char_unicode_categories"``: ``dict[str, int]`` — Unicode
                  category code (e.g. ``"Lu"``, ``"Ll"``, ``"Nd"``) to total
                  character count in that category across all non-null values.
                - ``"avg_string_length"``: ``float | None`` — mean character
                  count per non-null value; ``None`` when the column is
                  entirely null.
                - ``"sample_values"``: ``list[str]`` — first 5 non-null values
                  as strings; empty list when ``include_sample_values=False``.
            Returns an empty ``dict`` when no categorical columns are present.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        import unicodedata

        self._check_fitted()
        n_words = top_n_words or settings.eda_text_analysis_top_n_words
        n_chars = top_n_chars or settings.eda_text_analysis_top_n_chars

        df = self._train_X
        cat_cols = [
            c
            for c in df.columns
            if c not in self._bool_cols
            and c not in self._datetime_cols
            and not pd.api.types.is_numeric_dtype(df[c])
        ]

        result: dict[str, dict[str, Any]] = {}
        for col in cat_cols:
            s = df[col].dropna().astype(str)
            avg_len: float | None = float(s.str.len().mean()) if len(s) > 0 else None

            word_counts: dict[str, int] = {}
            char_counts: dict[str, int] = {}
            unicode_cats: dict[str, int] = {}
            for val in s:
                for word in val.lower().split():
                    word_counts[word] = word_counts.get(word, 0) + 1
                for ch in val:
                    char_counts[ch] = char_counts.get(ch, 0) + 1
                    cat_code = unicodedata.category(ch)
                    unicode_cats[cat_code] = unicode_cats.get(cat_code, 0) + 1

            result[col] = {
                "word_freq": dict(
                    sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)[:n_words]
                ),
                "char_freq": dict(
                    sorted(char_counts.items(), key=lambda kv: kv[1], reverse=True)[:n_chars]
                ),
                "char_unicode_categories": unicode_cats,
                "avg_string_length": avg_len,
                "sample_values": s.head(5).tolist() if include_sample_values else [],
            }

        return result

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing results.")
