"""Standalone metric utilities used across the dscompanion package."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)

__all__ = [
    "ks_statistic",
    "gini_coefficient",
    "psi_score",
    "feature_psi_table",
    "freeze_feature_reference",
    "csi_score",
    "feature_csi_table",
    "iv_score",
    "woe_bins",
    "expected_calibration_error",
    "decile_table",
]

DECILE_TABLE_COLUMNS = [
    "decile",
    "count",
    "events",
    "event_rate",
    "cumulative_count",
    "cumulative_events",
    "cumulative_event_rate",
    "pct_of_total_events",
    "cumulative_pct_of_total_events",
    "lift",
    "cumulative_lift",
]


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic as the maximum vertical
    distance between the true-positive-rate and false-positive-rate curves
    derived from the ROC curve, measuring how well the model separates the
    positive and negative classes.

    Args:
        y_true (np.ndarray): Binary ground-truth labels of shape
            ``(n_samples,)`` with values in ``{0, 1}``.
        y_prob (np.ndarray): Predicted positive-class probabilities of shape
            ``(n_samples,)`` with values in ``[0, 1]``.

    Returns:
        float: KS statistic in the range ``[0, 1]``.  A value of ``0``
        indicates no separation; ``1`` indicates perfect separation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Gini coefficient as ``2 * ROC-AUC - 1``, providing a
    normalised measure of rank-ordering ability that ranges from ``-1``
    (perfectly inverted) to ``1`` (perfect discrimination), with ``0``
    representing a random classifier.

    Args:
        y_true (np.ndarray): Binary ground-truth labels of shape
            ``(n_samples,)`` with values in ``{0, 1}``.
        y_prob (np.ndarray): Predicted positive-class probabilities of shape
            ``(n_samples,)`` with values in ``[0, 1]``.

    Returns:
        float: Gini coefficient in the range ``[-1, 1]``.  A well-calibrated
        binary classifier typically yields values in ``[0, 1]``.
    """
    from sklearn.metrics import roc_auc_score

    return float(2 * roc_auc_score(y_true, y_prob) - 1)


def psi_score(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Population Stability Index (PSI) between a reference score
    distribution and a comparison distribution, quantifying how much the
    distribution of model scores has shifted between two time periods or
    datasets.

    Bins are constructed using the percentiles of the ``expected`` distribution
    so that bin boundaries are data-driven.  Duplicate percentile breakpoints
    are collapsed via ``np.unique`` to avoid zero-width bins.  A small epsilon
    ``1e-8`` is applied to all proportions before log-computation to prevent
    division-by-zero or log-of-zero errors.

    Standard interpretation thresholds: PSI < 0.1 is stable; 0.1–0.25
    indicates moderate shift; above 0.25 indicates significant shift.

    Args:
        expected (np.ndarray): Reference (baseline) score distribution of
            shape ``(n_ref_samples,)``; typically model output scores on the
            training set.
        actual (np.ndarray): Comparison score distribution of shape
            ``(n_cmp_samples,)``; typically model output scores on an
            out-of-time holdout or production population.
        n_bins (int): Number of equal-frequency bins derived from the
            ``expected`` distribution for discretisation.  Defaults to
            ``10``.  Fewer unique breakpoints may result in fewer effective
            bins.

    Returns:
        float: PSI value in ``[0, +inf)``.  Returns ``0.0`` when both
        distributions are identical.  Very large values indicate near-complete
        distribution divergence.
    """
    eps = 1e-8
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def feature_psi_table(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    n_bins: int = 0,
) -> pd.DataFrame:
    """Compute per-column Population Stability Index between two DataFrames.

    For each column present in both ``reference`` and ``comparison``:
    numeric columns use equal-frequency bins derived from the reference
    distribution (via ``psi_score``); categorical/object columns use
    category-frequency proportions as the bin representation.  Columns
    absent from either frame are skipped with a DEBUG log.

    Args:
        reference (pd.DataFrame): Baseline feature matrix; typically the
            training set.  Used to derive bin boundaries for numeric columns.
        comparison (pd.DataFrame): Comparison feature matrix; typically an
            out-of-time or production holdout.
        n_bins (int): Number of equal-frequency bins for numeric columns.
            A value of ``0`` delegates to ``settings.psi_feature_n_bins``.

    Returns:
        pd.DataFrame: One row per evaluated column with columns
        ``feature`` (str), ``psi`` (float, rounded to 4 d.p.), and
        ``flag`` (bool, ``True`` when ``psi`` exceeds
        ``settings.psi_alert_threshold``).  Sorted by ``psi`` descending.
        Returns an empty DataFrame with those columns when no shared columns
        exist or both frames are empty.

    Raises:
        TypeError: If ``reference`` or ``comparison`` is not a
            ``pd.DataFrame``.
    """
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("reference must be a pd.DataFrame, got %s" % type(reference).__name__)
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pd.DataFrame, got %s" % type(comparison).__name__)

    from dscompanion.config import settings as _settings

    _n_bins = n_bins or _settings.psi_feature_n_bins
    _COLS = ["feature", "psi", "flag"]

    if reference.empty or comparison.empty:
        return pd.DataFrame(columns=_COLS)

    import logging as _logging

    _log = _logging.getLogger(__name__)

    rows = []
    shared = [c for c in reference.columns if c in comparison.columns]
    for col in shared:
        ref_col = reference[col].dropna()
        cmp_col = comparison[col].dropna()
        if len(ref_col) == 0 or len(cmp_col) == 0:
            _log.debug("feature_psi_table: skipping %r — empty after dropna", col)
            continue
        try:
            if pd.api.types.is_numeric_dtype(ref_col):
                v = psi_score(ref_col.values, cmp_col.values, n_bins=_n_bins)
            else:
                # Categorical PSI: use category-frequency proportions
                eps = 1e-8
                categories = set(ref_col.astype(str).unique()) | set(cmp_col.astype(str).unique())
                ref_counts = ref_col.astype(str).value_counts()
                cmp_counts = cmp_col.astype(str).value_counts()
                ref_pct = np.array(
                    [ref_counts.get(c, 0) / max(len(ref_col), 1) for c in categories]
                )
                cmp_pct = np.array(
                    [cmp_counts.get(c, 0) / max(len(cmp_col), 1) for c in categories]
                )
                ref_pct = np.clip(ref_pct, eps, None)
                cmp_pct = np.clip(cmp_pct, eps, None)
                v = float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))
        except (ValueError, ArithmeticError) as exc:
            _log.debug("feature_psi_table: skipping %r — %s", col, exc)
            continue
        rows.append(
            {
                "feature": col,
                "psi": round(v, 4),
                "flag": v > _settings.psi_alert_threshold,
            }
        )

    return (
        pd.DataFrame(rows, columns=_COLS).sort_values("psi", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=_COLS)
    )


_LOW_RESOLUTION_BIN_RATIO = 0.5


def freeze_feature_reference(reference: pd.DataFrame, n_bins: int = 0) -> dict[str, dict[str, Any]]:
    """Compute and freeze a per-column reference distribution from a raw training DataFrame.

    Used for later feature-level drift (CSI) comparison via ``feature_csi_table``/
    ``csi_score`` — without persisting any raw rows, so the result is safe to
    embed (joblib-picklable) inside an artefact that may be copied to a
    cluster or API service. Numeric columns are binned via the same
    equal-frequency (quantile) approach ``psi_score`` already uses, but with
    the first/last edge extended to ``-inf``/``inf`` so any future
    out-of-range value lands in the correct extreme bucket rather than being
    silently dropped (the behaviour ``psi_score`` itself has today).
    Categorical/object columns get a category-to-proportion map instead.

    Args:
        reference (pd.DataFrame): Baseline feature matrix, typically the raw
            (pre-``feature_pipeline``) training set.
        n_bins (int): Number of equal-frequency bins for numeric columns. A
            value of ``0`` delegates to ``settings.psi_feature_n_bins``.

    Returns:
        dict[str, dict]: One entry per column (columns that are empty after
        ``dropna()`` are skipped entirely). Numeric columns:
        ``{"type": "numeric", "bin_edges": np.ndarray, "bin_proportions":
        np.ndarray}`` (``bin_edges`` has ``len(bin_proportions) + 1``
        entries, first/last equal to ``-inf``/``inf``). Categorical columns:
        ``{"type": "categorical", "category_proportions": dict[str, float]}``.
    """
    from dscompanion.config import settings as _settings

    _n_bins = n_bins or _settings.psi_feature_n_bins
    reference_distribution: dict[str, dict[str, Any]] = {}

    for col in reference.columns:
        s = reference[col].dropna()
        if len(s) == 0:
            logger.debug("freeze_feature_reference: skipping %r — empty after dropna", col)
            continue

        if pd.api.types.is_numeric_dtype(s):
            edges = np.unique(np.percentile(s.values, np.linspace(0, 100, _n_bins + 1)))
            if len(edges) < 2:
                edges = np.array([s.values[0], s.values[0]])

            effective_n_bins = len(edges) - 1
            if effective_n_bins < max(1, int(_n_bins * _LOW_RESOLUTION_BIN_RATIO)):
                logger.warning(
                    "freeze_feature_reference: column %r collapsed to %d effective bin(s) "
                    "(requested %d) after de-duplicating percentile edges — likely a large "
                    "point-mass (e.g. many exact-zero values); CSI on this column will be "
                    "lower-resolution than requested.",
                    col,
                    effective_n_bins,
                    _n_bins,
                )

            edges = edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            bin_proportions = np.histogram(s.values, bins=edges)[0] / len(s)
            reference_distribution[col] = {
                "type": "numeric",
                "bin_edges": edges,
                "bin_proportions": bin_proportions,
            }
        else:
            category_proportions = (s.astype(str).value_counts() / len(s)).to_dict()
            reference_distribution[col] = {
                "type": "categorical",
                "category_proportions": category_proportions,
            }

    return reference_distribution


def csi_score(column_reference: dict[str, Any], comparison: pd.Series) -> float:
    """Compute the Characteristic/Population Stability Index for a single column
    against its frozen reference distribution.

    Args:
        column_reference (dict): One column's entry from
            ``freeze_feature_reference``'s return value.
        comparison (pd.Series): New data for this same column.

    Returns:
        float: CSI/PSI value in ``[0, +inf)``.
    """
    eps = 1e-8
    comparison = comparison.dropna()

    if column_reference["type"] == "numeric":
        actual_pct = np.histogram(comparison.values, bins=column_reference["bin_edges"])[0] / len(
            comparison
        )
        expected_pct = column_reference["bin_proportions"]
        actual_pct = np.clip(actual_pct, eps, None)
        expected_pct = np.clip(expected_pct, eps, None)
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    ref_props = column_reference["category_proportions"]
    cmp_str = comparison.astype(str)
    cmp_counts = cmp_str.value_counts()
    categories = set(ref_props) | set(cmp_counts.index)
    ref_pct = np.array([ref_props.get(c, eps) for c in categories])
    cmp_pct = np.array([cmp_counts.get(c, 0) / len(comparison) for c in categories])
    ref_pct = np.clip(ref_pct, eps, None)
    cmp_pct = np.clip(cmp_pct, eps, None)
    return float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))


def feature_csi_table(
    feature_reference: dict[str, dict[str, Any]],
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-column CSI between a frozen training reference and new data.

    The feature-level counterpart to ``feature_psi_table``, but driven by a
    frozen reference distribution (from ``freeze_feature_reference``) instead
    of a live reference DataFrame, and flagged against
    ``settings.csi_alert_threshold`` (stricter than
    ``settings.psi_alert_threshold``, since CSI compares raw business
    features rather than model scores).

    Args:
        feature_reference (dict[str, dict]): Output of
            ``freeze_feature_reference``.
        comparison (pd.DataFrame): New data to compare against the frozen
            reference; typically an out-of-time or production batch.

    Returns:
        pd.DataFrame: One row per evaluated column with columns ``feature``
        (str), ``psi`` (float, rounded to 4 d.p.), and ``flag`` (bool,
        ``True`` when ``psi`` exceeds ``settings.csi_alert_threshold``).
        Sorted by ``psi`` descending. Returns an empty DataFrame with those
        columns when no shared columns exist or ``comparison`` is empty.

    Raises:
        TypeError: If ``feature_reference`` is not a ``dict`` or
            ``comparison`` is not a ``pd.DataFrame``.
    """
    if not isinstance(feature_reference, dict):
        raise TypeError(
            "feature_reference must be a dict, got %s" % type(feature_reference).__name__
        )
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pd.DataFrame, got %s" % type(comparison).__name__)

    from dscompanion.config import settings as _settings

    _COLS = ["feature", "psi", "flag"]
    if not feature_reference or comparison.empty:
        return pd.DataFrame(columns=_COLS)

    shared = [c for c in feature_reference if c in comparison.columns]
    rows = []
    for col in shared:
        cmp_col = comparison[col].dropna()
        if len(cmp_col) == 0:
            logger.debug("feature_csi_table: skipping %r — empty after dropna", col)
            continue
        try:
            v = csi_score(feature_reference[col], cmp_col)
        except (ValueError, ArithmeticError) as exc:
            logger.debug("feature_csi_table: skipping %r — %s", col, exc)
            continue
        rows.append({"feature": col, "psi": round(v, 4), "flag": v > _settings.csi_alert_threshold})

    return (
        pd.DataFrame(rows, columns=_COLS).sort_values("psi", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=_COLS)
    )


def woe_bins(
    feature: pd.Series,
    target: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute Weight of Evidence (WoE) and per-bin Information Value (IV)
    contribution for a single feature against a binary target, discretising
    numeric features into quantile bins and treating string-typed features as
    nominal categories.

    Rows with missing values in either ``feature`` or ``target`` are silently
    dropped before binning.  A small epsilon ``1e-8`` is applied to prevent
    log-of-zero errors when a bin contains only events or only non-events.

    Args:
        feature (pd.Series): The feature to analyse.  Numeric series are
            binned into ``n_bins`` equal-frequency quantile buckets via
            ``pd.qcut``; object/string series are treated as nominal
            categories with one bin per distinct value.  Duplicate bin edges
            are handled by ``pd.qcut(duplicates="drop")``.
        target (pd.Series): Binary target series of the same length as
            ``feature``, with values in ``{0, 1}``.
        n_bins (int): Number of quantile bins to use for numeric features.
            Has no effect on categorical features.  Defaults to ``10``.

    Returns:
        pd.DataFrame: DataFrame with one row per bin and columns
        ``bin`` (bin label or category value), ``count`` (int, total
        observations in the bin), ``event_rate`` (float, proportion of
        positives in the bin), ``woe`` (float, log-odds ratio for the bin),
        and ``iv_contrib`` (float, IV contribution of the bin).  Returns an
        empty DataFrame with those five columns when all rows are ``NaN`` or
        the feature/target share no non-null overlap.
    """
    eps = 1e-8
    df = pd.DataFrame({"feature": feature, "target": target}).dropna()
    total_events = df["target"].sum()
    total_non_events = len(df) - total_events

    if pd.api.types.is_numeric_dtype(feature):
        df["bin"] = pd.qcut(df["feature"], q=n_bins, duplicates="drop")
    else:
        df["bin"] = df["feature"].astype(str)

    grouped = (
        df.groupby("bin", observed=True)["target"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "events", "count": "count"})
    )
    grouped["non_events"] = grouped["count"] - grouped["events"]
    grouped["dist_events"] = grouped["events"] / max(total_events, eps)
    grouped["dist_non_events"] = grouped["non_events"] / max(total_non_events, eps)
    grouped["dist_events"] = grouped["dist_events"].clip(eps)
    grouped["dist_non_events"] = grouped["dist_non_events"].clip(eps)
    grouped["woe"] = np.log(grouped["dist_events"] / grouped["dist_non_events"])
    grouped["iv_contrib"] = (grouped["dist_events"] - grouped["dist_non_events"]) * grouped["woe"]
    grouped["event_rate"] = grouped["events"] / grouped["count"].clip(1)
    return grouped.reset_index()[["bin", "count", "event_rate", "woe", "iv_contrib"]]


def iv_score(
    feature: pd.Series,
    target: pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute the total Information Value (IV) for a feature by summing the
    per-bin IV contributions produced by ``woe_bins``, providing a single
    scalar measure of a feature's predictive power against a binary target.

    Standard interpretation thresholds: IV < 0.02 = not useful; 0.02–0.1 =
    weak; 0.1–0.3 = medium; above 0.3 = strong; above 0.5 may indicate
    data leakage.

    Args:
        feature (pd.Series): The feature to evaluate.  Numeric series are
            quantile-binned; string/object series are treated as nominal
            categories.  Rows with missing values are dropped.
        target (pd.Series): Binary target series of the same length as
            ``feature``, with values in ``{0, 1}``.
        n_bins (int): Number of quantile bins for numeric features, forwarded
            to ``woe_bins``.  Defaults to ``10``.

    Returns:
        float: Total IV value in ``[0, +inf)``.  Returns ``0.0`` when all
        rows are ``NaN`` or the feature has only one unique non-null value.
    """
    bins = woe_bins(feature, target, n_bins)
    return float(bins["iv_contrib"].sum())


def decile_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute a decile-based gains/lift table for a binary classifier.

    Sorts observations by predicted probability in descending order, splits
    them into ``n_bins`` equal-sized groups by rank (decile 1 = highest
    predicted risk/response), and computes per-decile and cumulative event
    capture statistics — the standard credit-risk and marketing-response
    model validation table. Ranking by row position (not by probability
    value) avoids uneven bin sizes when many observations share the same
    predicted probability, a common occurrence with tree-based models.

    Args:
        y_true (np.ndarray): Binary ground-truth labels of shape
            ``(n_samples,)`` with values in ``{0, 1}``.
        y_prob (np.ndarray): Predicted positive-class probabilities of shape
            ``(n_samples,)`` with values in ``[0, 1]``.
        n_bins (int): Number of equal-sized deciles. Defaults to ``10``.
            Fewer effective bins result when ``len(y_true) < n_bins``.

    Returns:
        pd.DataFrame: One row per decile with columns ``decile`` (int, 1 =
        highest-scoring group), ``count``, ``events``, ``event_rate``,
        ``cumulative_count``, ``cumulative_events``, ``cumulative_event_rate``,
        ``pct_of_total_events`` (this decile's share of all events, i.e. the
        per-decile capture rate), ``cumulative_pct_of_total_events`` (the
        cumulative gain curve), ``lift`` (``event_rate`` divided by the
        overall event rate), and ``cumulative_lift``. Returns an empty
        DataFrame with these columns when ``y_true`` is empty or contains no
        positive events (division-by-zero guard).
    """
    eps = 1e-8
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0 or y_true.sum() == 0:
        return pd.DataFrame(columns=DECILE_TABLE_COLUMNS)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, q=min(n_bins, len(df)), labels=False, duplicates="drop") + 1

    overall_event_rate = df["y_true"].mean()
    total_events = df["y_true"].sum()

    grouped = (
        df.groupby("decile")["y_true"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "events", "count": "count"})
    )
    grouped["event_rate"] = grouped["events"] / grouped["count"]
    grouped["cumulative_count"] = grouped["count"].cumsum()
    grouped["cumulative_events"] = grouped["events"].cumsum()
    grouped["cumulative_event_rate"] = grouped["cumulative_events"] / grouped["cumulative_count"]
    safe_total_events = max(total_events, eps)
    grouped["pct_of_total_events"] = grouped["events"] / safe_total_events
    grouped["cumulative_pct_of_total_events"] = grouped["cumulative_events"] / safe_total_events
    grouped["lift"] = grouped["event_rate"] / max(overall_event_rate, eps)
    grouped["cumulative_lift"] = grouped["cumulative_event_rate"] / max(overall_event_rate, eps)

    return grouped.reset_index()[DECILE_TABLE_COLUMNS]


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE), measuring how closely
    a model's predicted probabilities match the observed event rates, by
    partitioning predictions into equal-width bins and computing a
    sample-weighted average of the absolute difference between mean
    predicted confidence and empirical accuracy per bin.

    Empty bins (no predictions fall within their probability interval) are
    skipped and contribute ``0`` to the sum, so the result is not inflated
    by unused probability regions.

    Args:
        y_true (np.ndarray): Binary ground-truth labels of shape
            ``(n_samples,)`` with values in ``{0, 1}``.
        y_prob (np.ndarray): Predicted positive-class probabilities of shape
            ``(n_samples,)`` with values in ``[0, 1]``.
        n_bins (int): Number of equal-width bins spanning ``[0, 1]`` used to
            group predictions.  Defaults to ``10``.

    Returns:
        float: ECE value in ``[0, 1]``.  Returns ``0.0`` when every bin is
        empty (e.g. both arrays have length zero) or when the model is
        perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
