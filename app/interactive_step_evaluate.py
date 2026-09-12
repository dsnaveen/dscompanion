"""Interactive mode — Step 10: See How Well It Performed.

Read-only evaluation step.  Re-evaluates the final model (tuned or default
from Steps 8–9) on the processed split from Step 8, then presents every
metric with a plain-language gloss so a non-expert user can interpret the
numbers, and flags suspicious patterns (near-perfect scores, near-random
scores, overfitting) with a concrete next action.

No confirm gate on the metrics themselves — the user clicks
"Continue to Step 11" after reviewing.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging
import math

import pandas as pd
import streamlit as st
from interactive_state import add_audit_entry, reset_from_step, set_step_confirmed, set_step_status

logger = logging.getLogger(__name__)

__all__ = ["render_step_evaluate"]

# ── Metric display metadata ───────────────────────────────────────────────────

_METRIC_LABELS: dict[str, str] = {
    "roc_auc": "AUC-ROC",
    "gini": "Gini",
    "ks_statistic": "KS Statistic",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
    "log_loss": "Log Loss",
    "psi": "PSI",
}

_METRIC_GLOSSES: dict[str, str] = {
    "roc_auc": (
        "How well the model separates positives from negatives. "
        "0.5 = no better than guessing, 1.0 = perfect. "
        "In banking, 0.70+ is typically acceptable, 0.80+ is good."
    ),
    "gini": (
        "Gini = 2 × AUC − 1. Common in credit scoring. "
        "0 = no discriminating power, 1 = perfect separation. "
        "A Gini of 0.60 corresponds to an AUC of 0.80."
    ),
    "ks_statistic": (
        "Kolmogorov–Smirnov: the largest gap between the predicted-score "
        "distributions of positives and negatives. Higher means better separation."
    ),
    "f1": (
        "Balance between precision and recall. "
        "Useful when both false alarms and missed positives are costly. "
        "Ranges from 0 (worst) to 1 (best)."
    ),
    "precision": (
        "Of all customers flagged positive by the model, what fraction actually was? "
        "Low precision means many false alarms."
    ),
    "recall": (
        "Of all actual positives, what fraction did the model catch? "
        "Low recall means many positives are missed."
    ),
    "log_loss": (
        "Average penalty for wrong predictions, lower is better. "
        "Heavily penalises confident wrong predictions."
    ),
    "psi": (
        "Population Stability Index: how much the score distribution shifted vs. training. "
        "< 0.10 is stable, 0.10–0.25 needs monitoring, > 0.25 is unstable, possible drift."
    ),
}

# Headline metrics shown as st.metric tiles; remaining shown only in the table.
_HEADLINE_METRICS: tuple[str, ...] = ("roc_auc", "gini", "ks_statistic")

# Full table display order: headline metrics first, then supporting metrics.
_METRIC_ORDER: list[str] = [
    "roc_auc",
    "gini",
    "ks_statistic",
    "f1",
    "precision",
    "recall",
    "log_loss",
    "psi",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _final_model():
    """Return the final fitted model from Step 9 (tuned or default).

    Args:
        None

    Returns:
        object | None: The ``ClassificationModel`` stored at
        ``interactive.tuning_model``, or ``None`` if Step 9 has not confirmed.
    """
    return st.session_state.get("interactive.tuning_model")


def _get_metrics() -> pd.DataFrame | None:
    """Return the evaluation DataFrame for the final model, computing if needed.

    Caches the result in ``int.evaluate.metrics_df`` so repeated renders do not
    re-evaluate the model.  Evaluation uses the processed split from Step 8
    (``int.train.processed_split``) so the same feature-transformed data that
    was used for training is used here.

    Args:
        None

    Returns:
        pd.DataFrame | None: Tidy DataFrame with columns ``split``, ``metric``,
        ``value`` from ``model.evaluate(split)``, or ``None`` if the model or
        split is unavailable.
    """
    cached = st.session_state.get("int.evaluate.metrics_df")
    if cached is not None:
        return cached

    model = _final_model()
    split = st.session_state.get("int.train.processed_split")
    if model is None or split is None:
        return None

    try:
        metrics_df = model.evaluate(split)
        st.session_state["int.evaluate.metrics_df"] = metrics_df
        return metrics_df
    except Exception as exc:
        logger.warning("Step 10 evaluation failed: %s", exc)
        return None


def _primary_split(metrics_df: pd.DataFrame) -> str:
    """Return the best split name to use for headline metrics and flags.

    Prefers ``val`` (unseen held-out data), then ``test``, then ``train``
    as a last resort.

    Args:
        metrics_df (pd.DataFrame): Tidy evaluation DataFrame with a ``split``
            column.

    Returns:
        str: One of ``"val"``, ``"test"``, or ``"train"``.
    """
    for s in ("val", "test", "train"):
        if not metrics_df[metrics_df["split"] == s].empty:
            return s
    return "train"


def _get_metric_value(metrics_df: pd.DataFrame, split: str, metric: str) -> float | None:
    """Extract a single metric value from a tidy metrics DataFrame.

    Args:
        metrics_df (pd.DataFrame): Tidy DataFrame with columns ``split``,
            ``metric``, ``value``.
        split (str): Split name to look up (e.g. ``"val"``).
        metric (str): Metric name to look up (e.g. ``"roc_auc"``).

    Returns:
        float | None: The metric value, or ``None`` if not found.
    """
    row = metrics_df[(metrics_df["split"] == split) & (metrics_df["metric"] == metric)]
    if row.empty:
        return None
    val = float(row["value"].iloc[0])
    return None if math.isnan(val) else val


def _suspicious_flags(metrics_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return plain-language warnings for suspicious metric patterns.

    Checks for near-perfect scores (possible leakage), near-random scores
    (possible target/feature mismatch), and overfitting (large train–val gap).

    Args:
        metrics_df (pd.DataFrame): Tidy evaluation DataFrame.

    Returns:
        list[tuple[str, str]]: Each element is ``(level, message)`` where
        ``level`` is ``"error"``, ``"warning"``, or ``"info"`` and ``message``
        is a plain-language Markdown string.
    """
    flags: list[tuple[str, str]] = []
    primary = _primary_split(metrics_df)

    primary_auc = _get_metric_value(metrics_df, primary, "roc_auc")
    train_auc = _get_metric_value(metrics_df, "train", "roc_auc")

    if primary_auc is not None:
        if primary_auc > 0.99:
            flags.append(
                (
                    "error",
                    f"**AUC of {primary_auc:.3f} on {primary} is unusually high.** "
                    "This can happen when a feature accidentally contains the answer "
                    "(data leakage). Consider going back to **Step 6** and reviewing "
                    "which features were kept.",
                )
            )
        elif primary_auc < 0.55:
            flags.append(
                (
                    "warning",
                    f"**AUC of {primary_auc:.3f} on {primary} is close to random guessing "
                    "(0.50).** The model may not have learned anything useful. Consider "
                    "reviewing the target column in **Step 2** and the features in **Step 6**.",
                )
            )

    if train_auc is not None and primary_auc is not None and primary != "train":
        gap = train_auc - primary_auc
        if gap > 0.10:
            flags.append(
                (
                    "warning",
                    f"**Training AUC ({train_auc:.3f}) is much higher than {primary} AUC "
                    f"({primary_auc:.3f}), a gap of {gap:.3f}.** "
                    "This suggests the model memorised the training data (overfitting). "
                    "Try removing features in **Step 6**, or a simpler model in **Step 8**.",
                )
            )

    primary_psi = _get_metric_value(metrics_df, primary, "psi")
    if primary_psi is not None:
        if primary_psi > 0.25:
            flags.append(
                (
                    "warning",
                    f"**PSI of {primary_psi:.3f} on {primary} is high (> 0.25).** "
                    "The score distribution shifted significantly vs. training, a possible "
                    "distribution shift between train and evaluation data.",
                )
            )
        elif primary_psi > 0.10:
            flags.append(
                (
                    "info",
                    f"PSI of {primary_psi:.3f} on {primary} is in the monitoring zone (0.10–0.25). "
                    "Worth tracking if this model is deployed to production.",
                )
            )

    return flags


# ── Render sections ───────────────────────────────────────────────────────────


def _render_headline(metrics_df: pd.DataFrame) -> None:
    """Show AUC-ROC, Gini, and KS as st.metric tiles for the primary split.

    Also shows a delta vs. the Step 8 baseline if the model was tuned, so
    the effect of tuning is immediately visible.

    Args:
        metrics_df (pd.DataFrame): Tidy evaluation DataFrame.

    Returns:
        None
    """
    primary = _primary_split(metrics_df)
    tuned = st.session_state.get("interactive.tuning_tuned", False)

    # Baseline from Step 8 for delta comparison (only meaningful if tuned)
    baseline_df: pd.DataFrame | None = None
    if tuned:
        baseline_df = st.session_state.get("int.train.metrics_df")
        if baseline_df is None:
            winner = st.session_state.get("int.train.winner_model")
            split = st.session_state.get("int.train.processed_split")
            if winner is not None and split is not None:
                try:
                    baseline_df = winner.evaluate(split)
                except Exception:
                    pass

    split_label = f"on **{primary}** split"
    st.caption(
        f"Key metrics {split_label}: data the model did not see during training."
        if primary != "train"
        else "Key metrics on **training** data. Validation/test splits are unavailable."
    )

    cols = st.columns(len(_HEADLINE_METRICS))
    for col, metric in zip(cols, _HEADLINE_METRICS):
        val = _get_metric_value(metrics_df, primary, metric)
        label = _METRIC_LABELS[metric]
        if val is None:
            col.metric(label, "—")
            continue

        delta = None
        if baseline_df is not None:
            baseline_val = _get_metric_value(baseline_df, primary, metric)
            if baseline_val is not None:
                delta = f"{val - baseline_val:+.4f} vs. default"

        col.metric(label, f"{val:.4f}", delta=delta)


def _render_full_breakdown(metrics_df: pd.DataFrame) -> None:
    """Show all splits side-by-side in a single metric table.

    Columns: Metric, What it means, then one column per available split
    (Train / Val / Test / OOT).  Eliminates the need to tab-switch to
    compare values across splits.

    Args:
        metrics_df (pd.DataFrame): Tidy evaluation DataFrame.

    Returns:
        None
    """
    splits = [
        s for s in ("train", "val", "test", "oot") if not metrics_df[metrics_df["split"] == s].empty
    ]
    if not splits:
        st.caption("No evaluation data available.")
        return

    # Build one row per metric; value columns named by split label.
    split_labels = {s: s.upper() for s in splits}
    rows = []
    for m in _METRIC_ORDER:
        row: dict = {
            "Metric": _METRIC_LABELS.get(m, m),
            "What it means": _METRIC_GLOSSES.get(m, ""),
        }
        found = False
        for s in splits:
            sub = metrics_df[(metrics_df["split"] == s) & (metrics_df["metric"] == m)]
            if sub.empty:
                row[split_labels[s]] = "—"
            else:
                val = float(sub["value"].iloc[0])
                row[split_labels[s]] = "—" if math.isnan(val) else f"{val:.4f}"
                found = True
        if found:
            rows.append(row)

    st.markdown("**Full Metric Breakdown**")
    if rows:
        col_config: dict = {
            "Metric": st.column_config.TextColumn(width="small"),
            "What it means": st.column_config.TextColumn(width="large"),
        }
        for label in split_labels.values():
            col_config[label] = st.column_config.TextColumn(width="small")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config=col_config,
        )
    else:
        st.caption("No metrics recorded.")


def _render_flags(metrics_df: pd.DataFrame) -> None:
    """Show plain-language warnings for any suspicious metric patterns.

    Args:
        metrics_df (pd.DataFrame): Tidy evaluation DataFrame.

    Returns:
        None
    """
    flags = _suspicious_flags(metrics_df)
    if not flags:
        st.success("No suspicious patterns found. The scores look plausible.")
        return

    st.markdown("**Things to Check**")
    for level, msg in flags:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)


# ── Confirm ───────────────────────────────────────────────────────────────────


def _confirm_step10(metrics_df: pd.DataFrame) -> None:
    """Record Step 10 completion and write an audit entry.

    Args:
        metrics_df (pd.DataFrame): Tidy evaluation DataFrame — used to
            extract the headline AUC for the audit log.

    Returns:
        None
    """
    primary = _primary_split(metrics_df)
    auc = _get_metric_value(metrics_df, primary, "roc_auc")
    flags = _suspicious_flags(metrics_df)

    flag_suffix = ""
    if any(lvl in ("error", "warning") for lvl, _ in flags):
        n = sum(1 for lvl, _ in flags if lvl in ("error", "warning"))
        flag_suffix = f" ({n} flag{'s' if n > 1 else ''} noted)"

    auc_str = f"{auc:.4f}" if auc is not None else "n/a"
    add_audit_entry(
        "evaluate",
        f"Evaluation reviewed: {primary} AUC {auc_str}.{flag_suffix}",
    )
    set_step_confirmed("evaluate", True)
    set_step_status("evaluate", "done")
    logger.info("Interactive mode — Step 10 confirmed: %s AUC=%s", primary, auc_str)


# ── Main entrypoint ───────────────────────────────────────────────────────────


def render_step_evaluate() -> bool:
    """Render Step 10 (See How Well It Performed) and report confirmation.

    Evaluates the final model from Step 9 on the processed split from Step 8,
    displays headline metrics, a full per-split breakdown with plain-language
    glosses, and any suspicious-result flags.  Confirmation is a single
    "Continue to Step 11" click with no substantive decision attached.

    Args:
        None

    Returns:
        bool: ``True`` once the user clicks Continue; ``False`` while still
        reviewing.
    """
    if st.session_state["interactive.step_confirmed"]["evaluate"]:
        return True

    algorithm = st.session_state.get("interactive.train_algorithm", "model")
    tuned = st.session_state.get("interactive.tuning_tuned", False)
    tuned_label = " (tuned)" if tuned else ""

    st.subheader("Step 10: How well it performed")
    st.caption(f"Evaluating **{algorithm}{tuned_label}** on train, validation, and test splits.")

    metrics_df = _get_metrics()

    if metrics_df is None or metrics_df.empty:
        st.error(
            "Could not evaluate the model — the final model or processed split from "
            "Step 8 is missing. Please go back and re-run Steps 8–9."
        )
        if st.button("Go back to Step 8 and retrain", key="int.evaluate.back_to_train"):
            reset_from_step("train")
            st.rerun()
        return False

    _render_headline(metrics_df)
    st.divider()
    _render_full_breakdown(metrics_df)
    st.divider()
    _render_flags(metrics_df)

    st.divider()
    if st.button("Continue to Step 11", key="int.evaluate.continue", type="primary"):
        _confirm_step10(metrics_df)

    return st.session_state["interactive.step_confirmed"]["evaluate"]
