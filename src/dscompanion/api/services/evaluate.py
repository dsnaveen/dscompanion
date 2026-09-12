"""Step 10 (Evaluate) service functions — the REST equivalent of
``dscompanion/app/interactive_step_evaluate.py``, with every ``streamlit`` call stripped out.

Read-only per the Phase D.3 REST contract: ``preview_evaluate`` recomputes the full
metric breakdown and suspicious-pattern flags fresh on every call (cheap relative to
training/tuning), ``confirm_evaluate`` is a flag-flip with no persisted user choice.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from dscompanion.api.schemas import (
    EvaluateBaseline,
    EvaluateConfirmResponse,
    EvaluateFlag,
    EvaluateMetricRow,
    EvaluatePreviewResponse,
)
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id
from dscompanion.config import settings
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["preview_evaluate", "confirm_evaluate"]

# Display metadata — structural copy, not a tunable threshold, so a plain module dict
# rather than settings (matches services/train.py's _ALGO_NOTES precedent).
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
        "Gini = 2 x AUC - 1. Common in credit scoring. "
        "0 = no discriminating power, 1 = perfect separation. "
        "A Gini of 0.60 corresponds to an AUC of 0.80."
    ),
    "ks_statistic": (
        "Kolmogorov-Smirnov: the largest gap between the predicted-score "
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
        "< 0.10 is stable, 0.10-0.25 needs monitoring, > 0.25 is unstable, possible drift."
    ),
}

_HEADLINE_METRICS: tuple[str, ...] = ("roc_auc", "gini", "ks_statistic")

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


def _primary_split(metrics_df: pd.DataFrame) -> str:
    """Return the best split name to use for headline metrics and flags.

    Prefers ``val`` (unseen held-out data), then ``test``, then ``train`` as
    a last resort — mirrors ``interactive_step_evaluate.py``'s
    ``_primary_split`` exactly.

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
        float | None: The metric value, or ``None`` if not found or ``NaN``.
    """
    row = metrics_df[(metrics_df["split"] == split) & (metrics_df["metric"] == metric)]
    if row.empty:
        return None
    val = float(row["value"].iloc[0])
    return None if math.isnan(val) else val


def _metric_rows(metrics_df: pd.DataFrame) -> list[EvaluateMetricRow]:
    """Build the full per-split, per-metric display rows, headline metrics first.

    Args:
        metrics_df (pd.DataFrame): Tidy evaluation DataFrame.

    Returns:
        list[EvaluateMetricRow]: One row per ``(split, metric)`` pair actually
            present in ``metrics_df``, in ``_METRIC_ORDER`` order within each
            split.
    """
    splits = [
        s for s in ("train", "val", "test", "oot") if not metrics_df[metrics_df["split"] == s].empty
    ]
    rows: list[EvaluateMetricRow] = []
    for split in splits:
        for metric in _METRIC_ORDER:
            value = _get_metric_value(metrics_df, split, metric)
            rows.append(
                EvaluateMetricRow(
                    split=split,
                    metric=metric,
                    label=_METRIC_LABELS.get(metric, metric),
                    gloss=_METRIC_GLOSSES.get(metric, ""),
                    value=value,
                )
            )
    return rows


def _suspicious_flags(metrics_df: pd.DataFrame) -> list[EvaluateFlag]:
    """Return plain-language warnings for suspicious metric patterns.

    Checks for near-perfect scores (possible leakage), near-random scores
    (possible target/feature mismatch), overfitting (large train-primary
    gap), and PSI drift — mirrors ``interactive_step_evaluate.py``'s
    ``_suspicious_flags`` exactly, with the four hardcoded thresholds routed
    through ``settings`` instead.

    Args:
        metrics_df (pd.DataFrame): Tidy evaluation DataFrame.

    Returns:
        list[EvaluateFlag]: Empty when no suspicious pattern is found.
    """
    flags: list[EvaluateFlag] = []
    primary = _primary_split(metrics_df)

    primary_auc = _get_metric_value(metrics_df, primary, "roc_auc")
    train_auc = _get_metric_value(metrics_df, "train", "roc_auc")

    if primary_auc is not None:
        if primary_auc > settings.evaluate_leakage_auc_threshold:
            flags.append(
                EvaluateFlag(
                    level="error",
                    message=(
                        f"AUC of {primary_auc:.3f} on {primary} is unusually high. "
                        "This can happen when a feature accidentally contains the answer "
                        "(data leakage). Consider going back to Step 6 and reviewing "
                        "which features were kept."
                    ),
                )
            )
        elif primary_auc < settings.evaluate_near_random_auc_threshold:
            flags.append(
                EvaluateFlag(
                    level="warning",
                    message=(
                        f"AUC of {primary_auc:.3f} on {primary} is close to random guessing "
                        "(0.50). The model may not have learned anything useful. Consider "
                        "reviewing the target column in Step 2 and the features in Step 6."
                    ),
                )
            )

    if train_auc is not None and primary_auc is not None and primary != "train":
        gap = train_auc - primary_auc
        if gap > settings.evaluate_overfitting_gap_threshold:
            flags.append(
                EvaluateFlag(
                    level="warning",
                    message=(
                        f"Training AUC ({train_auc:.3f}) is much higher than {primary} AUC "
                        f"({primary_auc:.3f}), a gap of {gap:.3f}. This suggests the model "
                        "memorised the training data (overfitting). Try removing features "
                        "in Step 6, or a simpler model in Step 8."
                    ),
                )
            )

    primary_psi = _get_metric_value(metrics_df, primary, "psi")
    if primary_psi is not None:
        if primary_psi > settings.evaluate_psi_unstable_threshold:
            flags.append(
                EvaluateFlag(
                    level="warning",
                    message=(
                        f"PSI of {primary_psi:.3f} on {primary} is high (> "
                        f"{settings.evaluate_psi_unstable_threshold}). The score distribution "
                        "shifted significantly vs. training, a possible distribution shift "
                        "between train and evaluation data."
                    ),
                )
            )
        elif primary_psi > settings.evaluate_psi_monitor_threshold:
            flags.append(
                EvaluateFlag(
                    level="info",
                    message=(
                        f"PSI of {primary_psi:.3f} on {primary} is in the monitoring zone "
                        f"({settings.evaluate_psi_monitor_threshold}-"
                        f"{settings.evaluate_psi_unstable_threshold}). Worth tracking if this "
                        "model is deployed to production."
                    ),
                )
            )

    return flags


def _baseline_rows(run: RunState, primary: str) -> list[EvaluateBaseline]:
    """Build the tuned-vs-default delta baseline, reusing Step 8's untuned model.

    ``confirm_tuning`` never overwrites ``run.artifacts["trained_model"]`` — it
    only adds ``"final_model"`` — so Step 8's original untuned model stays
    available here for exactly this comparison, matching
    ``interactive_step_evaluate.py``'s ``_render_headline`` re-evaluating its
    own cached baseline. No new persistence needed.

    Args:
        run (RunState): The run holding Step 8's untuned model and the
            processed split.
        primary (str): The split to evaluate the baseline on.

    Returns:
        list[EvaluateBaseline]: One row per headline metric, empty if the
            untuned model or processed split is unavailable.
    """
    baseline_model = run.artifacts.get("trained_model")
    processed_split: DataSplit | None = run.artifacts.get("processed_split")
    if baseline_model is None or processed_split is None:
        return []

    baseline_df = baseline_model.evaluate(processed_split)
    return [
        EvaluateBaseline(metric=metric, value=_get_metric_value(baseline_df, primary, metric))
        for metric in _HEADLINE_METRICS
    ]


# ── Preview ───────────────────────────────────────────────────────────────────


def preview_evaluate(run: RunState) -> EvaluatePreviewResponse:
    """Evaluates the final model from Step 9 and reports the full breakdown.

    Args:
        run (RunState): The run holding Step 9's final model and Step 8's
            processed split.

    Returns:
        EvaluatePreviewResponse: Full per-split metric breakdown, the
        tuned-vs-default baseline (when tuned), and any suspicious-pattern
        flags.

    Raises:
        ValueError: If Step 9 has not been confirmed, or the processed
            split is unavailable, for this run.
    """
    model = run.artifacts.get("final_model")
    processed_split: DataSplit | None = run.artifacts.get("processed_split")
    algorithm = run.step_data.get("train", {}).get("algorithm")
    if model is None or processed_split is None or algorithm is None:
        raise ValueError("Step 9 must be confirmed before Step 10.")

    metrics_df = model.evaluate(processed_split)
    primary = _primary_split(metrics_df)
    tuned = bool(run.step_data.get("tuning", {}).get("tuned", False))

    logger.info(
        "run_id=%s previewed Step 10: algorithm=%s primary_split=%s", run.run_id, algorithm, primary
    )
    return EvaluatePreviewResponse(
        algorithm=algorithm,
        tuned=tuned,
        primary_split=primary,
        metrics=_metric_rows(metrics_df),
        baseline=_baseline_rows(run, primary) if tuned else [],
        flags=_suspicious_flags(metrics_df),
    )


# ── Confirm ───────────────────────────────────────────────────────────────────


def confirm_evaluate(run: RunState) -> EvaluateConfirmResponse:
    """Records Step 10 completion — a flag-flip only, no persisted user choice.

    Args:
        run (RunState): The run to persist into.

    Returns:
        EvaluateConfirmResponse: Confirmation.

    Raises:
        ValueError: If Step 9 has not been confirmed, or the processed
            split is unavailable, for this run.
    """
    model = run.artifacts.get("final_model")
    processed_split: DataSplit | None = run.artifacts.get("processed_split")
    if model is None or processed_split is None:
        raise ValueError("Step 9 must be confirmed before Step 10.")

    metrics_df = model.evaluate(processed_split)
    primary = _primary_split(metrics_df)
    auc = _get_metric_value(metrics_df, primary, "roc_auc")
    flags = _suspicious_flags(metrics_df)

    flag_suffix = ""
    n_flags = sum(1 for f in flags if f.level in ("error", "warning"))
    if n_flags:
        flag_suffix = f" ({n_flags} flag{'s' if n_flags > 1 else ''} noted)"

    auc_str = f"{auc:.4f}" if auc is not None else "n/a"
    run.step_confirmed["evaluate"] = True
    run.step_status["evaluate"] = "done"
    nxt = next_step_id("evaluate")
    if nxt is not None:
        run.step_status[nxt] = "current"
    run.audit_trail.append(
        ("evaluate", f"Evaluation reviewed: {primary} AUC {auc_str}.{flag_suffix}")
    )

    logger.info("run_id=%s confirmed Step 10: %s AUC=%s", run.run_id, primary, auc_str)
    return EvaluateConfirmResponse(confirmed=True)
