"""Standalone production-readiness diagnostics for the check_model_readiness MCP tool.

Deliberately reimplements dscompanion.docs.model_card.ModelCard's
_performance_diagnostics() logic rather than importing it -- that method is a
private implementation detail requiring a full ModelCard construction, and per
its own docstring is already independently duplicated twice elsewhere in the
codebase for module-boundary independence (dscompanion.api vs. dscompanion.docs).
This is a deliberate fourth copy, following that established convention, kept
calibrated to the same settings.evaluate_* thresholds. Unlike the original
(which only reports flags when something's wrong), this returns all 4 checks
always, including passing ones, since the tool's whole purpose is a complete
checklist.
"""

from __future__ import annotations

from typing import Any

from dscompanion.config import settings

__all__ = ["readiness_checks"]


def _metric_value(metrics_df: Any, split: str, metric: str) -> float | None:
    row = metrics_df[(metrics_df["split"] == split) & (metrics_df["metric"] == metric)]
    if row.empty:
        return None
    val = float(row["value"].iloc[0])
    return None if val != val else val  # NaN check without importing numpy here


def readiness_checks(model: Any, split: Any) -> list[dict[str, str]]:
    """Run production-readiness diagnostics on a fitted model against a DataSplit.

    Args:
        model (Any): A fitted ``BaseDSCompanionModel`` subclass instance.
        split (Any): A ``DataSplit`` with at least a train and one held-out
            partition (val or test).

    Returns:
        list[dict[str, str]]: Always exactly 4 ``{"name": str, "status":
        "pass"|"warn"|"fail", "detail": str}`` records -- one each for
        "leakage", "near_random", "overfitting_gap", "psi" -- present whether
        they passed or not.
    """
    metrics_df = model.evaluate(split)
    if metrics_df.empty:
        return [
            {
                "name": name,
                "status": "warn",
                "detail": "Model could not be evaluated against the provided data.",
            }
            for name in ("leakage", "near_random", "overfitting_gap", "psi")
        ]

    primary = next(
        (s for s in ("val", "test", "train") if (metrics_df["split"] == s).any()), "train"
    )
    primary_auc = _metric_value(metrics_df, primary, "roc_auc")
    train_auc = _metric_value(metrics_df, "train", "roc_auc")
    primary_psi = _metric_value(metrics_df, primary, "psi")

    checks: list[dict[str, str]] = []

    if primary_auc is None:
        checks.append(
            {
                "name": "leakage",
                "status": "warn",
                "detail": "No roc_auc metric available -- leakage check skipped.",
            }
        )
        checks.append(
            {
                "name": "near_random",
                "status": "warn",
                "detail": "No roc_auc metric available -- near-random check skipped.",
            }
        )
    else:
        if primary_auc > settings.evaluate_leakage_auc_threshold:
            checks.append(
                {
                    "name": "leakage",
                    "status": "fail",
                    "detail": (
                        f"AUC of {primary_auc:.3f} on {primary} is unusually high -- "
                        "possible data leakage."
                    ),
                }
            )
        else:
            checks.append(
                {
                    "name": "leakage",
                    "status": "pass",
                    "detail": f"AUC of {primary_auc:.3f} on {primary} shows no leakage signal.",
                }
            )

        if primary_auc < settings.evaluate_near_random_auc_threshold:
            checks.append(
                {
                    "name": "near_random",
                    "status": "fail",
                    "detail": (
                        f"AUC of {primary_auc:.3f} on {primary} is close to random guessing."
                    ),
                }
            )
        else:
            checks.append(
                {
                    "name": "near_random",
                    "status": "pass",
                    "detail": f"AUC of {primary_auc:.3f} on {primary} is above random guessing.",
                }
            )

    if train_auc is None or primary_auc is None or primary == "train":
        checks.append(
            {
                "name": "overfitting_gap",
                "status": "warn",
                "detail": "No separate held-out split available -- overfitting check skipped.",
            }
        )
    else:
        gap = train_auc - primary_auc
        if gap > settings.evaluate_overfitting_gap_threshold:
            checks.append(
                {
                    "name": "overfitting_gap",
                    "status": "fail",
                    "detail": (
                        f"Training AUC ({train_auc:.3f}) exceeds {primary} AUC "
                        f"({primary_auc:.3f}) by {gap:.3f} -- possible overfitting."
                    ),
                }
            )
        else:
            checks.append(
                {
                    "name": "overfitting_gap",
                    "status": "pass",
                    "detail": f"Train-{primary} AUC gap of {gap:.3f} is within tolerance.",
                }
            )

    if primary_psi is None:
        checks.append(
            {
                "name": "psi",
                "status": "warn",
                "detail": "No psi metric available -- stability check skipped.",
            }
        )
    elif primary_psi > settings.evaluate_psi_unstable_threshold:
        checks.append(
            {
                "name": "psi",
                "status": "fail",
                "detail": (
                    f"PSI of {primary_psi:.3f} on {primary} indicates an unstable "
                    "score distribution."
                ),
            }
        )
    elif primary_psi > settings.evaluate_psi_monitor_threshold:
        checks.append(
            {
                "name": "psi",
                "status": "warn",
                "detail": f"PSI of {primary_psi:.3f} on {primary} is in the monitoring zone.",
            }
        )
    else:
        checks.append(
            {
                "name": "psi",
                "status": "pass",
                "detail": (
                    f"PSI of {primary_psi:.3f} on {primary} shows a stable score distribution."
                ),
            }
        )

    return checks
