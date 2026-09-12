"""Step 7 (Class Imbalance Handling) service functions — the REST equivalent of
``dscompanion/app/interactive_step_imbalance.py``, with every ``streamlit`` call stripped
out. No resampling actually runs here — the handler fires at training time (Step 8);
this step only records the strategy choice and (for row-count-changing strategies)
previews the estimated before/after training-set size.
"""

from __future__ import annotations

import logging

from dscompanion.api.schemas import (
    ImbalanceConfirmRequest,
    ImbalanceConfirmResponse,
    ImbalancePreviewResponse,
)
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id
from dscompanion.config import settings
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["preview_imbalance", "confirm_imbalance"]

# Strategies that don't change row counts — no row-count preview needed.
_NO_RESAMPLE_STRATEGIES = {"class_weight", "none"}

_AUDIT_MSGS = {
    "class_weight": "Imbalance handling: class weight (model-internal, no data change).",
    "smote": "Imbalance handling: SMOTE — training set will be oversampled at training time.",
    "undersample": (
        "Imbalance handling: undersample — training set will be undersampled at training time."
    ),
    "none": "Imbalance handling: none — training data unchanged.",
}


def _preview_row_counts(
    n_train: int, minority_count: int, majority_count: int, strategy: str
) -> int:
    """Estimates the post-resample training-set row count.

    Uses the ``sampling_strategy="auto"`` assumption — ``ImbalanceHandler``'s default —
    which balances both classes to the same count.

    Args:
        n_train (int): Training-set row count before resampling.
        minority_count (int): Minority-class row count.
        majority_count (int): Majority-class row count.
        strategy (str): One of ``"smote"`` or ``"undersample"``.

    Returns:
        int: Estimated post-resample row count.
    """
    if strategy == "smote":
        return 2 * majority_count
    return 2 * minority_count


def preview_imbalance(run: RunState) -> ImbalancePreviewResponse:
    """Reports the confirmed training split's class balance, without persisting anything.

    Args:
        run (RunState): The run holding Step 2's confirmed task and Step 3's split.

    Returns:
        ImbalancePreviewResponse: Class balance and whether it's roughly even.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed for this run.
    """
    split: DataSplit = run.artifacts.get("split")
    if split is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 7.")

    task = run.artifacts.get("task", "classification")
    if task != "classification":
        return ImbalancePreviewResponse(
            applicable=False,
            n_train=len(split.train_y),
            event_rate=0.0,
            is_balanced=True,
            class_counts={},
            balanced_min_event_rate=settings.imbalance_balanced_min_event_rate,
        )

    train_y = split.train_y
    n_train = len(train_y)
    event_rate = float(train_y.mean())
    band = settings.imbalance_balanced_min_event_rate
    is_balanced = band <= event_rate <= (1.0 - band)
    counts = train_y.value_counts().sort_index()

    logger.info("run_id=%s previewed Step 7 imbalance: event_rate=%.3f", run.run_id, event_rate)
    return ImbalancePreviewResponse(
        applicable=True,
        n_train=n_train,
        event_rate=event_rate,
        is_balanced=is_balanced,
        class_counts={str(k): int(v) for k, v in counts.items()},
        balanced_min_event_rate=band,
    )


def confirm_imbalance(run: RunState, request: ImbalanceConfirmRequest) -> ImbalanceConfirmResponse:
    """Persists the confirmed imbalance strategy for this run.

    Args:
        run (RunState): The run to persist into.
        request (ImbalanceConfirmRequest): The chosen strategy.

    Returns:
        ImbalanceConfirmResponse: Confirmation and the row-count preview (when the
        strategy changes row counts).

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed for this run.
    """
    split: DataSplit = run.artifacts.get("split")
    if split is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 7.")

    task = run.artifacts.get("task", "classification")
    strategy = request.strategy if task == "classification" else "none"

    train_y = split.train_y
    rows_before = len(train_y)
    rows_after: int | None = None
    if strategy not in _NO_RESAMPLE_STRATEGIES:
        counts = train_y.value_counts()
        rows_after = _preview_row_counts(
            rows_before, int(counts.min()), int(counts.max()), strategy
        )

    run.step_data["imbalance"] = {"strategy": strategy}
    run.artifacts["imbalance_strategy"] = strategy
    run.step_confirmed["imbalance"] = True
    run.step_status["imbalance"] = "done"
    nxt = next_step_id("imbalance")
    if nxt is not None:
        run.step_status[nxt] = "current"
    run.audit_trail.append(
        ("imbalance", _AUDIT_MSGS.get(strategy, f"Imbalance handling: {strategy}."))
    )

    logger.info("run_id=%s confirmed Step 7: strategy=%s", run.run_id, strategy)
    return ImbalanceConfirmResponse(
        confirmed=True, strategy=strategy, rows_before=rows_before, rows_after=rows_after
    )
