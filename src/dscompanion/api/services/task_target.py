"""Step 2 (Task & Target) service functions — the REST equivalent of
``dscompanion/app/interactive_step2.py``'s validation rules, with every ``streamlit`` call
stripped out. Classification only, same restriction as the UI (see that module's
docstring for why).
"""

from __future__ import annotations

import logging

import pandas as pd

from dscompanion.api.schemas import (
    TaskTargetConfirmRequest,
    TaskTargetConfirmResponse,
    TaskTargetPreviewRequest,
    TaskTargetPreviewResponse,
)
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id

logger = logging.getLogger(__name__)

__all__ = ["preview_task_target", "confirm_task_target"]


def _validate_target(df: pd.DataFrame, target: str) -> list[str]:
    """Validates a target column against strict binary classification rules.

    Rules (all must pass): present, no missing values, numeric dtype, exactly
    ``{0, 1}`` as unique values, class 1 is the minor (less frequent) class.

    Args:
        df (pd.DataFrame): Working dataframe.
        target (str): Candidate target column name.

    Returns:
        list[str]: One message per failed rule. Empty when every rule passes.
    """
    if target not in df.columns:
        return [f"Target '{target}' is not a column in the loaded data."]

    series = df[target]
    errors: list[str] = []

    n_missing = int(series.isna().sum())
    if n_missing > 0:
        errors.append(f"Target '{target}' has {n_missing:,} missing value(s).")
        return errors

    if not pd.api.types.is_numeric_dtype(series):
        errors.append(f"Target '{target}' has dtype {series.dtype}, must be numeric {{0, 1}}.")
        return errors

    unique_vals = set(series.unique().tolist())
    if unique_vals != {0, 1}:
        errors.append(
            f"Target '{target}' contains {sorted(unique_vals)} — must be exactly {{0, 1}}."
        )
        return errors

    counts = series.value_counts()
    n_ones = int(counts.get(1, 0))
    n_zeros = int(counts.get(0, 0))
    if n_ones > n_zeros:
        errors.append(
            f"Positive class (1) has {n_ones:,} rows vs. {n_zeros:,} for class 0 — "
            "1 must be the minority class."
        )
    return errors


def preview_task_target(
    run: RunState, request: TaskTargetPreviewRequest
) -> TaskTargetPreviewResponse:
    """Validates a candidate task/target choice against Step 1's loaded dataframe,
    without persisting anything to the run.

    Args:
        run (RunState): The run holding Step 1's confirmed dataframe.
        request (TaskTargetPreviewRequest): Candidate task, target, and identifier
            columns.

    Returns:
        TaskTargetPreviewResponse: Whether the target is valid, any validation
        errors, and its class value counts.

    Raises:
        ValueError: If Step 1 has not been confirmed for this run.
    """
    df = run.artifacts.get("df")
    if df is None:
        raise ValueError("Step 1 (Load Data) must be confirmed before Step 2.")

    errors = _validate_target(df, request.target)
    value_counts: dict[str, int] = {}
    if request.target in df.columns:
        value_counts = {
            str(k): int(v) for k, v in df[request.target].value_counts(dropna=True).items()
        }
    logger.info("Previewed target '%s': valid=%s", request.target, not errors)
    return TaskTargetPreviewResponse(
        valid=not errors, errors=errors, target_value_counts=value_counts
    )


def confirm_task_target(
    run: RunState, request: TaskTargetConfirmRequest
) -> TaskTargetConfirmResponse:
    """Re-validates and persists the task/target choice for this run.

    Args:
        run (RunState): The run holding Step 1's confirmed dataframe.
        request (TaskTargetConfirmRequest): Task, target, and identifier columns.

    Returns:
        TaskTargetConfirmResponse: The confirmed task and target.

    Raises:
        ValueError: If Step 1 has not been confirmed, or the target fails validation.
    """
    df = run.artifacts.get("df")
    if df is None:
        raise ValueError("Step 1 (Load Data) must be confirmed before Step 2.")

    errors = _validate_target(df, request.target)
    if errors:
        raise ValueError("; ".join(errors))

    run.artifacts["target"] = request.target
    run.artifacts["task"] = request.task
    run.artifacts["identifier_columns"] = request.identifier_columns
    run.step_data["task_target"] = {
        "task": request.task,
        "target": request.target,
        "identifier_columns": request.identifier_columns,
    }
    run.step_confirmed["task_target"] = True
    run.step_status["task_target"] = "done"
    nxt = next_step_id("task_target")
    if nxt is not None:
        run.step_status[nxt] = "current"
    counts = df[request.target].value_counts()
    run.audit_trail.append(
        (
            "task_target",
            f"Task: {request.task}, target: '{request.target}' "
            f"({int(counts.get(1, 0)):,} events / {int(counts.get(0, 0)):,} non-events).",
        )
    )
    logger.info(
        "run_id=%s confirmed Step 2: task=%s target=%s", run.run_id, request.task, request.target
    )
    return TaskTargetConfirmResponse(confirmed=True, task=request.task, target=request.target)
