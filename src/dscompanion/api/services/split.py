"""Step 3 (Split) service functions — the REST equivalent of
``dscompanion/app/interactive_step3.py``'s ``DataSplitter`` wiring, with every ``streamlit``
call stripped out.
"""

from __future__ import annotations

import logging

import pandas as pd

from dscompanion.api.schemas import (
    SplitConfirmRequest,
    SplitConfirmResponse,
    SplitPreviewRequest,
    SplitPreviewResponse,
)
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id
from dscompanion.split import DataSplit, DataSplitter

logger = logging.getLogger(__name__)

__all__ = ["preview_split", "confirm_split"]


def _run_split(df: pd.DataFrame, target: str, request: SplitPreviewRequest) -> DataSplit:
    """Computes a split for the given dataframe/target/request.

    Args:
        df (pd.DataFrame): Working dataframe, from Step 1.
        target (str): Confirmed target column, from Step 2.
        request (SplitPreviewRequest): Split method and parameters.

    Returns:
        DataSplit: The computed split.

    Raises:
        ValueError: If the split leaves no training rows, or ``DataSplitter`` itself
            rejects the configuration.
    """
    if request.date_col is not None and not pd.api.types.is_datetime64_any_dtype(
        df[request.date_col]
    ):
        # DataSplitter's temporal split compares the raw column against the cutoff
        # without parsing it — parse here first, same workaround as
        # interactive_step3.py's _run_split().
        df = df.copy()
        df[request.date_col] = pd.to_datetime(df[request.date_col])

    splitter = DataSplitter(
        strategy=request.method,
        test_size=request.test_size,
        val_size=request.val_size,
        date_col=request.date_col,
        group_col=request.group_col,
        oot_cutoff=request.oot_cutoff,
        target_col=target,
    )
    split = splitter.fit_split(df)
    if split.metadata.split_sizes.get("train", 0) == 0:
        raise ValueError("This split leaves no rows for training — lower the test/validation size.")
    return split


def preview_split(run: RunState, request: SplitPreviewRequest) -> SplitPreviewResponse:
    """Computes a split preview against Step 1/2's confirmed dataframe/target, without
    persisting anything to the run.

    Args:
        run (RunState): The run holding Step 1's dataframe and Step 2's target.
        request (SplitPreviewRequest): Split method and parameters.

    Returns:
        SplitPreviewResponse: Row counts and event rates per split partition.

    Raises:
        ValueError: If Step 1/2 have not been confirmed for this run, or the split
            configuration is invalid.
    """
    df = run.artifacts.get("df")
    target = run.artifacts.get("target")
    if df is None or target is None:
        raise ValueError("Steps 1 and 2 must be confirmed before Step 3.")

    split = _run_split(df, target, request)
    sizes = split.metadata.split_sizes
    rates = split.metadata.class_rates or {}
    logger.info("Previewed split (method=%s): sizes=%s", request.method, sizes)
    return SplitPreviewResponse(row_counts=dict(sizes), event_rates=dict(rates))


def confirm_split(run: RunState, request: SplitConfirmRequest) -> SplitConfirmResponse:
    """Recomputes and persists the split for this run.

    Args:
        run (RunState): The run holding Step 1's dataframe and Step 2's target.
        request (SplitConfirmRequest): Split method and parameters.

    Returns:
        SplitConfirmResponse: Row counts per split partition.

    Raises:
        ValueError: If Step 1/2 have not been confirmed for this run, or the split
            configuration is invalid.
    """
    df = run.artifacts.get("df")
    target = run.artifacts.get("target")
    if df is None or target is None:
        raise ValueError("Steps 1 and 2 must be confirmed before Step 3.")

    split = _run_split(df, target, request)
    sizes = split.metadata.split_sizes

    run.artifacts["split"] = split
    run.step_data["split"] = {
        "method": request.method,
        "test_size": request.test_size,
        "val_size": request.val_size,
        "date_col": request.date_col,
        "group_col": request.group_col,
    }
    run.step_confirmed["split"] = True
    run.step_status["split"] = "done"
    nxt = next_step_id("split")
    if nxt is not None:
        run.step_status[nxt] = "current"
    parts = [
        f"{name}: {sizes.get(name, 0):,} rows"
        for name in ("train", "val", "test")
        if sizes.get(name, 0)
    ]
    if sizes.get("oot", 0):
        parts.append(f"out-of-time: {sizes['oot']:,} rows")
    run.audit_trail.append(
        ("split", f"Split applied ({request.method}): " + " / ".join(parts) + ".")
    )
    logger.info(
        "run_id=%s confirmed Step 3 split (method=%s): sizes=%s", run.run_id, request.method, sizes
    )
    return SplitConfirmResponse(confirmed=True, row_counts=dict(sizes))
