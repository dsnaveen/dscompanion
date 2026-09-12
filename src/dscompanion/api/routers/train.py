"""Step 8 (Train a Model) endpoints — thin HTTP wrapper around
``dscompanion.api.services.train``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service, get_state_store
from dscompanion.api.schemas import (
    LeaderboardStartResponse,
    TrainConfirmRequest,
    TrainConfirmResponse,
    TrainPreviewResponse,
)
from dscompanion.api.services.train import confirm_train, preview_train, start_leaderboard_job
from dscompanion.api.state import RunState, RunStateStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/train",
    tags=["train"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=TrainPreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> TrainPreviewResponse:
    """Reports the available algorithms and feature-column count.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        TrainPreviewResponse: Algorithm picker options and feature count.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed for this run.
    """
    try:
        return preview_train(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/leaderboard/start", response_model=LeaderboardStartResponse)
def leaderboard_start(
    run: RunState = Depends(get_pipeline_service),
    store: RunStateStore = Depends(get_state_store),
) -> LeaderboardStartResponse:
    """Starts a background leaderboard comparison across every supported algorithm.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.
        store (RunStateStore): Injected via ``get_state_store``.

    Returns:
        LeaderboardStartResponse: The new job's id — poll
        ``GET /api/runs/{run_id}/jobs/{job_id}`` for progress and the result.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed, or no
            feature columns remain.
    """
    try:
        return start_leaderboard_job(run, store)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=TrainConfirmResponse)
def confirm(
    body: TrainConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> TrainConfirmResponse:
    """Trains (single-algorithm path) or accepts an already-trained leaderboard
    winner, then persists the result.

    Args:
        body (TrainConfirmRequest): The algorithm to confirm.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        TrainConfirmResponse: Confirmation and a brief metrics summary.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed, or training
            fails.
    """
    try:
        return confirm_train(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
