"""Step 9 (Hyperparameter Tuning) endpoints — thin HTTP wrapper around
``dscompanion.api.services.tuning``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service, get_state_store
from dscompanion.api.schemas import (
    TuningConfirmRequest,
    TuningConfirmResponse,
    TuningPreviewResponse,
    TuningStartRequest,
    TuningStartResponse,
)
from dscompanion.api.services.tuning import confirm_tuning, preview_tuning, start_tuning_job
from dscompanion.api.state import RunState, RunStateStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/tuning",
    tags=["tuning"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=TuningPreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> TuningPreviewResponse:
    """Reports whether tuning is available for Step 8's confirmed algorithm.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        TuningPreviewResponse: Availability and slider bounds.

    Raises:
        HTTPException: 400, if Step 8 has not been confirmed for this run.
    """
    try:
        return preview_tuning(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/tune/start", response_model=TuningStartResponse)
def tune_start(
    body: TuningStartRequest,
    run: RunState = Depends(get_pipeline_service),
    store: RunStateStore = Depends(get_state_store),
) -> TuningStartResponse:
    """Starts a background Optuna tuning run.

    Args:
        body (TuningStartRequest): Number of trials to run.
        run (RunState): Injected via ``get_pipeline_service``.
        store (RunStateStore): Injected via ``get_state_store``.

    Returns:
        TuningStartResponse: The new job's id — poll
        ``GET /api/runs/{run_id}/jobs/{job_id}`` for progress and the result.

    Raises:
        HTTPException: 400, if Step 8 has not been confirmed for this run.
    """
    try:
        return start_tuning_job(run, store, body.n_trials)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=TuningConfirmResponse)
def confirm(
    body: TuningConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> TuningConfirmResponse:
    """Persists the final model choice (tuned or Step 8's default).

    Args:
        body (TuningConfirmRequest): Whether to accept the tuned model.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        TuningConfirmResponse: Confirmation, whether tuned, and the best score.

    Raises:
        HTTPException: 400, if Step 8 has not been confirmed, or
            ``use_tuned=True`` but no tuned model is available.
    """
    try:
        return confirm_tuning(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
