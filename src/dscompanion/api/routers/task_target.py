"""Step 2 (Task & Target) endpoints — thin HTTP wrapper around
``dscompanion.api.services.task_target``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    TaskTargetConfirmRequest,
    TaskTargetConfirmResponse,
    TaskTargetPreviewRequest,
    TaskTargetPreviewResponse,
)
from dscompanion.api.services.task_target import confirm_task_target, preview_task_target
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/task_target",
    tags=["task_target"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=TaskTargetPreviewResponse)
def preview(
    body: TaskTargetPreviewRequest, run: RunState = Depends(get_pipeline_service)
) -> TaskTargetPreviewResponse:
    """Validates a candidate task/target choice without persisting anything.

    Args:
        body (TaskTargetPreviewRequest): Candidate task, target, identifier columns.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        TaskTargetPreviewResponse: Validity, errors, target value counts.

    Raises:
        HTTPException: 400, if Step 1 has not been confirmed for this run.
    """
    try:
        return preview_task_target(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=TaskTargetConfirmResponse)
def confirm(
    body: TaskTargetConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> TaskTargetConfirmResponse:
    """Re-validates and persists the task/target choice for this run.

    Args:
        body (TaskTargetConfirmRequest): Task, target, identifier columns.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        TaskTargetConfirmResponse: The confirmed task and target.

    Raises:
        HTTPException: 400, if Step 1 has not been confirmed or the target fails
            validation.
    """
    try:
        return confirm_task_target(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
