"""Step 3 (Split) endpoints — thin HTTP wrapper around ``dscompanion.api.services.split``."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    SplitConfirmRequest,
    SplitConfirmResponse,
    SplitPreviewRequest,
    SplitPreviewResponse,
)
from dscompanion.api.services.split import confirm_split, preview_split
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/split", tags=["split"], dependencies=[Depends(get_auth)]
)

__all__ = ["router"]


@router.post("/preview", response_model=SplitPreviewResponse)
def preview(
    body: SplitPreviewRequest, run: RunState = Depends(get_pipeline_service)
) -> SplitPreviewResponse:
    """Computes a split preview without persisting anything.

    Args:
        body (SplitPreviewRequest): Split method and parameters.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        SplitPreviewResponse: Row counts and event rates per partition.

    Raises:
        HTTPException: 400, if Steps 1/2 have not been confirmed or the split
            configuration is invalid.
    """
    try:
        return preview_split(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=SplitConfirmResponse)
def confirm(
    body: SplitConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> SplitConfirmResponse:
    """Recomputes and persists the split for this run.

    Args:
        body (SplitConfirmRequest): Split method and parameters.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        SplitConfirmResponse: Row counts per partition.

    Raises:
        HTTPException: 400, if Steps 1/2 have not been confirmed or the split
            configuration is invalid.
    """
    try:
        return confirm_split(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
