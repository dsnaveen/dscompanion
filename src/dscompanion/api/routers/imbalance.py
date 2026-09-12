"""Step 7 (Class Imbalance Handling) endpoints — thin HTTP wrapper around
``dscompanion.api.services.imbalance``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    ImbalanceConfirmRequest,
    ImbalanceConfirmResponse,
    ImbalancePreviewResponse,
)
from dscompanion.api.services.imbalance import confirm_imbalance, preview_imbalance
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/imbalance",
    tags=["imbalance"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=ImbalancePreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> ImbalancePreviewResponse:
    """Reports the confirmed training split's class balance.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        ImbalancePreviewResponse: Class balance and whether it's roughly even.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed for this run.
    """
    try:
        return preview_imbalance(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=ImbalanceConfirmResponse)
def confirm(
    body: ImbalanceConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> ImbalanceConfirmResponse:
    """Persists the confirmed imbalance strategy for this run.

    Args:
        body (ImbalanceConfirmRequest): The chosen strategy.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        ImbalanceConfirmResponse: Confirmation and the row-count preview.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed for this run.
    """
    try:
        return confirm_imbalance(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
