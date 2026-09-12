"""Step 11 (Calibration) endpoints — thin HTTP wrapper around
``dscompanion.api.services.calibration``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    CalibrationConfirmRequest,
    CalibrationConfirmResponse,
    CalibrationPreviewResponse,
)
from dscompanion.api.services.calibration import confirm_calibration, preview_calibration
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/calibration",
    tags=["calibration"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=CalibrationPreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> CalibrationPreviewResponse:
    """Fits calibration on the final model and reports ECE before/after.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        CalibrationPreviewResponse: Availability, ECE before/after, and the
        apply/skip recommendation.

    Raises:
        HTTPException: 400, if Step 9 has not been confirmed, or no
            held-out split is available, for this run.
    """
    try:
        return preview_calibration(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=CalibrationConfirmResponse)
def confirm(
    body: CalibrationConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> CalibrationConfirmResponse:
    """Applies or skips calibration and persists ``calibrated_model``.

    Args:
        body (CalibrationConfirmRequest): Whether to apply calibration.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        CalibrationConfirmResponse: Confirmation and whether calibration was
        actually applied.

    Raises:
        HTTPException: 400, if Step 9 has not been confirmed, or no
            held-out split is available, for this run.
    """
    try:
        return confirm_calibration(run, body.apply)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
