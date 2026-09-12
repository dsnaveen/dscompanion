"""Step 12 (Explainability / SHAP) endpoints — thin HTTP wrapper around
``dscompanion.api.services.shap``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    ShapConfirmRequest,
    ShapConfirmResponse,
    ShapPreviewResponse,
    ShapRunResponse,
)
from dscompanion.api.services.shap import confirm_shap, preview_shap, run_shap
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/shap",
    tags=["shap"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=ShapPreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> ShapPreviewResponse:
    """Reports whether SHAP is available for this run, without computing anything.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        ShapPreviewResponse: Availability — always ``True`` today.

    Raises:
        HTTPException: 400, if Step 11 has not been confirmed for this run.
    """
    try:
        return preview_shap(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/run", response_model=ShapRunResponse)
def shap_run(run: RunState = Depends(get_pipeline_service)) -> ShapRunResponse:
    """Fits ``SHAPExplainer`` on a sample of the held-out split and returns the summary.

    Explicit opt-in the caller must trigger — never called automatically by
    ``preview``, since SHAP computation is relatively expensive.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        ShapRunResponse: The summary chart, top-10 feature table, split
        used, and row count.

    Raises:
        HTTPException: 400, if Step 11 has not been confirmed, or no
            held-out data is available, for this run.
    """
    try:
        return run_shap(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=ShapConfirmResponse)
def confirm(
    body: ShapConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> ShapConfirmResponse:
    """Records Step 12 completion.

    Args:
        body (ShapConfirmRequest): Whether SHAP was actually run.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        ShapConfirmResponse: Confirmation.
    """
    return confirm_shap(run, body.shap_run)
