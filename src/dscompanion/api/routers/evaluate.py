"""Step 10 (Evaluate) endpoints — thin HTTP wrapper around
``dscompanion.api.services.evaluate``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import EvaluateConfirmResponse, EvaluatePreviewResponse
from dscompanion.api.services.evaluate import confirm_evaluate, preview_evaluate
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/evaluate",
    tags=["evaluate"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=EvaluatePreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> EvaluatePreviewResponse:
    """Evaluates the final model from Step 9 and reports the full breakdown.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        EvaluatePreviewResponse: Full per-split metric breakdown, the
        tuned-vs-default baseline, and any suspicious-pattern flags.

    Raises:
        HTTPException: 400, if Step 9 has not been confirmed for this run.
    """
    try:
        return preview_evaluate(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=EvaluateConfirmResponse)
def confirm(run: RunState = Depends(get_pipeline_service)) -> EvaluateConfirmResponse:
    """Records Step 10 completion — a flag-flip only, no persisted user choice.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        EvaluateConfirmResponse: Confirmation.

    Raises:
        HTTPException: 400, if Step 9 has not been confirmed for this run.
    """
    try:
        return confirm_evaluate(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
