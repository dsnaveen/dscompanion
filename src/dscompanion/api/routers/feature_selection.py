"""Step 6 (Feature Selection) endpoints — thin HTTP wrapper around
``dscompanion.api.services.feature_selection``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    FeatureSelectionConfirmRequest,
    FeatureSelectionConfirmResponse,
    FeatureSelectionPreviewResponse,
)
from dscompanion.api.services.feature_selection import (
    confirm_feature_selection,
    preview_feature_selection,
)
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/feature_selection",
    tags=["feature_selection"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=FeatureSelectionPreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> FeatureSelectionPreviewResponse:
    """Runs feature selection read-only and reports the proposed drop-list.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        FeatureSelectionPreviewResponse: Audit table and removal waterfall chart.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed, or selection itself
            fails.
    """
    try:
        return preview_feature_selection(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=FeatureSelectionConfirmResponse)
def confirm(
    body: FeatureSelectionConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> FeatureSelectionConfirmResponse:
    """Persists the final removed-columns decision for this run.

    Args:
        body (FeatureSelectionConfirmRequest): Final removed columns, or a skip flag.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        FeatureSelectionConfirmResponse: Confirmation, status, and removed columns.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed, or the removal would
            leave no features to train on.
    """
    try:
        return confirm_feature_selection(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
