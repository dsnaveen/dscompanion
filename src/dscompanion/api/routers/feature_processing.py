"""Step 5 (Feature Processing) endpoints — thin HTTP wrapper around
``dscompanion.api.services.feature_processing``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    FeatureProcessingConfirmRequest,
    FeatureProcessingConfirmResponse,
    FeatureProcessingPreviewResponse,
    FeatureProcessingTransformPreviewRequest,
    FeatureProcessingTransformPreviewResponse,
)
from dscompanion.api.services.feature_processing import (
    confirm_feature_processing,
    preview_feature_processing,
    preview_transform,
)
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/feature_processing",
    tags=["feature_processing"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=FeatureProcessingPreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> FeatureProcessingPreviewResponse:
    """Computes every feature column's recommended recipe, without persisting anything.

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        FeatureProcessingPreviewResponse: The full candidate list.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed for this run.
    """
    try:
        return preview_feature_processing(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/transform-preview", response_model=FeatureProcessingTransformPreviewResponse)
def transform_preview(
    body: FeatureProcessingTransformPreviewRequest, run: RunState = Depends(get_pipeline_service)
) -> FeatureProcessingTransformPreviewResponse:
    """Fits a real transform chain on the client's current accepted recipes and returns
    a before/after sample.

    Args:
        body (FeatureProcessingTransformPreviewRequest): Current accepted recipes.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        FeatureProcessingTransformPreviewResponse: Before/after sample rows.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed, or the recipes fail
            to fit.
    """
    try:
        return preview_transform(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=FeatureProcessingConfirmResponse)
def confirm(
    body: FeatureProcessingConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> FeatureProcessingConfirmResponse:
    """Persists the final recipe/drop decisions for this run.

    Args:
        body (FeatureProcessingConfirmRequest): Final accepted recipes and dropped
            columns.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        FeatureProcessingConfirmResponse: Confirmation and accepted/dropped/rejected
        counts.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed for this run.
    """
    try:
        return confirm_feature_processing(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
