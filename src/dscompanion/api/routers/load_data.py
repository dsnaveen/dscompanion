"""Step 1 (Load Data) endpoints — thin HTTP wrapper around
``dscompanion.api.services.load_data``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    LoadDataConfirmRequest,
    LoadDataConfirmResponse,
    LoadDataPreviewRequest,
    LoadDataPreviewResponse,
)
from dscompanion.api.services.load_data import confirm_load_data, preview_load_data
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/load_data",
    tags=["load_data"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.post("/preview", response_model=LoadDataPreviewResponse)
def preview(
    body: LoadDataPreviewRequest, run: RunState = Depends(get_pipeline_service)
) -> LoadDataPreviewResponse:
    """Loads and previews a dataset without persisting anything.

    Args:
        body (LoadDataPreviewRequest): Path, format, and sheet name to load.
        run (RunState): Injected via ``get_pipeline_service`` (unused — preview does
            not touch the run — but required so an unknown ``run_id`` still 404s).

    Returns:
        LoadDataPreviewResponse: Columns, dtypes, row count, duplicate columns.

    Raises:
        HTTPException: 400, if the file can't be read.
    """
    try:
        return preview_load_data(body)
    except Exception as exc:
        logger.warning("Step 1 preview failed: %s", type(exc).__name__)
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=LoadDataConfirmResponse)
def confirm(
    body: LoadDataConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> LoadDataConfirmResponse:
    """Loads a dataset, applies duplicate-column resolutions, and persists it as this
    run's working dataframe.

    Args:
        body (LoadDataConfirmRequest): Path, format, sheet name, duplicate
            resolutions.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        LoadDataConfirmResponse: Final columns and row count.

    Raises:
        HTTPException: 400, if the file can't be read or resolutions are invalid.
    """
    try:
        return confirm_load_data(run, body)
    except Exception as exc:
        logger.warning("Step 1 confirm failed: %s", type(exc).__name__)
        raise HTTPException(status_code=400, detail=str(exc))
