"""Step 4 (Explore the Data / EDA) endpoints — thin HTTP wrapper around
``dscompanion.api.services.eda``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.schemas import (
    EdaColumnChartRequest,
    EdaColumnChartResponse,
    EdaConfirmRequest,
    EdaConfirmResponse,
    EdaPreviewResponse,
)
from dscompanion.api.services.eda import confirm_eda, get_column_chart, preview_eda
from dscompanion.api.state import RunState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps/eda", tags=["eda"], dependencies=[Depends(get_auth)]
)

__all__ = ["router"]


@router.post("/preview", response_model=EdaPreviewResponse)
def preview(run: RunState = Depends(get_pipeline_service)) -> EdaPreviewResponse:
    """Runs the full data quality check without persisting anything (though the fitted
    report is cached server-side for the column-chart endpoint below).

    Args:
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        EdaPreviewResponse: Overview, missing-values, univariate, bivariate, and
        multivariate results.

    Raises:
        HTTPException: 400, if Steps 1-3 have not been confirmed for this run.
    """
    try:
        return preview_eda(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/column-chart", response_model=EdaColumnChartResponse)
def column_chart(
    body: EdaColumnChartRequest, run: RunState = Depends(get_pipeline_service)
) -> EdaColumnChartResponse:
    """Builds the chart for one column/feature, on demand.

    Args:
        body (EdaColumnChartRequest): Chart kind and column/feature name.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        EdaColumnChartResponse: The requested chart.

    Raises:
        HTTPException: 400, if preview hasn't been called yet or the column is invalid.
    """
    try:
        return get_column_chart(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/confirm", response_model=EdaConfirmResponse)
def confirm(
    body: EdaConfirmRequest, run: RunState = Depends(get_pipeline_service)
) -> EdaConfirmResponse:
    """Marks Step 4 confirmed — a flag-flip plus audit entries, no data mutation.

    Args:
        body (EdaConfirmRequest): Whether the check was skipped entirely.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        EdaConfirmResponse: Confirmation and resulting step status.

    Raises:
        HTTPException: 400, if not skipped and preview hasn't been called yet.
    """
    try:
        return confirm_eda(run, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
