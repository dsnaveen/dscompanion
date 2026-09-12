"""Data-file listing endpoint — not run-scoped, thin wrapper around
``dscompanion.api.services.data_files``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from dscompanion.api.deps import get_auth
from dscompanion.api.schemas import DataFilesResponse
from dscompanion.api.services.data_files import list_data_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data-files", tags=["data-files"], dependencies=[Depends(get_auth)])

__all__ = ["router"]


@router.get("", response_model=DataFilesResponse)
def get_data_files() -> DataFilesResponse:
    """Lists server-side data files available for Step 1.

    Args:
        None

    Returns:
        DataFilesResponse: Sorted relative paths of available data files.
    """
    return list_data_files()
