"""Lists server-side data files available for Step 1 — the REST equivalent of
``dscompanion/app/interactive_step1.py``'s ``_list_data_files()``, with the ``streamlit`` cache
decorator and UI calls stripped out.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dscompanion.api.schemas import DataFilesResponse

logger = logging.getLogger(__name__)

__all__ = ["list_data_files"]

_DATA_ROOT = Path(__file__).resolve().parents[4] / "data"
_SUPPORTED_SUFFIXES = (".csv", ".parquet", ".xlsx")


def list_data_files() -> DataFilesResponse:
    """Lists every supported data file under the repo's ``data/`` directory.

    Args:
        None

    Returns:
        DataFilesResponse: Sorted relative paths of every ``.csv``/``.parquet``/``.xlsx``
        file found. Empty when the data root doesn't exist.
    """
    if not _DATA_ROOT.exists():
        logger.warning("Data root does not exist: %s", _DATA_ROOT)
        return DataFilesResponse(files=[])

    files: list[Path] = []
    for suffix in _SUPPORTED_SUFFIXES:
        files.extend(_DATA_ROOT.rglob(f"*{suffix}"))

    relative = sorted(str(f.relative_to(_DATA_ROOT)) for f in files)
    logger.info("Listed %d data file(s) under %s", len(relative), _DATA_ROOT)
    return DataFilesResponse(files=relative)
