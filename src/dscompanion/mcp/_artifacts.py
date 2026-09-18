"""Shared run-directory and split-persistence helpers for MCP tool calls.

Every MCP tool call is stateless, so state that must survive between calls
(a trained model, the exact DataSplit it was trained/evaluated on) is passed
via files under a per-call run directory rather than in-memory objects.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import joblib

from dscompanion.config import settings

logger = logging.getLogger(__name__)

__all__ = ["new_run_dir", "save_split", "load_split"]


def new_run_dir() -> Path:
    """Create and return a new, uniquely-named run directory for one MCP tool call.

    Args:
        None

    Returns:
        pathlib.Path: The newly-created, empty run directory, nested under
        ``settings.mcp_output_dir``.
    """
    base = Path(settings.mcp_output_dir)
    run_dir = base / f"run_{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created MCP run directory: %s", run_dir)
    return run_dir


def save_split(split: Any, path: str | Path) -> Path:
    """Persist a DataSplit to a joblib file so later tool calls can reuse the exact
    same train/held-out partitions -- avoids re-splitting the source data (which
    could leak rows the model already trained on into what a later call treats as
    held-out).

    Args:
        split (Any): A ``DataSplit`` instance (``dscompanion.split.DataSplit``).
        path (str | Path): Destination file path; parent directories are created
            automatically if needed.

    Returns:
        pathlib.Path: Resolved path of the file that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(split, path)
    return path


def load_split(path: str | Path) -> Any:
    """Load a DataSplit previously saved with ``save_split``.

    Args:
        path (str | Path): Path to an existing joblib file produced by
            ``save_split``.

    Returns:
        Any: The deserialised ``DataSplit`` instance.
    """
    return joblib.load(path)
