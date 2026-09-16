"""Shared run-lifecycle helpers — run id/directory generation and log capture —
used by both PipelineRunner (training) and ScoringRunner (batch scoring) so both
get the identical, audited run-folder convention.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dscompanion.config import settings

__all__ = ["generate_run_id_and_dir", "attach_run_log_handler", "detach_run_log_handler"]


def generate_run_id_and_dir(output_dir: str | Path) -> tuple[str, Path]:
    """Generate a fresh, collision-safe run id and its output directory.

    Uses ``yyyymmdd_hhmmss`` rather than a random uuid — sortable
    chronologically in a Workspace file browser, and directly readable as
    "when did this run happen" without opening it. Timestamped in
    ``settings.run_id_timezone`` (defaults to IST) rather than the running
    machine's local time, so run folder names read consistently whether the
    pipeline runs on a local laptop or a UTC-default cluster. Only appends a
    short random disambiguating suffix in the rare case two runs start in
    the same second against the same ``output_dir`` (checked via directory
    existence) — the common case stays a clean, readable id.

    Args:
        output_dir (str | Path): Root directory this run's output lives
            under.

    Returns:
        tuple[str, Path]: ``(run_id, run_dir)`` — ``run_dir`` is always
        ``Path(output_dir) / run_id`` and is guaranteed not to already exist
        at the time this returns.

    Raises:
        ZoneInfoNotFoundError: If ``settings.run_id_timezone`` is not a
            valid IANA timezone name.
    """
    output_dir = Path(output_dir)
    run_id = datetime.now(ZoneInfo(settings.run_id_timezone)).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    if run_dir.exists():
        run_id = f"{run_id}_{uuid.uuid4().hex[:4]}"
        run_dir = output_dir / run_id
    return run_id, run_dir


def attach_run_log_handler(log_path: Path) -> tuple[logging.Handler, int]:
    """Attach a ``FileHandler`` to the ``dscompanion`` logger for the duration of a run.

    Also guarantees the ``dscompanion`` logger's effective level is INFO for
    the run's duration — confirmed on Databricks 2026-09-10 that a notebook
    host can pre-install its own root logger handler *before* ``import
    dscompanion`` runs, which silently makes ``dscompanion/__init__.py``'s
    own ``logging.basicConfig(level=INFO)`` a no-op (per Python's own docs:
    ``basicConfig()`` does nothing if the root logger already has
    handlers). Left unguarded, ``dscompanion``'s effective level then falls
    back to root's default ``WARNING``, so every ``logger.info(...)`` call
    in ``dscompanion`` is filtered out *before a ``LogRecord`` is even
    created* — no handler, however attached, can capture what was never
    created. Only raises the level when it's currently coarser than INFO;
    never narrows an existing, more verbose setting (e.g. a
    caller-configured DEBUG).

    Args:
        log_path (Path): Destination file — parent directory must already
            exist.

    Returns:
        tuple[logging.Handler, int]: The attached handler and the
        ``dscompanion`` logger's level *before* this call — pass both to
        ``detach_run_log_handler`` when the run finishes.
    """
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    dscompanion_logger = logging.getLogger("dscompanion")
    dscompanion_logger.addHandler(handler)
    prev_level = dscompanion_logger.level
    if dscompanion_logger.getEffectiveLevel() > logging.INFO:
        dscompanion_logger.setLevel(logging.INFO)
    return handler, prev_level


def detach_run_log_handler(handler: logging.Handler, prev_level: int) -> None:
    """Remove a handler and restore the level ``attach_run_log_handler`` set.

    Call from a ``finally`` block so both are undone even when a stage
    raises — never leaks across multiple runs in the same process/notebook
    session.

    Args:
        handler (logging.Handler): The handler to remove.
        prev_level (int): The level returned by ``attach_run_log_handler``,
            to restore.

    Returns:
        None
    """
    dscompanion_logger = logging.getLogger("dscompanion")
    dscompanion_logger.removeHandler(handler)
    dscompanion_logger.setLevel(prev_level)
    handler.close()
