"""Local, logging-based run tracking for dscompanion experiments.

Replaces the earlier mlflow-backed tracker. Every signal a real tracking
server would have recorded — run start/finish, metrics, hyperparameters,
artifact paths, tags — is now a single structured `logger.info` line
instead of a network call. There is no queryable run store: this module
only ever writes to whatever log handler the caller has configured. A
caller that needs to browse past runs needs a different mechanism (e.g.
a log aggregator, or a purpose-built local run-store) — this module
does not provide one.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["tracking_run", "log_metrics", "log_params", "log_artifact"]


@contextmanager
def tracking_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    enabled: bool = True,
    run_id: str | None = None,
) -> Generator[Any, None, None]:
    """Context manager that logs a run's start and finish via stdlib logging.

    Unlike a real tracking server, this never fails and never blocks on a
    network call — it always yields a handle, logs one INFO line on entry
    and one on exit, and calling code can log metrics/params/artifacts
    freely inside the block without any connection state to manage.

    Args:
        run_name (str): Human-readable label for the run, included in the
            start/finish log lines.
        tags (dict, optional): Arbitrary string key/value pairs logged
            alongside the run name at start. Defaults to ``None`` (no
            tags).
        enabled (bool): When ``False``, skips logging entirely — no log
            line at any level, immediately yields a ``_NoopRun()``. For
            callers that want silence (e.g. a scratch run, or a test).
            Defaults to ``True``.
        run_id (str, optional): Use this exact id instead of generating a
            fresh one. For callers (e.g. ``PipelineRunner``) that need the
            same id known before this context manager opens — such as
            naming an output directory after it. Defaults to ``None``
            (generates a fresh ``uuid.uuid4().hex``, unchanged behavior).

    Yields:
        RunHandle or _NoopRun: A handle exposing ``info.run_id`` (``run_id``
        verbatim when supplied, else a fresh ``uuid.uuid4().hex`` string)
        and ``info.experiment_id`` (always ``"0"`` — there is no
        experiment/namespace concept without a tracking server). ``_NoopRun``
        is yielded instead when ``enabled=False``, with the same attribute
        shape so callers never need to branch on type.

    Example::

        with tracking_run("my_model_v1", tags={"owner": "your_name"}) as run:
            log_params({"algorithm": "xgboost", "max_depth": 5})
            log_metrics({"roc_auc": 0.82})
    """
    if not enabled:
        yield _NoopRun()
        return

    run = RunHandle(run_id=run_id if run_id is not None else uuid.uuid4().hex)
    logger.info("Run started — %s (%s) tags=%s", run.info.run_id[:8], run_name, tags or {})
    try:
        yield run
    finally:
        logger.info("Run finished — %s", run.info.run_id[:8])


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of floating-point metrics, one INFO line per key.

    Args:
        metrics (dict): Mapping of metric name to numeric value.
        step (int, optional): Step index for time-series logging (e.g.
            epoch number), included in each log line. Defaults to
            ``None``.

    Returns:
        None: Always returns ``None``. Never raises — an empty ``metrics``
        dict simply logs nothing.
    """
    for key, value in metrics.items():
        logger.info("metric %s=%s step=%s", key, value, step)


def log_params(params: dict[str, Any]) -> None:
    """Log a dictionary of hyperparameters or configuration values as one line.

    Non-scalar values (lists, dicts, objects) are coerced to ``str()``
    before logging, matching the earlier mlflow-backed behavior — keeps
    the log line readable rather than dumping a repr of a nested object.

    Args:
        params (dict): Mapping of parameter name to value. Scalar types
            (``int``, ``float``, ``str``, ``bool``) are logged as-is;
            everything else is coerced to ``str``.

    Returns:
        None: Always returns ``None``.
    """
    safe = {k: v if isinstance(v, (int, float, str, bool)) else str(v) for k, v in params.items()}
    logger.info("params %s", safe)


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """Log that a local file was produced, and where it conceptually belongs.

    This does not copy or upload anything — the file already lives at
    ``local_path`` on disk. It only records, via logging, that it was
    produced and its intended logical location, mirroring what an
    artifact store's directory structure would have been.

    Args:
        local_path (str): Path to the local file that was written.
        artifact_path (str, optional): Logical subdirectory the artifact
            would have been filed under. Defaults to ``None`` (root).

    Returns:
        None: Always returns ``None``.
    """
    logger.info("artifact %s -> %s", local_path, artifact_path or "(root)")


class RunHandle:
    """Handle for an active local run — mirrors the shape callers expect.

    Represents a single completed or in-progress local experiment run,
    providing a simple object that callers can inspect to retrieve the
    run's unique identifier and experiment scope. Used as the yield value
    of ``tracking_run()`` when tracking is enabled.

    Attributes:
        info (object): Container with run metadata:
            - run_id (str): Fresh ``uuid.uuid4().hex`` string (32 characters)
              identifying this run uniquely.
            - experiment_id (str): Always ``"0"`` — there is no multi-experiment
              namespace in local tracking. Included for API compatibility.
    """

    def __init__(self, run_id: str) -> None:
        """Initialize a RunHandle with a fresh run ID.

        Args:
            run_id (str): The fresh ``uuid.uuid4().hex`` string uniquely
                identifying this run.

        Returns:
            None
        """

        class _Info:
            pass

        self.info = _Info()
        self.info.run_id = run_id
        self.info.experiment_id = "0"


class _NoopRun:
    """Stub returned when tracking is disabled via ``enabled=False``.

    When ``tracking_run(..., enabled=False)`` is invoked, this object is
    yielded instead of a ``RunHandle``. It has the same attribute shape
    (``info.run_id``, ``info.experiment_id``) so calling code never needs
    to branch on type. All values are fixed to indicate a disabled/noop
    tracking session.

    Attributes:
        info (class): Container with fixed run metadata:
            - run_id (str): Always ``"noop-run-id"`` — a sentinel indicating
              tracking was disabled.
            - experiment_id (str): Always ``"0"``, matching the shape of
              ``RunHandle`` for API compatibility.
    """

    class info:
        run_id = "noop-run-id"
        experiment_id = "0"
