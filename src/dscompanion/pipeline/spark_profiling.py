"""Optional Spark-native full-dataset profiling stage for PipelineRunner.

Runs only when a user opts in (``DataConfig.use_spark_profiling=True``) AND the
running environment actually has PySpark, an active SparkSession, and the optional
``spark-data-profiler`` package available -- both conditions must hold, checked via
``spark_profiling_available()``. Either alone does nothing: this is an opportunistic
enhancement for datasets too large to comfortably profile in pandas, not a hard
requirement, so every failure mode here falls back to the plain pandas path silently
rather than raising.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["spark_profiling_available"]


def spark_profiling_available() -> bool:
    """Check whether Spark-native profiling can actually run in this environment.

    Args:
        None

    Returns:
        bool: ``True`` only if PySpark is importable, an active ``SparkSession``
        exists, and ``spark-data-profiler`` is importable. Never raises -- every
        failure mode (``ImportError``, no active session) returns ``False`` so
        callers can fall back to the plain pandas path.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return False

    if SparkSession.getActiveSession() is None:
        return False

    try:
        import spark_data_profiler  # noqa: F401
    except ImportError:
        return False

    return True
