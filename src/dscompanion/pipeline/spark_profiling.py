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
from pathlib import Path

import pandas as pd

from dscompanion.config import settings
from dscompanion.pipeline.loaders import _get_spark_dataframe, _sample_spark_dataframe

logger = logging.getLogger(__name__)

__all__ = ["spark_profiling_available", "profile_and_sample"]


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


def profile_and_sample(path: str, fmt: str, output_dir: str | Path) -> pd.DataFrame:
    """Profile the full dataset natively in Spark, save the report, and return a
    pandas-sized sample for the rest of the pipeline.

    Callers should check ``spark_profiling_available()`` before calling this --
    it assumes PySpark, an active SparkSession, and spark-data-profiler are all
    already available, and does not re-check or gracefully degrade itself.

    Args:
        path (str): Path to the Delta table or parquet source.
        fmt (str): One of ``"delta"``, ``"parquet"``.
        output_dir (str | Path): Directory the Spark profile report
            (``spark_profile.html``/``.json``) is written into; created if it
            doesn't already exist.

    Returns:
        pd.DataFrame: A pandas sample of at most
        ``settings.spark_profiling_sample_rows`` rows, drawn from the full Spark
        DataFrame.

    Raises:
        RuntimeError: If PySpark/spark-data-profiler aren't actually available, or
            ``fmt`` isn't ``"delta"``/``"parquet"`` -- see ``_get_spark_dataframe``.
    """
    from spark_data_profiler import SparkDataProfiler

    spark_df = _get_spark_dataframe(path, fmt, context="use_spark_profiling=True")

    profiler = SparkDataProfiler(
        spark_df,
        missing_threshold=settings.high_missing_threshold * 100,
        skew_threshold=settings.eda_skewness_alert_threshold,
        imbalance_threshold=settings.eda_imbalance_alert_threshold,
        corr_reject_threshold=settings.correlation_threshold,
    )
    profiler.profile()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler.to_html(str(output_dir / "spark_profile.html"))
    profiler.to_json(str(output_dir / "spark_profile.json"))
    logger.info("Spark profile report written to %s", output_dir)

    sampled_spark_df = _sample_spark_dataframe(
        spark_df, nrows=settings.spark_profiling_sample_rows, fraction_rows=None
    )
    return sampled_spark_df.toPandas()
