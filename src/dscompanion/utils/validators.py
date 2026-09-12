"""Common validation helpers."""

from __future__ import annotations

import pandas as pd

__all__ = [
    "validate_dataframe",
    "validate_binary_target",
    "is_databricks",
]


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str = "DataFrame",
) -> None:
    """Verify that a DataFrame contains all required columns, raising a
    descriptive error that lists every missing column name if any are absent,
    so callers receive an actionable message rather than a downstream
    ``KeyError``.

    Args:
        df (pd.DataFrame): The DataFrame whose columns are checked.
        required_cols (list[str]): Ordered list of column names that must be
            present in ``df.columns``.  An empty list causes an immediate
            return without error.
        name (str): Human-readable label for the DataFrame used in the error
            message (e.g. ``"training set"`` or ``"X_val"``).  Defaults to
            ``"DataFrame"``.

    Returns:
        None: Returns ``None`` silently when all required columns are
        present.

    Raises:
        ValueError: When one or more column names from ``required_cols``
            are not found in ``df.columns``.  The message includes the
            ``name`` label and the complete list of missing column names.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("%s must be a pd.DataFrame, got %s" % (name, type(df).__name__))
    if df.empty:
        raise ValueError("%s is empty — must contain at least one row" % name)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (name, missing))


def validate_binary_target(y: pd.Series) -> None:
    """Verify that a target Series contains exactly two distinct non-null
    values, which is a precondition for all binary classification models
    in dscompanion; raises an informative error listing the observed unique
    values if the check fails.

    ``NaN`` values are dropped before counting unique values, so a series
    containing ``{0, 1, NaN}`` is considered valid (two unique non-null
    values: ``0`` and ``1``).

    Args:
        y (pd.Series): The target variable to validate.  Expected to hold
            integer or float values ``{0, 1}``, but any dtype with exactly
            two unique non-null values is accepted.

    Returns:
        None: Returns ``None`` silently when the target is binary.

    Raises:
        ValueError: When the number of unique non-null values is not exactly
            two.  The message includes the sorted list of observed unique
            values to aid debugging.
    """
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pd.Series, got %s" % type(y).__name__)
    if len(y) == 0:
        raise ValueError("y is empty — must contain at least one element")
    unique = set(y.dropna().unique())
    if len(unique) != 2:
        raise ValueError(f"Binary target must have exactly 2 unique values; found {sorted(unique)}")


def is_databricks() -> bool:
    """Detect whether the current process is running inside a Databricks
    notebook environment by inspecting the active IPython shell type and,
    as a fallback, querying the active Spark session's cluster name tag.

    No network calls or file-system reads are made; detection is based
    solely on in-process Python object inspection.  All exceptions are
    swallowed and result in ``False`` being returned, so this function is
    safe to call in any environment.

    Args:
        None

    Returns:
        bool: ``True`` when an active IPython shell whose type string
        contains ``"databricks"`` is detected, or when the active Spark
        session's ``spark.databricks.clusterUsageTags.clusterName``
        configuration key contains ``"databricks"``.  Returns ``False``
        in all other cases, including when IPython or PySpark is not
        installed.
    """
    try:
        # dbutils is injected into the Databricks notebook namespace
        import IPython

        ip = IPython.get_ipython()
        if ip is None:
            return False
        return "databricks" in str(type(ip)).lower() or _check_spark_conf()
    except Exception:
        return False


def _check_spark_conf() -> bool:
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            return False
        return (
            "databricks"
            in spark.conf.get("spark.databricks.clusterUsageTags.clusterName", "").lower()
        )
    except Exception:
        return False
