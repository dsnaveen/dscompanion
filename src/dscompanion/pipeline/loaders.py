"""Shared raw-file loading helpers, used by both PipelineRunner (training) and
ScoringRunner (batch scoring) — one implementation of "read this path/format into
a DataFrame" for both.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["load_raw_data"]


def load_raw_data(
    path: str, fmt: str, sheet_name: str | int | list[str | int] | None = None
) -> pd.DataFrame:
    """Load a single dataset from ``path`` in the given format.

    Args:
        path (str): Path to the data file or Delta table.
        fmt (str): One of ``"parquet"``, ``"csv"``, ``"excel"``, ``"delta"``.
        sheet_name (str | int | list[str | int] | None): Only used when
            ``fmt="excel"``; ignored otherwise. See ``_load_excel``.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        ValueError: If ``fmt`` is not one of the supported formats.
        RuntimeError: If loading fails for any reason (wraps the underlying
            exception with the path/format for context).
    """
    try:
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "csv":
            return pd.read_csv(path)
        if fmt == "excel":
            return _load_excel(path, sheet_name)
        if fmt == "delta":
            return _load_delta(path)
        raise ValueError(f"Unsupported format: {fmt!r}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load data from {path!r}: {exc}") from exc


def _load_excel(path: str, sheet_name: str | int | list[str | int] | None) -> pd.DataFrame:
    """Load one Excel file, optionally combining several named/indexed sheets.

    Args:
        path (str): Path to the ``.xlsx``/``.xls`` file.
        sheet_name (str | int | list[str | int] | None): A single sheet reads
            directly; a list reads each sheet and vertically concatenates
            them (every sheet must have identical columns); ``None`` reads
            the first sheet (index ``0``), not every sheet in the workbook —
            matching every other ``fmt`` here reading exactly one dataset
            from one ``path``.

    Returns:
        pd.DataFrame: The loaded (and, for a list of sheets, concatenated)
        data.

    Raises:
        ValueError: If a list of sheets is given and their column sets don't
            all match.
    """
    result = pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    if isinstance(result, dict):
        frames = list(result.items())
        first_name, first_df = frames[0]
        for name, df in frames[1:]:
            if set(df.columns) != set(first_df.columns):
                raise ValueError(
                    f"sheet_name list requires identical columns across sheets — "
                    f"sheet {name!r} has columns {sorted(df.columns)}, but sheet "
                    f"{first_name!r} has {sorted(first_df.columns)}."
                )
        return pd.concat([df for _, df in frames], ignore_index=True)
    return result


def _load_delta(path: str) -> pd.DataFrame:
    """Load a Delta table via an active SparkSession and convert to pandas.

    Args:
        path (str): Path to the Delta table.

    Returns:
        pd.DataFrame: The loaded data, via ``spark.read.format("delta").load(path).toPandas()``.

    Raises:
        RuntimeError: If PySpark isn't importable, or no active SparkSession
            is found.
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found.")
        return spark.read.format("delta").load(path).toPandas()
    except ImportError:
        raise RuntimeError(
            "format='delta' requires PySpark. Use format='parquet' for local development."
        )
