"""Shared raw-file loading helpers, used by both PipelineRunner (training) and
ScoringRunner (batch scoring) — one implementation of "read this path/format into
a DataFrame" for both.
"""

from __future__ import annotations

import pandas as pd

from dscompanion.config import settings

__all__ = ["load_raw_data"]


def load_raw_data(
    path: str,
    fmt: str,
    sheet_name: str | int | list[str | int] | None = None,
    nrows: int | None = None,
    fraction_rows: float | None = None,
) -> pd.DataFrame:
    """Load a single dataset from ``path`` in the given format, with optional
    dev-only row sampling.

    ``nrows``/``fraction_rows`` are random (not "first N"), seeded via
    ``settings.random_state``, and mutually exclusive (enforced upstream by
    ``DataConfig``, not re-checked here). For ``fmt="delta"``, sampling is
    pushed into Spark *before* ``.toPandas()`` — Delta is the only format
    with a distributed engine sitting in front of the driver, so this is the
    only case where a dev-only row-count guard can actually reduce driver
    memory pressure, rather than collecting the full table to the driver
    first and discarding most of it afterward. See ``_load_delta`` for the
    exact sampling strategy and its precision caveats. For ``"parquet"``/
    ``"csv"``/``"excel"``, sampling happens after the (necessarily full)
    pandas load — there's no distributed layer to push it into.

    Args:
        path (str): Path to the data file or Delta table.
        fmt (str): One of ``"parquet"``, ``"csv"``, ``"excel"``, ``"delta"``.
        sheet_name (str | int | list[str | int] | None): Only used when
            ``fmt="excel"``; ignored otherwise. See ``_load_excel``.
        nrows (int | None): If set, a random sample of this many rows (or
            the full dataset if it has fewer).
        fraction_rows (float | None): If set, a random fraction of the full
            dataset.

    Returns:
        pd.DataFrame: The loaded (and, if requested, sampled) data.

    Raises:
        ValueError: If ``fmt`` is not one of the supported formats.
        RuntimeError: If loading fails for any reason (wraps the underlying
            exception with the path/format for context).
    """
    try:
        if fmt == "delta":
            return _load_delta(path, nrows=nrows, fraction_rows=fraction_rows)

        if fmt == "parquet":
            df = pd.read_parquet(path)
        elif fmt == "csv":
            df = pd.read_csv(path)
        elif fmt == "excel":
            df = _load_excel(path, sheet_name)
        else:
            raise ValueError(f"Unsupported format: {fmt!r}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load data from {path!r}: {exc}") from exc

    if nrows:
        df = df.sample(n=min(nrows, len(df)), random_state=settings.random_state)
    elif fraction_rows:
        df = df.sample(frac=fraction_rows, random_state=settings.random_state)
    return df


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


def _load_delta(
    path: str,
    nrows: int | None = None,
    fraction_rows: float | None = None,
) -> pd.DataFrame:
    """Load a Delta table via an active SparkSession and convert to pandas.

    When ``nrows``/``fraction_rows`` is set, sampling happens in Spark
    *before* ``.toPandas()`` — Delta is the only format with a distributed
    engine sitting in front of the driver, so this is the only place a
    dev-only row-count guard can actually reduce driver memory pressure,
    rather than collecting the full table first and discarding most of it
    afterward.

    ``fraction_rows`` uses Spark's own probabilistic ``.sample(fraction=...)``
    — unlike pandas' exact-count ``.sample(frac=...)``, the returned row
    count is only approximately ``fraction_rows * total_rows``, not exact.
    ``nrows`` uses an estimate-and-oversample strategy (a ``.count()`` plus a
    Bernoulli sample with a 10% safety margin, trimmed to exactly ``nrows``
    via ``.limit()``) rather than a full ``orderBy(rand()).limit(n)``
    shuffle, to avoid sorting the entire table just to draw a small sample —
    for a very small ``nrows`` relative to the table size, an unlucky
    Bernoulli draw can occasionally undershoot the margin and return
    slightly fewer than ``nrows`` rows.

    Args:
        path (str): Path to the Delta table.
        nrows (int | None): If set, a random sample of (very likely, given
            the safety margin, but not strictly guaranteed) this many rows,
            or the full table if it has fewer.
        fraction_rows (float | None): If set, an approximate random fraction
            of the full table.

    Returns:
        pd.DataFrame: The loaded (and, if requested, sampled) data.

    Raises:
        RuntimeError: If PySpark isn't importable, or no active SparkSession
            is found.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise RuntimeError(
            "format='delta' requires PySpark. Use format='parquet' for local development."
        )

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found.")

    df = spark.read.format("delta").load(path)

    if nrows:
        total = df.count()
        if nrows < total:
            oversample_fraction = min(1.0, (nrows / total) * 1.1)
            df = df.sample(fraction=oversample_fraction, seed=settings.random_state).limit(nrows)
    elif fraction_rows:
        df = df.sample(fraction=fraction_rows, seed=settings.random_state)

    return df.toPandas()
