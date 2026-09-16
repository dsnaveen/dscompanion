"""Shared raw-file loading helpers, used by both PipelineRunner (training) and
ScoringRunner (batch scoring) — one implementation of "read this path/format into
a DataFrame" for both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dscompanion.config import settings

__all__ = ["load_raw_data"]


def load_raw_data(
    path: str,
    fmt: str,
    sheet_name: str | int | list[str | int] | None = None,
    nrows: int | None = None,
    fraction_rows: float | None = None,
    read_via_spark: bool = False,
    row_group_sample: bool = False,
) -> pd.DataFrame:
    """Load a single dataset from ``path`` in the given format, with optional
    dev-only row sampling.

    ``nrows``/``fraction_rows`` are random (not "first N"), seeded via
    ``settings.random_state``, and mutually exclusive (enforced upstream by
    ``DataConfig``, not re-checked here). Three sampling execution paths
    exist, depending on ``fmt``/flags:

    - ``fmt="delta"``: always pushed into Spark before ``.toPandas()`` — the
      only way to read Delta at all. See ``_load_delta``.
    - ``fmt="parquet"``, ``read_via_spark=True``: also pushed into Spark
      before ``.toPandas()``, via ``spark.read.parquet(path)``. See
      ``_load_parquet_via_spark``. Requires an active ``SparkSession``.
    - ``fmt="parquet"``, ``row_group_sample=True``: sampled at the parquet
      row-group level via ``pyarrow``, no Spark needed. See
      ``_load_parquet_via_row_groups`` for the precision caveats.
    - Every other case (``"parquet"`` with neither flag, ``"csv"``,
      ``"excel"``): sampling happens after the (necessarily full) pandas
      load — there's no distributed/metadata-only layer to push it into.

    ``read_via_spark``/``row_group_sample`` are mutually exclusive and only
    meaningful for ``fmt="parquet"`` (enforced upstream by ``DataConfig``,
    not re-checked here).

    Args:
        path (str): Path to the data file or Delta table.
        fmt (str): One of ``"parquet"``, ``"csv"``, ``"excel"``, ``"delta"``.
        sheet_name (str | int | list[str | int] | None): Only used when
            ``fmt="excel"``; ignored otherwise. See ``_load_excel``.
        nrows (int | None): If set, a random sample of this many rows (or
            the full dataset if it has fewer).
        fraction_rows (float | None): If set, a random fraction of the full
            dataset.
        read_via_spark (bool): ``fmt="parquet"`` only — read via Spark
            instead of pandas, pushing sampling down before ``.toPandas()``.
        row_group_sample (bool): ``fmt="parquet"`` only — sample at the
            row-group level via ``pyarrow``, no Spark needed.

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
            if read_via_spark:
                return _load_parquet_via_spark(path, nrows=nrows, fraction_rows=fraction_rows)
            if row_group_sample:
                return _load_parquet_via_row_groups(path, nrows=nrows, fraction_rows=fraction_rows)
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


def _get_active_spark_session(context: str):
    """Import PySpark and return the active SparkSession, or raise a clear error.

    Args:
        context (str): Short phrase naming the caller, used only in the
            ``ImportError`` fallback message (e.g. ``"format='delta'"``,
            ``"read_via_spark=True"``).

    Returns:
        pyspark.sql.SparkSession: The active session.

    Raises:
        RuntimeError: If PySpark isn't importable, or no active SparkSession
            is found.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise RuntimeError(f"{context} requires PySpark. Use a plain pandas format instead.")

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found.")
    return spark


def _sample_spark_dataframe(df, nrows: int | None, fraction_rows: float | None):
    """Apply ``nrows``/``fraction_rows`` sampling to a Spark DataFrame before collection.

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
        df: A Spark DataFrame.
        nrows (int | None): If set, a random sample of (very likely, given
            the safety margin, but not strictly guaranteed) this many rows,
            or the full DataFrame if it has fewer.
        fraction_rows (float | None): If set, an approximate random fraction
            of the full DataFrame.

    Returns:
        The (possibly sampled) Spark DataFrame — still lazy, not collected.
    """
    if nrows:
        total = df.count()
        if nrows < total:
            oversample_fraction = min(1.0, (nrows / total) * 1.1)
            df = df.sample(fraction=oversample_fraction, seed=settings.random_state).limit(nrows)
    elif fraction_rows:
        df = df.sample(fraction=fraction_rows, seed=settings.random_state)
    return df


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
    afterward. See ``_sample_spark_dataframe`` for the sampling strategy and
    its precision caveats.

    Args:
        path (str): Path to the Delta table.
        nrows (int | None): See ``_sample_spark_dataframe``.
        fraction_rows (float | None): See ``_sample_spark_dataframe``.

    Returns:
        pd.DataFrame: The loaded (and, if requested, sampled) data.

    Raises:
        RuntimeError: If PySpark isn't importable, or no active SparkSession
            is found.
    """
    spark = _get_active_spark_session("format='delta'")
    df = spark.read.format("delta").load(path)
    df = _sample_spark_dataframe(df, nrows, fraction_rows)
    return df.toPandas()


def _load_parquet_via_spark(
    path: str,
    nrows: int | None = None,
    fraction_rows: float | None = None,
) -> pd.DataFrame:
    """Load a parquet source via an active SparkSession and convert to pandas.

    ``DataConfig.read_via_spark=True``'s implementation — same
    memory-saving motivation and sampling strategy as ``_load_delta``, just
    for a source that doesn't strictly require Spark to read (unlike Delta,
    which needs Spark to resolve its transaction log). Useful when an active
    Spark session is already available (e.g. on Databricks) and the source
    is large enough that collecting it to the driver before sampling would
    be wasteful.

    Args:
        path (str): Path to a single parquet file or a partitioned
            directory of parquet files.
        nrows (int | None): See ``_sample_spark_dataframe``.
        fraction_rows (float | None): See ``_sample_spark_dataframe``.

    Returns:
        pd.DataFrame: The loaded (and, if requested, sampled) data.

    Raises:
        RuntimeError: If PySpark isn't importable, or no active SparkSession
            is found.
    """
    spark = _get_active_spark_session("read_via_spark=True")
    df = spark.read.parquet(path)
    df = _sample_spark_dataframe(df, nrows, fraction_rows)
    return df.toPandas()


def _load_parquet_via_row_groups(
    path: str,
    nrows: int | None = None,
    fraction_rows: float | None = None,
) -> pd.DataFrame:
    """Sample a parquet source at the row-group level via ``pyarrow``, no Spark needed.

    ``DataConfig.row_group_sample=True``'s implementation. Reads only the
    parquet footer metadata to discover row groups across however many
    files the source has (one file's worth for a single path, many for a
    partitioned directory — ``pyarrow.dataset`` handles both uniformly, and
    ``ParquetFileFragment.split_by_row_group()`` further splits each file
    into its individual row groups), then randomly selects enough of those
    row groups to cover the requested ``nrows``/``fraction_rows`` with a 10%
    oversample margin, reads *only* the selected row groups, and trims to
    the exact requested count.

    Coarser than genuinely uniform row-level sampling: the randomness is
    over which row groups get picked, not over individual rows within them
    — a source with few row groups/files sees little memory benefit (most
    of it has to be read anyway to assemble even a small sample), and if
    rows within a group aren't independently distributed (e.g. written in
    time order), the sample can be subtly biased in a way Spark's row-level
    Bernoulli sampling isn't.

    Args:
        path (str): Path to a single parquet file or a partitioned
            directory of parquet files.
        nrows (int | None): If set, a random sample of (very likely, given
            the oversample margin, but not strictly guaranteed) this many
            rows, or the full dataset if it has fewer.
        fraction_rows (float | None): If set, an approximate random fraction
            of the full dataset. Ignored if ``nrows`` is also set.

    Returns:
        pd.DataFrame: The loaded (and, if requested, sampled) data. The full
        dataset, unsampled, if neither ``nrows`` nor ``fraction_rows`` is
        set, or if the requested amount is at or above the total row count.
    """
    import pyarrow as pa
    import pyarrow.dataset as pa_dataset

    dataset = pa_dataset.dataset(path, format="parquet")

    if not nrows and not fraction_rows:
        return dataset.to_table().to_pandas()

    row_group_fragments = []
    for fragment in dataset.get_fragments():
        splitter = getattr(fragment, "split_by_row_group", None)
        row_group_fragments.extend(splitter() if splitter is not None else [fragment])

    row_counts = [f.count_rows() for f in row_group_fragments]
    total_rows = sum(row_counts)
    target = nrows if nrows else round(total_rows * (fraction_rows or 0))

    if target >= total_rows:
        return dataset.to_table().to_pandas()

    rng = np.random.RandomState(settings.random_state)
    order = rng.permutation(len(row_group_fragments))
    selected: list[int] = []
    running_total = 0
    for i in order:
        selected.append(int(i))
        running_total += row_counts[i]
        if running_total >= target:
            break

    table = pa.concat_tables([row_group_fragments[i].to_table() for i in selected])
    df = table.to_pandas()
    if len(df) > target:
        df = df.sample(n=target, random_state=settings.random_state)
    return df
