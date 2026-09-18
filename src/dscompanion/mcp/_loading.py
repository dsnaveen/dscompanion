"""Local dataset loading for MCP tools.

CSV and Parquet only, deliberately -- this is a local-first tool: no Delta/Spark
path, matching the "runs entirely on your machine" design constraint (a Delta path
would imply a Spark session and cluster access, which is out of scope for a local
MCP server).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dscompanion.pipeline.loaders import load_raw_data

__all__ = ["load_dataframe"]

_EXTENSION_TO_FORMAT = {".csv": "csv", ".parquet": "parquet", ".pq": "parquet"}


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a local CSV or Parquet file into a DataFrame, inferring format from
    the file extension.

    Args:
        path (str): Path to a local ``.csv``, ``.parquet``, or ``.pq`` file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        ValueError: If the file extension isn't one of the supported formats.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in _EXTENSION_TO_FORMAT:
        raise ValueError(
            f"Unsupported file extension {suffix!r} for {path!r} -- "
            f"expected one of {sorted(_EXTENSION_TO_FORMAT)}"
        )
    return load_raw_data(path, fmt=_EXTENSION_TO_FORMAT[suffix])
