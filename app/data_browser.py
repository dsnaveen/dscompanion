"""Dataset selection widgets: browse the repo's data/ directory or upload a local file.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

__all__ = ["render_data_source", "peek_columns"]

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
_SUPPORTED_TASKS = {"classification", "regression", "clustering"}


@st.cache_data
def _list_data_files(data_root_str: str) -> list[str]:
    """List every CSV/parquet file under ``data_root_str``, relative to it.

    Args:
        data_root_str (str): Absolute path to the data directory, passed as
            a string so the result is cacheable by ``st.cache_data``.

    Returns:
        list[str]: Sorted POSIX-style relative paths of every ``.csv`` and
        ``.parquet`` file found. Empty list if the directory doesn't exist
        or contains no matching files.
    """
    root = Path(data_root_str)
    if not root.exists():
        return []
    files = list(root.rglob("*.csv")) + list(root.rglob("*.parquet"))
    return sorted(str(f.relative_to(root)) for f in files)


def peek_columns(path: str | Path) -> list[str]:
    """Read just enough of a data file to list its column names.

    Args:
        path (str | Path): Path to a CSV or parquet file.

    Returns:
        list[str]: Column names. Empty list if the file can't be read.
    """
    path = Path(path)
    try:
        if path.suffix == ".parquet":
            return list(pd.read_parquet(path).columns)
        return list(pd.read_csv(path, nrows=5).columns)
    except Exception as exc:
        logger.warning("Column peek failed for %s: %s", path.name, type(exc).__name__)
        return []


def _load_sibling_metadata(path: Path) -> dict[str, Any] | None:
    """Load a ``metadata.json`` sitting next to ``path``, if present.

    Args:
        path (Path): Path to the selected data file.

    Returns:
        dict[str, Any] | None: Parsed metadata, or ``None`` if no sibling
        ``metadata.json`` exists or it fails to parse.
    """
    meta_path = path.parent / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception as exc:
        logger.warning("metadata.json parse failed for %s: %s", meta_path, type(exc).__name__)
        return None


def _apply_metadata_prefill(metadata: dict[str, Any] | None) -> None:
    """Pre-fill the target-column and task widgets from UCI metadata.

    Overwrites any existing ``cfg.data.target``/``cfg.model.task`` widget
    state — switching datasets makes the previous target column choice
    invalid anyway, so merging stale state is not useful here.

    Args:
        metadata (dict[str, Any] | None): Parsed ``metadata.json`` contents,
            or ``None`` when no sibling metadata file exists.

    Returns:
        None
    """
    if metadata is None:
        return
    target_cols = metadata.get("target_col") or []
    if target_cols:
        st.session_state["cfg.data.target"] = target_cols[0]
    tasks = metadata.get("tasks") or []
    if tasks:
        task = str(tasks[0]).lower()
        if task in _SUPPORTED_TASKS:
            st.session_state["cfg.model.task"] = task


def _get_upload_tmp_dir() -> tempfile.TemporaryDirectory:
    """Return the session's lazily-created upload temp directory.

    The ``TemporaryDirectory`` object itself (not just its ``.name``) is
    kept in ``st.session_state`` — letting the object get garbage collected
    would delete the directory out from under any path that still
    references it.

    Args:
        None

    Returns:
        tempfile.TemporaryDirectory: The session-scoped temp directory.
    """
    if st.session_state.get("upload_tmp_dir") is None:
        st.session_state["upload_tmp_dir"] = tempfile.TemporaryDirectory(
            prefix="dscompanion_streamlit_"
        )
    return st.session_state["upload_tmp_dir"]


def render_data_source() -> tuple[str | None, str | None]:
    """Render the data-source widgets and resolve the selected file.

    Renders a mode radio ("Browse Existing Data" vs "Upload a File") plus the
    corresponding picker, and applies UCI ``metadata.json`` pre-fill to the
    target/task widgets when a browsed file has a sibling metadata file.

    Args:
        None

    Returns:
        tuple[str | None, str | None]: ``(path, data_format)`` where
        ``data_format`` is ``"csv"`` or ``"parquet"`` inferred from the file
        suffix. Both are ``None`` when no file is selected yet.
    """
    mode = st.radio(
        "Data source", ["Browse Existing Data", "Upload a File"], key="data_source_mode"
    )

    path: str | None = None
    if mode == "Browse Existing Data":
        files = _list_data_files(str(_DATA_ROOT))
        if not files:
            st.warning(f"No .csv/.parquet files found under {_DATA_ROOT}")
            return None, None
        choice = st.selectbox("Dataset", files, key="browse_choice")
        full_path = _DATA_ROOT / choice
        path = str(full_path)
        # Pre-fill only runs the first time this exact file is selected — not
        # on every rerun — so it never silently overwrites a manual edit the
        # user made to target/task after the initial selection.
        if st.session_state.get("_prefilled_for_path") != path:
            metadata = _load_sibling_metadata(full_path)
            st.session_state["uci_metadata"] = metadata
            _apply_metadata_prefill(metadata)
            st.session_state["_prefilled_for_path"] = path
        metadata = st.session_state.get("uci_metadata")
        if metadata is not None:
            st.caption(f"Detected metadata: {metadata.get('name', choice)}")
    else:
        uploaded = st.file_uploader("Upload a CSV or Parquet file", type=["csv", "parquet"])
        if uploaded is None:
            return None, None
        tmp_dir = _get_upload_tmp_dir()
        dest = Path(tmp_dir.name) / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        st.session_state["uci_metadata"] = None
        path = str(dest)

    st.session_state["selected_data_path"] = path
    data_format = "parquet" if path.endswith(".parquet") else "csv"
    return path, data_format
