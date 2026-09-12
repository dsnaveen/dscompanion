"""Interactive mode — Step 1: Load Data.

Supports CSV, Parquet, and XLSX (with sheet selection), surfaces load errors
in place for a retry, previews the loaded data, and flags+resolves duplicate
column names found in the file's raw header before the user can continue.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from data_browser import _get_upload_tmp_dir
from interactive_state import (
    add_audit_entry,
    go_to_step,
    next_step_id,
    reset_from_step,
    set_step_confirmed,
    set_step_status,
)

logger = logging.getLogger(__name__)

__all__ = ["render_step_load_data"]

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
_SUPPORTED_SUFFIXES = (".csv", ".parquet", ".xlsx")


@st.cache_data
def _list_data_files(data_root_str: str) -> list[str]:
    """List every CSV/Parquet/XLSX file under ``data_root_str``, relative to it.

    Args:
        data_root_str (str): Absolute path to the data directory, passed as
            a string so the result is cacheable by ``st.cache_data``.

    Returns:
        list[str]: Sorted POSIX-style relative paths of every matching file.
        Empty list if the directory doesn't exist or has no matches.
    """
    root = Path(data_root_str)
    if not root.exists():
        return []
    files: list[Path] = []
    for suffix in _SUPPORTED_SUFFIXES:
        files.extend(root.rglob(f"*{suffix}"))
    return sorted(str(f.relative_to(root)) for f in files)


def _infer_format(path: str) -> str:
    """Infer the data format from a file path's suffix.

    Args:
        path (str): File path.

    Returns:
        str: One of ``"csv"``, ``"parquet"``, ``"xlsx"``.
    """
    suffix = Path(path).suffix.lower()
    return {".csv": "csv", ".parquet": "parquet", ".xlsx": "xlsx"}.get(suffix, "csv")


def _get_sheet_names(path: str) -> list[str]:
    """List sheet names in an XLSX workbook.

    Args:
        path (str): Path to the XLSX file.

    Returns:
        list[str]: Sheet names in workbook order. Empty list on read failure.
    """
    try:
        return pd.ExcelFile(path).sheet_names
    except Exception as exc:
        logger.warning(
            "Could not read sheet names from %s: %s", Path(path).name, type(exc).__name__
        )
        return []


def _read_raw_header(path: str, fmt: str, sheet_name: str | None) -> list[str]:
    """Read a file's literal header row, before pandas de-duplicates repeated names.

    pandas silently appends ``.1``/``.2`` to repeated column names at parse
    time for both ``read_csv`` and ``read_excel`` — by the time ``df.columns``
    is inspected, true duplicates are already gone. This reads the raw header
    directly so duplicate detection sees the original names.

    Args:
        path (str): Path to the data file.
        fmt (str): ``"csv"``, ``"parquet"``, or ``"xlsx"``.
        sheet_name (str | None): Sheet name, required when ``fmt == "xlsx"``.

    Returns:
        list[str]: Raw header values in column order. For ``"parquet"``
        (whose schema enforces unique column names already) this returns
        the same as the parsed dataframe's columns — duplicates are not
        possible there.
    """
    if fmt == "csv":
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        try:
            delimiter = csv.Sniffer().sniff(first_line).delimiter
        except csv.Error:
            delimiter = ","
        return next(csv.reader([first_line], delimiter=delimiter))
    if fmt == "xlsx":
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True)
        try:
            ws = wb[sheet_name] if sheet_name else wb.active
            row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
        finally:
            wb.close()
        return [str(v) for v in row]
    return list(pd.read_parquet(path).columns)


def _load_dataframe(path: str, fmt: str, sheet_name: str | None) -> pd.DataFrame:
    """Read the full dataframe for the chosen file/format/sheet.

    Args:
        path (str): Path to the data file.
        fmt (str): ``"csv"``, ``"parquet"``, or ``"xlsx"``.
        sheet_name (str | None): Sheet name, used only when ``fmt == "xlsx"``.

    Returns:
        pd.DataFrame: The loaded data, exactly as pandas parses it (duplicate
        column names already mangled by pandas at this point — resolved
        separately via the raw header comparison).
    """
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "xlsx":
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_parquet(path)


def _find_duplicate_names(raw_header: list[str]) -> list[str]:
    """Find column names that appear more than once in a raw header.

    Args:
        raw_header (list[str]): Raw header values in column order.

    Returns:
        list[str]: Distinct names that occur 2+ times, in first-seen order.
    """
    counts: dict[str, int] = {}
    for name in raw_header:
        counts[name] = counts.get(name, 0) + 1
    return [name for name, count in counts.items() if count > 1]


def _render_duplicate_resolution(
    dup_names: list[str], raw_header: list[str]
) -> dict[int, tuple[str, str | None]]:
    """Render a rename/drop choice for every occurrence of every duplicate name.

    Args:
        dup_names (list[str]): Distinct duplicated names.
        raw_header (list[str]): Raw header values in column order.

    Returns:
        dict[int, tuple[str, str | None]]: Maps column position (index into
        ``raw_header``) to ``("rename", new_name)`` or ``("drop", None)``.
        Only positions belonging to a duplicated name are present.
    """
    st.warning(
        f"{len(dup_names)} column name(s) appear more than once in this file: "
        f"{dup_names}. Choose what to do with each before continuing."
    )
    choices: dict[int, tuple[str, str | None]] = {}
    for name in dup_names:
        occurrence_positions = [i for i, n in enumerate(raw_header) if n == name]
        for occ_idx, col_idx in enumerate(occurrence_positions):
            default_label = f"{name}.{occ_idx}" if occ_idx > 0 else name
            action = st.radio(
                f"Column {name!r} (occurrence {occ_idx + 1} of {len(occurrence_positions)})",
                ["Rename", "Drop This Occurrence"],
                key=f"int.load_data.dup.{col_idx}.action",
                horizontal=True,
            )
            if action == "Rename":
                new_name = st.text_input(
                    "New name", value=default_label, key=f"int.load_data.dup.{col_idx}.rename"
                )
                choices[col_idx] = ("rename", new_name)
            else:
                choices[col_idx] = ("drop", None)
    return choices


def _apply_duplicate_resolution(
    df: pd.DataFrame, raw_header: list[str], choices: dict[int, tuple[str, str | None]]
) -> pd.DataFrame:
    """Apply confirmed rename/drop choices for duplicate columns.

    Args:
        df (pd.DataFrame): Loaded dataframe (pandas-mangled column names).
        raw_header (list[str]): Raw header values in column order — used as
            the base column-name list, since it reflects the user's actual
            intent rather than pandas' auto-generated ``.1``/``.2`` suffixes.
        choices (dict[int, tuple[str, str | None]]): Per-position resolution
            from ``_render_duplicate_resolution``.

    Returns:
        pd.DataFrame: Copy of ``df`` with duplicate columns renamed/dropped.
    """
    if len(raw_header) != len(df.columns):
        logger.warning(
            "Raw header length (%d) != parsed column count (%d) — skipping duplicate resolution",
            len(raw_header),
            len(df.columns),
        )
        return df
    new_columns = list(raw_header)
    for col_idx, (action, value) in choices.items():
        if action == "rename":
            new_columns[col_idx] = value
    # Select kept columns by position, not by name — renamed/un-renamed
    # duplicates can still collide on name at this point (e.g. one
    # occurrence dropped, the other kept under its original name), and
    # `df.drop(columns=[name])` would remove every column sharing that
    # name rather than just the targeted position.
    keep_positions = [
        i for i in range(len(new_columns)) if choices.get(i, ("rename", None))[0] != "drop"
    ]
    df = df.iloc[:, keep_positions].copy()
    df.columns = [new_columns[i] for i in keep_positions]
    return df


def _render_preview(df: pd.DataFrame) -> None:
    """Render the post-load preview: counts, dtype summary, and a data sample.

    Args:
        df (pd.DataFrame): Loaded dataframe.

    Returns:
        None
    """
    numeric_n = df.select_dtypes(include="number").shape[1]
    other_n = len(df.columns) - numeric_n
    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
    st.caption(f"{numeric_n} numeric column(s), {other_n} other column(s).")
    st.dataframe(df.head(10))


def render_step_load_data() -> bool:
    """Render Step 1 (Load Data) and report whether it's confirmed.

    Args:
        None

    Returns:
        bool: ``True`` once the user has loaded data and clicked
        "Confirm & continue" (any duplicate columns resolved first).
        ``False`` while still picking/loading/reviewing.
    """
    st.subheader("Step 1: Load data")

    mode = st.radio(
        "Data source", ["Browse Existing Data", "Upload a File"], key="int.load_data.source_mode"
    )

    path: str | None = None
    display_name: str | None = None
    if mode == "Browse Existing Data":
        files = _list_data_files(str(_DATA_ROOT))
        if not files:
            st.warning(f"No .csv/.parquet/.xlsx files found under {_DATA_ROOT}")
            return False
        choice = st.selectbox("Dataset", files, key="int.load_data.browse_choice")
        path = str(_DATA_ROOT / choice)
        display_name = choice
    else:
        uploaded = st.file_uploader(
            "Upload a CSV, Parquet, or XLSX file",
            type=["csv", "parquet", "xlsx"],
            key="int.load_data.uploader",
        )
        if uploaded is None:
            st.caption("Pick a file to continue.")
            return False
        tmp_dir = _get_upload_tmp_dir()
        dest = Path(tmp_dir.name) / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        path = str(dest)
        display_name = uploaded.name

    fmt = _infer_format(path)
    sheet_name: str | None = None
    if fmt == "xlsx":
        sheets = _get_sheet_names(path)
        if not sheets:
            st.error("Could not read sheet names from this file. Is it a valid XLSX workbook?")
            return False
        sheet_name = st.selectbox("Sheet name", sheets, key="int.load_data.sheet_name")

    selection_id = (path, fmt, sheet_name)

    if st.button("Load", key="int.load_data.load_button"):
        try:
            raw_header = _read_raw_header(path, fmt, sheet_name)
            df = _load_dataframe(path, fmt, sheet_name)
        except Exception as exc:
            st.session_state["int.load_data.error"] = str(exc)
            st.session_state["int.load_data.loaded"] = False
        else:
            st.session_state["int.load_data.error"] = None
            st.session_state["int.load_data.loaded"] = True
            st.session_state["int.load_data.raw_df"] = df
            st.session_state["int.load_data.raw_header"] = raw_header
            st.session_state["int.load_data.display_name"] = display_name
            st.session_state["int.load_data.selection_id"] = selection_id
            for k in list(st.session_state.keys()):
                if k.startswith("int.load_data.dup."):
                    del st.session_state[k]

    error = st.session_state.get("int.load_data.error")
    if error:
        st.error(
            f"Couldn't load this file: {error}. Change the format/sheet/file above and click "
            "Load again."
        )
        return False

    if not st.session_state.get("int.load_data.loaded"):
        st.caption("Pick a source and click Load to continue.")
        return False

    df = st.session_state["int.load_data.raw_df"]
    raw_header = st.session_state["int.load_data.raw_header"]
    _render_preview(df)

    dup_names = _find_duplicate_names(raw_header)
    dup_choices: dict[int, tuple[str, str | None]] = {}
    if dup_names:
        dup_choices = _render_duplicate_resolution(dup_names, raw_header)

    if st.button("Confirm & continue", key="int.load_data.confirm"):
        final_df = _apply_duplicate_resolution(df, raw_header, dup_choices) if dup_names else df

        nxt = next_step_id("load_data")

        prior_selection = st.session_state.get("interactive.load_data_selection_id")
        loaded_selection = st.session_state["int.load_data.selection_id"]
        if prior_selection is not None and prior_selection != loaded_selection and nxt is not None:
            reset_from_step(nxt)

        st.session_state["interactive.df"] = final_df
        st.session_state["interactive.load_data_selection_id"] = loaded_selection
        set_step_confirmed("load_data", True)
        set_step_status("load_data", "done")
        if nxt is not None:
            set_step_status(nxt, "current")
            go_to_step(nxt)
        add_audit_entry(
            "load_data",
            f"Loaded {st.session_state['int.load_data.display_name']}: "
            f"{len(final_df):,} rows × {len(final_df.columns)} columns.",
        )

    return st.session_state["interactive.step_confirmed"]["load_data"]
