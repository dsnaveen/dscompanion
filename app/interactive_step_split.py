"""Interactive mode — Step 3: Train / Validation / Test Split.

Wraps ``dscompanion.split.DataSplitter`` directly against the plain dataframe
confirmed at Step 2 (target already guaranteed binary 0/1 by that point, so
the pre-existing multiclass `.mean()` bug in
``DataSplitter._check_class_ratio()`` never triggers here). The split is
recomputed live from the current widget values and shown as a preview before
anything is confirmed — nothing downstream is touched until the user clicks
"Apply this split and continue".

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from interactive_state import add_audit_entry, set_step_confirmed, set_step_status

from dscompanion.split import DataSplit, DataSplitter

logger = logging.getLogger(__name__)

__all__ = ["render_step_split"]

_METHOD_HELP = {
    "stratified": "Stratified (recommended): keeps the same proportion of each class in "
    "every split.",
    "temporal": "Temporal: everything on or after a cutoff date becomes a held-out "
    "'out-of-time' set, used to check the model still works on the most recent data.",
    "grouped": "Grouped: you choose exactly which population becomes the test set (for "
    "example, a specific region or segment), to check the model still works on a population "
    "it never trained on. Validation is a normal sample from everything else.",
}

_GROUP_FLAG_COL = "__group_split_flag__"


def _default_cutoff(df: pd.DataFrame, date_col: str) -> "pd.Timestamp | None":
    """Suggest a cutoff date that leaves roughly 20% of rows as out-of-time.

    Args:
        df (pd.DataFrame): Working dataframe.
        date_col (str): Candidate date column name.

    Returns:
        pd.Timestamp | None: The 80th-percentile date, or ``None`` if the
        column can't be parsed as dates.
    """
    try:
        parsed = pd.to_datetime(df[date_col])
    except Exception:
        return None
    return parsed.quantile(0.8)


def _render_group_population_picker(
    df: pd.DataFrame, group_col: str
) -> tuple[pd.DataFrame, str, set]:
    """Render a dtype-adaptive widget for choosing the deliberate test population.

    Categorical/object/bool columns get a multiselect of their distinct values — the
    test population is exactly the selected value(s). Numeric and date columns get a
    threshold instead (hand-picking individual values from a near-continuous column
    isn't practical): a temporary boolean flag column is derived on a copy of ``df``
    so the same ``group_test_values={True}`` mechanism covers both cases uniformly.

    Args:
        df (pd.DataFrame): Working dataframe confirmed at Step 2.
        group_col (str): Column chosen to define the population boundary.

    Returns:
        tuple[pd.DataFrame, str, set]: ``(working_df, effective_group_col,
        group_test_values)`` to pass straight through to ``_run_split``.
        ``working_df`` is ``df`` unchanged for the categorical case, or a copy with
        one derived boolean column for the numeric/date case.
    """
    series = df[group_col]
    n_unique = series.nunique(dropna=True)
    st.caption(f"{n_unique} distinct value(s) in '{group_col}'.")

    is_date = pd.api.types.is_datetime64_any_dtype(series)
    if is_date or pd.api.types.is_numeric_dtype(series):
        if is_date:
            default = series.quantile(0.8)
            threshold = pd.Timestamp(
                st.date_input(
                    "Threshold", value=default.date(), key="int.split.group_threshold_date"
                )
            )
            values = series
        else:
            lo, hi = float(series.min()), float(series.max())
            threshold = st.slider(
                "Threshold",
                lo,
                hi,
                float(series.quantile(0.8)),
                key="int.split.group_threshold_numeric",
            )
            values = series

        test_is_above = (
            st.radio(
                "Test population direction",
                ["At or above the threshold", "Below the threshold"],
                key="int.split.group_direction",
                horizontal=True,
            )
            == "At or above the threshold"
        )
        is_test = values >= threshold if test_is_above else values < threshold
        working_df = df.copy()
        working_df[_GROUP_FLAG_COL] = is_test
        st.caption(
            f"Test population: '{group_col}' {'≥' if test_is_above else '<'} {threshold} "
            f"({int(is_test.sum()):,} rows)."
        )
        return working_df, _GROUP_FLAG_COL, {True}

    options = sorted(series.dropna().unique().tolist())
    selected = st.multiselect(
        f"Test population: Select one or more '{group_col}' value(s)",
        options,
        key="int.split.group_values",
    )
    if n_unique > 2:
        st.caption(
            "Tip: a column with exactly 2 distinct values gives the cleanest population split."
        )
    if selected:
        st.caption(
            f"Test population: '{group_col}' ∈ {selected} ({len(selected)} of {n_unique} value(s))."
        )
    return df, group_col, set(selected)


def _run_split(
    df: pd.DataFrame,
    target: str,
    method: str,
    test_size: float,
    val_size: float,
    date_col: str | None,
    group_col: str | None,
    oot_cutoff: str | None,
    group_test_values: set | None = None,
) -> tuple[DataSplit | None, str | None]:
    """Compute a split, translating any failure into a plain-language message.

    Args:
        df (pd.DataFrame): Working dataframe.
        target (str): Confirmed target column name.
        method (str): One of ``"stratified"``, ``"temporal"``, ``"grouped"``.
        test_size (float): Test fraction, in ``(0, 1)``.
        val_size (float): Validation fraction, in ``(0, 1)``.
        date_col (str | None): Date column — required for ``"temporal"``.
        group_col (str | None): Group column — required for ``"grouped"``.
        oot_cutoff (str | None): ISO date string — required for ``"temporal"``.
        group_test_values (set | None): Deliberate test-population values for
            ``"grouped"`` — see ``_render_group_population_picker``. ``None``
            falls back to ``DataSplitter``'s original random group assignment.

    Returns:
        tuple[DataSplit | None, str | None]: ``(split, error_message)`` —
        exactly one of the two is ``None``.
    """
    if date_col is not None and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        # DataSplitter._validate() parses date_col via pd.to_datetime() before
        # checking OOT row counts, but _temporal_split() compares the raw
        # column against the cutoff without parsing it — crashes whenever
        # date_col is stored as strings (the normal case for CSV-loaded
        # data). Worked around here rather than in dscompanion core: parse it
        # ourselves before handing the frame to DataSplitter.
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    try:
        splitter = DataSplitter(
            strategy=method,
            test_size=test_size,
            val_size=val_size,
            date_col=date_col,
            group_col=group_col,
            group_test_values=group_test_values,
            oot_cutoff=oot_cutoff,
            target_col=target,
        )
        split = splitter.fit_split(df)
    except Exception as exc:
        logger.exception("Interactive mode — split failed (method=%s)", method)
        return None, str(exc)

    if group_col == _GROUP_FLAG_COL:
        # The threshold-derived flag column exists purely to drive DataSplitter's
        # grouping — DataSplitter has no way to know it isn't a real feature (it
        # only ever excludes target_col from X), so it would otherwise ride along
        # into every downstream step as a fake, perfectly-leaky column.
        split.train_X = split.train_X.drop(columns=[_GROUP_FLAG_COL])
        split.val_X = split.val_X.drop(columns=[_GROUP_FLAG_COL])
        split.test_X = split.test_X.drop(columns=[_GROUP_FLAG_COL])
        if len(split.oot_X) > 0:
            split.oot_X = split.oot_X.drop(columns=[_GROUP_FLAG_COL])
        split.metadata.n_features -= 1

    if split.metadata.split_sizes.get("train", 0) == 0:
        return None, (
            "This split leaves no rows for training. Lower the test/validation size and "
            "try again."
        )
    return split, None


def _render_split_preview(split: DataSplit) -> None:
    """Show row counts and event rate per split — the human-readable version of what
    ``DataSplitter._check_class_ratio()`` checks internally, surfaced before the split
    is locked in.

    Args:
        split (DataSplit): Computed split.

    Returns:
        None
    """
    sizes = split.metadata.split_sizes
    rates = split.metadata.class_rates or {}
    rows = []
    for name in ["train", "val", "test", "oot"]:
        n = sizes.get(name, 0)
        if n == 0:
            continue
        rate = rates.get(name)
        rows.append(
            {
                "Split": name.upper() if name == "oot" else name.capitalize(),
                "Rows": n,
                "Event Rate": f"{rate:.1%}" if rate is not None else "n/a",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if sizes.get("oot", 0) > 0:
        st.caption(
            "OOT (out-of-time) is a held-out period from before/after the cutoff date, "
            "used later to check the model still works on the most recent data, separate "
            "from the regular test set."
        )


def render_step_split(df: pd.DataFrame, target: str) -> bool:
    """Render Step 3 (Split) and report whether it's confirmed.

    Args:
        df (pd.DataFrame): Working dataframe confirmed at Step 2.
        target (str): Confirmed target column name.

    Returns:
        bool: ``True`` once a split has been computed and the user clicked
        "Apply this split and continue". ``False`` while still deciding.
    """
    st.subheader("Step 3: Train / validation / test split")

    method = st.selectbox("Split method", list(_METHOD_HELP), key="int.split.method")
    st.caption(_METHOD_HELP[method])

    other_cols = [c for c in df.columns if c != target]
    date_col: str | None = None
    group_col: str | None = None
    selected_group_col: str | None = None
    oot_cutoff: str | None = None
    group_test_values: set | None = None
    working_df = df

    if method == "temporal":
        if not other_cols:
            st.error("No columns other than the target are available to use as a date column.")
            return False
        date_col = st.selectbox("Date column", other_cols, key="int.split.date_col")
        default_cutoff = _default_cutoff(df, date_col)
        if default_cutoff is None:
            st.error(f"Column '{date_col}' doesn't look like dates. Pick a different date column.")
            return False
        cutoff_date = st.date_input(
            "Out-of-time cutoff date", value=default_cutoff.date(), key="int.split.oot_cutoff"
        )
        oot_cutoff = str(cutoff_date)
        st.caption(
            f"Rows on or after {cutoff_date} become the out-of-time set; everything before "
            "that is split into train/validation/test as usual."
        )
    elif method == "grouped":
        if not other_cols:
            st.error("No columns other than the target are available to use as a group column.")
            return False
        selected_group_col = st.selectbox("Group column", other_cols, key="int.split.group_col")
        working_df, group_col, group_test_values = _render_group_population_picker(
            df, selected_group_col
        )
        if not group_test_values:
            st.caption("Pick at least one test-population value above to continue.")
            return False

    test_size = st.slider(
        "Test size (fraction of rows)", 0.01, 0.99, 0.2, 0.01, key="int.split.test_size"
    )
    val_size = st.slider(
        "Validation size (fraction of the remaining rows)",
        0.01,
        0.99,
        0.1,
        0.01,
        key="int.split.val_size",
    )

    if test_size + val_size >= 0.95:
        st.warning(
            "Test size plus validation size is too high. There would be almost nothing left "
            "to train on. Lower one or both sliders."
        )
        return False

    split, error = _run_split(
        working_df,
        target,
        method,
        test_size,
        val_size,
        date_col,
        group_col,
        oot_cutoff,
        group_test_values,
    )
    if error:
        st.error(f"Couldn't create this split: {error} Adjust the settings above and try again.")
        return False

    sizes = split.metadata.split_sizes
    parts = [f"Train: {sizes.get('train', 0):,} rows", f"Validation: {sizes.get('val', 0):,} rows"]
    parts.append(f"Test: {sizes.get('test', 0):,} rows")
    if sizes.get("oot", 0) > 0:
        parts.append(f"Out-of-Time: {sizes['oot']:,} rows")
    st.caption(" · ".join(parts))

    _render_split_preview(split)

    if st.button("Apply this split and continue", key="int.split.confirm"):
        sizes = split.metadata.split_sizes
        # Snapshot widget values into plain session state — widget keys are cleared
        # by Streamlit once this step stops rendering, so the config preview would
        # otherwise read None for method/test_size/val_size after confirmation.
        st.session_state["int.split._method"] = method
        st.session_state["int.split._test_size"] = test_size
        st.session_state["int.split._val_size"] = val_size
        st.session_state["interactive.split"] = split
        set_step_confirmed("split", True)
        set_step_status("split", "done")
        population_note = (
            f" Test population: '{selected_group_col}', {len(group_test_values)} value(s)."
            if method == "grouped"
            else ""
        )
        add_audit_entry(
            "split",
            f"Split applied ({method}): {sizes.get('train', 0):,} train / "
            f"{sizes.get('val', 0):,} validation / {sizes.get('test', 0):,} test"
            + (f" / {sizes.get('oot', 0):,} out-of-time" if sizes.get("oot", 0) > 0 else "")
            + " rows."
            + population_note,
        )

    return st.session_state["interactive.step_confirmed"]["split"]
