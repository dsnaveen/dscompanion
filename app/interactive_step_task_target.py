"""Interactive mode — Step 2: Choose Task & Target Variable.

Classification only for this slice (regression/clustering target handling is
not yet defined — see streamlit.md). Strict binary classification rules:

- Target must have no missing values.
- Target must be numeric.
- Target must contain exactly {0, 1}.
- The positive class (1) must be the minor (less frequent) class.

If any rule is violated, Step 2 blocks with a clear error and recovery
options — no silent fixes or remapping UI.

Also collects unique identifier columns (e.g. a customer/account ID) here,
alongside the target, so they can be excluded from modelling
everywhere downstream — Step 4's EDA, Step 5's missing-value table, and
the final feature list.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from interactive_state import (
    STEP_ORDER,
    add_audit_entry,
    go_to_step,
    next_step_id,
    reset_from_step,
    set_step_confirmed,
    set_step_status,
)

logger = logging.getLogger(__name__)

__all__ = ["render_step_task_target"]

_SUPPORTED_TASKS = ["classification", "regression", "clustering"]


def _render_target_preview(df: pd.DataFrame, target: str) -> None:
    """Show dtype, unique-value count, and (for low-cardinality columns) value counts.

    Args:
        df (pd.DataFrame): Working dataframe.
        target (str): Selected target column name.

    Returns:
        None
    """
    series = df[target]
    n_unique = series.nunique(dropna=True)
    st.caption(f"Type: {series.dtype} · {n_unique} unique value(s) (excluding missing).")
    if n_unique <= 10:
        counts = series.value_counts(dropna=True).rename("count")
        percent = (counts / counts.sum() * 100).round(1).rename("percent")
        st.dataframe(pd.concat([counts, percent], axis=1))


def _recovery_buttons(target_key_reset: bool = True) -> None:
    """Render standard recovery buttons shown on every validation error.

    Args:
        target_key_reset (bool): When ``True``, the first button resets the
            target selection so the user can pick a different column.

    Returns:
        None
    """
    col1, col2 = st.columns(2)
    if target_key_reset:
        with col1:
            if st.button(
                "Choose a different target column", key="int.task_target.pick_other_target"
            ):
                st.session_state.pop("int.task_target.target", None)
                st.rerun()
    with col2:
        if st.button(
            "Go back to Step 1 and load different data", key="int.task_target.back_to_step1"
        ):
            reset_from_step(STEP_ORDER[0])
            st.rerun()


def _validate_target(df: pd.DataFrame, target: str) -> bool:
    """Validate that the target column satisfies strict binary classification rules.

    Rules (all must pass before Step 2 can advance):

    1. No missing values.
    2. Numeric dtype.
    3. Exactly ``{0, 1}`` as unique values.
    4. Class 1 is the minor (less frequent) class.

    Args:
        df (pd.DataFrame): Working dataframe.
        target (str): Selected target column name.

    Returns:
        bool: ``True`` when all four rules pass; ``False`` when any rule fails
        (the failure is already rendered to the UI before returning).
    """
    series = df[target]

    # Rule 1 — no nulls
    n_missing = int(series.isna().sum())
    if n_missing > 0:
        st.error(
            f"Target '{target}' has {n_missing:,} missing value(s). "
            "Remove them from your dataset before loading."
        )
        _recovery_buttons()
        return False

    # Rule 2 — numeric dtype
    if not pd.api.types.is_numeric_dtype(series):
        st.error(
            f"Target '{target}' has dtype {series.dtype}. "
            "Binary classification requires a numeric target with values {{0, 1}}."
        )
        _recovery_buttons()
        return False

    # Rule 3 — exactly {0, 1}
    unique_vals = set(series.unique().tolist())
    if unique_vals != {0, 1}:
        st.error(
            f"Target '{target}' contains {sorted(unique_vals)}. " "Must contain exactly {0, 1}."
        )
        _recovery_buttons()
        return False

    # Rule 4 — class 1 is minor
    counts = series.value_counts()
    n_ones = int(counts.get(1, 0))
    n_zeros = int(counts.get(0, 0))
    event_rate = n_ones / len(series) if len(series) > 0 else 0.0
    if n_ones > n_zeros:
        st.error(
            f"The positive class (1) has {n_ones:,} rows but the negative class (0) "
            f"has {n_zeros:,} rows. Class 1 must be the minority class. "
            "Recode your target so the event (1) is less frequent than the non-event (0)."
        )
        _recovery_buttons()
        return False

    st.success(
        f"Target '{target}' is valid: {{0, 1}}, " f"{n_ones:,} events, event rate {event_rate:.1%}."
    )
    return True


def render_step_task_target(df: pd.DataFrame) -> bool:
    """Render Step 2 (Task & Target) and report whether it's confirmed.

    Args:
        df (pd.DataFrame): Working dataframe confirmed at Step 1.

    Returns:
        bool: ``True`` once a task and target pass all validation rules and
        the user clicked "Confirm & continue". ``False`` while still deciding.
    """
    st.subheader("Step 2: Choose task & target variable")

    task = st.selectbox("Task", _SUPPORTED_TASKS, key="int.task_target.task")
    if task != "classification":
        st.warning(
            f"'{task}' is not yet supported in Interactive mode. Only classification is "
            "implemented so far. Switch to Full Pipeline mode for this task, or pick "
            "'classification' to continue here."
        )
        return False

    target = st.selectbox(
        "Target variable",
        list(df.columns),
        key="int.task_target.target",
        index=None,
        placeholder="Choose a column...",
    )
    if not target:
        st.caption("Pick the column you want to predict.")
        return False

    if st.session_state.get("int.task_target.logged_target") != target:
        add_audit_entry("task_target", f"Target variable selected: '{target}'")
        st.session_state["int.task_target.logged_target"] = target

    _render_target_preview(df, target)

    identifier_columns = st.multiselect(
        "Select columns to exclude",
        [c for c in df.columns if c != target],
        key="int.task_target.identifier_columns",
    )
    st.caption(
        "Picked here so they're excluded everywhere downstream: EDA (Step 4), the missing-value "
        "decision table (Step 5), and the final feature list. Never used as model features."
    )

    if not _validate_target(df, target):
        return False

    if st.button("Confirm & continue", key="int.task_target.confirm"):
        st.session_state["interactive.df"] = df
        st.session_state["interactive.target"] = target
        st.session_state["interactive.task"] = task
        st.session_state["interactive.identifier_columns"] = identifier_columns
        set_step_confirmed("task_target", True)
        set_step_status("task_target", "done")
        counts = df[target].value_counts()
        add_audit_entry(
            "task_target",
            f"Task: {task}, target: '{target}' "
            f"({int(counts.get(1, 0)):,} events / {int(counts.get(0, 0)):,} non-events).",
        )
        if identifier_columns:
            add_audit_entry(
                "task_target",
                f"Excluded from modelling ({len(identifier_columns)} column"
                f"{'s' if len(identifier_columns) > 1 else ''}): "
                f"{', '.join(identifier_columns)}.",
            )
        nxt = next_step_id("task_target")
        if nxt is not None:
            go_to_step(nxt)

    return st.session_state["interactive.step_confirmed"]["task_target"]
