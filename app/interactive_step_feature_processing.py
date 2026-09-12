"""Interactive mode — Step 5: Feature Processing (ColumnRecipe decisions).

Every feature column gets a recommended, theory-driven ``ColumnRecipe``
(``dscompanion.features.recommend.recommend_column_recipe``) with a per-column
three-way choice: **Accept** (recipe applies, editable via add/remove/
reorder), **Reject** (raw passthrough, unchanged), or **Drop** (removed
entirely). Columns are split into two presentational groups — "Needing
Attention" (missing values or an EDA red flag, shown expanded) and "Rest"
(everything else, shown collapsed with a bulk accept action) — both use the
same three-way mechanism; the grouping only affects how much attention a
column demands from the user.

This replaces the earlier Impute/Drop-only version once the
``ColumnRecipe``/``FeatureTransformChain`` per-column transform system
existed but had no UI. Confirm-before-apply gate applies here (design rule
#1 in streamlit.md), since these choices change what data actually reaches
the model.

Computes missingness/EDA-flag candidates directly from ``split.train_X``
rather than assuming Step 4 ran — Step 4 can be skipped (its toggle
defaults to on but the user may turn it off, or it may fail), and this step
must work either way, falling back to a fresh ``UnivariateAnalyser`` fit
when no ``EDAReport`` is available.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from interactive_state import add_audit_entry, set_step_confirmed, set_step_status

from dscompanion.eda.univariate import UnivariateAnalyser
from dscompanion.features import (
    ColumnRecipe,
    FeatureTransformChain,
    TransformStep,
    recommend_column_recipe,
)
from dscompanion.features.registry import registered_transformer_names
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["render_step_feature_processing"]

_CHOICE_ACCEPT = "Accept Recommended"
_CHOICE_REJECT = "Leave As-Is"
_CHOICE_DROP = "Drop Column"
_CHOICES = [_CHOICE_ACCEPT, _CHOICE_REJECT, _CHOICE_DROP]


def _missing_value_candidates(
    train_X: pd.DataFrame, excluded_cols: set[str], suggested_reasons: dict[str, list[str]]
) -> pd.DataFrame:
    """Builds the "Needing Attention" table: missing values, plus suggested drops.

    Args:
        train_X (pd.DataFrame): Training feature DataFrame (may still
            contain the date column and identifier columns —
            ``DataSplitter`` only drops the target from ``train_X``).
        excluded_cols (set[str]): Columns to leave out of both candidate
            groups entirely — the date column chosen at Step 3, any
            identifier columns chosen at Step 2, and any boolean-dtype
            columns (already model-ready, out of scope for recipe
            treatment). Neither is a feature-processing decision.
        suggested_reasons (dict[str, list[str]]): Output of
            ``EDAReport.recommended_drops_with_reasons()`` (empty dict when
            Step 4 was skipped/failed, or nothing was flagged). A column
            named here appears in the table even with zero missing values,
            since "constant"/"near_zero_variance"/"low_iv" don't require
            any missingness at all.

    Returns:
        pd.DataFrame: Columns ``column`` (str), ``missing_rows`` (int),
        ``missing_pct`` (float, 0-100 scale), ``is_numeric`` (bool),
        ``suggested_reasons`` (list[str], empty when not suggested). One row
        per column with ``missing_rows > 0`` or a non-empty
        ``suggested_reasons``, in original column order. Empty (but
        correctly columned) when no column needs attention.
    """
    candidate_cols = [c for c in train_X.columns if c not in excluded_cols]
    missing_counts = train_X[candidate_cols].isna().sum()
    relevant_cols = [
        col for col in candidate_cols if missing_counts[col] > 0 or suggested_reasons.get(col)
    ]
    rows = [
        {
            "column": col,
            "missing_rows": int(missing_counts[col]),
            "missing_pct": round(100 * missing_counts[col] / len(train_X), 2),
            "is_numeric": pd.api.types.is_numeric_dtype(train_X[col]),
            "suggested_reasons": suggested_reasons.get(col, []),
        }
        for col in relevant_cols
    ]
    return pd.DataFrame(
        rows, columns=["column", "missing_rows", "missing_pct", "is_numeric", "suggested_reasons"]
    )


def _get_univariate_summaries(split: DataSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (numeric_summary, categorical_summary), reusing Step 4's EDAReport if present.

    Args:
        split (DataSplit): Step-3-confirmed split — a fresh
            ``UnivariateAnalyser`` is fitted on ``split`` when Step 4 was
            skipped or failed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``UnivariateAnalyser.numeric_summary()``
        and ``.categorical_summary()`` output — see those methods'
        docstrings for the full column schema. Both are needed by
        ``recommend_column_recipe()`` to build a recommendation per column.
    """
    eda_report = st.session_state.get("interactive.eda_report")
    if eda_report is not None:
        return eda_report.numeric_summary(), eda_report.categorical_summary()
    analyser = UnivariateAnalyser().fit(split)
    return analyser.numeric_summary(), analyser.categorical_summary()


def _stats_row(
    col: str, is_numeric: bool, numeric_summary: pd.DataFrame, categorical_summary: pd.DataFrame
) -> pd.Series:
    """Looks up one column's summary-statistics row for ``recommend_column_recipe()``.

    Args:
        col (str): Column name.
        is_numeric (bool): Which summary table to search.
        numeric_summary (pd.DataFrame): ``UnivariateAnalyser.numeric_summary()`` output.
        categorical_summary (pd.DataFrame): ``UnivariateAnalyser.categorical_summary()`` output.

    Returns:
        pd.Series: The matching ``feature == col`` row.

    Raises:
        IndexError: If ``col`` is not present in the relevant summary table
            (should not happen for any numeric/categorical candidate column
            passed in from this file).
    """
    table = numeric_summary if is_numeric else categorical_summary
    return table.loc[table["feature"] == col].iloc[0]


def _recommended_recipe(
    col: str, is_numeric: bool, numeric_summary: pd.DataFrame, categorical_summary: pd.DataFrame
) -> ColumnRecipe:
    """Computes the recommended ``ColumnRecipe`` for one column.

    Pure/deterministic given ``split.train_X`` — safe to recompute on every
    Streamlit rerun rather than caching, since the underlying summary
    statistics don't change within Step 5.

    Args:
        col (str): Column name.
        is_numeric (bool): Selects which summary table/recommender rules apply.
        numeric_summary (pd.DataFrame): ``UnivariateAnalyser.numeric_summary()`` output.
        categorical_summary (pd.DataFrame): ``UnivariateAnalyser.categorical_summary()`` output.

    Returns:
        ColumnRecipe: ``source="recommended"``, possibly zero steps (e.g. a
        clean numeric column with no missingness/skew).
    """
    stats = _stats_row(col, is_numeric, numeric_summary, categorical_summary)
    dtype = "numeric" if is_numeric else "categorical"
    return recommend_column_recipe(col, stats, dtype)


def _init_recipe_steps_state(col: str, recommended: ColumnRecipe) -> None:
    """Initializes this column's editable step list from its recommendation, once.

    Args:
        col (str): Column name.
        recommended (ColumnRecipe): This column's freshly-computed recommendation.

    Returns:
        None
    """
    steps_key = f"int.feature_processing.recipe_steps.{col}"
    if steps_key not in st.session_state:
        st.session_state[steps_key] = [
            {"uid": i, "transformer": s.transformer, "params": s.params}
            for i, s in enumerate(recommended.steps)
        ]
        st.session_state[f"int.feature_processing.next_uid.{col}"] = len(recommended.steps)


def _current_transform_steps(col: str) -> list[TransformStep]:
    """Reads this column's current editable step list back as real ``TransformStep`` objects.

    Args:
        col (str): Column name.

    Returns:
        list[TransformStep]: The user's current (possibly edited) step list.
    """
    raw = st.session_state.get(f"int.feature_processing.recipe_steps.{col}", [])
    return [TransformStep(transformer=s["transformer"], params=s.get("params", {})) for s in raw]


def _render_recipe_editor(col: str) -> None:
    """Renders the add/remove/reorder step editor for one column's recipe.

    Each step row keys its widgets off a stable per-step ``uid`` (assigned
    at initialization/append time), not its list position — positional keys
    would silently mix up widget state across reruns whenever a step is
    removed or reordered, since Streamlit keeps prior widget values keyed by
    identity, not by the list's current shape.

    Args:
        col (str): Column name — the step list lives at
            ``st.session_state["int.feature_processing.recipe_steps.{col}"]``.

    Returns:
        None
    """
    steps_key = f"int.feature_processing.recipe_steps.{col}"
    steps = st.session_state[steps_key]
    registry_names = registered_transformer_names()

    for pos, step in enumerate(steps):
        row = st.columns([3, 0.5, 0.5, 0.5])
        new_transformer = row[0].selectbox(
            "Step",
            registry_names,
            index=registry_names.index(step["transformer"]),
            key=f"int.feature_processing.recipe_step.{col}.{step['uid']}",
            label_visibility="collapsed",
        )
        step["transformer"] = new_transformer
        if row[1].button(
            "↑", key=f"int.feature_processing.step_up.{col}.{step['uid']}", disabled=(pos == 0)
        ):
            steps[pos - 1], steps[pos] = steps[pos], steps[pos - 1]
            st.rerun()
        if row[2].button(
            "↓",
            key=f"int.feature_processing.step_down.{col}.{step['uid']}",
            disabled=(pos == len(steps) - 1),
        ):
            steps[pos + 1], steps[pos] = steps[pos], steps[pos + 1]
            st.rerun()
        if row[3].button(
            ":material/delete:", key=f"int.feature_processing.step_remove.{col}.{step['uid']}"
        ):
            steps.pop(pos)
            st.rerun()

    if st.button("+ Add step", key=f"int.feature_processing.step_add.{col}"):
        next_uid = st.session_state.get(f"int.feature_processing.next_uid.{col}", len(steps))
        steps.append({"uid": next_uid, "transformer": registry_names[0], "params": {}})
        st.session_state[f"int.feature_processing.next_uid.{col}"] = next_uid + 1
        st.rerun()


def _render_column_control(
    col: str,
    is_numeric: bool,
    missing_rows: int,
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
) -> None:
    """Renders one column's full Accept/Reject/Drop control, recommendation, and editor.

    Args:
        col (str): Column name.
        is_numeric (bool): Selects recommender rules and the "Reject" safety-warning text.
        missing_rows (int): Missing-row count for this column on the training set —
            drives the "Reject" safety warning for numeric columns.
        numeric_summary (pd.DataFrame): ``UnivariateAnalyser.numeric_summary()`` output.
        categorical_summary (pd.DataFrame): ``UnivariateAnalyser.categorical_summary()`` output.

    Returns:
        None
    """
    recommended = _recommended_recipe(col, is_numeric, numeric_summary, categorical_summary)
    _init_recipe_steps_state(col, recommended)
    choice_key = f"int.feature_processing.choice.{col}"
    st.session_state.setdefault(choice_key, _CHOICE_ACCEPT)

    st.markdown(f"**{col}**")
    choice = st.radio(
        "Action", _CHOICES, key=choice_key, horizontal=True, label_visibility="collapsed"
    )

    if choice == _CHOICE_ACCEPT:
        if recommended.steps:
            chips = " → ".join(s.transformer for s in recommended.steps)
            st.caption(f"Recommended: {chips}")
        st.caption(recommended.rationale)
        with st.expander("Edit steps"):
            _render_recipe_editor(col)
    elif choice == _CHOICE_REJECT:
        if not is_numeric:
            st.warning(
                f"'{col}' will pass through as raw text. Most models cannot train on "
                "unencoded categorical values."
            )
        elif missing_rows > 0:
            st.warning(
                f"'{col}' has {missing_rows:,} missing value(s) and will pass through "
                "unchanged. Most models cannot train with missing values present."
            )
    # _CHOICE_DROP: nothing further to render.


def _accept_all_recommended(cols: list[str]) -> None:
    """Bulk-sets every listed column to Accept its recommendation, discarding prior edits.

    Args:
        cols (list[str]): Column names to reset (typically the "Rest" group).

    Returns:
        None
    """
    for col in cols:
        st.session_state[f"int.feature_processing.choice.{col}"] = _CHOICE_ACCEPT
        st.session_state.pop(f"int.feature_processing.recipe_steps.{col}", None)
        st.session_state.pop(f"int.feature_processing.next_uid.{col}", None)


def _render_live_preview(split: DataSplit, all_cols: list[str]) -> None:
    """Shows a before/after sample using a real ``FeatureTransformChain`` fit on current choices.

    Args:
        split (DataSplit): Step-3-confirmed split — the chain is fit on
            ``split.train_X`` only, mirroring how ``PipelineRunner`` fits
            feature-processing transformers on train data alone.
        all_cols (list[str]): Every candidate column (both groups) to
            derive the current accepted-recipes dict from.

    Returns:
        None
    """
    accepted = {
        col: ColumnRecipe(column=col, steps=_current_transform_steps(col))
        for col in all_cols
        if st.session_state.get(f"int.feature_processing.choice.{col}") == _CHOICE_ACCEPT
        and _current_transform_steps(col)
    }
    if not accepted:
        st.caption("No column has an accepted recipe yet. Nothing to preview.")
        return

    sample = split.train_X[list(accepted)].head(5)
    try:
        chain = FeatureTransformChain(recipes=accepted)
        after = chain.fit_transform(split.train_X)
        after_sample = after.loc[sample.index]
    except Exception as exc:
        logger.warning("Step 5 preview unavailable: %s", exc)
        st.caption("Preview unavailable. The current recipe selections could not be fit.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before**")
        st.dataframe(sample.reset_index(drop=True), width="stretch")
    with col2:
        st.markdown("**After**")
        st.dataframe(after_sample.reset_index(drop=True), width="stretch")
    st.caption(
        "First 5 training rows, columns with an accepted recipe only. Column-expanding "
        "steps (e.g. one-hot encoding) show their expanded output columns."
    )


def render_step_feature_processing(split: DataSplit, target: str) -> bool:
    """Render Step 5 (Feature Processing — ColumnRecipe decisions), report whether confirmed.

    Args:
        split (DataSplit): Split confirmed at Step 3 — recommendations and
            the live preview are computed from ``split.train_X`` only,
            mirroring how ``PipelineRunner`` fits feature-processing
            transformers on train data alone.
        target (str): Confirmed target column name. Unused directly here
            (``train_X`` already excludes it) — accepted for signature
            symmetry with the other interactive steps. Identifier columns
            chosen at Step 2 (``interactive.identifier_columns``), the date
            column chosen at Step 3 (``int.split.date_col``), and any
            boolean/datetime-dtype column are excluded from both candidate
            groups — none of these is a feature-processing decision.

    Returns:
        bool: ``True`` once the user clicks "Continue". ``False`` while
        still reviewing.
    """
    st.subheader("Step 5: Feature processing")
    st.caption(
        "Every feature column gets a recommended transformation. Accept it (and edit if "
        "needed), leave the column as-is, or drop it entirely. This changes the data the "
        "model actually trains on."
    )

    date_col = st.session_state.get("int.split.date_col")
    identifier_columns = st.session_state.get("interactive.identifier_columns") or []
    bool_cols = split.train_X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    # Any datetime-dtype column, not just the one chosen as int.split.date_col — a
    # second incidental date column has no row in either UnivariateAnalyser summary
    # table (both explicitly exclude datetime_cols), so recommend_column_recipe()
    # has nothing to look up for it. DateFeatureExtractor is the separate mechanism
    # for date features; out of scope for this recipe system, same as bool columns.
    datetime_cols = split.train_X.select_dtypes(include="datetime").columns.tolist()
    excluded_cols = {c for c in [date_col, *identifier_columns, *bool_cols, *datetime_cols] if c}

    eda_report = st.session_state.get("interactive.eda_report")
    suggested_reasons = (
        eda_report.recommended_drops_with_reasons() if eda_report is not None else {}
    )
    attention_table = _missing_value_candidates(split.train_X, excluded_cols, suggested_reasons)
    attention_cols = set(attention_table["column"])
    rest_cols = [
        c for c in split.train_X.columns if c not in excluded_cols and c not in attention_cols
    ]

    numeric_summary, categorical_summary = _get_univariate_summaries(split)
    all_candidate_cols = list(attention_table["column"]) + rest_cols

    if not all_candidate_cols:
        st.success("No feature columns to configure. Nothing to decide here.")
        if st.button("Continue", key="int.feature_processing.continue_empty"):
            st.session_state["interactive.feature_processing_recipes"] = {}
            st.session_state["interactive.feature_processing_dropped_columns"] = []
            set_step_confirmed("feature_processing", True)
            set_step_status("feature_processing", "done")
            add_audit_entry("feature_processing", "Feature processing: no columns to configure.")
        return st.session_state["interactive.step_confirmed"]["feature_processing"]

    st.markdown(f"#### Columns Needing Attention ({len(attention_table)})")
    if attention_table.empty:
        st.caption("No column has missing values or an EDA red flag.")
    else:
        for row in attention_table.itertuples():
            _render_column_control(
                row.column, row.is_numeric, row.missing_rows, numeric_summary, categorical_summary
            )
            st.divider()

    st.markdown(f"#### Other Columns ({len(rest_cols)})")
    with st.expander(f"Review {len(rest_cols)} other column(s)", expanded=False):
        if rest_cols and st.button(
            "Accept all recommended", key="int.feature_processing.accept_all_rest"
        ):
            _accept_all_recommended(rest_cols)
            st.rerun()
        for col in rest_cols:
            is_numeric = pd.api.types.is_numeric_dtype(split.train_X[col])
            _render_column_control(col, is_numeric, 0, numeric_summary, categorical_summary)
            st.divider()

    with st.expander("Preview: How the data will look after these decisions", expanded=True):
        _render_live_preview(split, all_candidate_cols)

    if st.button("Continue", key="int.feature_processing.continue"):
        recipes: dict[str, ColumnRecipe] = {}
        dropped_columns: list[str] = []
        n_accepted = 0
        n_rejected = 0

        for col in all_candidate_cols:
            choice = st.session_state.get(f"int.feature_processing.choice.{col}", _CHOICE_ACCEPT)
            if choice == _CHOICE_DROP:
                dropped_columns.append(col)
                continue
            if choice == _CHOICE_REJECT:
                n_rejected += 1
                continue

            is_numeric = pd.api.types.is_numeric_dtype(split.train_X[col])
            recommended = _recommended_recipe(col, is_numeric, numeric_summary, categorical_summary)
            current_steps = _current_transform_steps(col)
            if not current_steps:
                continue
            is_edited = current_steps != recommended.steps
            recipes[col] = ColumnRecipe(
                column=col,
                steps=current_steps,
                source="manual" if is_edited else "recommended",
                rationale=None if is_edited else recommended.rationale,
            )
            n_accepted += 1

        st.session_state["interactive.feature_processing_recipes"] = recipes
        st.session_state["interactive.feature_processing_dropped_columns"] = dropped_columns
        set_step_confirmed("feature_processing", True)
        set_step_status("feature_processing", "done")

        parts = []
        if dropped_columns:
            parts.append(f"dropped {len(dropped_columns)} column(s) ({', '.join(dropped_columns)})")
        if n_accepted:
            parts.append(f"accepted a recipe for {n_accepted} column(s)")
        if n_rejected:
            parts.append(f"left {n_rejected} column(s) as-is")
        add_audit_entry("feature_processing", "Feature processing: " + "; ".join(parts) + ".")

    return st.session_state["interactive.step_confirmed"]["feature_processing"]
