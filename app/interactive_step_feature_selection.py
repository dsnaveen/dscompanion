"""Interactive mode — Step 6: Feature Selection.

Runs dscompanion's real ``FeatureSelectionPipeline`` (the same selector chain
``PipelineRunner._build_selection_pipeline()`` uses for a real run) read-only
first, shows the proposed drop-list with plain-language reasons, and lets the
user override per-row before committing — design rule from streamlit.md's
Step 6 spec ("run selectors read-only first, then show the proposed drop-list
before removing anything").

Runs on the output of a ``FeatureTransformChain`` built from Step 5's
confirmed ``interactive.feature_processing_recipes`` — a reversal of this
step's earlier design (raw ``split.train_X``), made once Step 5 gained a real
per-column transform system: selection metrics (IV, correlation, cardinality)
are far more meaningful post-encoding than on raw categorical strings, and
this is what the columns actually look like at train time. Column-expanding
recipe steps (e.g. one-hot encoding) mean the audit table and "Remove?" rows
below operate on the *transformed* column names, not the original ones —
consistent with how ``PipelineRunner``'s own batch-mode selection stage
already runs after (not before) feature processing.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging

import streamlit as st
from interactive_state import add_audit_entry, dark_fig, set_step_confirmed, set_step_status

from dscompanion.features import ColumnRecipe, FeatureTransformChain
from dscompanion.selection import (
    CardinalitySelector,
    ConstantSelector,
    CorrelationSelector,
    FeatureSelectionPipeline,
    IVSelector,
    NullRateSelector,
)
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["render_step_feature_selection"]

# Selector class name -> plain-language stage label, per streamlit.md's
# "Reasons in plain language, not bare selector class names" rule.
_STAGE_LABELS = {
    "NullRateSelector": "Missing-Rate Check",
    "ConstantSelector": "Constant Check",
    "CardinalitySelector": "Cardinality Check",
    "CorrelationSelector": "Correlation Check",
    "IVSelector": "Predictive Value Check",
}
_STAGE_AUDIT_NOUNS = {
    "NullRateSelector": "column(s) with too many missing values",
    "ConstantSelector": "constant/near-constant column(s)",
    "CardinalitySelector": "high-cardinality column(s)",
    "CorrelationSelector": "highly-correlated column(s)",
    "IVSelector": "low-predictive-value column(s)",
}


def _build_selectors(task: str) -> list:
    """Mirror ``PipelineRunner._build_selection_pipeline()``'s default selector chain.

    Args:
        task (str): Confirmed task ("classification"/"regression"/
            "clustering") — ``IVSelector`` only applies to classification,
            matching the real pipeline's conditional wiring exactly.

    Returns:
        list: Ordered ``BaseSelector`` instances built from dscompanion's
        settings-driven defaults — no per-experiment threshold overrides in
        Interactive mode yet, matching streamlit.md's Step 6 scope (the
        override is per-row accept/reject, not per-selector threshold
        tuning).
    """
    selectors = [
        NullRateSelector(),
        ConstantSelector(),
        CardinalitySelector(),
        CorrelationSelector(),
    ]
    if task == "classification":
        selectors.append(IVSelector())
    return selectors


def render_step_feature_selection(split: DataSplit, target: str) -> bool:
    """Render Step 6 (Feature Selection) and report whether it's confirmed.

    Args:
        split (DataSplit): Split confirmed at Step 3 — selection fits a
            ``FeatureTransformChain`` (from Step 5's confirmed recipes) on
            ``split.train_X``, then runs selectors on that transformed
            output — mirroring how ``PipelineRunner`` runs feature selection
            after (not before) feature processing. Identifier columns
            (Step 2), the date column (Step 3), and columns already dropped
            at Step 5 are excluded from the candidate set entirely — none of
            those is a feature selection decision.
        target (str): Confirmed target column name. Unused directly here
            (``train_X`` already excludes it) — accepted for signature
            symmetry with ``render_step4``/``render_step5``.

    Returns:
        bool: ``True`` once the user clicks "Continue". ``False`` while
        still reviewing.
    """
    st.subheader("Step 6: Feature selection")
    st.caption(
        "dscompanion runs a chain of checks (missing rate, constant/near-constant values, "
        "high cardinality, high correlation, and, for classification, low predictive "
        "value) to propose features to drop. Nothing is removed until you confirm, "
        "and you can keep any flagged feature by unchecking its row below."
    )

    date_col = st.session_state.get("int.split.date_col")
    identifier_columns = st.session_state.get("interactive.identifier_columns") or []
    dropped_at_step5 = st.session_state.get("interactive.feature_processing_dropped_columns") or []
    excluded_cols = {c for c in [date_col, *identifier_columns, *dropped_at_step5] if c}
    candidate_cols = [c for c in split.train_X.columns if c not in excluded_cols]
    task = st.session_state.get("interactive.task", "classification")

    recipes: dict[str, ColumnRecipe] = (
        st.session_state.get("interactive.feature_processing_recipes") or {}
    )
    recipes_for_candidates = {c: r for c, r in recipes.items() if c in candidate_cols}

    try:
        chain = FeatureTransformChain(recipes=recipes_for_candidates)
        transformed_X = chain.fit_transform(split.train_X[candidate_cols])
        sel_pipeline = FeatureSelectionPipeline(selectors=_build_selectors(task))
        sel_pipeline.fit(transformed_X, split.train_y)
    except Exception as exc:
        logger.exception("Interactive mode — feature selection failed")
        st.error(f"Couldn't run feature selection: {exc}")
        if st.button(
            "Continue without feature selection", key="int.feature_selection.continue_after_error"
        ):
            st.session_state["interactive.feature_selection_removed_columns"] = []
            set_step_confirmed("feature_selection", True)
            set_step_status("feature_selection", "flagged")
            add_audit_entry("feature_selection", "Feature selection failed. Skipped.")
        return st.session_state["interactive.step_confirmed"]["feature_selection"]

    audit = sel_pipeline.audit_report()

    if audit.empty:
        st.success("No features were flagged for removal. Nothing to review here.")
        if st.button("Continue", key="int.feature_selection.continue_empty"):
            st.session_state["interactive.feature_selection_removed_columns"] = []
            set_step_confirmed("feature_selection", True)
            set_step_status("feature_selection", "done")
            add_audit_entry(
                "feature_selection", "Feature selection: no columns flagged for removal."
            )
        return st.session_state["interactive.step_confirmed"]["feature_selection"]

    st.plotly_chart(dark_fig(sel_pipeline.plot_removal_waterfall()), width="stretch", theme=None)

    if len(audit) >= transformed_X.shape[1]:
        st.error(
            "All features were flagged for removal, which would leave nothing to train "
            "on. Review the boxes below and uncheck at least one, or go back and adjust "
            "Step 5."
        )

    header = st.columns([2, 1.6, 3, 1])
    for label, c in zip(["Feature", "Check", "Reason", "Remove?"], header):
        c.markdown(f"**{label}**")
    st.divider()

    n_to_remove = 0
    for row in audit.itertuples():
        cols = st.columns([2, 1.6, 3, 1])
        cols[0].markdown(f"**{row.feature}**")
        cols[1].write(_STAGE_LABELS.get(row.removed_by, row.removed_by))
        cols[2].write(row.reason)
        remove = cols[3].checkbox(
            "Remove",
            value=True,
            key=f"int.feature_selection.remove.{row.feature}",
            label_visibility="collapsed",
        )
        if remove:
            n_to_remove += 1
        st.divider()

    if n_to_remove == 0:
        st.warning("Every proposed removal is unchecked. Nothing will be removed if you continue.")

    if st.button(
        f"Remove these {n_to_remove} feature(s) and continue", key="int.feature_selection.continue"
    ):
        removed_columns = [
            row.feature
            for row in audit.itertuples()
            if st.session_state.get(f"int.feature_selection.remove.{row.feature}", True)
        ]
        if len(removed_columns) >= transformed_X.shape[1]:
            st.error("Cannot remove every feature. Uncheck at least one row above.")
        else:
            st.session_state["interactive.feature_selection_removed_columns"] = removed_columns
            set_step_confirmed("feature_selection", True)
            set_step_status("feature_selection", "done")

            by_stage: dict[str, int] = {}
            for row in audit.itertuples():
                if row.feature in removed_columns:
                    by_stage[row.removed_by] = by_stage.get(row.removed_by, 0) + 1
            if by_stage:
                parts = [
                    f"removed {n} {_STAGE_AUDIT_NOUNS.get(stage, stage)}"
                    for stage, n in by_stage.items()
                ]
                add_audit_entry("feature_selection", "Feature selection: " + "; ".join(parts) + ".")
            else:
                add_audit_entry(
                    "feature_selection", "Feature selection: no columns removed (all kept)."
                )

    return st.session_state["interactive.step_confirmed"]["feature_selection"]
