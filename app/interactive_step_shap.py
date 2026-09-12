"""Interactive mode — Step 12: Explainability (SHAP).

Read-only step — no confirm gate on the chart itself.  A toggle (on by default)
lets the user skip SHAP if they don't need it.  When run, fits
``SHAPExplainer`` on a sample of the validation split and shows a summary bar
chart of mean |SHAP| importance with a plain-language interpretation guide
above it.  The result is cached so repeated rerenders don't re-compute.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from interactive_state import (
    add_audit_entry,
    dark_fig,
    reset_from_step,
    set_step_confirmed,
    set_step_status,
)

logger = logging.getLogger(__name__)

__all__ = ["render_step_shap"]

# Max rows sampled from the split before running SHAP — keeps UI responsive.
_SHAP_MAX_ROWS = 500


# ── Data / model helpers ──────────────────────────────────────────────────────


def _final_model():
    """Return the model from Step 11 (calibrated or raw).

    Args:
        None

    Returns:
        object | None: The model stored at ``int.calibration.model``, or ``None``
        if Step 11 has not confirmed.
    """
    return st.session_state.get("int.calibration.model")


def _shap_data() -> tuple[pd.DataFrame, str] | None:
    """Select a sample from the best available held-out split for SHAP.

    Prefers val, then test, then train.  Caps rows at ``_SHAP_MAX_ROWS`` to
    keep computation time acceptable in the interactive UI.

    Args:
        None

    Returns:
        tuple[pd.DataFrame, str] | None: ``(X_sample, split_name)`` or ``None``
        if no processed split is available.
    """
    split = st.session_state.get("int.train.processed_split")
    if split is None:
        return None

    for attr_x, name in (("val_X", "val"), ("test_X", "test"), ("train_X", "train")):
        X = getattr(split, attr_x, None)
        if X is not None and len(X) > 0:
            if len(X) > _SHAP_MAX_ROWS:
                X = X.sample(_SHAP_MAX_ROWS, random_state=42)
            return X, name
    return None


# ── SHAP computation ──────────────────────────────────────────────────────────


def _run_shap() -> tuple | None:
    """Fit (or return cached) SHAPExplainer and the summary figure.

    Caches the fitted explainer in ``int.shap.explainer`` and the Plotly
    figure in ``int.shap.fig`` so rerenders don't re-compute.

    Args:
        None

    Returns:
        tuple | None: ``(explainer, fig, split_name, n_rows)`` or ``None``
        if the model, split, or SHAP import fails.
    """
    cached_exp = st.session_state.get("int.shap.explainer")
    cached_fig = st.session_state.get("int.shap.fig")
    if cached_exp is not None and cached_fig is not None:
        return (
            cached_exp,
            cached_fig,
            st.session_state.get("int.shap.split_used", "val"),
            st.session_state.get("int.shap.n_rows", 0),
        )

    model = _final_model()
    data = _shap_data()
    if model is None or data is None:
        return None

    X, split_name = data

    try:
        from dscompanion.explain.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(model)
        explainer.fit(X)
        fig = explainer.summary_plot()

        st.session_state["int.shap.explainer"] = explainer
        st.session_state["int.shap.fig"] = fig
        st.session_state["int.shap.split_used"] = split_name
        st.session_state["int.shap.n_rows"] = len(X)
        logger.info(
            "Step 12: SHAP fitted on %s split, %d rows, %d features",
            split_name,
            len(X),
            len(explainer.feature_names_),
        )
        return explainer, fig, split_name, len(X)
    except Exception as exc:
        logger.warning("Step 12 SHAP computation failed: %s", exc)
        return None


# ── Confirm ───────────────────────────────────────────────────────────────────


def _confirm_step12(shap_run: bool, top_feature: str | None = None) -> None:
    """Record Step 12 completion and write an audit entry.

    Args:
        shap_run (bool): Whether SHAP analysis was actually computed.
        top_feature (str | None): Name of the most important feature, for
            the audit trail.  Pass ``None`` when SHAP was skipped.

    Returns:
        None
    """
    if shap_run and top_feature:
        note = f"SHAP run — top feature: {top_feature}."
    else:
        note = "SHAP skipped by user."

    add_audit_entry("shap", note)
    set_step_confirmed("shap", True)
    set_step_status("shap", "done")
    logger.info("Interactive mode — Step 12 confirmed: %s", note)


# ── Main entrypoint ───────────────────────────────────────────────────────────


def render_step_shap() -> bool:
    """Render Step 12 (Explainability — SHAP) and report confirmation.

    Shows a toggle (on by default) that lets the user opt out.  When on,
    fits ``SHAPExplainer`` on a sample of the validation split, renders the
    summary bar chart with a plain-language guide above it, and offers a
    "Continue to Step 13" button.  No confirm gate on the chart — the step
    is read-only.

    Args:
        None

    Returns:
        bool: ``True`` once the user clicks Continue; ``False`` while
        reviewing.
    """
    if st.session_state["interactive.step_confirmed"]["shap"]:
        return True

    st.subheader("Step 12: Explainability (SHAP)")
    st.caption(
        "See which features influenced the model's predictions the most. "
        "This doesn't change the model. It only helps you understand it."
    )
    st.info(
        "Off by default: SHAP computation is relatively expensive, so it's opt-in "
        "rather than run automatically."
    )

    run_shap = st.checkbox(
        "Run SHAP analysis",
        value=False,
        key="int.shap.toggle",
        help="Off by default on this cluster — see the note above. Check to run SHAP anyway.",
    )

    if not run_shap:
        st.info("SHAP analysis is turned off. Click Continue when ready.")
        if st.button("Continue to Step 13", key="int.shap.continue_skip", type="primary"):
            _confirm_step12(shap_run=False)
        return st.session_state["interactive.step_confirmed"]["shap"]

    # ── Run SHAP ───────────────────────────────────────────────────────────
    with st.spinner("Computing SHAP values. This may take up to a minute for tree models…"):
        result = _run_shap()

    if result is None:
        st.error(
            "Could not compute SHAP values. The model from Step 11 or the processed "
            "split from Step 8 is missing. Please go back and re-run Steps 8–11."
        )
        if st.button("Go back to Step 8 and retrain", key="int.shap.back_to_train"):
            reset_from_step("train")
            st.rerun()
        return False

    explainer, fig, split_name, n_rows = result

    # ── Interpretation guide ───────────────────────────────────────────────
    st.markdown("**How to Read This Chart**")
    st.caption(
        "Each bar represents one feature. "
        "**Features at the top had the biggest average impact** on the model's predictions: "
        "longer bar = stronger influence. "
        "The values shown are mean absolute SHAP values: on average, how much that feature "
        "moved the model's score, across all rows in the sample. "
        f"Computed on a sample of **{n_rows} rows** from the **{split_name}** split."
    )

    # ── Chart ─────────────────────────────────────────────────────────────
    st.plotly_chart(dark_fig(fig), width="stretch")

    # ── Top feature table ──────────────────────────────────────────────────
    importance_df = (
        explainer.mean_abs_shap()
        .head(10)
        .rename(columns={"feature": "Feature", "mean_abs_shap": "Mean |SHAP|", "rank": "Rank"})
    )
    with st.expander("Top 10 features: Full numbers", expanded=False):
        st.dataframe(
            importance_df[["Rank", "Feature", "Mean |SHAP|"]],
            hide_index=True,
            width="stretch",
        )

    top_feature = (
        explainer.mean_abs_shap()["feature"].iloc[0] if len(explainer.feature_names_) > 0 else None
    )

    st.divider()
    if st.button("Continue to Step 13", key="int.shap.continue", type="primary"):
        _confirm_step12(shap_run=True, top_feature=top_feature)

    return st.session_state["interactive.step_confirmed"]["shap"]
