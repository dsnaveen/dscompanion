"""Interactive mode — Step 13: Review & Generate Report.

The final step in the Interactive pipeline. Shows the full audit trail of
every confirmed decision as the centrepiece, lets the user add optional
author / use-case metadata, then generates an Excel model card, an HTML
report, and a YAML config snapshot on demand — reusing dscompanion's existing
``ModelCard`` machinery. ``ModelCard.generate()`` + ``to_excel()``/``to_html()``
are expensive; the output bytes are cached in session state on first
generation and served from cache on subsequent rerenders.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st
import yaml
from interactive_state import add_audit_entry, set_step_confirmed, set_step_status

from dscompanion.docs.model_card import ModelCard

logger = logging.getLogger(__name__)

__all__ = ["render_step_report"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config_yaml() -> str:
    """Assemble a YAML snapshot of the confirmed pipeline choices.

    Reads directly from Interactive-mode session state so no import from
    ``streamlit_app`` is needed (avoiding circular imports).  Covers Steps
    1-12 — only sections confirmed so far are included.  ``data.feature_columns``
    (Step 5's surviving candidate set), ``eda`` (thresholds actually used),
    and ``feature_recipes`` (the exact per-column decisions) are included
    so the exported YAML can reproduce what this session actually did, not
    just its top-level model/split/tuning choices.

    Args:
        None

    Returns:
        str: YAML-formatted config string ready for download.
    """
    cfg: dict = {}

    sel = st.session_state.get("interactive.load_data_selection_id")
    if sel:
        path, fmt, sheet = sel
        cfg["data"] = {"path": str(path), "format": fmt}
        if sheet:
            cfg["data"]["sheet_name"] = sheet

    step_confirmed = st.session_state["interactive.step_confirmed"]

    if step_confirmed["task_target"]:
        cfg.setdefault("data", {})["target"] = st.session_state.get("interactive.target")
        cfg["model"] = {"task": st.session_state.get("interactive.task")}

    if step_confirmed["split"]:
        cfg["split"] = {
            "method": st.session_state.get("int.split._method"),
            "test_size": st.session_state.get("int.split._test_size"),
            "val_size": st.session_state.get("int.split._val_size"),
        }
        group_col = st.session_state.get("int.split.group_col")
        if group_col:
            cfg["split"]["group_column"] = group_col

    if step_confirmed["eda"]:
        thresholds = st.session_state.get("interactive.eda_thresholds") or {}
        cfg["eda"] = {
            # The recipes/drops this threshold pass informed are already
            # captured explicitly below (feature_recipes, feature_columns) —
            # a batch reproduction of this exact run applies those directly
            # rather than re-running the (expensive) EDA stage itself.
            "enabled": False,
            "high_missing_threshold": thresholds.get("high_missing"),
            "near_zero_variance_threshold": thresholds.get("near_zero_variance"),
        }

    if step_confirmed["feature_processing"]:
        recipes = st.session_state.get("interactive.feature_processing_recipes") or {}
        if recipes:
            cfg["feature_recipes"] = {
                col: recipe.model_dump(exclude_none=True) for col, recipe in recipes.items()
            }

        raw_split = st.session_state.get("interactive.split")
        if raw_split is not None:
            date_col = st.session_state.get("int.split.date_col")
            identifier_columns = st.session_state.get("interactive.identifier_columns") or []
            dropped_columns = (
                st.session_state.get("interactive.feature_processing_dropped_columns") or []
            )
            excluded = {c for c in [date_col, *identifier_columns, *dropped_columns] if c}
            feature_columns = [c for c in raw_split.train_X.columns if c not in excluded]
            cfg.setdefault("data", {})["feature_columns"] = feature_columns

    if step_confirmed["imbalance"]:
        cfg.setdefault("target", {})["imbalance"] = {
            "strategy": st.session_state.get("interactive.imbalance_strategy", "class_weight")
        }

    if step_confirmed["train"]:
        cfg.setdefault("model", {})["algorithm"] = st.session_state.get(
            "interactive.train_algorithm"
        )

    if step_confirmed["tuning"]:
        cfg["tuning"] = {"enabled": bool(st.session_state.get("interactive.tuning_tuned"))}

    # Calibration is not a PipelineConfig field — PipelineRunner always
    # calibrates unconditionally (Stage 10 in runner.py) for classification
    # tasks, so there is no on/off knob to emit here. Whether this session
    # applied it is still visible in the audit trail and the model card's
    # own calibration section.

    if step_confirmed["shap"]:
        shap_run = st.session_state.get("int.shap.explainer") is not None
        cfg["explain"] = {"shap_enabled": shap_run}

    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _build_model_card(author: str, use_case: str) -> ModelCard:
    """Assemble and generate a ``ModelCard`` from Interactive-mode session state.

    Args:
        author (str): Free-text author name for the report header.
        use_case (str): Free-text use-case description for the report header.

    Returns:
        ModelCard: A fully generated (``generate()``-called) model card.

    Raises:
        Exception: Propagated from ``ModelCard.generate()`` — caught and shown
            by the caller.
    """
    model = st.session_state.get("int.calibration.model")
    # The processed split (post feature-processing/selection), not the raw
    # Step 3 split — the model was fit and evaluated against this column set
    # throughout Steps 8-12, so ModelCard.generate()'s internal
    # model.evaluate(self.split)/predict_proba(self.split.X_test) calls need
    # the same one or every metrics/decile/stability/EDA section silently
    # fails on a column mismatch and falls back to an empty "note" placeholder.
    split = st.session_state.get("int.train.processed_split")
    explainer = st.session_state.get("int.shap.explainer")
    calibrator = (
        st.session_state.get("int.calibration.calibrator")
        if st.session_state.get("interactive.calibration_applied")
        else None
    )
    eda_report = st.session_state.get("interactive.eda_report")
    tuner = (
        st.session_state.get("int.tuning.tuner")
        if st.session_state.get("interactive.tuning_tuned")
        else None
    )
    return ModelCard(
        model=model,
        split=split,
        explainer=explainer,
        calibrator=calibrator,
        eda_report=eda_report,
        tuner=tuner,
        author=author,
        use_case=use_case,
    ).generate()


def _render_pipeline_summary() -> None:
    """Render a four-tile quick-glance summary of the final pipeline choices.

    Args:
        None

    Returns:
        None
    """
    algorithm = st.session_state.get("interactive.train_algorithm", "—")
    tuned = bool(st.session_state.get("interactive.tuning_tuned"))
    calibrated = bool(st.session_state.get("interactive.calibration_applied"))
    shap_run = st.session_state.get("int.shap.explainer") is not None

    cols = st.columns(4)
    cols[0].metric("Algorithm", algorithm)
    cols[1].metric("Tuned", "Yes" if tuned else "No")
    cols[2].metric("Calibrated", "Yes" if calibrated else "No")
    cols[3].metric("SHAP", "Yes" if shap_run else "No")


def _render_audit_trail_expanded() -> None:
    """Render the full audit trail as an expanded list — the centrepiece of Step 13.

    Unlike ``render_audit_trail()`` in ``interactive_state.py``, this renders
    the entries flat and always-visible (not inside a collapsible expander)
    because Step 13's purpose is to review before committing.

    Args:
        None

    Returns:
        None
    """
    trail = st.session_state.get("interactive.audit_trail", [])
    if not trail:
        st.caption("No decisions were recorded during this session.")
        return
    for _step, text in trail:
        st.write(f"- {text}")


def _generate_and_cache(author: str, use_case: str) -> tuple[bool, str | None]:
    """Generate the model card and cache Excel + HTML bytes in session state.

    Runs ``ModelCard.generate()`` once, then ``to_excel()``/``to_html()``
    in a single temporary directory.  Bytes are stored under
    ``int.report.xlsx_bytes``/``int.report.html_bytes`` to avoid re-running
    the expensive generation on subsequent rerenders. Each export format
    fails independently — an HTML export failure does not block the Excel
    download, and vice versa.

    Args:
        author (str): Author name passed to ``ModelCard``.
        use_case (str): Use-case description passed to ``ModelCard``.

    Returns:
        tuple[bool, str | None]: ``(success, error_message)`` — exactly one
        is meaningful; ``error_message`` is ``None`` on success.
    """
    try:
        card = _build_model_card(author=author, use_case=use_case)
    except Exception as exc:
        logger.exception("Interactive mode — ModelCard.generate() failed")
        return False, str(exc)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            xlsx_bytes = card.to_excel(tmp_path / "model_card.xlsx").read_bytes()
            st.session_state["int.report.xlsx_bytes"] = xlsx_bytes
        except Exception as exc:
            logger.warning("Excel report generation failed: %s", type(exc).__name__)
            st.session_state["int.report.xlsx_bytes"] = None

        try:
            html_bytes = card.to_html(tmp_path / "model_card.html").read_bytes()
            st.session_state["int.report.html_bytes"] = html_bytes
        except Exception as exc:
            logger.warning("HTML report generation failed: %s", type(exc).__name__)
            st.session_state["int.report.html_bytes"] = None

    st.session_state["int.report.report_generated"] = True
    return True, None


def _render_download_buttons(config_yaml: str) -> None:
    """Render the three download buttons (YAML config, Excel report, HTML report).

    Args:
        config_yaml (str): Pre-built YAML config string.

    Returns:
        None
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download config (YAML)",
            config_yaml.encode(),
            file_name="interactive_config.yaml",
            mime="text/yaml",
            width="stretch",
        )

    xlsx_bytes = st.session_state.get("int.report.xlsx_bytes")
    with col2:
        if xlsx_bytes is not None:
            st.download_button(
                "Download Excel report",
                xlsx_bytes,
                file_name="model_card.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        else:
            st.caption("Excel report could not be generated.")

    html_bytes = st.session_state.get("int.report.html_bytes")
    with col3:
        if html_bytes is not None:
            st.download_button(
                "Download HTML report",
                html_bytes,
                file_name="model_card.html",
                mime="text/html",
                width="stretch",
            )
        else:
            st.caption("HTML report could not be generated.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_step_report() -> bool:
    """Render Step 13 (Review & Generate Report) and report whether it is done.

    Args:
        None

    Returns:
        bool: ``True`` once the model card has been generated successfully.
        ``False`` while still waiting for the user to generate.
    """
    st.subheader("Step 13: Review & generate report")
    st.caption(
        "Review every decision made during this session, add your name and a short "
        "use-case description, then generate the model card."
    )

    st.markdown("**Decisions Made During This Session**")
    _render_audit_trail_expanded()

    st.divider()
    st.markdown("**Pipeline Summary**")
    _render_pipeline_summary()

    st.divider()
    st.markdown("**Report Metadata (Optional)**")
    author = st.text_input(
        "Author name",
        key="int.report.author",
        placeholder="e.g. Jane Doe",
    )
    use_case = st.text_area(
        "Use case description",
        key="int.report.use_case",
        placeholder=("e.g. Customer churn model, Q3 2026"),
        height=80,
    )

    st.divider()

    config_yaml = _build_config_yaml()

    if not st.session_state.get("int.report.report_generated"):
        st.info(
            "Review the decisions above. When ready, click **Generate Report** to "
            "create the Excel and HTML model card."
        )
        col_gen, col_yaml = st.columns([2, 1])
        with col_gen:
            if st.button(
                "Generate report",
                key="int.report.generate",
                type="primary",
                width="stretch",
            ):
                with st.spinner("Generating model card. This may take a moment..."):
                    ok, err = _generate_and_cache(author=author, use_case=use_case)
                if ok:
                    set_step_status("report", "done")
                    add_audit_entry("report", "Model card generated and downloaded.")
                    set_step_confirmed("report", True)
                    st.rerun()
                else:
                    st.error(f"Report generation failed: {err}")
                    st.caption(
                        "The model card could not be created. See the error above. "
                        "You can still download the YAML config."
                    )
        with col_yaml:
            st.download_button(
                "Download config (YAML)",
                config_yaml.encode(),
                file_name="interactive_config.yaml",
                mime="text/yaml",
                width="stretch",
            )
        return False

    st.success(
        "Model card generated. Download your reports below. "
        "You can regenerate with updated metadata at any time."
    )
    _render_download_buttons(config_yaml)

    if st.button("Regenerate report", key="int.report.regenerate"):
        st.session_state.pop("int.report.report_generated", None)
        st.session_state.pop("int.report.xlsx_bytes", None)
        st.session_state.pop("int.report.html_bytes", None)
        st.rerun()

    return True
