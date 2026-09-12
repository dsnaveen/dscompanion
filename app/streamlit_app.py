"""DS Companion — local Streamlit UI.

Pick a dataset, set a task/algorithm, tweak config via widgets, run the full
dscompanion pipeline, and view results — all in a browser. Local-dev-only tool:
streamlit is not an approved Databricks cluster package and this directory
is never imported by dscompanion's runtime code.

Run with::

    streamlit run dscompanion/app/streamlit_app.py
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

# streamlit run adds this script's directory to sys.path automatically, but
# other entry points (e.g. AppTest.from_file, used in smoke tests) don't —
# insert it explicitly so sibling-module imports below always resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import yaml
from data_browser import peek_columns, render_data_source
from glossary import render_glossary
from interactive_state import (
    STEP_ORDER,
    StepId,
    init_interactive_state,
    render_audit_trail,
    render_back_button,
    render_completed_steps,
    render_stepper,
)
from interactive_step_calibration import render_step_calibration
from interactive_step_eda import render_step_eda
from interactive_step_evaluate import render_step_evaluate
from interactive_step_feature_processing import render_step_feature_processing
from interactive_step_feature_selection import render_step_feature_selection
from interactive_step_imbalance import render_step_imbalance
from interactive_step_load_data import render_step_load_data
from interactive_step_report import render_step_report
from interactive_step_shap import render_step_shap
from interactive_step_split import render_step_split
from interactive_step_task_target import render_step_task_target
from interactive_step_train import render_step_train
from interactive_step_tuning import render_step_tuning
from pydantic import ValidationError
from results_view import render_results
from widgets import build_config_dict, render_advanced_widgets, render_core_widgets

from dscompanion.pipeline import PipelineConfig, PipelineRunner

logger = logging.getLogger(__name__)

st.set_page_config(page_title="DS Companion", page_icon=":material/model_training:", layout="wide")

# Icons for the top-level run modes, applied via st.segmented_control's
# format_func in main() rather than embedded in the options list directly — this keeps
# the widget's returned value the plain "Interactive"/"Full Pipeline"/"Glossary"
# string every existing `if run_mode == "..."` check already compares against.
_RUN_MODE_ICONS = {
    "Interactive": ":material/checklist:",
    "Full Pipeline": ":material/settings:",
    "Glossary": ":material/menu_book:",
}

# One no-arg closure per step slug — kept explicit rather than a generic loop
# because (unlike React's step components, which self-serve from a shared store
# with no props) each render_stepN() function here needs different prior-step
# arguments (render_step_split needs df/target, render_step_eda needs
# split/target, etc.). Session-state lookups happen inside each lambda body,
# not at dict-construction time, so building this table never risks a
# KeyError for a not-yet-confirmed step's data.
_STEP_DISPATCH: dict[StepId, Callable[[], None]] = {
    "load_data": render_step_load_data,
    "task_target": lambda: render_step_task_target(st.session_state["interactive.df"]),
    "split": lambda: render_step_split(
        st.session_state["interactive.df"], st.session_state["interactive.target"]
    ),
    "eda": lambda: render_step_eda(
        st.session_state["interactive.split"], st.session_state["interactive.target"]
    ),
    "feature_processing": lambda: render_step_feature_processing(
        st.session_state["interactive.split"], st.session_state["interactive.target"]
    ),
    "feature_selection": lambda: render_step_feature_selection(
        st.session_state["interactive.split"], st.session_state["interactive.target"]
    ),
    "imbalance": lambda: render_step_imbalance(st.session_state["interactive.split"]),
    "train": lambda: render_step_train(st.session_state["interactive.split"]),
    "tuning": render_step_tuning,
    "evaluate": render_step_evaluate,
    "calibration": render_step_calibration,
    "shap": render_step_shap,
    "report": render_step_report,
}


def _init_session_state() -> None:
    """Set every session-state default the app relies on, once per session.

    Args:
        None

    Returns:
        None
    """
    defaults = {
        "run_result": None,
        "run_error": None,
        "is_running": False,
        "upload_tmp_dir": None,
        "uci_metadata": None,
        "selected_data_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _build_partial_config_dict() -> dict:
    """Assemble the slice of a real ``PipelineConfig`` dict decided so far.

    Mirrors ``PipelineConfig``'s actual nested field names (``data.path``,
    ``model.task``, ``split.method``, ...) so what's shown is recognizably
    "config.py", not an app-invented summary — but only includes a section
    once the step that decides it has been confirmed, so the document grows
    exactly in step with the user's own confirmed decisions.

    Args:
        None

    Returns:
        dict: Nested dict in ``PipelineConfig``'s shape, containing only the
        fields decided by confirmed steps so far.
    """
    cfg: dict = {}
    step_confirmed = st.session_state["interactive.step_confirmed"]

    if step_confirmed["load_data"]:
        path, fmt, sheet_name = st.session_state.get(
            "interactive.load_data_selection_id", (None, None, None)
        )
        cfg["data"] = {"path": path, "format": fmt}
        if sheet_name:
            # Not a real PipelineConfig field (XLSX isn't a PipelineRunner-
            # native format, see streamlit.md's Step 1 design) — called out
            # as app-only so this doesn't look like a config.py field.
            cfg["data"]["_sheet_name (app-only, not a PipelineConfig field)"] = sheet_name

    if step_confirmed["task_target"]:
        cfg.setdefault("data", {})["target"] = st.session_state["interactive.target"]
        cfg["model"] = {"task": st.session_state["interactive.task"]}
        encoding = st.session_state.get("interactive.target_encoding")
        if encoding:
            cfg["data"]["_target_encoding (app-only, not a PipelineConfig field)"] = encoding
        identifier_columns = st.session_state.get("interactive.identifier_columns") or []
        if identifier_columns:
            # Not a real PipelineConfig field on its own — it's folded into
            # data.feature_columns once Step 5 confirms (the earliest point
            # the full exclusion list, including date_column and missing-
            # value drops, is known). Shown here immediately so the choice
            # is visible right away, per this preview's own "familiarize as
            # you go" purpose.
            cfg["data"]["ignored_columns"] = identifier_columns

    if step_confirmed["split"]:
        split_cfg = {
            "method": st.session_state.get("int.split._method"),
            "test_size": st.session_state.get("int.split._test_size"),
            "val_size": st.session_state.get("int.split._val_size"),
        }
        group_col = st.session_state.get("int.split.group_col")
        if group_col:
            split_cfg["group_column"] = group_col
        cfg["split"] = split_cfg

    if step_confirmed["eda"]:
        eda_cfg = {"enabled": bool(st.session_state.get("int.eda.run_eda"))}
        thresholds = st.session_state.get("interactive.eda_thresholds")
        if thresholds:
            eda_cfg["high_missing_threshold"] = thresholds["high_missing"]
            eda_cfg["near_zero_variance_threshold"] = thresholds["near_zero_variance"]
        cfg["eda"] = eda_cfg

    if step_confirmed["feature_processing"]:
        dropped = st.session_state.get("interactive.feature_processing_dropped_columns") or []
        identifier_columns = st.session_state.get("interactive.identifier_columns") or []
        if dropped or identifier_columns:
            all_cols = list(st.session_state["interactive.split"].train_X.columns)
            date_col = st.session_state.get("int.split.date_col")
            excluded = {c for c in [date_col, *dropped, *identifier_columns] if c}
            data_cfg = cfg.setdefault("data", {})
            data_cfg.pop("ignored_columns", None)
            data_cfg["feature_columns"] = [c for c in all_cols if c not in excluded]
        recipes = st.session_state.get("interactive.feature_processing_recipes") or {}
        if recipes:
            cfg["feature_recipes"] = {
                col: recipe.model_dump(exclude_defaults=True) for col, recipe in recipes.items()
            }

    if step_confirmed["feature_selection"]:
        removed_at_step6 = (
            st.session_state.get("interactive.feature_selection_removed_columns") or []
        )
        if removed_at_step6:
            dropped = st.session_state.get("interactive.feature_processing_dropped_columns") or []
            identifier_columns = st.session_state.get("interactive.identifier_columns") or []
            all_cols = list(st.session_state["interactive.split"].train_X.columns)
            date_col = st.session_state.get("int.split.date_col")
            # removed_at_step6 names are post-FeatureTransformChain column
            # names — a column-expanding recipe step (e.g. one-hot encoding)
            # means most of them aren't raw column names at all, so they
            # can't be expressed via data.feature_columns (a pre-transform
            # allowlist). Only the subset that happen to still be raw column
            # names (1:1-transform or passthrough features) can be excluded
            # this way — a known limitation of the exported config's
            # fidelity versus what Step 6 actually removed; re-running
            # PipelineRunner's own selection stage on this config will
            # independently re-derive a similar (not necessarily identical)
            # removal set from the same default selectors.
            removable_raw_names = {c for c in removed_at_step6 if c in all_cols}
            excluded = {
                c for c in [date_col, *dropped, *identifier_columns, *removable_raw_names] if c
            }
            cfg.setdefault("data", {})["feature_columns"] = [
                c for c in all_cols if c not in excluded
            ]

    if step_confirmed["imbalance"]:
        strategy = st.session_state.get("interactive.imbalance_strategy", "class_weight")
        cfg.setdefault("target", {})["imbalance"] = {"strategy": strategy}

    if step_confirmed["train"]:
        algorithm = st.session_state.get("interactive.train_algorithm", "xgboost")
        cfg.setdefault("model", {})["algorithm"] = algorithm

    if step_confirmed["tuning"]:
        tuned = st.session_state.get("interactive.tuning_tuned", False)
        cfg.setdefault("tuning", {})["enabled"] = tuned

    return cfg


def _render_config_preview() -> None:
    """Render the live, growing ``PipelineConfig`` YAML for decisions confirmed so far.

    Lets the user watch their own step-by-step choices take shape as the
    real config object the pipeline will eventually run with — visible and
    expanded by default, since the point is familiarization, not an
    incidental audit detail.

    Args:
        None

    Returns:
        None
    """
    partial = _build_partial_config_dict()
    with st.expander(
        ":material/description: Current config.py state",
        expanded=True,
    ):
        if not partial:
            st.caption("Nothing confirmed yet. This fills in as you complete each step.")
            return
        st.code(
            yaml.dump(partial, default_flow_style=False, sort_keys=False, allow_unicode=True),
            language="yaml",
        )
        st.caption(
            "Only sections decided by a confirmed step are shown. Everything else "
            "(model algorithm, feature processing, and so on) is still undecided."
        )


def _first_unconfirmed_step(step_confirmed: dict[StepId, bool]) -> StepId | None:
    """Returns the first not-yet-confirmed step slug in wizard order, or ``None``
    once every step is confirmed.

    Args:
        step_confirmed (dict[StepId, bool]): This run's confirmation flags.

    Returns:
        StepId | None: The step slug to render next, or ``None`` in the
        terminal (all-steps-confirmed) state.
    """
    return next((s for s in STEP_ORDER if not step_confirmed[s]), None)


def _render_interactive_mode() -> None:
    """Render Interactive mode: one pipeline stage at a time, gated on confirmation.

    Args:
        None

    Returns:
        None
    """
    init_interactive_state()
    render_stepper()
    render_audit_trail()

    # Stacked above the active step, each collapsed by default — lets a
    # completed step's decisions stay reachable without occupying the
    # screen space its live form used to. See render_completed_steps()'s
    # own docstring for how this maps to the notebook-stacking pattern.
    render_completed_steps()

    step_confirmed = st.session_state["interactive.step_confirmed"]

    # Computed before the dispatch loop so the Back button sits above the step
    # content and labels the correct target step.
    render_back_button(_first_unconfirmed_step(step_confirmed))

    # Reserve the visual slot here (top of the page, alongside the audit
    # trail) but fill it only after the dispatch loop below — a step
    # confirmed during *this* script pass (e.g. clicking Step 1's "Confirm &
    # Continue") only flips its state flag inside that loop, further down;
    # filling the placeholder now would always show last run's state.
    config_preview_slot = st.empty()

    # render_step_*() only mutates state — it never forces a rerun across a
    # step boundary, forward or backward (e.g. Task & Target's "Go back to
    # Step 1" recovery button just resets load_data's confirmed flag). Looping
    # here and re-checking confirmation flags after each render lets a single
    # click jump straight to whichever step is now current, within the same
    # script pass. Stop as soon as a render leaves the flags unchanged — that
    # means the step is still waiting on user input, so rendering it again
    # would create duplicate widgets with the same keys.
    for _ in range(len(STEP_ORDER) + 1):
        before = dict(step_confirmed)
        current = _first_unconfirmed_step(step_confirmed)
        if current is not None:
            _STEP_DISPATCH[current]()
        else:
            # Report has no re-render guard (unlike every other step) — it's
            # designed to stay visible after confirmation so the download
            # buttons remain reachable. Keep calling it here instead of
            # replacing it with this summary; the summary below explicitly
            # says "Download your reports from Step 13 above", so Report must
            # actually still be rendered.
            render_step_report()
            algorithm = st.session_state.get("interactive.train_algorithm", "?")
            tuned = st.session_state.get("interactive.tuning_tuned", False)
            calibrated = st.session_state.get("interactive.calibration_applied", False)
            st.success(
                f"All {len(STEP_ORDER)} steps are complete. "
                f"Task: {st.session_state['interactive.task']}. "
                f"Target: '{st.session_state['interactive.target']}'. "
                f"Model: '{algorithm}'"
                + (" (Tuned)" if tuned else "")
                + (" (Calibrated)" if calibrated else "")
                + ". Download your reports from Step 13 above."
            )
            break
        if dict(step_confirmed) == before:
            break

    with config_preview_slot.container():
        _render_config_preview()


def main() -> None:
    """Render the full app: mode toggle, then either Interactive steps or Full Pipeline.

    Args:
        None

    Returns:
        None
    """
    _init_session_state()
    st.title(":material/model_training: DS Companion")

    run_mode = st.segmented_control(
        "Run mode",
        ["Interactive", "Full Pipeline", "Glossary"],
        format_func=lambda mode: f"{_RUN_MODE_ICONS[mode]} {mode}",
        default="Interactive",
        required=True,
        key="run_mode",
        label_visibility="collapsed",
    )
    if run_mode == "Interactive":
        _render_interactive_mode()
        return
    if run_mode == "Glossary":
        render_glossary()
        return

    st.header(":material/upload_file: 1. Data")
    data_path, data_format = render_data_source()
    columns = peek_columns(data_path) if data_path else []

    st.header(":material/tune: 2. Configuration")
    render_core_widgets(columns)
    render_advanced_widgets(columns)

    st.header(":material/play_arrow: 3. Run")
    run_clicked = st.button(
        "Run pipeline",
        icon=":material/rocket_launch:",
        disabled=st.session_state["is_running"] or not data_path,
    )

    if not data_path:
        st.caption("Select or upload a dataset above before running.")

    if run_clicked and data_path:
        st.session_state["is_running"] = True
        st.session_state["run_error"] = None
        try:
            config_dict = build_config_dict(data_path, data_format)
            try:
                cfg = PipelineConfig(**config_dict)
            except ValidationError as exc:
                st.session_state["run_error"] = exc
                st.session_state["run_result"] = None
            else:
                with st.spinner("Running pipeline..."):
                    try:
                        result = PipelineRunner(cfg).run()
                    except Exception as exc:
                        logger.exception("Pipeline run failed")
                        st.session_state["run_error"] = exc
                        st.session_state["run_result"] = None
                    else:
                        st.session_state["run_result"] = result
        finally:
            st.session_state["is_running"] = False

    error = st.session_state["run_error"]
    if isinstance(error, ValidationError):
        st.error("Configuration is invalid:")
        for err in error.errors():
            loc = ".".join(str(p) for p in err["loc"])
            st.error(f"- **{loc}**: {err['msg']}")
    elif error is not None:
        st.error(str(error))

    if st.session_state["run_result"] is not None:
        st.header(":material/insights: 4. Results")
        render_results(st.session_state["run_result"])


if __name__ == "__main__":
    main()
