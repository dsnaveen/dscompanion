"""Interactive-mode plumbing: step stepper, audit trail, downstream-invalidation.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code. See
``streamlit.md``'s "v2 Planning — Interactive (step-by-step) mode" section for
the full design this module implements (five UX rules: confirm-before-apply,
plain language, recovery action on every dead end, preview-before-decide,
always-visible step navigation).

Step identity is a stable slug (``StepId``), not its position — mirrors
``dscompanion/dscompanion/api/steps.py`` and ``frontend/enterprise/src/constants/steps.ts``
exactly (no shared runtime code between the three UIs; update all three by hand
in the same commit if this changes). ``STEP_ORDER`` is a separate, freely
reorderable list — inserting a step becomes "add one slug to one list."
"""

from __future__ import annotations

import logging
from typing import Literal

import streamlit as st

logger = logging.getLogger(__name__)

__all__ = [
    "StepId",
    "STEP_ORDER",
    "STEP_NAMES",
    "step_number",
    "next_step_id",
    "prev_step_id",
    "steps_from",
    "init_interactive_state",
    "add_audit_entry",
    "render_audit_trail",
    "render_completed_steps",
    "set_step_status",
    "set_step_confirmed",
    "reset_from_step",
    "render_back_button",
    "render_stepper",
    "current_step",
    "go_to_step",
    "dark_fig",
]

StepId = Literal[
    "load_data",
    "task_target",
    "split",
    "eda",
    "feature_processing",
    "feature_selection",
    "imbalance",
    "train",
    "tuning",
    "evaluate",
    "calibration",
    "shap",
    "report",
]

STEP_ORDER: list[StepId] = [
    "load_data",
    "task_target",
    "split",
    "eda",
    "feature_processing",
    "feature_selection",
    "imbalance",
    "train",
    "tuning",
    "evaluate",
    "calibration",
    "shap",
    "report",
]

STEP_NAMES: dict[StepId, str] = {
    "load_data": "Load data",
    "task_target": "Task & target",
    "split": "Split",
    "eda": "Explore data",
    "feature_processing": "Feature processing",
    "feature_selection": "Feature selection",
    "imbalance": "Imbalance",
    "train": "Train",
    "tuning": "Tuning",
    "evaluate": "Evaluate",
    "calibration": "Calibration",
    "shap": "SHAP",
    "report": "Report",
}

_STATUS_ICON = {
    "done": ":material/check_circle:",
    "current": ":material/radio_button_checked:",
    "pending": ":material/radio_button_unchecked:",
    "flagged": ":material/warning:",
}

_STEP_INDEX: dict[StepId, int] = {step: i for i, step in enumerate(STEP_ORDER)}


def step_number(step: StepId) -> int:
    """Returns ``step``'s 1-based wizard number, for "Step N of 13" display.

    Args:
        step (StepId): The step slug.

    Returns:
        int: 1-based position, e.g. ``1`` for ``"load_data"``.
    """
    return _STEP_INDEX[step] + 1


def next_step_id(step: StepId) -> StepId | None:
    """Returns the step slug immediately after ``step``, or ``None`` if ``step`` is last.

    Args:
        step (StepId): The step slug.

    Returns:
        StepId | None: The next step's slug, or ``None`` when ``step`` is the last
        entry in ``STEP_ORDER``.
    """
    idx = _STEP_INDEX[step]
    return STEP_ORDER[idx + 1] if idx + 1 < len(STEP_ORDER) else None


def prev_step_id(step: StepId) -> StepId | None:
    """Returns the step slug immediately before ``step``, or ``None`` if ``step`` is first.

    Args:
        step (StepId): The step slug.

    Returns:
        StepId | None: The previous step's slug, or ``None`` when ``step`` is the
        first entry in ``STEP_ORDER``.
    """
    idx = _STEP_INDEX[step]
    return STEP_ORDER[idx - 1] if idx > 0 else None


def steps_from(step: StepId) -> list[StepId]:
    """Returns ``step`` and every step after it, in wizard order.

    Args:
        step (StepId): The first step slug to include.

    Returns:
        list[StepId]: ``step`` and all later slugs, in ``STEP_ORDER`` order.
    """
    return STEP_ORDER[_STEP_INDEX[step] :]


# Categorical palette (dark-mode steps) — same 8 hues as
# dscompanion/dscompanion/utils/plotting.py's light-mode _PALETTE, stepped for the dark
# surface per the dataviz skill's validated reference palette. Swapped in by
# dark_fig() via colorway= so multi-series charts stay CVD-safe in dark mode
# too, not just transparent-background-with-light-mode-hues.
_PALETTE_DARK = [
    "#3987e5",
    "#008300",
    "#d55181",
    "#c98500",
    "#199e70",
    "#d95926",
    "#9085e9",
    "#e66767",
]


def dark_fig(fig):
    """Apply transparent backgrounds and light-coloured text to a Plotly figure.

    Call this on every figure before passing it to ``st.plotly_chart`` so
    charts blend into Streamlit's dark UI instead of showing a white box.
    ``apply_dscompanion_theme``'s margins are preserved — backgrounds, font
    colour, grid lines, legend styling, and the categorical palette
    (``colorway``) are all overridden with dark-mode-appropriate values.

    Args:
        fig: Any Plotly ``Figure`` instance.

    Returns:
        The same figure, mutated in-place and returned for chaining.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c8c8c8"},
        colorway=_PALETTE_DARK,
        legend={"bgcolor": "rgba(40,40,40,0.7)", "bordercolor": "#555", "borderwidth": 1},
        xaxis={"gridcolor": "#2c2c2a", "linecolor": "#383835"},
        yaxis={"gridcolor": "#2c2c2a", "linecolor": "#383835"},
    )
    return fig


def init_interactive_state() -> None:
    """Set every Interactive-mode session-state default, once per session.

    Args:
        None

    Returns:
        None
    """
    st.session_state.setdefault("interactive.current_step", STEP_ORDER[0])
    st.session_state.setdefault("interactive.step_status", {step: "pending" for step in STEP_ORDER})
    if st.session_state["interactive.step_status"].get(STEP_ORDER[0]) == "pending":
        st.session_state["interactive.step_status"][STEP_ORDER[0]] = "current"
    st.session_state.setdefault("interactive.step_confirmed", {step: False for step in STEP_ORDER})
    st.session_state.setdefault("interactive.audit_trail", [])
    st.session_state.setdefault("interactive.df", None)
    st.session_state.setdefault("interactive.load_data_selection_id", None)


def current_step() -> StepId:
    """Return the step slug the user is currently on.

    Args:
        None

    Returns:
        StepId: Current step slug.
    """
    return st.session_state["interactive.current_step"]


def go_to_step(step: StepId) -> None:
    """Move the stepper to ``step`` without changing any step's status.

    Args:
        step (StepId): Step slug to navigate to.

    Returns:
        None
    """
    st.session_state["interactive.current_step"] = step


def set_step_status(step: StepId, status: str) -> None:
    """Set a step's stepper status.

    Args:
        step (StepId): Step slug.
        status (str): One of ``"done"``, ``"current"``, ``"pending"``,
            ``"flagged"``.

    Returns:
        None
    """
    st.session_state["interactive.step_status"][step] = status


def set_step_confirmed(step: StepId, value: bool) -> None:
    """Set a step's confirmation flag.

    Args:
        step (StepId): Step slug.
        value (bool): The new confirmation state.

    Returns:
        None
    """
    st.session_state["interactive.step_confirmed"][step] = value


def add_audit_entry(step: StepId, text: str) -> None:
    """Append one plain-language line to the "Decisions so far" trail.

    Args:
        step (StepId): Step slug this decision belongs to — used by
            ``reset_from_step`` to truncate the trail when an earlier
            decision changes.
        text (str): Plain-language description of the confirmed action.
            Never include raw row values — counts/column names only, per
            ``~/.claude/python_rules.md``'s logging rule (the same
            constraint applies here since this trail is shown on screen).

    Returns:
        None
    """
    st.session_state["interactive.audit_trail"].append((step, text))
    logger.info("Interactive mode — step %s confirmed: %s", step, text)


def reset_from_step(step: StepId) -> None:
    """Invalidate ``step`` and every step after it.

    Used when a confirmed earlier decision changes (e.g. a different file is
    loaded on Step 1 after Step 2 already ran) — downstream state can no
    longer be trusted, so every later step's status resets to ``"pending"``
    and its audit-trail entries are dropped.

    Args:
        step (StepId): The first step slug to invalidate. This step and every
            step after it (per ``STEP_ORDER``) is reset.

    Returns:
        None
    """
    reset_steps = steps_from(step)
    statuses = st.session_state["interactive.step_status"]
    confirmed = st.session_state["interactive.step_confirmed"]
    for s in reset_steps:
        statuses[s] = "pending"
        confirmed[s] = False
    reset_set = set(reset_steps)
    trail = st.session_state["interactive.audit_trail"]
    st.session_state["interactive.audit_trail"] = [
        (s, text) for s, text in trail if s not in reset_set
    ]
    if step == STEP_ORDER[0]:
        st.session_state["interactive.df"] = None
    st.session_state["interactive.current_step"] = step
    st.session_state.pop("int.back_pending_to_step", None)
    logger.info("Interactive mode — reset from step %s onward", step)


def render_back_button(from_step: StepId | None) -> None:
    """Render a Back button with a confirm-before-invalidate gate.

    Placed once above the dispatch loop in ``streamlit_app.py`` so it appears
    consistently on every step without editing each individual step file.
    Going back always calls ``reset_from_step()`` on the target step, which
    invalidates it and all later ones.

    The flow is two-phase to avoid accidental destructive navigation:

    - **Phase 1** — user clicks "← Back to Step N-1": sets
      ``int.back_pending_to_step`` in session state and reruns.
    - **Phase 2** — warning + "Yes, go back" / "Cancel" buttons appear in
      place of the normal Back button.  "Yes" calls ``reset_from_step`` and
      clears the pending flag; "Cancel" only clears the flag.

    Args:
        from_step (StepId | None): The step currently being rendered —
            computed from the run's confirmed-step state before the dispatch
            loop runs. ``None`` means "past the last step" (all 13 steps
            confirmed) — the target is then the last real step, matching the
            terminal-state behaviour before this refactor. The first step
            gets no Back button (``prev_step_id`` returns ``None``).

    Returns:
        None
    """
    to_step = STEP_ORDER[-1] if from_step is None else prev_step_id(from_step)
    if to_step is None:
        return

    pending_to = st.session_state.get("int.back_pending_to_step")

    if pending_to == to_step:
        statuses = st.session_state["interactive.step_status"]
        done_steps = [s for s in STEP_ORDER if statuses.get(s) == "done"]
        last_done = done_steps[-1] if done_steps else to_step
        st.warning(
            f"Going back to **Step {step_number(to_step)}: {STEP_NAMES[to_step]}** will clear "
            f"your confirmed decisions from Step {step_number(to_step)} through "
            f"Step {step_number(last_done)}. This cannot be undone."
        )
        col_yes, col_cancel, *_ = st.columns([2, 2, 8])
        if col_yes.button("Yes, Go Back", key="int.back_confirm_yes"):
            reset_from_step(to_step)
            st.rerun()
        if col_cancel.button("Cancel", key="int.back_confirm_cancel"):
            st.session_state.pop("int.back_pending_to_step", None)
            st.rerun()
    else:
        if st.button(
            f"← Back to Step {step_number(to_step)}: {STEP_NAMES[to_step]}",
            key=f"int.back_btn.{from_step or 'terminal'}",
        ):
            st.session_state["int.back_pending_to_step"] = to_step
            st.rerun()


def render_stepper() -> None:
    """Render the always-visible step-status panel in the sidebar.

    Each step appears on its own row with a status icon and full name.
    The current step is shown in bold; completed, pending, and flagged
    steps each use their own Material Symbols icon (see ``_STATUS_ICON``).
    Using the sidebar (rather than
    a 13-column horizontal row in the main area) keeps every step name
    readable on standard screens and satisfies the "always-visible step
    navigation" UX rule even when step content is long enough to scroll.

    Args:
        None

    Returns:
        None
    """
    statuses = st.session_state["interactive.step_status"]
    with st.sidebar:
        st.markdown("### Pipeline Progress")
        st.divider()
        for step in STEP_ORDER:
            status = statuses.get(step, "pending")
            icon = _STATUS_ICON[status]
            name = STEP_NAMES[step]
            if status == "current":
                st.markdown(
                    f"{icon} &nbsp;**Step {step_number(step)} · {name}**", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"{icon} &nbsp;Step {step_number(step)} · {name}", unsafe_allow_html=True
                )
        st.divider()


def render_audit_trail() -> None:
    """Render the collapsible "Decisions so far" audit trail in the sidebar.

    Args:
        None

    Returns:
        None
    """
    trail = st.session_state["interactive.audit_trail"]
    with st.sidebar:
        with st.expander(f":material/fact_check: Decisions So Far ({len(trail)})", expanded=False):
            if not trail:
                st.caption("No decisions recorded yet.")
            for _step, text in trail:
                st.write(f"- {text}")


def render_completed_steps() -> None:
    """Render each confirmed step as a collapsed card the user can reopen.

    Confirmed steps stop rendering their live form (the dispatch loop in
    ``streamlit_app.py`` only calls the first *unconfirmed* step), so
    without this, a completed step's decisions are only visible via the
    sidebar's bundled "Decisions so far" list. This gives each one a
    dedicated, collapsed-by-default slot inline in the main flow —
    mirroring the notebook-style "stack and stay" pattern already shipped
    for ``frontend/enterprise/``'s ``NotebookColumn``. Deliberately shows
    only the audit-trail text already recorded for that step (not a
    reconstruction of its full form) — same "generic, not per-step-rich"
    scoping call made for that component's ``StepSummaryCard``, since
    re-deriving a richer summary from raw session-state would risk
    drifting out of sync with what ``add_audit_entry`` actually recorded.

    Args:
        None

    Returns:
        None
    """
    statuses = st.session_state["interactive.step_status"]
    trail = st.session_state["interactive.audit_trail"]
    for step in STEP_ORDER:
        if statuses.get(step) != "done":
            continue
        entries = [text for s, text in trail if s == step]
        with st.expander(
            f":material/check_circle: Step {step_number(step)} · {STEP_NAMES[step]}",
            expanded=False,
        ):
            if not entries:
                st.caption("Confirmed. No recorded decisions for this step.")
            for text in entries:
                st.write(f"- {text}")
