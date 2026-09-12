"""Step identity vs. step position — the single source of truth for the backend.

Before this module, a step's *position* in the 13-step wizard (an int) doubled as its
*identity* everywhere: ``RunState.step_status: dict[int, str]``, REST routes
(``/steps/{n}/...``), and ``STEP_NAMES: dict[int, str]``. Inserting a new step in the
middle would have required renumbering every step after it, across three separately
maintained UIs (this API, the React frontend, the Streamlit app).

``StepId`` is now the real identity (a stable slug). ``STEP_ORDER`` is a separate,
freely-reorderable list — inserting a step becomes "add one slug to one list."

React (``frontend/enterprise/src/constants/steps.ts``) and Streamlit
(``dscompanion/app/interactive_state.py``) each keep their own mirrored copy of this same
shape — there is no shared runtime code between the three UIs, so if this module
changes, update both by hand in the same commit.
"""

from __future__ import annotations

from typing import Final, Literal

__all__ = [
    "StepId",
    "STEP_ORDER",
    "STEP_NAMES",
    "PORTED_STEPS",
    "step_index",
    "step_number",
    "total_steps",
    "next_step_id",
    "prev_step_id",
    "steps_from",
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

STEP_ORDER: Final[list[StepId]] = [
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

STEP_NAMES: Final[dict[StepId, str]] = {
    "load_data": "Load Data",
    "task_target": "Task & Target",
    "split": "Split",
    "eda": "Explore Data",
    "feature_processing": "Feature Processing",
    "feature_selection": "Feature Selection",
    "imbalance": "Imbalance",
    "train": "Train",
    "tuning": "Tuning",
    "evaluate": "Evaluate",
    "calibration": "Calibration",
    "shap": "SHAP",
    "report": "Report",
}

# Steps with a real dscompanion/api/ router; the rest return 501 (see routers/steps_stub.py).
PORTED_STEPS: Final[set[StepId]] = {
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
}

_INDEX: Final[dict[StepId, int]] = {step: i for i, step in enumerate(STEP_ORDER)}


def step_index(step: StepId) -> int:
    """Returns ``step``'s 0-based position in ``STEP_ORDER``.

    Args:
        step (StepId): The step slug.

    Returns:
        int: 0-based position, e.g. ``0`` for ``"load_data"``.

    Raises:
        KeyError: If ``step`` is not a known slug.
    """
    return _INDEX[step]


def step_number(step: StepId) -> int:
    """Returns ``step``'s 1-based wizard number, for "Step N of 13" display.

    Args:
        step (StepId): The step slug.

    Returns:
        int: 1-based position, e.g. ``1`` for ``"load_data"``.

    Raises:
        KeyError: If ``step`` is not a known slug.
    """
    return _INDEX[step] + 1


def total_steps() -> int:
    """Returns the total number of steps in the wizard.

    Args:
        None

    Returns:
        int: ``len(STEP_ORDER)``.
    """
    return len(STEP_ORDER)


def next_step_id(step: StepId) -> StepId | None:
    """Returns the step slug immediately after ``step``, or ``None`` if ``step`` is last.

    Args:
        step (StepId): The step slug.

    Returns:
        StepId | None: The next step's slug, or ``None`` when ``step`` is the last
        entry in ``STEP_ORDER``.

    Raises:
        KeyError: If ``step`` is not a known slug.
    """
    idx = _INDEX[step]
    return STEP_ORDER[idx + 1] if idx + 1 < len(STEP_ORDER) else None


def prev_step_id(step: StepId) -> StepId | None:
    """Returns the step slug immediately before ``step``, or ``None`` if ``step`` is first.

    Args:
        step (StepId): The step slug.

    Returns:
        StepId | None: The previous step's slug, or ``None`` when ``step`` is the
        first entry in ``STEP_ORDER``.

    Raises:
        KeyError: If ``step`` is not a known slug.
    """
    idx = _INDEX[step]
    return STEP_ORDER[idx - 1] if idx > 0 else None


def steps_from(step: StepId) -> list[StepId]:
    """Returns ``step`` and every step after it, in wizard order.

    Replaces the old ``range(step, N_STEPS + 1)`` idiom used by
    ``RunState.reset_from_step()`` and Streamlit's ``reset_from_step()``.

    Args:
        step (StepId): The first step slug to include.

    Returns:
        list[StepId]: ``step`` and all later slugs, in ``STEP_ORDER`` order.

    Raises:
        KeyError: If ``step`` is not a known slug.
    """
    return STEP_ORDER[_INDEX[step] :]
