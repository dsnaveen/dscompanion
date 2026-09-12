"""Tests for dscompanion/api/steps.py — the step-identity/step-position source of truth.
"""

from __future__ import annotations

import pytest

from dscompanion.api.steps import (
    STEP_NAMES,
    STEP_ORDER,
    next_step_id,
    prev_step_id,
    step_index,
    step_number,
    steps_from,
    total_steps,
)


def test_step_order_and_names_have_no_duplicates() -> None:
    assert len(STEP_ORDER) == len(set(STEP_ORDER))
    assert set(STEP_ORDER) == set(STEP_NAMES)


def test_total_steps_matches_step_order_length() -> None:
    assert total_steps() == len(STEP_ORDER) == 13


@pytest.mark.parametrize("expected_index,step", list(enumerate(STEP_ORDER)))
def test_step_index_round_trips_for_every_step(expected_index: int, step: str) -> None:
    assert step_index(step) == expected_index  # type: ignore[arg-type]
    assert step_number(step) == expected_index + 1  # type: ignore[arg-type]


def test_next_step_id_of_last_step_is_none() -> None:
    assert next_step_id(STEP_ORDER[-1]) is None


def test_prev_step_id_of_first_step_is_none() -> None:
    assert prev_step_id(STEP_ORDER[0]) is None


def test_next_and_prev_step_id_are_inverses() -> None:
    for step in STEP_ORDER[:-1]:
        nxt = next_step_id(step)
        assert nxt is not None
        assert prev_step_id(nxt) == step


def test_steps_from_first_step_returns_full_order() -> None:
    assert steps_from(STEP_ORDER[0]) == STEP_ORDER


def test_steps_from_last_step_returns_single_element() -> None:
    assert steps_from(STEP_ORDER[-1]) == [STEP_ORDER[-1]]


def test_steps_from_middle_step_returns_suffix() -> None:
    mid = STEP_ORDER[4]
    assert steps_from(mid) == STEP_ORDER[4:]


def test_step_index_unknown_slug_raises_key_error() -> None:
    with pytest.raises(KeyError):
        step_index("not_a_real_step")  # type: ignore[arg-type]
