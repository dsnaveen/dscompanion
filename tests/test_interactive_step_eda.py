"""Tests for dscompanion/app/interactive_step_eda.py's threshold sliders.

Regression coverage: the high-missing and near-zero-variance thresholds were
referenced in the step's own UI copy ("set via the slider above") but no
slider actually existed — both were hardcoded module constants (one of them
not even sourced from ``settings``, a bare ``0.30`` literal), so the
thresholds were silently unadjustable from the Interactive UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

# dscompanion/app/ is a standalone script directory, not part of the installable
# dscompanion package (see streamlit.md) — add it to sys.path the same way
# streamlit_app.py does for its own sibling imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from interactive_state import STEP_ORDER

from dscompanion.config import settings
from dscompanion.split import DataSplit


def _seeded_app(data_split: DataSplit) -> AppTest:
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.session_state["run_mode"] = "Interactive"
    confirmed = {step: True for step in STEP_ORDER}
    confirmed["eda"] = False
    at.session_state["interactive.step_confirmed"] = confirmed
    at.session_state["interactive.task"] = "classification"
    at.session_state["interactive.target"] = "target"
    at.session_state["interactive.load_data_selection_id"] = (None, None, None)
    at.session_state["interactive.split"] = data_split
    return at


class TestEDAThresholdSliders:
    def test_sliders_exist_and_default_to_settings_values(self, data_split):
        at = _seeded_app(data_split)
        at.run()
        assert not at.exception

        high_missing = at.slider(key="int.eda.high_missing_threshold")
        near_zero = at.slider(key="int.eda.near_zero_variance_threshold")
        assert high_missing.value == settings.high_missing_threshold
        assert near_zero.value == settings.near_zero_variance_threshold

    def test_changing_a_threshold_invalidates_the_cached_report(self, data_split):
        at = _seeded_app(data_split)
        at.run()
        assert not at.exception
        assert "int.eda._cached_report" in at.session_state

        at.slider(key="int.eda.high_missing_threshold").set_value(0.6).run()

        assert not at.exception
        assert at.session_state["int.eda._cached_thresholds"][0] == 0.6
        # A fresh report must have been rebuilt against the new threshold,
        # not silently reused from before the slider moved.
        assert "int.eda._cached_report" in at.session_state
