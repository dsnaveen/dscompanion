"""Tests for dscompanion/app/interactive_step_calibration.py — Calibration step.

Regression test for a real bug: ``_fit_calibrator()`` called
``Calibrator(method="isotonic", cv="prefit")`` — ``cv`` isn't a parameter
``Calibrator`` accepts at all (it always assumes a pre-fitted model, unlike
sklearn's ``CalibratedClassifierCV(cv="prefit")`` convention this was likely
copied from), so every Calibration render raised ``TypeError``, silently
caught and surfaced to the user as a misleading "model or split missing"
message — reported as "calibration step is not working well" while using
``data/bank_marketing_clean.parquet``, though the bug isn't dataset-specific.

``dscompanion.calibration.calibrator.Calibrator`` itself already has correct test
coverage (``test_calibration.py``) that never exercised this exact call site
— the bug lived purely in the app layer. This is the first ``AppTest``-based
test committed in this codebase (see ``test_interactive_step_train.py`` — no
such precedent existed until now); it's used here specifically because this
class of bug — an app-level call site drifting from the real library
signature — only manifests through the module's actual render entrypoint,
not through unit-testing ``Calibrator`` in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

# dscompanion/app/ is a standalone script directory, not part of the installable
# dscompanion package (see streamlit.md) — add it to sys.path the same way
# streamlit_app.py does for its own sibling imports, since
# interactive_step_calibration.py imports `from interactive_state import
# add_audit_entry, set_step_status`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from interactive_state import STEP_ORDER

from dscompanion.models import ModelFactory


def _render_step_calibration_script() -> None:
    from interactive_step_calibration import render_step_calibration

    render_step_calibration()


class TestRenderStepCalibration:
    def test_fits_calibrator_without_crashing(self, numeric_split):
        """Regression test for the Calibrator(cv="prefit") TypeError.

        Seeds session state with a real fitted model + processed split
        (mirroring what Train/Tuning leave behind) and drives Calibration's
        real entrypoint end-to-end — this is exactly the path that silently
        raised before the fix.
        """
        model = ModelFactory.build(task="classification", algorithm="logistic")
        model.fit(numeric_split.train_X, numeric_split.train_y)

        at = AppTest.from_function(_render_step_calibration_script, default_timeout=30)
        at.session_state["interactive.step_confirmed"] = {step: False for step in STEP_ORDER}
        at.session_state["interactive.task"] = "classification"
        at.session_state["interactive.tuning_model"] = model
        at.session_state["int.train.processed_split"] = numeric_split
        at.run()

        assert not at.exception
        assert at.session_state["int.calibration.calibrator"] is not None
        # A genuine failure (bad constructor kwarg) prints exactly this
        # message from render_step_calibration()'s `result is None` branch —
        # assert it's absent, not just that no exception was raised, since
        # the bug was silently caught and never surfaced as an at.exception.
        assert not any("Please go back and re-run Steps 8" in e.value for e in at.error)

    def test_shows_ece_before_and_after(self, numeric_split):
        model = ModelFactory.build(task="classification", algorithm="logistic")
        model.fit(numeric_split.train_X, numeric_split.train_y)

        at = AppTest.from_function(_render_step_calibration_script, default_timeout=30)
        at.session_state["interactive.step_confirmed"] = {step: False for step in STEP_ORDER}
        at.session_state["interactive.task"] = "classification"
        at.session_state["interactive.tuning_model"] = model
        at.session_state["int.train.processed_split"] = numeric_split
        at.run()

        assert not at.exception
        metric_labels = [m.label for m in at.get("metric")]
        assert "Before Calibration" in metric_labels
        assert "After Calibration (Isotonic)" in metric_labels
