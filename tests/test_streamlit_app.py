"""Tests for dscompanion/app/streamlit_app.py — Interactive mode's dispatch loop.

Regression test for a real bug: once all 13 steps were confirmed, the dispatch
loop in ``_render_interactive_mode()`` stopped calling ``render_step_report()``
entirely (its branch was only taken while Report was *not yet* confirmed) and
fell through to a generic "All 13 steps complete" message instead — one that
explicitly says "Download your reports from Step 13 above" while Report's own
download buttons were no longer being rendered anywhere on the page.

``render_step_report()`` has no re-render guard (unlike every other step's
``if already_confirmed: return True`` early exit), confirming it was always
meant to keep rendering after confirmation — the dispatch loop just never
let it.
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


class TestInteractiveModeDispatchLoop:
    def test_step13_download_ui_still_renders_after_all_steps_confirmed(self):
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
        at.session_state["run_mode"] = "Interactive"
        at.session_state["interactive.step_confirmed"] = {step: True for step in STEP_ORDER}
        at.session_state["interactive.task"] = "classification"
        at.session_state["interactive.target"] = "subscribed"
        # Avoids _build_partial_config_dict() unpacking a missing tuple for
        # a load_data-confirmed branch — unrelated to this bug, just a
        # prerequisite for a clean run with no confirmed steps' real data.
        at.session_state["interactive.load_data_selection_id"] = (None, None, None)
        at.session_state["interactive.train_algorithm"] = "xgboost"
        at.session_state["interactive.tuning_tuned"] = False
        at.session_state["interactive.calibration_applied"] = True
        at.session_state["int.report.report_generated"] = True
        at.session_state["int.report.xlsx_bytes"] = b"fake-xlsx-bytes"

        at.run()

        assert not at.exception
        success_text = " ".join(s.value for s in at.success)
        assert "Model card generated" in success_text  # Step 13's own view
        assert "All 13 steps are complete" in success_text  # the terminal summary
        download_labels = {b.label for b in at.get("download_button")}
        assert "Download Excel report" in download_labels
        assert "Download config (YAML)" in download_labels
        assert "Download HTML report" not in download_labels  # HTML is not offered
