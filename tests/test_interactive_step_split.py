"""Tests for dscompanion/app/interactive_step_split.py's "grouped" split UI (UIR.4).

Drives the real Step 3 render entrypoint via ``AppTest`` rather than unit-testing
``_render_group_population_picker``/``_run_split`` in isolation — this step's confirm
button and column widgets are Streamlit-native, so this exercises the actual user flow
(pick method → pick group column → pick population → confirm) the way a real session
would, matching this app's other AppTest-based step tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from interactive_state import STEP_ORDER


@pytest.fixture
def bank_marketing_df() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "data" / "bank_marketing_clean.parquet"
    if not data_path.exists():
        pytest.skip(f"optional local dataset not present: {data_path}")
    return pd.read_parquet(data_path)


def _at_on_split_step(bank_marketing_df: pd.DataFrame) -> AppTest:
    """Fresh AppTest session parked on Step 3, with Steps 1-2 already confirmed."""
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.session_state["run_mode"] = "Interactive"
    at.session_state["interactive.df"] = bank_marketing_df
    at.session_state["interactive.load_data_selection_id"] = ("x", "parquet", None)
    at.session_state["interactive.task"] = "classification"
    at.session_state["interactive.target"] = "subscribed"
    at.session_state["interactive.step_confirmed"] = {
        s: (s in ("load_data", "task_target")) for s in STEP_ORDER
    }
    at.session_state["interactive.step_status"] = {s: "pending" for s in STEP_ORDER}
    at.session_state["interactive.step_status"]["load_data"] = "done"
    at.session_state["interactive.step_status"]["task_target"] = "done"
    at.session_state["interactive.step_status"]["split"] = "current"
    at.session_state["interactive.current_step"] = "split"
    return at


class TestGroupedSplitCategoricalPath:
    def test_multiselect_confirms_deliberate_test_population(self, bank_marketing_df):
        at = _at_on_split_step(bank_marketing_df)
        at.run()
        assert not at.exception

        [s for s in at.selectbox if s.label == "Split method"][0].select("grouped").run()
        assert not at.exception
        [s for s in at.selectbox if s.label == "Group column"][0].select("job").run()
        assert not at.exception

        multiselects = at.multiselect
        assert multiselects, "expected a multiselect for a categorical group column"
        multiselects[0].select("management").run()
        assert not at.exception

        confirm = [b for b in at.button if "Apply" in (b.label or "")]
        assert confirm
        confirm[0].click().run()
        assert not at.exception
        assert at.session_state["interactive.step_confirmed"]["split"] is True

        split = at.session_state["interactive.split"]
        assert (split.test_X["job"] == "management").all()
        assert not (split.train_X["job"] == "management").any()
        assert not (split.val_X["job"] == "management").any()
        # The group column itself is a real feature (matches the existing
        # customer_id-style precedent) — only the numeric/date derived flag
        # column (see below) is stripped, not a genuine categorical column.
        assert "job" in split.train_X.columns

    def test_no_selection_blocks_confirmation(self, bank_marketing_df):
        at = _at_on_split_step(bank_marketing_df)
        at.run()
        [s for s in at.selectbox if s.label == "Split method"][0].select("grouped").run()
        [s for s in at.selectbox if s.label == "Group column"][0].select("job").run()
        assert not at.exception
        assert at.session_state["interactive.step_confirmed"]["split"] is False
        assert not [b for b in at.button if "Apply" in (b.label or "")]


class TestGroupedSplitNumericPath:
    def test_slider_threshold_confirms_and_strips_derived_flag_column(self, bank_marketing_df):
        at = _at_on_split_step(bank_marketing_df)
        at.run()
        [s for s in at.selectbox if s.label == "Split method"][0].select("grouped").run()
        [s for s in at.selectbox if s.label == "Group column"][0].select("age").run()
        assert not at.exception

        sliders = [s for s in at.slider if s.label == "Threshold"]
        assert sliders
        sliders[0].set_value(45).run()
        assert not at.exception

        confirm = [b for b in at.button if "Apply" in (b.label or "")]
        confirm[0].click().run()
        assert not at.exception
        assert at.session_state["interactive.step_confirmed"]["split"] is True

        split = at.session_state["interactive.split"]
        # Default direction is "At or above the threshold".
        assert split.test_X["age"].min() >= 45
        assert split.train_X["age"].max() < 45
        # The threshold-derived flag column must never leak into the feature set.
        assert "__group_split_flag__" not in split.train_X.columns
        assert "__group_split_flag__" not in split.test_X.columns
        assert split.metadata.n_features == 16
