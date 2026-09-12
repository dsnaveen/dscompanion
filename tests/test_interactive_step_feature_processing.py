"""Tests for dscompanion/app/interactive_step_feature_processing.py's ColumnRecipe migration.

Covers the pure recommendation/candidate helpers directly (no Streamlit needed), plus the
real Accept/Reject/Drop user flow via ``AppTest``, matching this app's other per-step test
files' convention (see ``test_interactive_step_split.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dscompanion.eda.univariate import UnivariateAnalyser
from dscompanion.split import DataSplit

# dscompanion/app/ is a standalone script directory, not part of the installable
# dscompanion package (see streamlit.md) — add it to sys.path the same way
# streamlit_app.py does for its own sibling imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from interactive_state import STEP_ORDER
from interactive_step_feature_processing import (
    _missing_value_candidates,
    _recommended_recipe,
    _stats_row,
)

# ---------------------------------------------------------------------------
# Pure helpers — no Streamlit session required
# ---------------------------------------------------------------------------


class TestMissingValueCandidates:
    def test_columns_with_missing_values_included(self, data_split):
        table = _missing_value_candidates(
            data_split.train_X, excluded_cols=set(), suggested_reasons={}
        )
        assert {"f4", "f5"} <= set(table["column"])

    def test_clean_columns_excluded_without_suggested_reason(self, data_split):
        table = _missing_value_candidates(
            data_split.train_X, excluded_cols=set(), suggested_reasons={}
        )
        assert "f1" not in set(table["column"])

    def test_suggested_reason_includes_column_with_no_missingness(self, data_split):
        table = _missing_value_candidates(
            data_split.train_X,
            excluded_cols=set(),
            suggested_reasons={"constant_col": ["constant"]},
        )
        row = table.loc[table["column"] == "constant_col"].iloc[0]
        assert row["missing_rows"] == 0
        assert row["suggested_reasons"] == ["constant"]

    def test_excluded_columns_never_appear(self, data_split):
        table = _missing_value_candidates(
            data_split.train_X, excluded_cols={"f4", "f5"}, suggested_reasons={}
        )
        assert not ({"f4", "f5"} & set(table["column"]))

    def test_empty_when_nothing_qualifies(self, data_split):
        clean_df = data_split.train_X[["f1", "f2", "f3"]]
        table = _missing_value_candidates(clean_df, excluded_cols=set(), suggested_reasons={})
        assert table.empty
        assert list(table.columns) == [
            "column",
            "missing_rows",
            "missing_pct",
            "is_numeric",
            "suggested_reasons",
        ]


class TestRecommendedRecipe:
    @pytest.fixture
    def summaries(self, data_split):
        analyser = UnivariateAnalyser().fit(data_split)
        return analyser.numeric_summary(), analyser.categorical_summary()

    def test_stats_row_finds_matching_feature(self, summaries):
        numeric_summary, categorical_summary = summaries
        row = _stats_row("f4", True, numeric_summary, categorical_summary)
        assert row["feature"] == "f4"

    def test_numeric_column_with_missingness_recommends_impute(self, data_split, summaries):
        numeric_summary, categorical_summary = summaries
        recipe = _recommended_recipe("f4", True, numeric_summary, categorical_summary)
        assert any(s.transformer == "impute" for s in recipe.steps)
        assert recipe.source == "recommended"

    def test_clean_numeric_column_recommends_nothing(self, data_split, summaries):
        numeric_summary, categorical_summary = summaries
        recipe = _recommended_recipe("f2", True, numeric_summary, categorical_summary)
        assert recipe.steps == []

    def test_high_cardinality_categorical_recommends_target_encode_chain(
        self, data_split, summaries
    ):
        numeric_summary, categorical_summary = summaries
        recipe = _recommended_recipe("cat_high", False, numeric_summary, categorical_summary)
        transformers = [s.transformer for s in recipe.steps]
        assert "rare_group" in transformers
        assert "target_encode" in transformers

    def test_low_cardinality_categorical_recommends_onehot(self, data_split, summaries):
        numeric_summary, categorical_summary = summaries
        recipe = _recommended_recipe("cat_low", False, numeric_summary, categorical_summary)
        assert [s.transformer for s in recipe.steps] == ["onehot_encode"]

    def test_constant_column_recommends_no_steps(self, data_split, summaries):
        numeric_summary, categorical_summary = summaries
        recipe = _recommended_recipe("constant_col", True, numeric_summary, categorical_summary)
        assert recipe.steps == []
        assert "onstant" in recipe.rationale


# ---------------------------------------------------------------------------
# Real Accept/Reject/Drop flow via AppTest
# ---------------------------------------------------------------------------


def _at_on_feature_processing_step(data_split: DataSplit) -> AppTest:
    """Fresh AppTest session parked on Step 5, with Steps 1-4 already confirmed."""
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.session_state["run_mode"] = "Interactive"
    at.session_state["interactive.split"] = data_split
    at.session_state["interactive.target"] = "target"
    at.session_state["interactive.task"] = "classification"
    at.session_state["interactive.identifier_columns"] = []
    at.session_state["int.split.date_col"] = None
    at.session_state["interactive.eda_report"] = None
    # Avoids _build_partial_config_dict() unpacking a missing tuple for a
    # load_data-confirmed branch — same prerequisite test_streamlit_app.py
    # already needs, unrelated to Step 5 itself.
    at.session_state["interactive.load_data_selection_id"] = (None, None, None)
    at.session_state["interactive.step_confirmed"] = {
        s: (s in ("load_data", "task_target", "split", "eda")) for s in STEP_ORDER
    }
    at.session_state["interactive.step_status"] = {s: "pending" for s in STEP_ORDER}
    for s in ("load_data", "task_target", "split", "eda"):
        at.session_state["interactive.step_status"][s] = "done"
    at.session_state["interactive.step_status"]["feature_processing"] = "current"
    at.session_state["interactive.current_step"] = "feature_processing"
    return at


class TestFeatureProcessingAcceptDefault:
    def test_default_continue_produces_recipes_for_flagged_columns(self, data_split):
        at = _at_on_feature_processing_step(data_split)
        at.run()
        assert not at.exception

        [b for b in at.button if b.key == "int.feature_processing.continue"][0].click().run()
        assert not at.exception
        assert at.session_state["interactive.step_confirmed"]["feature_processing"] is True

        recipes = at.session_state["interactive.feature_processing_recipes"]
        assert "f4" in recipes
        assert any(s.transformer == "impute" for s in recipes["f4"].steps)
        # A clean column with an empty recommended recipe contributes nothing
        # (accepting zero steps is a no-op, equivalent to Reject).
        assert "f1" not in recipes
        assert at.session_state["interactive.feature_processing_dropped_columns"] == []


class TestFeatureProcessingDrop:
    def test_drop_choice_removes_column_from_recipes_and_lists_it_dropped(self, data_split):
        at = _at_on_feature_processing_step(data_split)
        at.run()
        assert not at.exception

        drop_radio = [r for r in at.radio if r.key == "int.feature_processing.choice.f4"][0]
        drop_radio.set_value("Drop Column").run()
        assert not at.exception

        [b for b in at.button if b.key == "int.feature_processing.continue"][0].click().run()
        assert not at.exception

        assert "f4" not in at.session_state["interactive.feature_processing_recipes"]
        assert "f4" in at.session_state["interactive.feature_processing_dropped_columns"]


class TestFeatureProcessingReject:
    def test_reject_choice_leaves_column_out_of_recipes_and_not_dropped(self, data_split):
        at = _at_on_feature_processing_step(data_split)
        at.run()
        assert not at.exception

        reject_radio = [r for r in at.radio if r.key == "int.feature_processing.choice.f4"][0]
        reject_radio.set_value("Leave As-Is").run()
        assert not at.exception
        # Safety warning: f4 has missing values and would break training if
        # left raw — must be surfaced, not silently allowed.
        warning_text = " ".join(w.value for w in at.warning)
        assert "f4" in warning_text

        [b for b in at.button if b.key == "int.feature_processing.continue"][0].click().run()
        assert not at.exception

        assert "f4" not in at.session_state["interactive.feature_processing_recipes"]
        assert "f4" not in at.session_state["interactive.feature_processing_dropped_columns"]


class TestFeatureProcessingRestGroupBulkAccept:
    def test_accept_all_rest_marks_clean_columns_accepted(self, data_split):
        at = _at_on_feature_processing_step(data_split)
        at.run()
        assert not at.exception

        bulk_button = [b for b in at.button if b.key == "int.feature_processing.accept_all_rest"]
        assert bulk_button, "expected the Rest group's bulk-accept button to be rendered"
        bulk_button[0].click().run()
        assert not at.exception
        assert at.session_state["int.feature_processing.choice.cat_low"] == "Accept Recommended"

        [b for b in at.button if b.key == "int.feature_processing.continue"][0].click().run()
        assert not at.exception

        recipes = at.session_state["interactive.feature_processing_recipes"]
        assert "cat_low" in recipes
        assert recipes["cat_low"].steps[-1].transformer == "onehot_encode"
