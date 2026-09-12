"""Tests for dscompanion/app/interactive_step_train.py's ``_leaderboard_worker`` — the
background training loop behind #32's non-blocking leaderboard comparison.

``_leaderboard_worker`` takes no Streamlit dependency by design (it runs in a background
thread — see its docstring), so it's called directly here as a plain function; no threading
or Streamlit session-state mocking needed to exercise its logic.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

# dscompanion/app/ is a standalone script directory, not part of the installable
# dscompanion package (see streamlit.md) — add it to sys.path the same way
# streamlit_app.py does for its own sibling imports, since
# interactive_step_train.py imports `from interactive_state import
# add_audit_entry, set_step_status`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from interactive_state import STEP_ORDER
from interactive_step_train import _leaderboard_worker

from dscompanion.models import ModelFactory


class TestLeaderboardWorker:
    def test_populates_progress_and_result_on_success(self, numeric_split):
        progress: list[dict] = []
        result: dict = {}
        _leaderboard_worker(
            numeric_split,
            None,
            "classification",
            ["logistic", "decision_tree"],
            progress,
            threading.Lock(),
            result,
        )

        assert [p["algorithm"] for p in progress] == ["logistic", "decision_tree"]
        assert all(p["status"] in ("ok", "failed") for p in progress)  # none left "running"
        assert result["error"] is None
        assert result["winner"] in ("logistic", "decision_tree")
        assert result["winner_model"] is not None
        assert set(result["lb_df"]["algorithm"]) == {"logistic", "decision_tree"}
        # Every successfully-trained algorithm's fitted model must be retained (not
        # just the winner's) so the UI can offer a choice without retraining.
        assert set(result["fitted_models"]) == {"logistic", "decision_tree"}
        assert result["fitted_models"][result["winner"]] is result["winner_model"]

    def test_all_algorithms_failing_sets_error_and_no_winner(self, numeric_split):
        progress: list[dict] = []
        result: dict = {}
        with patch("interactive_step_train.ModelFactory.build", side_effect=RuntimeError("boom")):
            _leaderboard_worker(
                numeric_split,
                None,
                "classification",
                ["logistic", "decision_tree"],
                progress,
                threading.Lock(),
                result,
            )

        assert all(p["status"] == "failed" for p in progress)
        assert result["winner"] is None
        assert result["winner_model"] is None
        assert result["fitted_models"] == {}
        assert "Every algorithm failed" in result["error"]

    def test_unexpected_crash_still_populates_result(self, numeric_split):
        """The outer try/except must guarantee `result` even on a totally unanticipated failure."""
        progress: list[dict] = []
        result: dict = {}
        # Patch only .sort_values (used once, post-loop) rather than the DataFrame
        # constructor itself — the except block's own recovery line
        # (`result["lb_df"] = pd.DataFrame()`) also constructs a DataFrame, and a
        # blanket constructor patch would break that recovery path too, not just
        # simulate an unanticipated failure.
        with patch("pandas.DataFrame.sort_values", side_effect=RuntimeError("unexpected")):
            _leaderboard_worker(
                numeric_split,
                None,
                "classification",
                ["logistic"],
                progress,
                threading.Lock(),
                result,
            )

        assert result["error"] is not None
        assert "Unexpected error" in result["error"]
        assert result["winner"] is None
        assert result["lb_df"].empty

    def test_progress_updates_in_place_not_appended_twice(self, numeric_split):
        """Each algorithm gets exactly one progress entry (running -> ok/failed in place)."""
        progress: list[dict] = []
        result: dict = {}
        _leaderboard_worker(
            numeric_split, None, "classification", ["logistic"], progress, threading.Lock(), result
        )
        assert len(progress) == 1


class TestLeaderboardResultsUI:
    """Regression coverage: leaderboard mode always used the auto-selected
    "winner" model with no way to pick a different one, and going back from
    Step 9 to Step 8 left the user stuck on frozen results with no way to
    rebuild.
    """

    def _seeded_app(self, numeric_split, fitted_models: dict, winner: str = "logistic"):
        lb_df = pd.DataFrame(
            [
                {
                    "algorithm": algo,
                    "status": "ok",
                    "fit_time_s": 0.1,
                    "roc_auc": 0.9 if algo == winner else 0.8,
                    "error": None,
                }
                for algo in fitted_models
            ]
        )
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
        at.session_state["run_mode"] = "Interactive"
        confirmed = {step: True for step in STEP_ORDER}
        confirmed["train"] = False
        at.session_state["interactive.step_confirmed"] = confirmed
        at.session_state["interactive.task"] = "classification"
        at.session_state["interactive.target"] = "target"
        at.session_state["interactive.load_data_selection_id"] = (None, None, None)
        at.session_state["interactive.split"] = numeric_split
        at.session_state["int.train.path"] = "leaderboard"
        at.session_state["int.train.leaderboard_df"] = lb_df
        at.session_state["int.train.winner"] = winner
        at.session_state["int.train.winner_model"] = fitted_models[winner]
        at.session_state["int.train.fitted_models"] = fitted_models
        at.session_state["int.train.processed_split"] = numeric_split
        return at

    def test_can_pick_a_different_algorithm_than_the_winner(self, numeric_split):
        model_a = ModelFactory.build(task="classification", algorithm="logistic")
        model_a.fit(numeric_split.train_X, numeric_split.train_y)
        model_b = ModelFactory.build(task="classification", algorithm="decision_tree")
        model_b.fit(numeric_split.train_X, numeric_split.train_y)
        fitted_models = {"logistic": model_a, "decision_tree": model_b}

        at = self._seeded_app(numeric_split, fitted_models, winner="logistic")
        at.run()
        assert not at.exception

        selectbox = at.selectbox(key="int.train.selected_algorithm")
        assert selectbox.value == "logistic"  # defaults to the recommended winner

        selectbox.set_value("decision_tree").run()
        assert not at.exception
        assert at.session_state["int.train.algorithm"] == "decision_tree"
        assert at.session_state["int.train.winner_model"] is model_b

    def test_compare_again_clears_state_so_the_user_can_rebuild(self, numeric_split):
        """Going back to Step 8 previously left the user stuck on frozen leaderboard
        results with no way to retrain — "Compare again" must reset far enough that
        the initial "how would you like to choose an algorithm" screen reappears.
        """
        model_a = ModelFactory.build(task="classification", algorithm="logistic")
        model_a.fit(numeric_split.train_X, numeric_split.train_y)

        at = self._seeded_app(numeric_split, {"logistic": model_a}, winner="logistic")
        at.run()
        assert not at.exception

        at.button(key="int.train.rerun_leaderboard").click().run()

        assert not at.exception
        assert "int.train.leaderboard_df" not in at.session_state
        assert "int.train.path" not in at.session_state
        assert "int.train.fitted_models" not in at.session_state
        # Back to the initial choice screen, not stuck on stale results.
        assert at.button(key="int.train.pick_leaderboard")
        assert at.button(key="int.train.pick_single")
