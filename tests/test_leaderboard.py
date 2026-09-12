"""Tests for dscompanion.leaderboard — Leaderboard multi-algorithm comparison."""

from __future__ import annotations

import pandas as pd
import pytest

from dscompanion.leaderboard import Leaderboard
from dscompanion.models import ClassificationModel, ModelFactory


@pytest.fixture
def numeric_data_split(data_split):
    """DataSplit with numeric-only X, real DataFrames throughout (no MagicMock) —
    Leaderboard.run() calls model.evaluate(split), which inspects every split's
    X_*/y_* directly, so each partition must be a genuine (possibly empty)
    DataFrame/Series rather than an auto-attribute mock.
    """
    from dscompanion.split import DataSplit

    def _numeric(df: pd.DataFrame) -> pd.DataFrame:
        return df.select_dtypes(include="number").fillna(0)

    return DataSplit(
        train_X=_numeric(data_split.train_X),
        train_y=data_split.train_y,
        val_X=_numeric(data_split.val_X),
        val_y=data_split.val_y,
        test_X=_numeric(data_split.test_X),
        test_y=data_split.test_y,
        oot_X=_numeric(data_split.oot_X),
        oot_y=data_split.oot_y,
        metadata=data_split.metadata,
    )


_FAST_ALGORITHMS = ["logistic", "decision_tree", "naive_bayes"]


class TestLeaderboardConstruction:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="task='classification'"):
            Leaderboard(task="regression")

    def test_unknown_include_algorithm_raises_on_run(self, numeric_data_split):
        lb = Leaderboard(include=["alien_net"])
        with pytest.raises(ValueError, match="Unknown algorithm"):
            lb.run(numeric_data_split)

    def test_best_model_before_run_raises_runtime_error(self):
        lb = Leaderboard()
        with pytest.raises(RuntimeError, match="run\\(\\) must be called"):
            lb.best_model()


class TestLeaderboardRun:
    def test_run_returns_expected_columns(self, numeric_data_split):
        lb = Leaderboard(include=_FAST_ALGORITHMS)
        result = lb.run(numeric_data_split)

        assert isinstance(result, pd.DataFrame)
        for col in ["algorithm", "status", "fit_time_seconds", "error", "roc_auc"]:
            assert col in result.columns
        assert set(result["algorithm"]) == set(_FAST_ALGORITHMS)
        assert result is lb.leaderboard_

    def test_run_sorts_by_default_metric_descending(self, numeric_data_split):
        lb = Leaderboard(include=_FAST_ALGORITHMS)
        result = lb.run(numeric_data_split)

        scores = result["roc_auc"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)

    def test_include_filters_algorithms(self, numeric_data_split):
        lb = Leaderboard(include=["logistic", "naive_bayes"])
        result = lb.run(numeric_data_split)
        assert set(result["algorithm"]) == {"logistic", "naive_bayes"}

    def test_exclude_filters_algorithms(self, numeric_data_split):
        all_algorithms = set(ModelFactory.SUPPORTED_ALGORITHMS["classification"])
        lb = Leaderboard(exclude=["svm", "xgboost"])
        result = lb.run(numeric_data_split)
        assert set(result["algorithm"]) == all_algorithms - {"svm", "xgboost"}

    def test_fitted_models_populated_for_successful_algorithms(self, numeric_data_split):
        lb = Leaderboard(include=_FAST_ALGORITHMS)
        lb.run(numeric_data_split)
        assert set(lb.fitted_models_) == set(_FAST_ALGORITHMS)
        for model in lb.fitted_models_.values():
            assert isinstance(model, ClassificationModel)

    def test_best_model_returns_top_ranked_fitted_model(self, numeric_data_split):
        lb = Leaderboard(include=_FAST_ALGORITHMS)
        result = lb.run(numeric_data_split)
        best = lb.best_model()
        assert isinstance(best, ClassificationModel)
        assert lb.fitted_models_[result.iloc[0]["algorithm"]] is best

    def test_best_algorithm_matches_best_model(self, numeric_data_split):
        lb = Leaderboard(include=_FAST_ALGORITHMS)
        lb.run(numeric_data_split)
        assert lb.fitted_models_[lb.best_algorithm()] is lb.best_model()

    def test_best_algorithm_before_run_raises_runtime_error(self):
        lb = Leaderboard()
        with pytest.raises(RuntimeError, match="run\\(\\) must be called"):
            lb.best_algorithm()

    def test_eval_split_defaults_to_val_when_non_empty(self, numeric_data_split):
        lb = Leaderboard(include=["logistic"])
        lb.run(numeric_data_split)
        assert lb._resolve_eval_split(numeric_data_split) == "val"

    def test_eval_split_falls_back_to_test_when_val_empty(self, numeric_data_split):
        numeric_data_split.val_X = pd.DataFrame(columns=numeric_data_split.train_X.columns)
        numeric_data_split.val_y = pd.Series(dtype=float)
        lb = Leaderboard(include=["logistic"])
        assert lb._resolve_eval_split(numeric_data_split) == "test"

    def test_explicit_eval_split_honoured(self, numeric_data_split):
        lb = Leaderboard(include=["logistic"], eval_split="oot")
        result = lb.run(numeric_data_split)
        assert result.loc[0, "status"] == "ok"

    def test_ascending_auto_resolves_for_log_loss(self, numeric_data_split):
        lb = Leaderboard(include=_FAST_ALGORITHMS, sort_metric="log_loss")
        result = lb.run(numeric_data_split)
        scores = result["log_loss"].dropna().tolist()
        assert scores == sorted(scores)

    def test_failed_algorithm_recorded_not_raised(self, numeric_data_split):
        lb = Leaderboard(
            include=["logistic", "knn"],
            params_overrides={"knn": {"n_neighbors": -1}},
        )
        result = lb.run(numeric_data_split)

        knn_row = result[result["algorithm"] == "knn"].iloc[0]
        assert knn_row["status"] == "failed"
        assert knn_row["error"] is not None
        assert "knn" not in lb.fitted_models_
        # the well-formed algorithm must still succeed despite knn's failure
        logistic_row = result[result["algorithm"] == "logistic"].iloc[0]
        assert logistic_row["status"] == "ok"

    def test_full_default_algorithm_set_runs_end_to_end(self, numeric_data_split):
        # lightgbm is an optional dependency not installed in this dev env — its row is
        # expected to fail with ImportError; every other algorithm must still succeed.
        lb = Leaderboard(exclude=["lightgbm"])
        result = lb.run(numeric_data_split)
        assert set(result["algorithm"]) == set(
            ModelFactory.SUPPORTED_ALGORITHMS["classification"]
        ) - {"lightgbm"}
        assert (result["status"] == "ok").all()
