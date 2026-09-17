"""Tests for dscompanion.tuning — Tuner with optuna/predefined backends."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dscompanion.models import ClassificationModel, ModelFactory
from dscompanion.tuning import Tuner


def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


def _hyperopt_available() -> bool:
    try:
        import hyperopt  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_split(data_split):
    """DataSplit with numeric-only X for fast model tests (no val set)."""
    split = MagicMock()
    split.X_train = data_split.X_train.select_dtypes(include="number").fillna(0)
    split.y_train = data_split.y_train
    split.X_val = None
    split.X_oot = data_split.X_oot.select_dtypes(include="number").fillna(0)
    split.y_oot = data_split.y_oot
    return split


@pytest.fixture
def numeric_split_with_val(data_split):
    """DataSplit with numeric-only X including a real validation set."""
    split = MagicMock()
    split.X_train = data_split.X_train.select_dtypes(include="number").fillna(0)
    split.y_train = data_split.y_train
    split.X_val = data_split.X_val.select_dtypes(include="number").fillna(0)
    split.y_val = data_split.y_val
    split.X_oot = data_split.X_oot.select_dtypes(include="number").fillna(0)
    split.y_oot = data_split.y_oot
    return split


# ---------------------------------------------------------------------------
# Predefined backend
# ---------------------------------------------------------------------------


class TestPredefinedBackend:
    def test_predefined_returns_fitted_model(self, numeric_split):
        model = ModelFactory.build("classification", "logistic")
        tuner = Tuner(
            model=model,
            backend="predefined",
            search_space="logistic_classification",
            cv=numeric_split,
        )
        best_model = tuner.run()
        assert isinstance(best_model, ClassificationModel)
        preds = best_model.predict(numeric_split.X_oot)
        assert len(preds) == len(numeric_split.X_oot)

    def test_predefined_best_params_non_empty(self, numeric_split):
        model = ModelFactory.build("classification", "logistic")
        tuner = Tuner(
            model=model,
            backend="predefined",
            search_space="logistic_classification",
            cv=numeric_split,
        )
        tuner.run()
        assert isinstance(tuner.best_params_, dict)
        assert len(tuner.best_params_) > 0


# ---------------------------------------------------------------------------
# Optuna backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _optuna_available(), reason="optuna not installed")
class TestOptunaBackend:
    def test_run_5_trials_no_error(self, numeric_split):
        model = ModelFactory.build("classification", "logistic")
        search_space = {
            "C": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        }
        tuner = Tuner(
            model=model,
            backend="optuna",
            search_space=search_space,
            n_trials=5,
            cv=numeric_split,
            metric="roc_auc",
        )
        best_model = tuner.run()
        assert isinstance(best_model, ClassificationModel)

    def test_best_params_is_dict(self, numeric_split):
        model = ModelFactory.build("classification", "logistic")
        tuner = Tuner(
            model=model,
            backend="optuna",
            search_space={"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=3,
            cv=numeric_split,
        )
        tuner.run()
        assert isinstance(tuner.best_params_, dict)
        assert len(tuner.best_params_) > 0

    def test_best_score_is_float(self, numeric_split):
        model = ModelFactory.build("classification", "logistic")
        tuner = Tuner(
            model=model,
            backend="optuna",
            search_space={"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=3,
            cv=numeric_split,
        )
        tuner.run()
        assert isinstance(tuner.best_score_, float)

    def test_returned_model_roc_auc_above_05(self, numeric_split):
        from unittest.mock import MagicMock

        from dscompanion.models import ModelFactory

        model = ModelFactory.build("classification", "logistic")
        tuner = Tuner(
            model=model,
            backend="optuna",
            search_space={"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=3,
            cv=numeric_split,
        )
        best_model = tuner.run()
        split = MagicMock()
        split.X_train = numeric_split.X_train
        split.y_train = numeric_split.y_train
        split.X_val = None
        split.X_oot = None
        eval_df = best_model.evaluate(split)
        auc_rows = eval_df[eval_df["metric"] == "roc_auc"]
        assert len(auc_rows) > 0
        assert auc_rows["value"].iloc[0] > 0.5

    def test_missing_cv_raises(self):
        model = ModelFactory.build("classification", "logistic")
        tuner = Tuner(model=model, backend="optuna", n_trials=3)
        with pytest.raises(ValueError, match="cv"):
            tuner.run()

    def test_best_trial_summary_raises_before_run(self, numeric_split):
        from sklearn.exceptions import NotFittedError

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(model, {}, n_trials=1, cv=numeric_split, metric="roc_auc")
        with pytest.raises(NotFittedError):
            backend.best_trial_summary()

    def test_best_trial_summary_returns_dataframe(self, numeric_split):
        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model,
            {"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=3,
            cv=numeric_split,
            metric="roc_auc",
        )
        backend.run()
        df = backend.best_trial_summary()
        assert list(df.columns) == ["trial_number", "params", "metric_value"]
        assert len(df) <= 3

    def test_best_trial_summary_empty_when_all_trials_fail(self, numeric_split):
        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(model, {}, n_trials=1, cv=numeric_split, metric="roc_auc")
        mock_trial = MagicMock()
        mock_trial.value = None
        backend.study_ = MagicMock()
        backend.study_.trials = [mock_trial]
        df = backend.best_trial_summary()
        assert df.empty
        assert list(df.columns) == ["trial_number", "params", "metric_value"]

    def test_optimization_history_plot_raises_before_run(self, numeric_split):
        from sklearn.exceptions import NotFittedError

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(model, {}, n_trials=1, cv=numeric_split, metric="roc_auc")
        with pytest.raises(NotFittedError):
            backend.optimization_history_plot()

    def test_optimization_history_plot_returns_figure(self, numeric_split):
        import plotly.graph_objects as go

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model,
            {"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=3,
            cv=numeric_split,
            metric="roc_auc",
        )
        backend.run()
        fig = backend.optimization_history_plot()
        assert isinstance(fig, go.Figure)

    def test_param_importance_plot_returns_figure(self, numeric_split):
        import plotly.graph_objects as go

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model,
            {"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=3,
            cv=numeric_split,
            metric="roc_auc",
        )
        backend.run()
        fig = backend.param_importance_plot()
        assert isinstance(fig, go.Figure)

    def test_run_with_val_split_exercises_eval_set(self, numeric_split_with_val):
        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model,
            {"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=2,
            cv=numeric_split_with_val,
            metric="roc_auc",
        )
        best_params = backend.run()
        assert isinstance(best_params, dict)

    def test_build_sampler_random(self, numeric_split):
        import optuna

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model, {}, n_trials=1, cv=numeric_split, metric="roc_auc", sampler_type="random"
        )
        assert isinstance(backend._build_sampler(optuna), optuna.samplers.RandomSampler)

    def test_build_sampler_cmaes(self, numeric_split):
        import optuna

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model, {}, n_trials=1, cv=numeric_split, metric="roc_auc", sampler_type="cmaes"
        )
        assert isinstance(backend._build_sampler(optuna), optuna.samplers.CmaEsSampler)

    def test_build_sampler_unknown_raises(self, numeric_split):
        import optuna

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model, {}, n_trials=1, cv=numeric_split, metric="roc_auc", sampler_type="gridsearch"
        )
        with pytest.raises(ValueError, match="sampler_type"):
            backend._build_sampler(optuna)

    def test_build_pruner_hyperband(self, numeric_split):
        import optuna

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model, {}, n_trials=1, cv=numeric_split, metric="roc_auc", pruner_type="hyperband"
        )
        assert isinstance(backend._build_pruner(optuna), optuna.pruners.HyperbandPruner)

    def test_build_pruner_none(self, numeric_split):
        import optuna

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model, {}, n_trials=1, cv=numeric_split, metric="roc_auc", pruner_type="none"
        )
        assert isinstance(backend._build_pruner(optuna), optuna.pruners.NopPruner)

    def test_build_pruner_unknown_raises(self, numeric_split):
        import optuna

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(
            model, {}, n_trials=1, cv=numeric_split, metric="roc_auc", pruner_type="invalid"
        )
        with pytest.raises(ValueError, match="pruner_type"):
            backend._build_pruner(optuna)


# ---------------------------------------------------------------------------
# Search space auto-inference
# ---------------------------------------------------------------------------


class TestSearchSpaceInference:
    """XGBClassifier/XGBRegressor abbreviate "xgboost" to "xgb", which used to
    break the substring-token match against the "xgboost_classification" /
    "xgboost_regression" search-space keys — auto-inference silently returned
    None for the most common algorithm. See _CLASS_NAME_ALIASES in tuner.py.
    """

    def test_all_max_depth_ranges_capped_at_10(self):
        """max_depth is capped at 10 for every algorithm — deep trees overfit
        on typical tabular dataset sizes.
        """
        from dscompanion.tuning.search_spaces import SEARCH_SPACES

        for backend, spaces in SEARCH_SPACES.items():
            for key, space in spaces.items():
                if "max_depth" in space:
                    high = space["max_depth"]["high"]
                    assert (
                        high <= 10
                    ), f"{backend}.{key}.max_depth.high={high} exceeds the cap of 10"

    def test_xgboost_classification_resolves(self):
        from dscompanion.tuning.tuner import _infer_space_key

        model = ModelFactory.build("classification", "xgboost")
        assert _infer_space_key(model, "optuna") == "xgboost_classification"
        assert _infer_space_key(model, "hyperopt") == "xgboost_classification"

    def test_xgboost_regression_resolves(self):
        from dscompanion.tuning.tuner import _infer_space_key

        model = ModelFactory.build("regression", "xgboost")
        assert _infer_space_key(model, "optuna") == "xgboost_regression"

    def test_logistic_and_random_forest_still_resolve(self):
        from dscompanion.tuning.tuner import _infer_space_key

        logistic = ModelFactory.build("classification", "logistic")
        assert _infer_space_key(logistic, "optuna") == "logistic_classification"

        rf = ModelFactory.build("classification", "random_forest")
        assert _infer_space_key(rf, "optuna") == "random_forest_classification"

    @pytest.mark.parametrize(
        "algorithm,expected_key",
        [
            ("svm", "svm_classification"),
            ("knn", "knn_classification"),
            ("decision_tree", "decision_tree_classification"),
            ("extra_trees", "extra_trees_classification"),
            ("adaboost", "adaboost_classification"),
            ("naive_bayes", "naive_bayes_classification"),
            ("lightgbm", "lightgbm_classification"),
            ("gradient_boosting", "gradient_boosting_classification"),
            ("random_forest", "random_forest_classification"),
            ("lda", "lda_classification"),
            ("qda", "qda_classification"),
            ("mlp", "mlp_classification"),
        ],
    )
    def test_new_classification_algorithms_resolve(self, algorithm, expected_key):
        """SVC -> "svc" and GaussianNB -> "gaussiannb" don't contain their search-space
        key's "svm"/"naive_bayes" tokens verbatim -- same class of bug as the historical
        xgboost gap above. Covered by the "svc"/"kneighbors"/"gaussiannb" aliases in
        _CLASS_NAME_ALIASES. LinearDiscriminantAnalysis/QuadraticDiscriminantAnalysis ->
        "lineardiscriminantanalysis"/"quadraticdiscriminantanalysis" are similarly
        covered by their own aliases; MLPClassifier -> "mlpclassifier" needs no alias
        since "mlp" is already a verbatim substring.
        """
        from dscompanion.tuning.tuner import _infer_space_key

        model = ModelFactory.build("classification", algorithm)
        assert _infer_space_key(model, "optuna") == expected_key
        assert _infer_space_key(model, "hyperopt") == expected_key

    @pytest.mark.parametrize(
        "algorithm,expected_key",
        [
            ("elastic_net", "elastic_net_regression"),
            ("gradient_boosting", "gradient_boosting_regression"),
            ("svm", "svm_regression"),
            ("knn", "knn_regression"),
            ("decision_tree", "decision_tree_regression"),
            ("extra_trees", "extra_trees_regression"),
            ("adaboost", "adaboost_regression"),
            ("lightgbm", "lightgbm_regression"),
            ("random_forest", "random_forest_regression"),
            ("lasso", "lasso_regression"),
        ],
    )
    def test_new_regression_algorithms_resolve(self, algorithm, expected_key):
        """SVR -> "svr" doesn't contain the "svm_regression" key's "svm" token
        verbatim -- covered by the new "svr"->"svm" alias in _CLASS_NAME_ALIASES.
        """
        from dscompanion.tuning.tuner import _infer_space_key

        model = ModelFactory.build("regression", algorithm)
        assert _infer_space_key(model, "optuna") == expected_key
        assert _infer_space_key(model, "hyperopt") == expected_key

    def test_linear_regression_has_no_search_space(self):
        """Deliberately excluded, not a gap — LinearRegression has zero tunable
        hyperparameters in this codebase's usage
        (ModelFactory._DEFAULTS["regression"]["linear"] == {}), so there is
        nothing meaningful for Optuna/Hyperopt to search over.
        """
        from dscompanion.tuning.tuner import _infer_space_key

        model = ModelFactory.build("regression", "linear")
        assert _infer_space_key(model, "optuna") is None
        assert _infer_space_key(model, "hyperopt") is None

    @pytest.mark.skipif(not _optuna_available(), reason="optuna not installed")
    def test_tuner_run_without_explicit_search_space_on_xgboost(self, numeric_split):
        model = ModelFactory.build(
            "classification",
            "xgboost",
            params={"n_estimators": 20, "max_depth": 3, "early_stopping_rounds": None},
        )
        tuner = Tuner(model=model, backend="optuna", n_trials=2, cv=numeric_split, metric="roc_auc")
        best_model = tuner.run()
        assert isinstance(best_model, ClassificationModel)
        assert tuner.best_params_

    @pytest.mark.skipif(not _optuna_available(), reason="optuna not installed")
    def test_tuner_run_without_explicit_search_space_on_lightgbm(self, numeric_split):
        """Regression test — reproduces the real crash found on the cluster 2026-09-11:
        Leaderboard picked lightgbm as the winner, then Tuner raised
        'Could not infer search space for ClassificationModel' because
        lightgbm_classification had no entry in SEARCH_SPACES.
        """
        model = ModelFactory.build("classification", "lightgbm", params={"n_estimators": 20})
        tuner = Tuner(model=model, backend="optuna", n_trials=2, cv=numeric_split, metric="roc_auc")
        best_model = tuner.run()
        assert isinstance(best_model, ClassificationModel)
        assert tuner.best_params_

    def test_optimization_curve_populated_after_run(self, numeric_split):
        import pandas as pd

        model = ModelFactory.build(
            "classification",
            "xgboost",
            params={"n_estimators": 20, "max_depth": 3, "early_stopping_rounds": None},
        )
        tuner = Tuner(model=model, backend="optuna", n_trials=3, cv=numeric_split, metric="roc_auc")
        tuner.run()
        assert hasattr(tuner, "optimization_curve_")
        assert isinstance(tuner.optimization_curve_, pd.DataFrame)
        assert set(tuner.optimization_curve_.columns) == {
            "trial_number",
            "metric_value",
            "best_so_far",
        }
        assert len(tuner.optimization_curve_) > 0
        # best_so_far must be monotonically non-decreasing for maximize direction
        bsf = tuner.optimization_curve_["best_so_far"].tolist()
        assert all(bsf[i] <= bsf[i + 1] for i in range(len(bsf) - 1))


# ---------------------------------------------------------------------------
# Full Tuner.run() coverage — every classification/regression algorithm
# ---------------------------------------------------------------------------
#
# A resolved search-space key (TestSearchSpaceInference above) doesn't prove
# Optuna can actually sample from it and fit the estimator without error --
# that's exactly the class of gap that let the lightgbm registry hole through
# undetected until it hit the real cluster. This exercises a full Tuner.run()
# for every algorithm ModelFactory supports, not just its search-space lookup.


@pytest.fixture
def numeric_regression_split(data_split):
    """Numeric-only split with a continuous target, for regression-task
    Tuner.run() tests -- the shared synthetic fixtures only carry a binary
    classification target."""
    split = MagicMock()
    X_train = data_split.X_train.select_dtypes(include="number").fillna(0)
    X_oot = data_split.X_oot.select_dtypes(include="number").fillna(0)
    split.X_train = X_train
    split.y_train = X_train["f1"] + 0.5 * X_train["f6"] - 0.25 * X_train["f7"]
    split.X_val = None
    split.X_oot = X_oot
    split.y_oot = X_oot["f1"] + 0.5 * X_oot["f6"] - 0.25 * X_oot["f7"]
    return split


@pytest.mark.skipif(not _optuna_available(), reason="optuna not installed")
class TestTuningAcrossAllAlgorithms:
    @pytest.mark.parametrize(
        "algorithm",
        [
            "xgboost",
            "lightgbm",
            "logistic",
            "random_forest",
            "gradient_boosting",
            "svm",
            "knn",
            "decision_tree",
            "extra_trees",
            "adaboost",
            "naive_bayes",
            "lda",
            "qda",
            "mlp",
        ],
    )
    def test_tunes_successfully(self, algorithm, numeric_split):
        model = ModelFactory.build("classification", algorithm)
        tuner = Tuner(model=model, backend="optuna", n_trials=2, cv=numeric_split, metric="roc_auc")
        best_model = tuner.run()
        assert isinstance(best_model, ClassificationModel)
        assert tuner.best_params_

    @pytest.mark.parametrize(
        "algorithm",
        [
            "xgboost",
            "lightgbm",
            "ridge",
            "lasso",
            "elastic_net",
            "random_forest",
            "gradient_boosting",
            "svm",
            "knn",
            "decision_tree",
            "extra_trees",
            "adaboost",
        ],
    )
    def test_tunes_successfully_regression(self, algorithm, numeric_regression_split):
        from dscompanion.models import RegressionModel

        model = ModelFactory.build("regression", algorithm)
        tuner = Tuner(
            model=model,
            backend="optuna",
            n_trials=2,
            cv=numeric_regression_split,
            metric="r2",
            direction="maximize",
        )
        best_model = tuner.run()
        assert isinstance(best_model, RegressionModel)
        assert tuner.best_params_

    def test_linear_regression_run_raises_clear_error(self, numeric_regression_split):
        """Completes the contract test_linear_regression_has_no_search_space (which only
        checked _infer_space_key) by confirming .run() itself raises cleanly, rather than
        failing some other way, when no search space is registered.
        """
        model = ModelFactory.build("regression", "linear")
        tuner = Tuner(
            model=model,
            backend="optuna",
            n_trials=2,
            cv=numeric_regression_split,
            metric="r2",
            direction="maximize",
        )
        with pytest.raises(ValueError, match="Could not infer search space"):
            tuner.run()


# ---------------------------------------------------------------------------
# Full Tuner.run() coverage via the hyperopt backend
# ---------------------------------------------------------------------------
#
# Regression tests for two hyperopt-specific bugs found while adding lda/qda/mlp
# (2026-09-17), neither ever caught before because no existing test ran a full
# Tuner.run(backend="hyperopt") end to end:
#
# 1. HyperoptBackend.run()'s best_params extraction read hyperopt's raw internal
#    trials_.trials[i]["misc"]["vals"] representation directly -- for a
#    "categorical" param (hp.choice) that's the *index* into choices, not the
#    resolved value, so e.g. lda's "solver" came back as 0 instead of "eigen" and
#    crashed the final refit. Fixed via hyperopt.space_eval() in both run() and
#    best_trial_summary().
# 2. hp.quniform always returns a float even though every "quniform" usage in
#    this codebase represents an integer hyperparameter (n_neighbors, max_depth,
#    etc.) -- sklearn 1.9's strict param validation rejects the float, failing
#    every trial for knn/decision_tree/extra_trees/random_forest/adaboost. Fixed
#    via hyperopt.pyll.scope.int() in _build_hp_space.


@pytest.mark.skipif(not _hyperopt_available(), reason="hyperopt not installed")
class TestHyperoptBackendEndToEnd:
    @pytest.mark.parametrize(
        "algorithm",
        [
            "xgboost",
            "lightgbm",
            "logistic",
            "random_forest",
            "gradient_boosting",
            "svm",
            "knn",
            "decision_tree",
            "extra_trees",
            "adaboost",
            "naive_bayes",
            "lda",
            "qda",
            "mlp",
        ],
    )
    def test_tunes_successfully(self, algorithm, numeric_split):
        params = {"n_estimators": 20} if algorithm in ("xgboost", "lightgbm") else None
        model = ModelFactory.build("classification", algorithm, params=params)
        tuner = Tuner(
            model=model, backend="hyperopt", n_trials=2, cv=numeric_split, metric="roc_auc"
        )
        best_model = tuner.run()
        assert isinstance(best_model, ClassificationModel)
        assert tuner.best_params_

    def test_categorical_param_resolved_to_value_not_index(self, numeric_split):
        """lda's "solver" search-space entry is categorical (hp.choice) -- before the
        space_eval fix, best_params_["solver"] came back as a raw int index (e.g. 0)
        instead of "lsqr"/"eigen", crashing the final refit with an sklearn
        InvalidParameterError.
        """
        model = ModelFactory.build("classification", "lda")
        tuner = Tuner(
            model=model, backend="hyperopt", n_trials=3, cv=numeric_split, metric="roc_auc"
        )
        tuner.run()
        assert tuner.best_params_["solver"] in ("lsqr", "eigen")

    def test_quniform_param_resolved_to_int_not_float(self, numeric_split):
        """knn's "n_neighbors" search-space entry is quniform -- before the scope.int()
        fix, hyperopt returned a float (e.g. 25.0), which sklearn 1.9's strict param
        validation rejects for KNeighborsClassifier, failing every trial.
        """
        model = ModelFactory.build("classification", "knn")
        tuner = Tuner(
            model=model, backend="hyperopt", n_trials=3, cv=numeric_split, metric="roc_auc"
        )
        tuner.run()
        assert isinstance(tuner.best_params_["n_neighbors"], int)

    def test_best_trial_summary_resolves_categorical_params(self, numeric_split):
        """best_trial_summary()'s per-trial params dict had the same raw-index bug as
        run()'s best_params_ -- a second call site reading the same underlying
        trials_.trials[i]["misc"]["vals"] structure.
        """
        model = ModelFactory.build("classification", "lda")
        tuner = Tuner(
            model=model, backend="hyperopt", n_trials=3, cv=numeric_split, metric="roc_auc"
        )
        tuner.run()
        assert all(row["solver"] in ("lsqr", "eigen") for row in tuner.trials_dataframe_["params"])


# ---------------------------------------------------------------------------
# _suggest_param helper
# ---------------------------------------------------------------------------


class TestSuggestParam:
    """Direct unit tests for the module-level _suggest_param helper."""

    def test_categorical_type(self):
        from dscompanion.tuning.backends.optuna_backend import _suggest_param

        trial = MagicMock()
        trial.suggest_categorical.return_value = "B"
        result = _suggest_param(trial, "algo", {"type": "categorical", "choices": ["A", "B", "C"]})
        assert result == "B"
        trial.suggest_categorical.assert_called_once_with("algo", ["A", "B", "C"])

    def test_unknown_type_raises_value_error(self):
        from dscompanion.tuning.backends.optuna_backend import _suggest_param

        trial = MagicMock()
        with pytest.raises(ValueError, match="Unknown param type"):
            _suggest_param(trial, "p", {"type": "unknown_type"})


# ---------------------------------------------------------------------------
# Per-trial logging (stdlib logging, replaces mlflow nested-run logging)
# ---------------------------------------------------------------------------


class TestOptunaBackendTrialLogging:
    def test_log_trial_logs_params_and_score(self, caplog, numeric_split):
        import logging

        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(model, {}, n_trials=1, cv=numeric_split, metric="roc_auc")
        with caplog.at_level(logging.INFO, logger="dscompanion.tuning.backends.optuna_backend"):
            backend._log_trial(0, {"max_depth": 5}, 0.87)
        message = caplog.records[0].message
        assert "0" in message and "max_depth" in message and "0.87" in message

    def test_log_trial_never_raises_on_nan_score(self, numeric_split):
        from dscompanion.tuning.backends.optuna_backend import OptunaBackend

        model = ModelFactory.build("classification", "logistic")
        backend = OptunaBackend(model, {}, n_trials=1, cv=numeric_split, metric="roc_auc")
        backend._log_trial(0, {"max_depth": 5}, float("nan"))  # must not raise


class TestHyperoptBackendTrialLogging:
    def test_log_trial_logs_params_and_score(self, caplog, numeric_split):
        import logging

        from dscompanion.tuning.backends.hyperopt_backend import HyperoptBackend

        model = ModelFactory.build("classification", "logistic")
        backend = HyperoptBackend(model, {}, n_trials=1, cv=numeric_split, metric="roc_auc")
        with caplog.at_level(logging.INFO, logger="dscompanion.tuning.backends.hyperopt_backend"):
            backend._log_trial(0, {"max_depth": 5}, 0.87)
        message = caplog.records[0].message
        assert "0" in message and "max_depth" in message and "0.87" in message

    def test_log_trial_never_raises_on_nan_score(self, numeric_split):
        from dscompanion.tuning.backends.hyperopt_backend import HyperoptBackend

        model = ModelFactory.build("classification", "logistic")
        backend = HyperoptBackend(model, {}, n_trials=1, cv=numeric_split, metric="roc_auc")
        backend._log_trial(0, {"max_depth": 5}, float("nan"))  # must not raise
