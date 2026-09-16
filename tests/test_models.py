"""Tests for dscompanion.models — ModelFactory, ClassificationModel, evaluate, save/load."""

from __future__ import annotations

import inspect
import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from dscompanion.config import settings
from dscompanion.models import ClassificationModel, ClusteringModel, ModelFactory, RegressionModel
from dscompanion.models.base import BaseDSCompanionModel
from dscompanion.utils.metrics import psi_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_X_train(data_split):
    """Numeric-only feature subset for quick model tests."""
    X = data_split.X_train
    return X.select_dtypes(include="number").fillna(0)


@pytest.fixture
def numeric_X_oot(data_split):
    X = data_split.X_oot
    return X.select_dtypes(include="number").fillna(0)


@pytest.fixture
def numeric_X_test(data_split):
    X = data_split.X_test
    return X.select_dtypes(include="number").fillna(0)


@pytest.fixture
def y_train(data_split):
    return data_split.y_train


@pytest.fixture
def y_oot(data_split):
    return data_split.y_oot


@pytest.fixture
def y_test(data_split):
    return data_split.y_test


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------


class TestModelFactory:
    def test_build_xgboost_classification_returns_classification_model(self):
        model = ModelFactory.build(
            task="classification",
            algorithm="xgboost",
        )
        assert isinstance(model, ClassificationModel)

    def test_build_logistic_returns_classification_model(self):
        model = ModelFactory.build(
            task="classification",
            algorithm="logistic",
        )
        assert isinstance(model, ClassificationModel)

    def test_build_regression(self):
        model = ModelFactory.build(
            task="regression",
            algorithm="ridge",
        )
        assert isinstance(model, RegressionModel)

    def test_build_lasso_regression(self):
        model = ModelFactory.build(
            task="regression",
            algorithm="lasso",
        )
        assert isinstance(model, RegressionModel)
        assert model.estimator.__class__.__name__ == "Lasso"

    @pytest.mark.parametrize(
        "algorithm,estimator_cls_name",
        [
            ("svm", "SVC"),
            ("knn", "KNeighborsClassifier"),
            ("decision_tree", "DecisionTreeClassifier"),
            ("extra_trees", "ExtraTreesClassifier"),
            ("adaboost", "AdaBoostClassifier"),
            ("naive_bayes", "GaussianNB"),
        ],
    )
    def test_build_new_classification_algorithms(self, algorithm, estimator_cls_name):
        model = ModelFactory.build(
            task="classification",
            algorithm=algorithm,
        )
        assert isinstance(model, ClassificationModel)
        assert model.estimator.__class__.__name__ == estimator_cls_name

    def test_build_svm_classification_enables_probability(self):
        model = ModelFactory.build(
            task="classification",
            algorithm="svm",
        )
        assert model.estimator.probability is True

    @pytest.mark.parametrize(
        "algorithm,estimator_cls_name",
        [
            ("elastic_net", "ElasticNet"),
            ("gradient_boosting", "GradientBoostingRegressor"),
            ("svm", "SVR"),
            ("knn", "KNeighborsRegressor"),
            ("decision_tree", "DecisionTreeRegressor"),
            ("extra_trees", "ExtraTreesRegressor"),
            ("adaboost", "AdaBoostRegressor"),
        ],
    )
    def test_build_new_regression_algorithms(self, algorithm, estimator_cls_name):
        model = ModelFactory.build(
            task="regression",
            algorithm=algorithm,
        )
        assert isinstance(model, RegressionModel)
        assert model.estimator.__class__.__name__ == estimator_cls_name

    def test_default_params_xgboost_has_expected_keys(self):
        params = ModelFactory.default_params("classification", "xgboost")
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params

    def test_params_override(self):
        model = ModelFactory.build(
            task="classification",
            algorithm="logistic",
            params={"max_iter": 999},
        )
        assert model.estimator.max_iter == 999

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            ModelFactory.build(task="forecasting", algorithm="xgboost")

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown classification algorithm"):
            ModelFactory.build(task="classification", algorithm="alien_net")


# ---------------------------------------------------------------------------
# ClassificationModel — fit, predict, evaluate
# ---------------------------------------------------------------------------


class TestClassificationModel:
    def test_fit_completes(self, numeric_X_train, y_train):
        model = ModelFactory.build(
            task="classification",
            algorithm="logistic",
        )
        model.fit(numeric_X_train, y_train)
        assert model._fit_time is not None

    def test_predict_shape(self, numeric_X_train, numeric_X_oot, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        preds = model.predict(numeric_X_oot)
        assert preds.shape == (len(numeric_X_oot),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, numeric_X_train, numeric_X_oot, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        proba = model.predict_proba(numeric_X_oot)
        assert proba.shape[0] == len(numeric_X_oot)
        assert proba.shape[1] == 2

    def test_evaluate_returns_tidy_df(self, data_split, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)

        # Patch data_split with numeric X
        split = MagicMock()
        split.X_train = numeric_X_train
        split.y_train = y_train
        split.X_val = None
        split.X_oot = data_split.X_oot.select_dtypes(include="number").fillna(0)
        split.y_oot = data_split.y_oot

        eval_df = model.evaluate(split)
        assert "split" in eval_df.columns
        assert "metric" in eval_df.columns
        assert "value" in eval_df.columns
        assert len(eval_df) > 0

    def test_evaluate_includes_test_split(self, numeric_X_train, numeric_X_test, y_train, y_test):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)

        split = MagicMock()
        split.X_train = numeric_X_train
        split.y_train = y_train
        split.X_test = numeric_X_test
        split.y_test = y_test
        split.X_val = None
        split.X_oot = None

        eval_df = model.evaluate(split)
        assert "test" in eval_df["split"].unique()

    def test_evaluate_skips_test_split_when_absent(self, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)

        split = MagicMock()
        split.X_train = numeric_X_train
        split.y_train = y_train
        split.X_test = None
        split.X_val = None
        split.X_oot = None

        eval_df = model.evaluate(split)
        assert "test" not in eval_df["split"].unique()

    def test_roc_auc_better_than_random(self, data_split, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        split = MagicMock()
        split.X_train = numeric_X_train
        split.y_train = y_train
        split.X_val = None
        split.X_oot = None

        eval_df = model.evaluate(split)
        train_auc = eval_df.loc[
            (eval_df["split"] == "train") & (eval_df["metric"] == "roc_auc"), "value"
        ].values
        assert len(train_auc) > 0 and train_auc[0] > 0.5


# ---------------------------------------------------------------------------
# PSI — split_name gate
# ---------------------------------------------------------------------------


class TestClassificationPSI:
    """PSI is emitted only on non-train splits and uses the correct baseline."""

    def _build_split(self, X_train, y_train, X_test, y_test):
        split = MagicMock()
        split.X_train = X_train
        split.y_train = y_train
        split.X_test = X_test
        split.y_test = y_test
        split.X_val = None
        split.X_oot = None
        return split

    def test_psi_absent_from_train_metrics(self, numeric_X_train, numeric_X_test, y_train, y_test):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        split = self._build_split(numeric_X_train, y_train, numeric_X_test, y_test)
        eval_df = model.evaluate(split)
        train_metrics = eval_df.loc[eval_df["split"] == "train", "metric"].tolist()
        assert "psi" not in train_metrics

    def test_psi_present_on_test_split(self, numeric_X_train, numeric_X_test, y_train, y_test):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        split = self._build_split(numeric_X_train, y_train, numeric_X_test, y_test)
        eval_df = model.evaluate(split)
        test_metrics = eval_df.loc[eval_df["split"] == "test", "metric"].tolist()
        assert "psi" in test_metrics

    def test_psi_skipped_when_train_scores_none(
        self, numeric_X_train, numeric_X_test, y_train, y_test
    ):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        model._train_scores = None  # simulate no baseline stored
        split = self._build_split(numeric_X_train, y_train, numeric_X_test, y_test)
        eval_df = model.evaluate(split)
        assert "psi" not in eval_df["metric"].tolist()

    def test_compute_metrics_callable_as_staticmethod_matches_evaluate(
        self, numeric_X_train, numeric_X_test, y_train, y_test
    ):
        """ClassificationModel._compute_metrics is a pure staticmethod — callable
        directly on raw arrays (the exact shape a future monitoring job needs),
        with no fitted estimator involved, and produces the same psi value
        evaluate() would for the same split.
        """
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        split = self._build_split(numeric_X_train, y_train, numeric_X_test, y_test)
        eval_df = model.evaluate(split)
        expected_psi = eval_df.loc[
            (eval_df["split"] == "test") & (eval_df["metric"] == "psi"), "value"
        ].iloc[0]

        y_pred = model.predict(numeric_X_test)
        y_prob = model.predict_proba(numeric_X_test)[:, 1]
        direct = ClassificationModel._compute_metrics(
            y_test.values,
            y_pred,
            y_prob,
            split_name="test",
            train_scores=model._train_scores,
        )
        assert round(direct["psi"], settings.evaluate_round_precision) == expected_psi


# ---------------------------------------------------------------------------
# psi_score
# ---------------------------------------------------------------------------


class TestPsiScore:
    def test_identical_distributions_zero(self):
        rng = np.random.RandomState(0)
        scores = rng.uniform(0, 1, size=1000)
        assert psi_score(scores, scores.copy()) < 0.01

    def test_very_different_distributions_high(self):
        rng = np.random.RandomState(0)
        expected = rng.beta(2, 8, size=1000)  # concentrated near 0
        actual = rng.beta(8, 2, size=1000)  # concentrated near 1
        assert psi_score(expected, actual) > 0.25


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestModelSerialisation:
    def test_save_load_predictions_identical(
        self, numeric_X_train, numeric_X_oot, y_train, tmp_path
    ):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        path = model.save(tmp_path / "model.joblib")
        loaded = ClassificationModel.load(path)
        np.testing.assert_array_equal(
            model.predict(numeric_X_oot),
            loaded.predict(numeric_X_oot),
        )


# ---------------------------------------------------------------------------
# RegressionModel
# ---------------------------------------------------------------------------


class TestRegressionModel:
    @pytest.fixture
    def reg_data(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.randn(200), "f2": rng.randn(200)})
        y = pd.Series(3 * X["f1"] + rng.randn(200) * 0.1, name="target")
        return X, y

    def test_fit_and_predict(self, reg_data):
        X, y = reg_data
        model = ModelFactory.build("regression", "ridge")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_evaluate_contains_standard_metrics(self, reg_data):
        X, y = reg_data
        model = ModelFactory.build("regression", "ridge")
        model.fit(X, y)
        split = MagicMock()
        split.X_train = X
        split.y_train = y
        split.X_test = X.iloc[:50]
        split.y_test = y.iloc[:50]
        split.X_val = None
        split.X_oot = None
        eval_df = model.evaluate(split)
        metrics = eval_df["metric"].tolist()
        for m in ("rmse", "mae", "r2"):
            assert m in metrics

    def test_mape_is_nan_when_all_targets_zero(self, reg_data):
        X, _ = reg_data
        y_zero = pd.Series(np.zeros(len(X)), name="target")
        model = ModelFactory.build("regression", "ridge")
        model.fit(X, y_zero)
        result = model._compute_metrics(y_zero.values, model.predict(X))
        assert np.isnan(result["mape"])

    def test_residual_plot_returns_figure(self, reg_data):
        X, y = reg_data
        model = ModelFactory.build("regression", "ridge")
        model.fit(X, y)
        import plotly.graph_objects as go

        fig = model.residual_plot(X, y)
        assert isinstance(fig, go.Figure)

    def test_residual_plot_type_error_on_bad_input(self, reg_data):
        X, y = reg_data
        model = ModelFactory.build("regression", "ridge")
        model.fit(X, y)
        with pytest.raises(TypeError):
            model.residual_plot(X.values, y)
        with pytest.raises(TypeError):
            model.residual_plot(X, y.values)


# ---------------------------------------------------------------------------
# ClusteringModel
# ---------------------------------------------------------------------------


class TestClusteringModel:
    @pytest.fixture
    def cluster_data(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(
            {
                "f1": np.concatenate([rng.normal(0, 0.3, 100), rng.normal(5, 0.3, 100)]),
                "f2": np.concatenate([rng.normal(0, 0.3, 100), rng.normal(5, 0.3, 100)]),
            }
        )
        return X

    def test_fit_and_predict_shape(self, cluster_data):
        from sklearn.cluster import KMeans

        model = ClusteringModel(KMeans(n_clusters=2, random_state=0, n_init="auto"))
        model.fit(cluster_data, pd.Series(np.zeros(len(cluster_data))))
        labels = model.predict(cluster_data)
        assert labels.shape == (len(cluster_data),)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_evaluate_returns_clustering_metrics(self, cluster_data):
        from sklearn.cluster import KMeans

        model = ClusteringModel(KMeans(n_clusters=2, random_state=0, n_init="auto"))
        model.fit(cluster_data, pd.Series(np.zeros(len(cluster_data))))
        split = MagicMock()
        split.X_train = cluster_data
        eval_df = model.evaluate(split)
        assert "silhouette_score" in eval_df["metric"].tolist()
        assert "davies_bouldin_score" in eval_df["metric"].tolist()

    def test_predict_proba_raises(self, cluster_data):
        from sklearn.cluster import KMeans

        model = ClusteringModel(KMeans(n_clusters=2, random_state=0, n_init="auto"))
        model.fit(cluster_data, pd.Series(np.zeros(len(cluster_data))))
        with pytest.raises(AttributeError):
            model.predict_proba(cluster_data)

    def test_single_cluster_returns_empty_metrics(self, cluster_data):
        from sklearn.cluster import KMeans

        model = ClusteringModel(KMeans(n_clusters=1, random_state=0, n_init="auto"))
        model.fit(cluster_data, pd.Series(np.zeros(len(cluster_data))))
        split = MagicMock()
        split.X_train = cluster_data
        eval_df = model.evaluate(split)
        # silhouette/davies_bouldin undefined for 1 cluster
        assert "silhouette_score" not in eval_df["metric"].tolist()

    def test_predict_falls_back_to_labels_when_no_predict_method(self, cluster_data):
        from sklearn.cluster import DBSCAN

        estimator = DBSCAN(eps=0.5, min_samples=5)
        estimator.fit(cluster_data)
        model = ClusteringModel(estimator)
        labels = model.predict(cluster_data)
        np.testing.assert_array_equal(labels, estimator.labels_)

    def test_elbow_plot_returns_figure(self, cluster_data):
        import plotly.graph_objects as go
        from sklearn.cluster import KMeans

        model = ClusteringModel(KMeans(n_clusters=2, random_state=0, n_init="auto"))
        fig = model.elbow_plot(cluster_data, max_k=4)
        assert isinstance(fig, go.Figure)

    def test_classification_roc_curve_plot_returns_figure(self, numeric_X_train, y_train):
        import plotly.graph_objects as go

        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        fig = model.roc_curve_plot(numeric_X_train, y_train)
        assert isinstance(fig, go.Figure)

    def test_classification_score_distribution_plot_returns_figure(self, numeric_X_train, y_train):
        import plotly.graph_objects as go

        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        fig = model.score_distribution_plot(numeric_X_train, y_train)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# BaseDSCompanionModel — additional coverage (type guards, eval_set, evaluate paths)
# ---------------------------------------------------------------------------


class TestBaseModelGuards:
    """Drive the uncovered lines in models/base.py via the concrete ClassificationModel."""

    def test_fit_type_error_when_X_not_dataframe(self, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        with pytest.raises(TypeError, match="pd.DataFrame"):
            model.fit(numeric_X_train.values, y_train)

    def test_fit_type_error_when_y_not_series(self, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        with pytest.raises(TypeError, match="pd.Series"):
            model.fit(numeric_X_train, y_train.values)

    def test_predict_type_error_when_X_not_dataframe(self, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        with pytest.raises(TypeError, match="pd.DataFrame"):
            model.predict(numeric_X_train.values)

    def test_predict_proba_type_error_when_X_not_dataframe(self, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        with pytest.raises(TypeError, match="pd.DataFrame"):
            model.predict_proba(numeric_X_train.values)

    def test_evaluate_includes_val_split_when_present(self, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        split = MagicMock()
        split.X_train = numeric_X_train
        split.y_train = y_train
        split.X_test = numeric_X_train.iloc[:50]
        split.y_test = y_train.iloc[:50]
        split.X_val = numeric_X_train.iloc[50:100]
        split.y_val = y_train.iloc[50:100]
        split.X_oot = None
        eval_df = model.evaluate(split)
        assert "val" in eval_df["split"].tolist()

    def test_evaluate_includes_oot_split_when_present(
        self, numeric_X_train, numeric_X_oot, y_train, y_oot
    ):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        split = MagicMock()
        split.X_train = numeric_X_train
        split.y_train = y_train
        split.X_test = None
        split.X_val = None
        split.X_oot = numeric_X_oot
        split.y_oot = y_oot
        eval_df = model.evaluate(split)
        assert "oot" in eval_df["split"].tolist()

    def test_fit_with_eval_set_xgboost_path(self, numeric_X_train, y_train):
        """XGBoost accepts eval_set; logistic does not — both paths must not raise."""
        model = ModelFactory.build("classification", "xgboost")
        val = (numeric_X_train.iloc[:50], y_train.iloc[:50])
        model.fit(numeric_X_train, y_train, eval_set=[val])
        assert model._fit_time is not None

    def test_fit_with_eval_set_non_xgboost_falls_back(self, numeric_X_train, y_train):
        """LogisticRegression does not accept eval_set — fallback to plain fit."""
        model = ModelFactory.build("classification", "logistic")
        val = (numeric_X_train.iloc[:50], y_train.iloc[:50])
        model.fit(numeric_X_train, y_train, eval_set=[val])
        assert model._fit_time is not None

    def test_constructor_has_no_mlflow_params(self):
        # Assert on BaseDSCompanionModel.__init__ directly, not ClassificationModel's —
        # ClassificationModel.__init__ is `(self, estimator, **kwargs)` and would
        # never name these params regardless of what BaseDSCompanionModel looks like.
        sig = inspect.signature(BaseDSCompanionModel.__init__)
        assert "mlflow_experiment" not in sig.parameters
        assert "run_name" not in sig.parameters
        assert "log_to_mlflow" not in sig.parameters

    def test_fit_logs_params_at_debug_level(self, caplog):
        # Instantiated directly (not via ModelFactory.build) — ModelFactory.build
        # still unconditionally forwards a `mlflow_experiment` kwarg into the
        # wrapper constructor; stripping that call site is Task 4's job, not
        # this test's concern.
        #
        # Fit params are logged at DEBUG (not INFO) so that tuning/leaderboard
        # loops that fit many candidate models don't bury the per-fit INFO
        # signal ("%s fitted in %.2fs...") under a params dump for every
        # candidate. The separate "Fit time:" line was dropped outright — it
        # duplicated the fit-time value already carried by the "fitted in"
        # INFO line above it.
        from sklearn.linear_model import LogisticRegression

        model = ClassificationModel(LogisticRegression(max_iter=1000))
        X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 0, 1, 0]})
        y = pd.Series([0, 1, 0, 1])
        with caplog.at_level(logging.DEBUG, logger="dscompanion.models.base"):
            model.fit(X, y)
        # Assert on the exact log-line prefix written in base.py's fit() — not
        # a generic "fit"/"params" substring, which the pre-existing
        # "%s fitted in %.2fs..." line would also satisfy on its own.
        messages = [r.message for r in caplog.records]
        assert any(m.startswith("Fit params:") for m in messages)

    def test_save_creates_parent_dirs(self, numeric_X_train, y_train, tmp_path):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        nested_path = tmp_path / "subdir" / "model.joblib"
        saved = model.save(nested_path)
        assert saved.exists()

    def test_get_params_returns_dict(self, numeric_X_train, y_train):
        model = ModelFactory.build("classification", "logistic")
        model.fit(numeric_X_train, y_train)
        params = model._get_params()
        assert isinstance(params, dict)
        # All values must be primitive types
        for v in params.values():
            assert isinstance(v, (int, float, str, bool))
