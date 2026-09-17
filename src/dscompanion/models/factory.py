"""ModelFactory: public-facing model builder with sensible defaults."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

from dscompanion.config import settings
from dscompanion.models.classification import ClassificationModel
from dscompanion.models.clustering import ClusteringModel
from dscompanion.models.regression import RegressionModel

__all__ = ["ModelFactory"]

_LGBM_CLS = "lgb.LGBMClassifier"
_LGBM_REG = "lgb.LGBMRegressor"


def _import_lgbm(kind: str):
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier if kind == "classifier" else lgb.LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm is required. pip install lightgbm") from exc


class ModelFactory:
    """Centralised builder for dscompanion model wrappers with sensible algorithm defaults.

    Provides a single ``build`` class method that resolves a ``task`` +
    ``algorithm`` combination to the correct estimator class, merges
    caller-supplied ``params`` on top of built-in defaults, and returns an
    initialised ``BaseDSCompanionModel`` subclass instance ready to call ``fit`` on.
    No instance is required; all methods are static or class methods.

    Example::

        model = ModelFactory.build(
            task="classification",
            algorithm="xgboost",
        )
        model.fit(X_train, y_train)
    """

    # Single source of truth for which algorithm names each task accepts —
    # must stay in sync with the `if algorithm ==` dispatch in
    # `_build_estimator`. Consumed by reporting code (e.g. PipelineRunner's
    # Configuration Summary) so the "choices" shown for `model.algorithm`
    # are task-specific rather than the full cross-task union.
    SUPPORTED_ALGORITHMS: dict[str, list[str]] = {
        "classification": [
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
        "regression": [
            "xgboost",
            "lightgbm",
            "linear",
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
        "clustering": ["kmeans", "dbscan", "hierarchical"],
    }

    _DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
        "classification": {
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "min_child_weight": 5,
                "scale_pos_weight": 1,
                "eval_metric": "auc",
                "early_stopping_rounds": 20,
                "random_state": settings.random_state,
                "n_jobs": -1,
            },
            "logistic": {
                "max_iter": 2000,
                "random_state": settings.random_state,
                "solver": "saga",
            },
            "random_forest": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 20,
                "random_state": settings.random_state,
                "n_jobs": -1,
            },
            "gradient_boosting": {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "random_state": settings.random_state,
            },
            "svm": {
                "C": 1.0,
                "kernel": "rbf",
                "probability": True,
                "random_state": settings.random_state,
            },
            "knn": {
                "n_neighbors": 5,
                "n_jobs": -1,
            },
            "decision_tree": {
                "max_depth": 6,
                "min_samples_leaf": 20,
                "random_state": settings.random_state,
            },
            "extra_trees": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 20,
                "random_state": settings.random_state,
                "n_jobs": -1,
            },
            "adaboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": settings.random_state,
            },
            "naive_bayes": {},
            "lda": {
                "solver": "lsqr",
                "shrinkage": "auto",
            },
            "qda": {"reg_param": 0.01},
            "mlp": {
                "hidden_layer_sizes": (100,),
                "max_iter": 500,
                "early_stopping": True,
                "random_state": settings.random_state,
            },
        },
        "regression": {
            "linear": {},
            "ridge": {"alpha": 1.0},
            "lasso": {"alpha": 1.0, "random_state": settings.random_state},
            "elastic_net": {
                "alpha": 1.0,
                "l1_ratio": 0.5,
                "random_state": settings.random_state,
            },
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": settings.random_state,
                "n_jobs": -1,
            },
            "random_forest": {
                "n_estimators": 200,
                "min_samples_leaf": 20,
                "random_state": settings.random_state,
                "n_jobs": -1,
            },
            "gradient_boosting": {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "random_state": settings.random_state,
            },
            "svm": {
                "C": 1.0,
                "kernel": "rbf",
            },
            "knn": {
                "n_neighbors": 5,
                "n_jobs": -1,
            },
            "decision_tree": {
                "max_depth": 6,
                "min_samples_leaf": 20,
                "random_state": settings.random_state,
            },
            "extra_trees": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 20,
                "random_state": settings.random_state,
                "n_jobs": -1,
            },
            "adaboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": settings.random_state,
            },
        },
        "clustering": {
            "kmeans": {
                "n_clusters": 8,
                "random_state": settings.random_state,
                "n_init": "auto",
            },
            "dbscan": {"eps": 0.5, "min_samples": 5},
            "hierarchical": {"n_clusters": 5, "linkage": "ward"},
        },
    }

    @staticmethod
    def build(
        task: str,
        algorithm: str = "xgboost",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ClassificationModel | RegressionModel | ClusteringModel":
        """Build a fully configured dscompanion model wrapper for the requested task and algorithm.

        Retrieves built-in parameter defaults for the ``task`` + ``algorithm``
        combination, merges any caller-supplied ``params`` on top (caller
        values win), instantiates the underlying estimator via
        ``_build_estimator``, and wraps it in the appropriate
        ``BaseDSCompanionModel`` subclass.  Parameters not accepted by the target
        estimator's ``__init__`` are stripped automatically.  No side-effects
        beyond object construction.

        Args:
            task (str): Type of ML task.  Must be one of ``"classification"``,
                ``"regression"``, or ``"clustering"`` (case-insensitive).
            algorithm (str): Algorithm identifier within the task group.
                Supported values per task:

                - classification: ``"xgboost"``, ``"logistic"``,
                  ``"random_forest"``, ``"gradient_boosting"``, ``"lightgbm"``,
                  ``"svm"``, ``"knn"``, ``"decision_tree"``, ``"extra_trees"``,
                  ``"adaboost"``, ``"naive_bayes"``, ``"lda"``, ``"qda"``,
                  ``"mlp"``
                - regression: ``"xgboost"``, ``"linear"``, ``"ridge"``,
                  ``"lasso"``, ``"elastic_net"``, ``"random_forest"``,
                  ``"gradient_boosting"``, ``"lightgbm"``, ``"svm"``,
                  ``"knn"``, ``"decision_tree"``, ``"extra_trees"``,
                  ``"adaboost"``
                - clustering: ``"kmeans"``, ``"dbscan"``, ``"hierarchical"``

                Defaults to ``"xgboost"``.
            params (dict, optional): Hyper-parameter overrides merged on top
                of the built-in defaults.  Keys not recognised by the
                estimator are silently dropped.  Defaults to ``None`` (use
                built-in defaults only).
            **kwargs: Additional keyword arguments forwarded verbatim to the
                wrapper constructor (e.g. ``feature_names``).

        Returns:
            BaseDSCompanionModel: An instance of ``ClassificationModel``,
            ``RegressionModel``, or ``ClusteringModel`` depending on ``task``,
            with the estimator already initialised but not yet fitted.

        Raises:
            ValueError: If ``task`` is not one of the three supported values,
                or if ``algorithm`` is not recognised within the given task.
            ImportError: If ``xgboost`` or ``lightgbm`` is requested but not
                installed.
        """
        task = task.lower()
        algorithm = algorithm.lower()

        defaults = ModelFactory.default_params(task, algorithm)
        merged = {**defaults, **(params or {})}

        estimator = ModelFactory._build_estimator(task, algorithm, merged)
        wrapper_cls = {
            "classification": ClassificationModel,
            "regression": RegressionModel,
            "clustering": ClusteringModel,
        }.get(task)
        if wrapper_cls is None:
            raise ValueError(
                f"Unknown task: {task!r}. Choose classification/regression/clustering."
            )

        return wrapper_cls(
            estimator=estimator,
            **kwargs,
        )

    @staticmethod
    def default_params(task: str, algorithm: str) -> dict[str, Any]:
        """Return a copy of the built-in default parameters for a task and algorithm pair.

        Looks up the internal ``_DEFAULTS`` table for the given combination and
        returns a shallow copy so callers can mutate it freely.  No side-effects.

        Args:
            task (str): One of ``"classification"``, ``"regression"``, or
                ``"clustering"``.
            algorithm (str): Algorithm identifier within the task group (e.g.
                ``"xgboost"``, ``"logistic"``).

        Returns:
            dict: Shallow copy of the default hyper-parameter dict for the
            requested combination.  Returns an empty dict ``{}`` when the
            ``task`` or ``algorithm`` is not found in the defaults table.
        """
        return dict(ModelFactory._DEFAULTS.get(task, {}).get(algorithm, {}))

    @staticmethod
    def _build_estimator(
        task: str, algorithm: str, params: dict[str, Any]
    ) -> Any:  # Any: sklearn-compatible estimator
        from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis,
            QuadraticDiscriminantAnalysis,
        )
        from sklearn.ensemble import (
            AdaBoostClassifier,
            AdaBoostRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.linear_model import (
            ElasticNet,
            Lasso,
            LinearRegression,
            LogisticRegression,
            Ridge,
        )
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        # Strip params that the estimator doesn't understand (e.g. early_stopping_rounds
        # is XGBoost-only; LogisticRegression chokes on it)
        def _safe_build(cls, params):
            import inspect

            sig = inspect.signature(cls.__init__)
            valid = set(sig.parameters) - {"self"}
            filtered = {k: v for k, v in params.items() if k in valid or "kwargs" in str(sig)}
            return cls(**filtered)

        if task == "classification":
            if algorithm == "logistic":
                return _safe_build(LogisticRegression, params)
            if algorithm == "random_forest":
                return _safe_build(RandomForestClassifier, params)
            if algorithm == "gradient_boosting":
                return _safe_build(GradientBoostingClassifier, params)
            if algorithm == "svm":
                return _safe_build(SVC, params)
            if algorithm == "knn":
                return _safe_build(KNeighborsClassifier, params)
            if algorithm == "decision_tree":
                return _safe_build(DecisionTreeClassifier, params)
            if algorithm == "extra_trees":
                return _safe_build(ExtraTreesClassifier, params)
            if algorithm == "adaboost":
                return _safe_build(AdaBoostClassifier, params)
            if algorithm == "naive_bayes":
                return _safe_build(GaussianNB, params)
            if algorithm == "lda":
                return _safe_build(LinearDiscriminantAnalysis, params)
            if algorithm == "qda":
                return _safe_build(QuadraticDiscriminantAnalysis, params)
            if algorithm == "mlp":
                return _safe_build(MLPClassifier, params)
            if algorithm == "lightgbm":
                return _import_lgbm("classifier")(**params)
            if algorithm == "xgboost":
                from xgboost import XGBClassifier

                return XGBClassifier(**params)
            raise ValueError(f"Unknown classification algorithm: {algorithm!r}")

        if task == "regression":
            if algorithm == "linear":
                return LinearRegression()
            if algorithm == "ridge":
                return _safe_build(Ridge, params)
            if algorithm == "lasso":
                return _safe_build(Lasso, params)
            if algorithm == "elastic_net":
                return _safe_build(ElasticNet, params)
            if algorithm == "random_forest":
                return _safe_build(RandomForestRegressor, params)
            if algorithm == "gradient_boosting":
                return _safe_build(GradientBoostingRegressor, params)
            if algorithm == "svm":
                return _safe_build(SVR, params)
            if algorithm == "knn":
                return _safe_build(KNeighborsRegressor, params)
            if algorithm == "decision_tree":
                return _safe_build(DecisionTreeRegressor, params)
            if algorithm == "extra_trees":
                return _safe_build(ExtraTreesRegressor, params)
            if algorithm == "adaboost":
                return _safe_build(AdaBoostRegressor, params)
            if algorithm == "lightgbm":
                return _import_lgbm("regressor")(**params)
            if algorithm == "xgboost":
                from xgboost import XGBRegressor

                return XGBRegressor(**params)
            raise ValueError(f"Unknown regression algorithm: {algorithm!r}")

        if task == "clustering":
            if algorithm == "kmeans":
                return _safe_build(KMeans, params)
            if algorithm == "dbscan":
                return _safe_build(DBSCAN, params)
            if algorithm == "hierarchical":
                return _safe_build(AgglomerativeClustering, params)
            raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")

        raise ValueError(f"Unknown task: {task!r}")
