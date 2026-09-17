"""Tuner: public-facing hyperparameter tuning dispatcher."""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd

from dscompanion.models.base import BaseDSCompanionModel
from dscompanion.tuning.search_spaces import PREDEFINED_PARAMS, SEARCH_SPACES

__all__ = ["Tuner"]


# Estimator class names abbreviate or otherwise diverge from the
# SEARCH_SPACES key prefix (e.g. XGBClassifier -> "xgb", key prefix
# "xgboost"; SVC -> "svc", key prefix "svm") -- expand known abbreviations
# before token-matching below.
_CLASS_NAME_ALIASES = {
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "svc": "svm",
    "svr": "svm",
    "kneighbors": "knn",
    "gaussiannb": "naivebayes",
    "lineardiscriminantanalysis": "lda",
    "quadraticdiscriminantanalysis": "qda",
}


def _infer_space_key(model: BaseDSCompanionModel, backend: str) -> str | None:
    """Guess the search space key from the model's estimator class name.

    Requires every non-task token of a candidate key to match (not just any one
    of them) -- e.g. "tree" from "decision_tree_classification" is also a
    substring of "extratreesclassifier" ("extra" + "trees"), so an any-token
    match would wrongly resolve ExtraTreesClassifier to the decision-tree space.
    """
    cls_name = type(model.estimator).__name__.lower()
    for abbr, full in _CLASS_NAME_ALIASES.items():
        cls_name = cls_name.replace(abbr, full)
    task = type(model).__name__.lower().replace("model", "")  # "classification", etc.
    for key in SEARCH_SPACES.get(backend, {}):
        algo_tokens = [tok for tok in key.split("_") if tok != task]
        if task in key and algo_tokens and all(tok in cls_name for tok in algo_tokens):
            return key
    return None


class Tuner:
    """Dispatch hyperparameter search to the selected backend and return a best-params-fitted model.

    Supports three backends: ``"optuna"`` (TPE with optional pruning),
    ``"hyperopt"`` (TPE via the hyperopt library), and ``"predefined"``
    (directly apply a fixed param dict without search).  The search space can
    be supplied as a raw spec dict, as a string key into the ``SEARCH_SPACES``
    registry, or auto-inferred from the model's estimator class name.
    Side-effects: populates ``best_params_``, ``best_score_``, and
    ``trials_dataframe_`` after ``run()``.

    Args:
        model: ``BaseDSCompanionModel`` instance whose estimator will be tuned.
            The original instance is not mutated; a deep copy receives the
            best parameters.
        backend: Tuning engine to use.  One of ``"optuna"`` (default),
            ``"hyperopt"``, or ``"predefined"``.
        search_space: Hyperparameter specification.  May be a dict mapping
            parameter names to dscompanion spec dicts (keys: ``type``, ``low``,
            ``high``, etc.), a string key into ``SEARCH_SPACES[backend]``,
            or ``None`` to trigger auto-inference from the model type.
        n_trials: Number of search iterations.  Defaults to ``50`` for
            ``optuna``/``hyperopt`` and ``0`` for ``predefined``.
        cv: ``DataSplit`` object providing ``X_train``, ``y_train``, and
            optionally ``X_val`` / ``y_val`` for scoring.  Required for
            ``optuna`` and ``hyperopt`` backends; ignored for ``predefined``.
        metric: Name of the metric returned by ``model.evaluate()`` that the
            search should optimise.  Defaults to ``"roc_auc"``.
        direction: ``"maximize"`` (default) to maximise the metric, or
            ``"minimize"`` to minimise it.

    Attributes:
        best_params_: Dict of parameter name to best value found by the
            search.  Populated after ``run()``.
        best_score_: Objective metric value of the best trial as a float.
            ``float("nan")`` when not available (e.g. ``predefined`` backend).
        trials_dataframe_: DataFrame summarising the top 10 trials with
            columns ``trial_number``, ``params``, and ``metric_value``.
            Empty DataFrame for the ``predefined`` backend.
        optimization_curve_: DataFrame of all completed trials in
            chronological order with columns ``trial_number``,
            ``metric_value``, and ``best_so_far`` (running
            max/min depending on ``direction``).  Populated for the
            ``optuna`` backend only; empty DataFrame otherwise.
    """

    def __init__(
        self,
        model: BaseDSCompanionModel,
        backend: str = "optuna",
        search_space: dict[str, Any] | str | None = None,
        n_trials: int | None = None,
        cv: Any | None = None,
        metric: str = "roc_auc",
        direction: str = "maximize",
    ) -> None:
        self.model = model
        self.backend = backend
        self.search_space = search_space
        self.n_trials = n_trials or (50 if backend != "predefined" else 0)
        self.cv = cv
        self.metric = metric
        self.direction = direction

    def run(self) -> BaseDSCompanionModel:
        """Execute the hyperparameter search and return a new model fitted with the best parameters.

        For the ``predefined`` backend, the fixed parameters from
        ``PREDEFINED_PARAMS`` are applied directly without any search.  For
        ``optuna`` and ``hyperopt`` backends, the resolved search space is
        passed to the respective backend class which runs ``n_trials``
        evaluations.  The best parameters are then applied to a deep copy of
        ``self.model``, which is re-fitted on ``cv.X_train`` / ``cv.y_train``
        before being returned.  Side-effects: sets ``best_params_``,
        ``best_score_``, and ``trials_dataframe_`` on ``self``.

        Args:
            None

        Returns:
            A new ``BaseDSCompanionModel`` instance (deep copy of ``self.model``)
            with best parameters applied and fitted on the training split.

        Raises:
            ValueError: If ``backend`` is not one of ``"optuna"``,
                ``"hyperopt"``, or ``"predefined"``.
            ValueError: If the ``optuna`` or ``hyperopt`` backend is selected
                but ``cv`` is ``None``.
            ValueError: If the search space cannot be inferred and
                ``search_space`` was not provided explicitly.
        """
        if self.backend == "predefined":
            return self._run_predefined()
        if self.cv is None:
            raise ValueError("Tuner requires cv (DataSplit) for optuna/hyperopt backends.")

        resolved_space = self._resolve_space()
        if not resolved_space:
            raise ValueError(
                f"Could not infer search space for {type(self.model).__name__}. "
                "Pass search_space explicitly."
            )

        if self.backend == "optuna":
            from dscompanion.tuning.backends.optuna_backend import OptunaBackend

            be = OptunaBackend(
                model=self.model,
                search_space=resolved_space,
                n_trials=self.n_trials,
                cv=self.cv,
                metric=self.metric,
                direction=self.direction,
            )
        elif self.backend == "hyperopt":
            from dscompanion.tuning.backends.hyperopt_backend import HyperoptBackend

            be = HyperoptBackend(
                model=self.model,
                search_space=resolved_space,
                n_trials=self.n_trials,
                cv=self.cv,
                metric=self.metric,
                direction=self.direction,
            )
        else:
            raise ValueError(
                f"Unknown backend: {self.backend!r}. Choose optuna/hyperopt/predefined."
            )

        self.best_params_ = be.run()
        self.trials_dataframe_ = be.best_trial_summary()
        try:
            best_row = self.trials_dataframe_.iloc[0]
            self.best_score_ = float(best_row["metric_value"])
        except Exception:
            self.best_score_ = float("nan")

        # Build chronological optimization curve for UI charting.
        # be.study_ exists only for the optuna backend; falls back to empty
        # for hyperopt or any backend that doesn't expose study_.
        _empty_oc = pd.DataFrame(columns=["trial_number", "metric_value", "best_so_far"])
        if hasattr(be, "study_"):
            _rows = [
                {"trial_number": t.number, "metric_value": t.value}
                for t in be.study_.trials
                if t.value is not None
            ]
            if _rows:
                _oc = pd.DataFrame(_rows).sort_values("trial_number").reset_index(drop=True)
                _oc["best_so_far"] = (
                    _oc["metric_value"].cummax()
                    if self.direction == "maximize"
                    else _oc["metric_value"].cummin()
                )
                self.optimization_curve_ = _oc
            else:
                self.optimization_curve_ = _empty_oc
        else:
            self.optimization_curve_ = _empty_oc

        return self._build_best_model(self.best_params_)

    @property
    def best_params_(self) -> dict[str, Any]:
        return self._best_params

    @best_params_.setter
    def best_params_(self, value: dict[str, Any]) -> None:
        self._best_params = value

    @property
    def best_score_(self) -> float:
        return self._best_score

    @best_score_.setter
    def best_score_(self, value: float) -> None:
        self._best_score = value

    @property
    def trials_dataframe_(self) -> pd.DataFrame:
        return self._trials_df

    @trials_dataframe_.setter
    def trials_dataframe_(self, value: pd.DataFrame) -> None:
        self._trials_df = value

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_predefined(self) -> BaseDSCompanionModel:
        space_key = self._resolve_predefined_key()
        params = PREDEFINED_PARAMS.get(space_key, {})
        self.best_params_ = params
        self.best_score_ = float("nan")
        self.trials_dataframe_ = pd.DataFrame()
        model = self._build_best_model(params)
        logger.info("Predefined tuning: using %s defaults, fitting model", space_key)
        return model

    def _resolve_space(self) -> dict[str, Any] | None:
        if isinstance(self.search_space, dict):
            return self.search_space
        if isinstance(self.search_space, str):
            return SEARCH_SPACES.get(self.backend, {}).get(self.search_space)
        key = _infer_space_key(self.model, self.backend)
        if key:
            return SEARCH_SPACES.get(self.backend, {}).get(key)
        return None

    def _resolve_predefined_key(self) -> str:
        if isinstance(self.search_space, str):
            return self.search_space
        key = _infer_space_key(self.model, "optuna") or ""
        return key

    def _build_best_model(self, params: dict[str, Any]) -> BaseDSCompanionModel:
        new_model = copy.deepcopy(self.model)
        try:
            valid = new_model.estimator.get_params()
            filtered = {k: v for k, v in params.items() if k in valid}
            new_model.estimator.set_params(**filtered)
        except Exception as exc:
            logger.warning("Could not apply best params: %s", exc)

        if self.cv is not None:
            eval_set = None
            if self.cv.X_val is not None and len(self.cv.X_val) > 0:
                eval_set = [(self.cv.X_val, self.cv.y_val)]
            elif new_model.estimator.get_params().get("early_stopping_rounds") is not None:
                # Some estimators (e.g. XGBoost) raise outright when early_stopping_rounds
                # is set but no eval_set is available — degrade to a full-length fit
                # instead of crashing the final re-fit when cv has no val split.
                new_model.estimator.set_params(early_stopping_rounds=None)
            new_model.fit(self.cv.X_train, self.cv.y_train, eval_set=eval_set)
        return new_model
