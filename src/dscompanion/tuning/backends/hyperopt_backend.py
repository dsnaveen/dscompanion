"""HyperoptBackend: Hyperopt-based hyperparameter search."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["HyperoptBackend"]


def _build_hp_space(space_spec: dict[str, Any]):
    """Convert an dscompanion parameter spec dict into a Hyperopt ``hp.*`` expression dict.

    Iterates over each parameter in ``space_spec`` and maps the dscompanion
    ``type`` field to the corresponding ``hyperopt.hp`` distribution.  The
    ``"loguniform"`` type applies ``math.log`` to the bounds before calling
    ``hp.loguniform``.  The ``"float"`` type with ``log=True`` also uses
    ``hp.loguniform``; without the flag it uses ``hp.uniform``.

    Args:
        space_spec: Dict mapping parameter name (str) to a dscompanion spec dict.
            Supported ``type`` values and their required keys:
            - ``"uniform"`` — ``low`` (float), ``high`` (float)
            - ``"loguniform"`` — ``low`` (float, > 0), ``high`` (float, > 0)
            - ``"quniform"`` — ``low`` (float), ``high`` (float), optional ``q`` (float, default 1)
            - ``"int"`` — ``low`` (int), ``high`` (int)
            - ``"float"`` — ``low`` (float), ``high`` (float), optional ``log`` (bool)
            - ``"categorical"`` — ``choices`` (list)

    Returns:
        Dict mapping each parameter name (str) to the corresponding
        ``hyperopt.hp.*`` expression object, ready to pass to
        ``hyperopt.fmin``.

    Raises:
        ImportError: If the ``hyperopt`` package is not installed.
        ValueError: If an unrecognised ``type`` value is encountered in
            ``space_spec``.
    """
    try:
        from hyperopt import hp
        from hyperopt.pyll import scope
    except ImportError as exc:
        raise ImportError(
            "hyperopt is not available in this environment. "
            "Use OptunaBackend instead — optuna==3.4.0 is approved."
        ) from exc

    hp_space = {}
    for name, spec in space_spec.items():
        ptype = spec["type"]
        if ptype == "uniform":
            hp_space[name] = hp.uniform(name, spec["low"], spec["high"])
        elif ptype == "loguniform":
            hp_space[name] = hp.loguniform(name, math.log(spec["low"]), math.log(spec["high"]))
        elif ptype == "quniform":
            # hp.quniform always returns a float (e.g. 25.0) even though every current
            # "quniform" usage represents an integer hyperparameter (n_neighbors, max_depth,
            # etc.) -- scikit-learn 1.9's strict param validation rejects a float there
            # ("must be an int ... Got 25.0 instead"), failing every trial. scope.int()
            # casts inside the pyll graph so the resolved value is a real int.
            hp_space[name] = scope.int(
                hp.quniform(name, spec["low"], spec["high"], spec.get("q", 1))
            )
        elif ptype == "int":
            hp_space[name] = hp.randint(name, spec["high"] - spec["low"]) + spec["low"]
        elif ptype == "float":
            if spec.get("log", False):
                hp_space[name] = hp.loguniform(name, math.log(spec["low"]), math.log(spec["high"]))
            else:
                hp_space[name] = hp.uniform(name, spec["low"], spec["high"])
        elif ptype == "categorical":
            hp_space[name] = hp.choice(name, spec["choices"])
        else:
            raise ValueError(f"Unknown hyperopt param type: {ptype!r}")
    return hp_space


class HyperoptBackend:
    """Run hyperparameter optimisation using Hyperopt's Tree-structured Parzen Estimator (TPE).

    Converts the dscompanion parameter spec into native ``hyperopt.hp.*``
    expressions, then calls ``hyperopt.fmin`` for up to ``n_trials``
    evaluations.  Each trial fits a deep copy of the model on the training
    split and evaluates it on the validation split (or training split when no
    validation split is available).  Logs each trial's parameters and score
    via stdlib logging.  Side-effects: populates ``trials_`` after ``run()``.

    Args:
        model: ``BaseDSCompanionModel`` instance whose estimator will be tuned.
            Not mutated; a deep copy is used for each trial.
        search_space: Dict mapping parameter name to a dscompanion spec dict.
            Supported types: ``"uniform"``, ``"loguniform"``, ``"quniform"``,
            ``"int"``, ``"float"`` (with optional ``log`` flag), ``"categorical"``.
        n_trials: Maximum number of Hyperopt evaluations.
        cv: ``DataSplit`` providing ``X_train``, ``y_train``, and optionally
            ``X_val`` / ``y_val`` for per-trial scoring.
        metric: Metric key to optimise.  Must appear in the DataFrame returned
            by ``model.evaluate()``.
        direction: ``"maximize"`` (default) to maximise the metric, or
            ``"minimize"`` to minimise it.  Internally this is converted to a
            minimisation problem via sign flip when ``"maximize"`` is used.

    Attributes:
        trials_: Hyperopt ``Trials`` object holding the full history of every
            completed and failed trial.  Available after ``run()``.
    """

    def __init__(
        self,
        model: Any,
        search_space: dict[str, Any],
        n_trials: int,
        cv: Any,
        metric: str,
        direction: str = "maximize",
    ) -> None:
        self.model = model
        self.search_space = search_space
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.direction = direction

    def run(self) -> dict[str, Any]:
        """Execute Hyperopt TPE search for ``n_trials`` evaluations and return the best parameters.

        Builds the ``hp.*`` search space from ``self.search_space``, runs
        ``hyperopt.fmin`` with TPE, then extracts the best parameter values
        from ``self.trials_``.  Logs best score at INFO level.

        Args:
            None

        Returns:
            Dict mapping parameter name (str) to best value found across all
            trials.  List-valued Hyperopt results are unwrapped to scalars.

        Raises:
            ImportError: If the ``hyperopt`` package is not installed.
        """
        try:
            from hyperopt import Trials, fmin, space_eval, tpe
        except ImportError as exc:
            raise ImportError(
                "hyperopt is not available in this environment. "
                "Use OptunaBackend instead — optuna==3.4.0 is approved."
            ) from exc

        self.trials_ = Trials()
        self._trial_idx = [0]

        hp_space = _build_hp_space(self.search_space)
        self._hp_space = hp_space

        def objective(params):
            return self._objective(params)

        fmin(
            fn=objective,
            space=hp_space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=self.trials_,
            verbose=False,
        )

        # misc["vals"] holds hyperopt's raw internal representation -- for a
        # "categorical" param (hp.choice) that's the *index* into the choices
        # list, not the resolved value (e.g. 0 instead of "eigen"). space_eval
        # resolves indices back to their real values; without it, a categorical
        # best param is silently the wrong type and fails when passed to the
        # estimator (or worse, silently selects the wrong choice for an
        # estimator that doesn't validate its params strictly).
        best_idx = self.trials_.best_trial["tid"]
        raw_vals = {k: v[0] for k, v in self.trials_.trials[best_idx]["misc"]["vals"].items() if v}
        best_params = space_eval(hp_space, raw_vals)

        best_loss = self.trials_.best_trial["result"]["loss"]
        best_score = -best_loss if self.direction == "maximize" else best_loss
        logger.info(
            "Hyperopt finished — best %s=%.4f in %d trials",
            self.metric,
            best_score,
            self.n_trials,
        )
        return best_params

    def best_trial_summary(self) -> pd.DataFrame:
        """Return a summary of the top-10 trials from the completed Hyperopt search.

        Iterates over all trials stored in ``self.trials_``, converts the raw
        Hyperopt loss to the original metric direction, and sorts by metric
        value descending (maximise) or ascending (minimise).

        Args:
            None

        Returns:
            DataFrame with columns ``trial_number`` (int), ``params`` (dict),
            and ``metric_value`` (float), containing the top 10 trials.
            Returns an empty DataFrame if no trials completed successfully.
        """
        from hyperopt import space_eval

        rows = []
        for t in self.trials_.trials:
            loss = t["result"].get("loss")
            if loss is None:
                continue
            score = -loss if self.direction == "maximize" else loss
            raw_vals = {k: v[0] for k, v in t["misc"]["vals"].items() if v}
            params = space_eval(self._hp_space, raw_vals) if raw_vals else {}
            rows.append({"trial_number": t["tid"], "params": params, "metric_value": score})
        df = pd.DataFrame(rows).sort_values(
            "metric_value", ascending=(self.direction == "minimize")
        )
        return df.head(10).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _objective(self, params: dict[str, Any]) -> dict[str, Any]:
        import copy

        from hyperopt import STATUS_FAIL, STATUS_OK

        trial_num = self._trial_idx[0]
        self._trial_idx[0] += 1

        candidate = copy.deepcopy(self.model)
        try:
            candidate.estimator.set_params(
                **{k: v for k, v in params.items() if k in candidate.estimator.get_params()}
            )
            eval_set = None
            if self.cv.X_val is not None and len(self.cv.X_val) > 0:
                eval_set = [(self.cv.X_val, self.cv.y_val)]
            elif candidate.estimator.get_params().get("early_stopping_rounds") is not None:
                # Some estimators (e.g. XGBoost) raise outright when early_stopping_rounds
                # is set but no eval_set is available — degrade to a full-length fit
                # instead of failing every trial when the caller's cv has no val split.
                candidate.estimator.set_params(early_stopping_rounds=None)
            candidate.fit(self.cv.X_train, self.cv.y_train, eval_set=eval_set)
            eval_df = candidate.evaluate(self.cv)
            val_split = "val" if (self.cv.X_val is not None and len(self.cv.X_val) > 0) else "train"
            row = eval_df[(eval_df["split"] == val_split) & (eval_df["metric"] == self.metric)]
            score = float(row["value"].iloc[0]) if len(row) > 0 else float("nan")
        except Exception as exc:
            logger.warning("Hyperopt trial %d failed: %s", trial_num, exc)
            return {"status": STATUS_FAIL, "loss": float("inf")}

        loss = -score if self.direction == "maximize" else score
        self._log_trial(trial_num, params, score)
        return {"status": STATUS_OK, "loss": loss}

    def _log_trial(self, trial_num: int, params: dict, score: float) -> None:
        """Log one completed tuning trial's parameters and score.

        Args:
            trial_num (int): Zero-based trial index.
            params (dict): Hyperparameter values tried this trial.
            score (float): The metric value achieved, or NaN if the trial
                failed to produce one.

        Returns:
            None: Always returns ``None``. Never raises.
        """
        logger.info("Trial %d params=%s score=%s", trial_num, params, score)
