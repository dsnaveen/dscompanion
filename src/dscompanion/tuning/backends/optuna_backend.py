"""OptunaBackend: Optuna-based hyperparameter search."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.exceptions import NotFittedError

from dscompanion.config import settings

logger = logging.getLogger(__name__)

__all__ = ["OptunaBackend"]


def _suggest_param(trial, name: str, spec: dict[str, Any]):
    ptype = spec["type"]
    if ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown param type: {ptype!r}")


class OptunaBackend:
    """Run hyperparameter optimisation using Optuna's TPE sampler with optional pruning.

    Creates an Optuna ``Study``, samples hyperparameters from ``search_space``
    using the configured sampler, fits a deep copy of the model on the training
    split for each trial, and evaluates on the validation split (or training
    split when no validation split is available).  Logs each trial's
    parameters and score via stdlib logging.  Side-effects: populates
    ``study_`` after ``run()``.

    Args:
        model: ``BaseDSCompanionModel`` instance whose estimator will be tuned.
            Not mutated; a deep copy is used for each trial evaluation.
        search_space: Dict mapping parameter name (str) to a dscompanion spec dict
            with keys ``type`` (``"int"``, ``"float"``, or ``"categorical"``),
            ``low``, ``high``, and optional flags such as ``log``.
        n_trials: Number of Optuna trials to run.
        cv: ``DataSplit`` providing ``X_train``, ``y_train``, and optionally
            ``X_val`` / ``y_val`` for per-trial scoring.
        metric: Metric key to optimise.  Must appear in the DataFrame returned
            by ``model.evaluate()``.
        direction: ``"maximize"`` (default) to maximise the metric, or
            ``"minimize"`` to minimise it.
        pruner_type: Optuna pruner to use.  One of ``"median"`` (default),
            ``"hyperband"``, or ``"none"`` (``NopPruner``).
        sampler_type: Optuna sampler to use.  One of ``"tpe"`` (default),
            ``"random"``, or ``"cmaes"``.  All are seeded at 42 for
            reproducibility.
        n_jobs: Number of parallel Optuna trials.  Defaults to ``1``
            (sequential).

    Attributes:
        study_: Optuna ``Study`` object populated after ``run()``.  Exposes
            ``best_params``, ``best_value``, and the full ``trials`` list.
    """

    def __init__(
        self,
        model,
        search_space: dict[str, Any],
        n_trials: int,
        cv: Any,
        metric: str,
        direction: str = "maximize",
        pruner_type: str = "median",
        sampler_type: str = "tpe",
        n_jobs: int = 1,
    ) -> None:
        self.model = model
        self.search_space = search_space
        self.n_trials = n_trials
        self.cv = cv
        self.metric = metric
        self.direction = direction
        self.pruner_type = pruner_type
        self.sampler_type = sampler_type
        self.n_jobs = n_jobs

    def run(self) -> dict[str, Any]:
        """Create an Optuna study, run all trials, and return the best hyperparameters found.

        Initialises the sampler and pruner from ``sampler_type`` and
        ``pruner_type``, creates a study with the configured ``direction``,
        and calls ``study_.optimize`` for ``n_trials`` evaluations.  Logs the
        best metric value at INFO level after the study completes.

        Args:
            None

        Returns:
            Dict mapping parameter name (str) to the best value found across
            all completed trials, as provided by ``study_.best_params``.

        Raises:
            ImportError: If the ``optuna`` package is not installed.
        """
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError("optuna is required. pip install optuna") from exc

        sampler = self._build_sampler(optuna)
        pruner = self._build_pruner(optuna)

        self.study_ = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )
        self.study_.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )
        logger.info(
            "Optuna finished — best %s=%.4f in %d trials",
            self.metric,
            self.study_.best_value,
            self.n_trials,
        )
        return self.study_.best_params

    def best_trial_summary(self) -> pd.DataFrame:
        """Return a summary of the top-10 completed trials from the Optuna study.

        Iterates over all trials in ``study_.trials``, skips any whose value
        is ``None`` (failed or pruned before completion), and sorts the
        remainder by ``metric_value`` in the direction specified by
        ``self.direction``.

        Args:
            None

        Returns:
            DataFrame with columns ``trial_number`` (int), ``params`` (dict),
            and ``metric_value`` (float), containing at most 10 rows.
            Returns an empty DataFrame if no trials completed successfully.
        """
        if not hasattr(self, "study_"):
            raise NotFittedError(
                "%s has not been run yet. Call 'run()' first." % type(self).__name__
            )
        rows = [
            {
                "trial_number": t.number,
                "params": t.params,
                "metric_value": t.value,
            }
            for t in self.study_.trials
            if t.value is not None
        ]
        if not rows:
            return pd.DataFrame(columns=["trial_number", "params", "metric_value"])
        df = pd.DataFrame(rows).sort_values(
            "metric_value",
            ascending=(self.direction == "minimize"),
        )
        return df.head(10).reset_index(drop=True)

    def optimization_history_plot(self):
        """Build an Optuna optimisation-history chart showing how the best metric
        evolved across trials.

        Delegates to ``optuna.visualization.plot_optimization_history``.  If
        that call fails for any reason (e.g. the study has no completed
        trials), falls back to an empty ``go.Figure``.  If ``plotly`` is also
        unavailable, returns ``None``.

        Args:
            None

        Returns:
            Plotly ``go.Figure`` showing objective value per trial, or an empty
            ``go.Figure`` when visualisation fails, or ``None`` if Plotly is
            not installed.
        """
        if not hasattr(self, "study_"):
            raise NotFittedError(
                "%s has not been run yet. Call 'run()' first." % type(self).__name__
            )
        try:
            import optuna.visualization as vis

            return vis.plot_optimization_history(self.study_)
        except Exception as exc:
            logger.debug("optimization_history_plot failed: %s", exc)
            try:
                import plotly.graph_objects as go

                return go.Figure()
            except ImportError:
                return None

    def param_importance_plot(self):
        """Build an Optuna parameter-importance chart ranking each hyperparameter by
        its effect on the objective.

        Delegates to ``optuna.visualization.plot_param_importances``.  If that
        call fails (e.g. fewer than two completed trials), falls back to an
        empty ``go.Figure``.  If ``plotly`` is also unavailable, returns
        ``None``.

        Args:
            None

        Returns:
            Plotly ``go.Figure`` showing relative importance of each
            hyperparameter, or an empty ``go.Figure`` when visualisation
            fails, or ``None`` if Plotly is not installed.
        """
        if not hasattr(self, "study_"):
            raise NotFittedError(
                "%s has not been run yet. Call 'run()' first." % type(self).__name__
            )
        try:
            import optuna.visualization as vis

            return vis.plot_param_importances(self.study_)
        except Exception as exc:
            logger.debug("param_importance_plot failed: %s", exc)
            try:
                import plotly.graph_objects as go

                return go.Figure()
            except ImportError:
                return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _objective(self, trial) -> float:
        import copy

        params = {
            name: _suggest_param(trial, name, spec) for name, spec in self.search_space.items()
        }

        candidate = copy.deepcopy(self.model)
        candidate.estimator.set_params(
            **{k: v for k, v in params.items() if k in candidate.estimator.get_params()}
        )
        try:
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
            logger.warning("Trial %d failed: %s", trial.number, exc)
            score = float("nan")

        self._log_trial(trial.number, params, score)
        return score

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

    def _build_sampler(self, optuna):
        if self.sampler_type == "tpe":
            return optuna.samplers.TPESampler(seed=settings.random_state)
        if self.sampler_type == "random":
            return optuna.samplers.RandomSampler(seed=settings.random_state)
        if self.sampler_type == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=settings.random_state)
        raise ValueError(
            "Unknown sampler_type %r — expected 'tpe', 'random', or 'cmaes'." % self.sampler_type
        )

    def _build_pruner(self, optuna):
        if self.pruner_type == "median":
            return optuna.pruners.MedianPruner()
        if self.pruner_type == "hyperband":
            return optuna.pruners.HyperbandPruner()
        if self.pruner_type == "none":
            return optuna.pruners.NopPruner()
        raise ValueError(
            "Unknown pruner_type %r — expected 'median', 'hyperband', or 'none'." % self.pruner_type
        )
