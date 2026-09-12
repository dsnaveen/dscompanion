"""Leaderboard: compare multiple classification algorithms on a single DataSplit."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd

from dscompanion.config import settings
from dscompanion.models.classification import ClassificationModel
from dscompanion.models.factory import ModelFactory

__all__ = ["Leaderboard"]

# Metrics in ClassificationModel's suite where a lower value is better — every
# other metric (roc_auc, gini, ks_statistic, f1, precision, recall) is "higher
# is better", so this is the only direction lookup needed.
_LOWER_IS_BETTER = {"log_loss", "psi"}


class Leaderboard:
    """Trains and ranks multiple classification algorithms on a single DataSplit.

    Inspired by PyCaret's ``compare_models()`` and H2O AutoML's leaderboard,
    adapted to dscompanion's existing conventions: mirrors ``Tuner``'s
    constructor-config + ``run(split)`` shape rather than a flat function,
    and reuses ``BaseDSCompanionModel.evaluate()`` rather than introducing new
    cross-validation machinery — every algorithm is fit once on
    ``split.X_train``/``split.y_train`` and evaluated once on a single
    held-out partition of the supplied ``DataSplit``.

    Per-algorithm failures (e.g. a missing optional dependency, a solver that
    fails to converge) are caught and recorded as a failed row rather than
    aborting the whole comparison, mirroring H2O AutoML's behaviour of
    skipping rather than crashing on one bad candidate.

    Args:
        task (str): Currently only ``"classification"`` is supported; any
            other value raises ``ValueError`` at construction time. Kept as
            an explicit parameter, matching ``ModelFactory.build()``'s
            convention, so regression/clustering support can be added later
            without an API break.
        include (list[str], optional): If supplied, only these algorithm
            names are run — each must be a member of
            ``ModelFactory.SUPPORTED_ALGORITHMS[task]``. Defaults to
            ``None`` (run every algorithm supported for ``task``).
        exclude (list[str], optional): Algorithm names to skip. Ignored when
            ``include`` is supplied. Defaults to ``None``.
        params_overrides (dict, optional): Mapping of algorithm name to a
            hyperparameter override dict, forwarded to
            ``ModelFactory.build(params=...)`` for that algorithm only —
            algorithms not present in this mapping use
            ``ModelFactory``'s built-in defaults. Defaults to ``None``.
        sort_metric (str, optional): Metric column to rank by. Defaults to
            ``None``, which resolves to ``settings.classification_metrics[0]``
            (``"roc_auc"``).
        ascending (bool, optional): Sort direction. Defaults to ``None``,
            which auto-resolves from ``_LOWER_IS_BETTER`` (``True`` for
            ``"log_loss"``/``"psi"``, ``False`` otherwise).
        eval_split (str, optional): Which ``DataSplit`` partition to evaluate
            and rank on — one of ``"train"``, ``"val"``, ``"test"``,
            ``"oot"``. Defaults to ``None``, which resolves to ``"val"`` when
            non-empty, else ``"test"`` (always present on a ``DataSplit``).
            Deliberately never defaults to ``"oot"`` — using a true
            out-of-time holdout for algorithm *selection* would compromise
            its later use as an unbiased final check.

    Attributes:
        leaderboard_ (pd.DataFrame): Set after ``run()``. One row per
            algorithm, columns ``algorithm``, ``status`` (``"ok"`` or
            ``"failed"``), ``fit_time_seconds``, ``error`` (failure reason,
            ``None`` for successful rows), and every metric key returned by
            ``_compute_metrics`` for the resolved ``eval_split`` (``NaN`` for
            failed rows). Sorted by the resolved ``sort_metric``.
        fitted_models_ (dict[str, ClassificationModel]): Set after ``run()``.
            Algorithm name -> fitted model, for every algorithm that trained
            successfully — not just the top-ranked one.
    """

    def __init__(
        self,
        task: str = "classification",
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        params_overrides: dict[str, dict[str, Any]] | None = None,
        sort_metric: str | None = None,
        ascending: bool | None = None,
        eval_split: str | None = None,
    ) -> None:
        if task != "classification":
            raise ValueError(
                "Leaderboard currently only supports task='classification', got %r" % task
            )
        self.task = task
        self.include = include
        self.exclude = exclude
        self.params_overrides = params_overrides
        self.sort_metric = sort_metric
        self.ascending = ascending
        self.eval_split = eval_split
        self.leaderboard_: pd.DataFrame | None = None
        self.fitted_models_: dict[str, ClassificationModel] = {}

    def _resolve_algorithms(self) -> list[str]:
        all_algorithms = ModelFactory.SUPPORTED_ALGORITHMS[self.task]
        if self.include is not None:
            unknown = set(self.include) - set(all_algorithms)
            if unknown:
                raise ValueError(
                    "Unknown algorithm(s) for task %r: %s" % (self.task, sorted(unknown))
                )
            return list(self.include)
        if self.exclude is not None:
            return [a for a in all_algorithms if a not in self.exclude]
        return list(all_algorithms)

    def _resolve_eval_split(self, split: Any) -> str:  # Any: DataSplit — avoids circular import
        if self.eval_split is not None:
            return self.eval_split
        if split.X_val is not None and len(split.X_val) > 0:
            return "val"
        return "test"

    def _resolve_fit_eval_set(
        self, split: Any
    ) -> list[tuple[pd.DataFrame, pd.Series]] | None:  # Any: DataSplit
        """Pick a held-out partition to pass as ``model.fit(eval_set=...)``.

        Algorithms with built-in early stopping (e.g. XGBoost's default
        ``early_stopping_rounds=20``) raise if fit without an eval_set at
        all — independent of which partition the leaderboard ranks on, so
        this always prefers val/test regardless of ``self.eval_split``.
        """
        if split.X_val is not None and len(split.X_val) > 0:
            return [(split.X_val, split.y_val)]
        if split.X_test is not None and len(split.X_test) > 0:
            return [(split.X_test, split.y_test)]
        return None

    def run(self, split: Any) -> pd.DataFrame:  # Any: DataSplit — avoids circular import
        """Fit and evaluate every resolved algorithm, returning a ranked leaderboard.

        Side-effects: populates ``self.leaderboard_`` and
        ``self.fitted_models_``.

        Args:
            split (DataSplit): Object exposing ``X_train``/``y_train`` and the
                partition named by the resolved ``eval_split``.

        Returns:
            pd.DataFrame: Same object stored on ``self.leaderboard_`` — see
            the class docstring for the column schema.
        """
        algorithms = self._resolve_algorithms()
        eval_split = self._resolve_eval_split(split)
        fit_eval_set = self._resolve_fit_eval_set(split)
        sort_metric = (
            self.sort_metric if self.sort_metric is not None else settings.classification_metrics[0]
        )
        ascending = (
            self.ascending if self.ascending is not None else (sort_metric in _LOWER_IS_BETTER)
        )

        rows: list[dict[str, Any]] = []
        self.fitted_models_ = {}
        for algorithm in algorithms:
            params = (self.params_overrides or {}).get(algorithm)
            t0 = time.perf_counter()
            try:
                model = ModelFactory.build(
                    task=self.task,
                    algorithm=algorithm,
                    params=params,
                )
                model.fit(split.X_train, split.y_train, eval_set=fit_eval_set)
                fit_time = time.perf_counter() - t0
                metrics_df = model.evaluate(split)
                split_metrics = metrics_df[metrics_df["split"] == eval_split]
                row: dict[str, Any] = {
                    "algorithm": algorithm,
                    "status": "ok",
                    "fit_time_seconds": round(fit_time, settings.fit_time_round_precision),
                    "error": None,
                }
                row.update(dict(zip(split_metrics["metric"], split_metrics["value"])))
                self.fitted_models_[algorithm] = model
            except (ValueError, RuntimeError, ImportError, TypeError) as exc:
                logger.warning("Leaderboard: algorithm %s failed: %s", algorithm, exc)
                row = {
                    "algorithm": algorithm,
                    "status": "failed",
                    "fit_time_seconds": None,
                    "error": str(exc),
                }
            rows.append(row)

        leaderboard = pd.DataFrame(rows)
        if sort_metric in leaderboard.columns:
            leaderboard = leaderboard.sort_values(
                sort_metric, ascending=ascending, na_position="last"
            )
        leaderboard = leaderboard.reset_index(drop=True)

        self.leaderboard_ = leaderboard
        logger.info(
            "Leaderboard: %d/%d algorithms succeeded (eval_split=%s, sort_metric=%s)",
            len(self.fitted_models_),
            len(algorithms),
            eval_split,
            sort_metric,
        )
        return leaderboard

    def best_algorithm(self) -> str:
        """Return the top-ranked algorithm name from the most recent ``run()`` call.

        Args:
            None

        Returns:
            str: Algorithm name from the first (top-ranked) row of
            ``self.leaderboard_``.

        Raises:
            RuntimeError: If ``run()`` has not been called yet, or if every
                algorithm failed on the last ``run()`` call, so no winner
                is available.
        """
        if self.leaderboard_ is None:
            raise RuntimeError("Leaderboard.run() must be called before best_algorithm().")
        ok_rows = self.leaderboard_[self.leaderboard_["status"] == "ok"]
        if ok_rows.empty:
            raise RuntimeError("No algorithm completed successfully; no best model available.")
        return ok_rows.iloc[0]["algorithm"]

    def best_model(self) -> ClassificationModel:
        """Return the top-ranked fitted model from the most recent ``run()`` call.

        Args:
            None

        Returns:
            ClassificationModel: The fitted model corresponding to
            ``best_algorithm()``.

        Raises:
            RuntimeError: If ``run()`` has not been called yet, or if every
                algorithm failed on the last ``run()`` call, so no fitted
                model is available.
        """
        return self.fitted_models_[self.best_algorithm()]
