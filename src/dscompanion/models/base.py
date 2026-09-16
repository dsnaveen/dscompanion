"""BaseDSCompanionModel: abstract base for all dscompanion model wrappers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd

from dscompanion.config import settings

__all__ = ["BaseDSCompanionModel"]

_EVALUATE_COLS = ["split", "metric", "value"]


class BaseDSCompanionModel(ABC):
    """Abstract base class that every dscompanion model wrapper must subclass.

    Provides a uniform fit/predict/predict_proba interface, cross-split metric
    evaluation, and joblib-based serialisation.  Subclasses must implement
    ``_compute_metrics``.  No side-effects occur at construction time.

    Args:
        estimator (Any): An instantiated sklearn-compatible estimator (e.g.
            ``XGBClassifier``, ``RandomForestRegressor``).
        feature_names (list of str, optional): Ordered list of expected feature
            column names.  If ``None`` at fit time, it is inferred from
            ``X.columns``.
    """

    def __init__(
        self,
        estimator: Any,  # Any sklearn-compatible estimator with fit/predict interface
        feature_names: list[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.feature_names = feature_names
        self._fit_time: float | None = None

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        split_name: str = "",
    ) -> dict[str, float]:
        """Compute task-specific evaluation metrics and return them as a flat dict.

        Subclasses must implement this as a ``@staticmethod`` — it calculates
        whichever metrics are appropriate for the task (e.g. AUC for
        classification, RMSE for regression) purely from its arguments, with
        no dependency on a fitted estimator instance.  This lets callers
        outside of ``evaluate()`` (e.g. a monitoring job re-measuring
        performance on historical predictions) reuse the exact same metric
        formulas on raw arrays.  A concrete subclass's override may accept
        additional task-specific keyword arguments (e.g. ``train_scores`` —
        see ``ClassificationModel``).  No side-effects; only reads its
        arguments and returns a new dict.

        Args:
            y_true (np.ndarray): Ground-truth labels or continuous targets,
                shape ``(n_samples,)``.
            y_pred (np.ndarray): Hard predictions produced by the estimator,
                shape ``(n_samples,)``.
            y_prob (np.ndarray, optional): Predicted positive-class
                probabilities, shape ``(n_samples,)``.  Pass ``None`` when the
                estimator does not support probability output.

        Returns:
            dict: Mapping of metric name (str) to scalar float value.  Returns
            an empty dict if no metrics could be computed.
        """

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list | None = None,
    ) -> "BaseDSCompanionModel":
        """Fit the underlying estimator on the supplied training data.

        Side-effects: mutates ``self.estimator`` (training), sets
        ``self.feature_names`` from ``X.columns`` when not previously specified,
        records ``self._fit_time`` in seconds, and logs estimator parameters
        plus fit duration via stdlib logging.

        Args:
            X (pd.DataFrame): Training feature matrix, shape
                ``(n_samples, n_features)``.
            y (pd.Series): Training target vector, shape ``(n_samples,)``.
            eval_set (list, optional): List of ``(X_eval, y_eval)`` tuples
                forwarded to XGBoost-compatible estimators for early stopping.
                Ignored silently when the underlying estimator does not accept
                this argument.  Defaults to ``None``.

        Returns:
            BaseDSCompanionModel: ``self``, allowing method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame`` or ``y`` is not a
                ``pd.Series``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pd.Series, got %s" % type(y).__name__)

        self.feature_names = (
            self.feature_names if self.feature_names is not None else list(X.columns)
        )

        t0 = time.perf_counter()
        if eval_set is not None:
            try:
                self.estimator.fit(X, y, eval_set=eval_set, verbose=False)
            except TypeError:
                self.estimator.fit(X, y)
        else:
            self.estimator.fit(X, y)
        self._fit_time = time.perf_counter() - t0

        logger.info(
            "%s fitted in %.2fs on %d rows x %d cols",
            type(self).__name__,
            self._fit_time,
            X.shape[0],
            X.shape[1],
        )

        logger.debug("Fit params: %s", self._get_params())

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate hard predictions (labels or values) for the input features.

        Delegates directly to the underlying estimator's ``predict`` method
        with no additional post-processing.  No side-effects.

        Args:
            X (pd.DataFrame): Feature matrix, shape ``(n_samples, n_features)``.
                Column order must match the order seen during ``fit``.

        Returns:
            np.ndarray: Predicted labels (classification) or continuous values
            (regression), shape ``(n_samples,)``.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class membership probabilities for each sample.

        Delegates directly to the underlying estimator's ``predict_proba``
        method.  No side-effects.

        Args:
            X (pd.DataFrame): Feature matrix, shape ``(n_samples, n_features)``.
                Column order must match the order seen during ``fit``.

        Returns:
            np.ndarray: Probability matrix of shape ``(n_samples, n_classes)``
            where each row sums to 1.0.  For binary classifiers, column index 1
            contains the positive-class probability.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            AttributeError: If the underlying estimator does not implement
                ``predict_proba`` (e.g. SVMs without probability calibration).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        return self.estimator.predict_proba(X)

    def evaluate(self, split: Any) -> pd.DataFrame:  # Any: DataSplit — avoids circular import
        """Compute task-specific metrics across all available data splits.

        Iterates over ``train``, ``val`` (if present), and ``oot`` (if present)
        partitions of the supplied ``DataSplit``, calls ``predict`` and
        optionally ``predict_proba`` on each, then delegates to
        ``_compute_metrics``.  No side-effects beyond the underlying predict
        calls.

        Args:
            split (DataSplit): Object exposing ``X_train``, ``y_train``, and
                ``X_test`` / ``y_test`` (always present), plus optionally
                ``X_val`` / ``y_val`` / ``X_oot`` / ``y_oot``.  Test, validation,
                and OOT splits are included in output only when the
                corresponding ``X_*`` attribute is not ``None`` and non-empty.

        Returns:
            pd.DataFrame: Tidy frame with columns ``split`` (str — one of
            ``"train"``, ``"test"``, ``"val"``, ``"oot"``), ``metric`` (str),
            and ``value`` (float, rounded to
            ``settings.evaluate_round_precision`` decimal places).
            Returns an empty DataFrame with those columns if no metrics are
            produced by ``_compute_metrics``.
        """
        rows = []
        split_map: dict[str, tuple[pd.DataFrame, pd.Series]] = {
            "train": (split.X_train, split.y_train)
        }
        if split.X_test is not None and len(split.X_test) > 0:
            split_map["test"] = (split.X_test, split.y_test)
        if split.X_val is not None and len(split.X_val) > 0:
            split_map["val"] = (split.X_val, split.y_val)
        if split.X_oot is not None and len(split.X_oot) > 0:
            split_map["oot"] = (split.X_oot, split.y_oot)

        for split_name, (X, y) in split_map.items():
            y_pred = self.predict(X)
            y_prob: np.ndarray | None = None
            if hasattr(self.estimator, "predict_proba"):
                y_prob = self.predict_proba(X)[:, 1]
            extra_kwargs = (
                {"train_scores": self._train_scores} if hasattr(self, "_train_scores") else {}
            )
            metrics = self._compute_metrics(
                y.values, y_pred, y_prob, split_name=split_name, **extra_kwargs
            )
            for metric, value in metrics.items():
                rows.append(
                    {
                        "split": split_name,
                        "metric": metric,
                        "value": round(value, settings.evaluate_round_precision),
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_EVALUATE_COLS)

    def save(self, path: str | Path) -> Path:
        """Serialise the entire model wrapper (estimator + metadata) to a joblib file.

        Creates any missing parent directories before writing.  Side-effect:
        writes a file to disk and emits an INFO log message with the resolved
        path.

        Args:
            path (str or Path): Destination file path; conventionally ends with
                ``.joblib``.  Parent directories are created automatically if
                they do not exist.

        Returns:
            pathlib.Path: Resolved absolute path of the file that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseDSCompanionModel":
        """Deserialise a model wrapper previously saved with ``save``.

        Reads the joblib file from disk and emits an INFO log message on
        success.  No other side-effects.

        Args:
            path (str or Path): Path to an existing ``.joblib`` file produced
                by ``BaseDSCompanionModel.save``.

        Returns:
            BaseDSCompanionModel: The deserialised model wrapper instance.  The
            concrete subclass type (e.g. ``ClassificationModel``) is preserved
            as stored in the file.
        """
        obj = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_params(self) -> dict[str, Any]:
        try:
            params = self.estimator.get_params()
            return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
        except AttributeError:
            return {}
