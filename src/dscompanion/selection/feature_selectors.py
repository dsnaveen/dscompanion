"""Feature selectors: Constant, Cardinality, Correlation, IV, SHAP, RFE."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dscompanion.config import settings

__all__ = [
    "BaseSelector",
    "ConstantSelector",
    "NullRateSelector",
    "CardinalitySelector",
    "CorrelationSelector",
    "IVSelector",
    "SHAPSelector",
    "RFESelector",
]


class BaseSelector(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for all dscompanion feature selectors.

    Defines the shared fit/transform/introspection contract.  Concrete
    subclasses must implement ``fit`` and populate ``removed_features_``
    and ``selected_features_`` before returning.

    Attributes:
        removed_features_: Dict mapping feature name to a human-readable
            string explaining why that feature was removed.
        selected_features_: List of feature names retained after ``fit``.
    """

    removed_features_: dict[str, str]
    selected_features_: list[str]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaseSelector":
        """Analyse ``X`` and populate ``removed_features_`` and ``selected_features_``.

        Args:
            X: Training feature DataFrame.  Shape ``(n_samples, n_features)``.
            y: Optional target series.  Required by some subclasses (e.g.
                ``IVSelector``, ``RFESelector``); ignored by others.

        Returns:
            The fitted selector instance (``self``), enabling method chaining.
        """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` restricted to the features retained by this selector.

        Only columns that appear in both ``selected_features_`` and ``X.columns``
        are kept, so the method is safe to call on DataFrames that already lack
        some of the original columns.

        Args:
            X: Feature DataFrame to filter.  Shape ``(n_samples, n_features)``.

        Returns:
            DataFrame containing only the selected columns, preserving the
            original row order and index.  Returns an empty DataFrame (zero
            columns) if no selected feature is present in ``X``.
        """
        cols = [c for c in self.selected_features_ if c in X.columns]
        return X[cols]

    def get_support(self) -> list[str]:
        """Return the names of all features retained after fitting.

        Args:
            None

        Returns:
            List of feature name strings.  Returns an empty list if no features
            were retained.
        """
        return list(self.selected_features_)

    def get_removed(self) -> pd.DataFrame:
        """Return a DataFrame describing every feature removed during ``fit``.

        Args:
            None

        Returns:
            DataFrame with columns ``feature`` (str) and ``reason`` (str),
            one row per removed feature.  Returns an empty DataFrame with those
            two columns if no features were removed.
        """
        return pd.DataFrame(
            [{"feature": k, "reason": v} for k, v in self.removed_features_.items()]
        )


class ConstantSelector(BaseSelector):
    """Remove near-constant and zero-variance columns from a DataFrame.

    A column is flagged for removal when a single value accounts for at least
    ``threshold`` fraction of non-null rows, or when the column has zero
    variance (numeric only).  All-missing columns are also removed.
    Side-effects: populates ``removed_features_`` and ``selected_features_``
    on the instance.

    Args:
        threshold: Fraction of non-null rows a single value must represent to
            trigger removal.  Defaults to ``0.99`` (i.e. 99 % dominance).

    Attributes:
        removed_features_: Dict mapping feature name to a string describing
            the specific near-constant or zero-variance reason.
        selected_features_: List of column names that survived filtering.
    """

    def __init__(self, threshold: float = settings.quasi_constant_threshold) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ConstantSelector":
        """Scan each column in ``X`` and mark near-constant or zero-variance ones for removal.

        Columns where all values are missing, where one value dominates at or
        above ``threshold``, or where the numeric standard deviation is zero
        are added to ``removed_features_``.  Logs a summary at INFO level.

        Args:
            X: Training feature DataFrame.  Shape ``(n_samples, n_features)``.
            y: Ignored; present only for API compatibility with sklearn pipelines.

        Returns:
            The fitted ``ConstantSelector`` instance (``self``).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        self.removed_features_ = {}
        for col in X.columns:
            s = X[col].dropna()
            if len(s) == 0:
                self.removed_features_[col] = "constant: all missing"
                continue
            top_freq = s.value_counts(normalize=True).iloc[0]
            top_val = s.value_counts().index[0]
            if top_freq >= self.threshold:
                self.removed_features_[col] = f"constant: {top_freq*100:.1f}% is {top_val!r}"
            elif pd.api.types.is_numeric_dtype(X[col]) and X[col].std() == 0:
                self.removed_features_[col] = "zero variance"

        self.selected_features_ = [c for c in X.columns if c not in self.removed_features_]
        logger.info(
            "ConstantSelector: removed %d / %d features",
            len(self.removed_features_),
            len(X.columns),
        )
        return self


class NullRateSelector(BaseSelector):
    """Remove columns whose missing-value rate exceeds a threshold.

    A column is flagged for removal when the fraction of null values across
    ``X`` is strictly above ``threshold``. Side-effects: populates
    ``removed_features_``, ``selected_features_``, and ``null_rates_`` on
    the instance.

    Args:
        threshold: Missing-value rate above which a column is dropped.
            Defaults to ``settings.null_rate_threshold``.

    Attributes:
        removed_features_: Dict mapping feature name to a string describing
            its null rate vs. the threshold.
        selected_features_: List of column names that survived filtering.
        null_rates_: Dict mapping every column name (selected and removed)
            to its observed null rate.
    """

    def __init__(self, threshold: float = settings.null_rate_threshold) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NullRateSelector":
        """Scan each column in ``X`` and mark those above ``threshold`` null rate for removal.

        Args:
            X: Training feature DataFrame.  Shape ``(n_samples, n_features)``.
            y: Ignored; present only for API compatibility with sklearn pipelines.

        Returns:
            The fitted ``NullRateSelector`` instance (``self``).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        self.null_rates_: dict[str, float] = X.isna().mean().to_dict()
        self.removed_features_ = {}
        for col, rate in self.null_rates_.items():
            if rate > self.threshold:
                self.removed_features_[col] = (
                    f"null_rate={rate:.2f} > threshold={self.threshold:.2f}"
                )

        self.selected_features_ = [c for c in X.columns if c not in self.removed_features_]
        logger.info(
            "NullRateSelector: removed %d / %d features",
            len(self.removed_features_),
            len(X.columns),
        )
        return self


class CardinalitySelector(BaseSelector):
    """Remove non-numeric columns whose distinct-value count is too high or too low.

    High-cardinality categoricals (e.g. free-text IDs) rarely add signal and
    can cause memory issues.  Columns with only one distinct value carry no
    information.  Side-effects: populates ``removed_features_`` and
    ``selected_features_`` on the instance.

    Args:
        max_cardinality: Maximum number of distinct values allowed for a
            non-numeric column before it is removed.  Defaults to ``100``.
        remove_single_unique: When ``True`` (default), also remove any column
            — numeric or categorical — that has at most one unique value.

    Attributes:
        removed_features_: Dict mapping feature name to a string describing
            the cardinality violation.
        selected_features_: List of column names that survived filtering.
    """

    def __init__(
        self,
        max_cardinality: int = settings.high_cardinality_threshold,
        remove_single_unique: bool = True,
    ) -> None:
        self.max_cardinality = max_cardinality
        self.remove_single_unique = remove_single_unique

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CardinalitySelector":
        """Scan each column in ``X`` and flag those that violate cardinality constraints.

        Numeric columns are exempt from the ``max_cardinality`` check but are
        still subject to the ``remove_single_unique`` check.  Logs a summary
        at INFO level.

        Args:
            X: Training feature DataFrame.  Shape ``(n_samples, n_features)``.
            y: Ignored; present only for API compatibility with sklearn pipelines.

        Returns:
            The fitted ``CardinalitySelector`` instance (``self``).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        self.removed_features_ = {}
        for col in X.columns:
            n_unique = X[col].nunique()
            if self.remove_single_unique and n_unique <= 1:
                self.removed_features_[col] = f"single unique value (n_unique={n_unique})"
            elif not pd.api.types.is_numeric_dtype(X[col]) and n_unique > self.max_cardinality:
                self.removed_features_[col] = (
                    f"high cardinality: n_unique={n_unique} > max={self.max_cardinality}"
                )

        self.selected_features_ = [c for c in X.columns if c not in self.removed_features_]
        logger.info(
            "CardinalitySelector: removed %d / %d features",
            len(self.removed_features_),
            len(X.columns),
        )
        return self


class CorrelationSelector(BaseSelector):
    """Remove one column from each highly correlated numeric feature pair.

    For each pair whose absolute correlation exceeds ``threshold``, the column
    with lower Information Value is dropped; when IV is unavailable the column
    with lower variance is dropped.  Only numeric columns participate in the
    correlation check; non-numeric columns are always retained.  Side-effects:
    populates ``removed_features_``, ``selected_features_``, and
    ``corr_matrix_`` on the instance.

    Args:
        threshold: Absolute value of the correlation coefficient above which
            one of the pair is removed.  Defaults to ``0.85``.
        method: Correlation method passed to ``pd.DataFrame.corr``.  One of
            ``"pearson"`` (default), ``"spearman"``, or ``"kendall"``.
        iv_table: Optional DataFrame with columns ``feature`` and ``iv``.
            When provided, the column with higher IV is kept; otherwise the
            column with higher variance is kept.

    Attributes:
        removed_features_: Dict mapping feature name to a string describing
            the correlated pair and threshold.
        selected_features_: List of column names that survived filtering.
        corr_matrix_: Absolute correlation matrix computed on numeric columns
            during ``fit``.
    """

    def __init__(
        self,
        threshold: float = settings.correlation_threshold,
        method: str = "pearson",
        iv_table: pd.DataFrame | None = None,
    ) -> None:
        self.threshold = threshold
        self.method = method
        self.iv_table = iv_table

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CorrelationSelector":
        """Compute the correlation matrix and identify redundant numeric features.

        Iterates over the upper triangle of the absolute correlation matrix.
        When a pair exceeds ``threshold``, one column is marked for removal
        (see class docstring for the tie-breaking rule).  Stores the full
        correlation matrix in ``corr_matrix_``.  Logs each drop decision at
        DEBUG level and a summary at INFO level.

        Args:
            X: Training feature DataFrame.  Shape ``(n_samples, n_features)``.
            y: Ignored; present only for API compatibility with sklearn pipelines.

        Returns:
            The fitted ``CorrelationSelector`` instance (``self``).
        """
        num_X = X.select_dtypes(include="number")
        self.corr_matrix_ = num_X.corr(method=self.method).abs()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        iv_map: dict[str, float] = {}
        if self.iv_table is not None:
            iv_map = dict(zip(self.iv_table["feature"], self.iv_table["iv"]))

        removed: set = set()
        cols = list(self.corr_matrix_.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[i] in removed or cols[j] in removed:
                    continue
                r = self.corr_matrix_.iloc[i, j]
                if r > self.threshold:
                    keep, drop = self._pick_keep(cols[i], cols[j], num_X, iv_map)
                    removed.add(drop)
                    logger.debug("CorrelationSelector: drop %r (|r|=%.3f with %r)", drop, r, keep)

        self.removed_features_ = {
            col: f"correlated: |r|>{self.threshold} with a retained feature" for col in removed
        }
        self.selected_features_ = [c for c in X.columns if c not in removed]
        logger.info("CorrelationSelector: removed %d / %d features", len(removed), len(X.columns))
        return self

    def _pick_keep(
        self,
        col_a: str,
        col_b: str,
        X_num: pd.DataFrame,
        iv_map: dict[str, float],
    ) -> tuple:
        iv_a = iv_map.get(col_a, np.nan)
        iv_b = iv_map.get(col_b, np.nan)
        if not np.isnan(iv_a) and not np.isnan(iv_b):
            return (col_a, col_b) if iv_a >= iv_b else (col_b, col_a)
        # Fall back to higher variance
        var_a = X_num[col_a].var()
        var_b = X_num[col_b].var()
        return (col_a, col_b) if var_a >= var_b else (col_b, col_a)


class IVSelector(BaseSelector):
    """Remove features whose Information Value (IV) falls below a minimum threshold.

    IV measures the predictive power of a feature with respect to a binary
    target.  Features with IV below ``threshold`` are considered noise and
    dropped.  IV can be supplied externally via ``iv_table`` (e.g. pre-computed
    by a WOE encoder) or computed on the fly from ``y`` during ``fit``.
    Side-effects: populates ``removed_features_``, ``selected_features_``,
    and ``iv_`` on the instance.

    Args:
        threshold: Minimum IV a feature must have to be retained.
            Defaults to ``0.02`` (the conventional "almost no predictive power"
            boundary).
        iv_table: Optional DataFrame with columns ``feature`` (str) and ``iv``
            (float).  When provided, IVs are taken directly from this table
            and ``y`` is ignored.  When ``None``, ``y`` must be passed to
            ``fit`` and ``dscompanion.utils.metrics.iv_score`` is used for
            computation.

    Attributes:
        removed_features_: Dict mapping feature name to a string showing the
            computed IV and the threshold that was not met.
        selected_features_: List of column names that survived filtering.
        iv_: Dict mapping feature name to its IV value as computed or
            looked up during ``fit``.
    """

    def __init__(
        self,
        threshold: float = settings.iv_threshold,
        iv_table: pd.DataFrame | None = None,
    ) -> None:
        self.threshold = threshold
        self.iv_table = iv_table

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "IVSelector":
        """Compute or look up IV values and mark below-threshold features for removal.

        When ``iv_table`` is provided, IV values are taken from that table and
        ``y`` is ignored.  Otherwise, ``y`` is required and IV is computed via
        ``dscompanion.utils.metrics.iv_score`` for each column.  Features whose IV
        falls below ``threshold`` are added to ``removed_features_``.  Logs a
        summary at INFO level.

        Args:
            X: Training feature DataFrame.  Shape ``(n_samples, n_features)``.
            y: Binary target series.  Required when ``iv_table`` is ``None``.
                Ignored when ``iv_table`` is provided.

        Returns:
            The fitted ``IVSelector`` instance (``self``).

        Raises:
            ValueError: If both ``iv_table`` and ``y`` are ``None``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        if self.iv_table is not None:
            iv_series = pd.Series(dict(zip(self.iv_table["feature"], self.iv_table["iv"])))
        elif y is not None:
            from dscompanion.utils.metrics import iv_score

            iv_series = pd.Series({col: iv_score(X[col], y) for col in X.columns})
        else:
            raise ValueError("Either iv_table or y must be provided to IVSelector.fit.")

        self.iv_: dict[str, float] = iv_series.to_dict()
        self.removed_features_ = {
            col: f"low IV: {iv:.4f} < threshold {self.threshold}"
            for col, iv in self.iv_.items()
            if iv < self.threshold and col in X.columns
        }
        self.selected_features_ = [c for c in X.columns if c not in self.removed_features_]
        logger.info(
            "IVSelector: removed %d / %d features (threshold=%.4f)",
            len(self.removed_features_),
            len(X.columns),
            self.threshold,
        )
        return self


class SHAPSelector(BaseSelector):
    """Remove features whose mean absolute SHAP value falls below a threshold.

    Uses ``shap.Explainer`` to compute feature contributions for a fitted
    model on a random subsample of ``X`` (up to ``max_samples`` rows).  For
    multi-output models the mean absolute SHAP is averaged across output
    dimensions.  Side-effects: populates ``removed_features_``,
    ``selected_features_``, and ``shap_importance_`` on the instance.

    Args:
        model: A fitted sklearn-compatible estimator.  Must be provided;
            there is no sensible default.
        threshold: Features whose mean \\|SHAP\\| is strictly below this value
            are removed.  Defaults to ``0.0``, which removes only features
            with zero contribution.
        max_samples: Maximum number of rows to subsample from ``X`` before
            calling ``shap.Explainer``.  Defaults to ``2000``.  Larger values
            improve accuracy at the cost of runtime.

    Attributes:
        removed_features_: Dict mapping feature name to a string showing the
            mean SHAP importance and the threshold that was not met.
        selected_features_: List of column names that survived filtering.
        shap_importance_: Dict mapping feature name to its mean absolute SHAP
            value as computed during ``fit``.
    """

    def __init__(
        self,
        model=None,
        threshold: float = settings.shap_selector_min_importance,
        max_samples: int = settings.shap_selector_max_samples,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.max_samples = max_samples

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SHAPSelector":
        """Compute mean absolute SHAP values and mark below-threshold features for removal.

        Subsamples up to ``max_samples`` rows from ``X`` (random seed 42),
        creates a ``shap.Explainer`` from ``self.model`` and the sample,
        then computes SHAP values.  For multi-output SHAP arrays the last
        axis is averaged first.  Logs a summary at INFO level.

        Args:
            X: Feature DataFrame.  Should contain only numeric columns.
                Shape ``(n_samples, n_features)``.
            y: Ignored; present only for API compatibility with sklearn pipelines.

        Returns:
            The fitted ``SHAPSelector`` instance (``self``).

        Raises:
            ValueError: If ``model`` is ``None``.
            ImportError: If the ``shap`` package is not installed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        if self.model is None:
            raise ValueError("SHAPSelector requires a fitted model passed via model=.")
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required for SHAPSelector. pip install shap") from exc

        sample = X.sample(min(self.max_samples, len(X)), random_state=settings.random_state)
        explainer = shap.Explainer(self.model, sample)
        shap_values = explainer(sample)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=-1)

        self.shap_importance_: dict[str, float] = dict(zip(X.columns, mean_abs))
        self.removed_features_ = {
            col: f"low SHAP importance: {v:.6f} < {self.threshold}"
            for col, v in self.shap_importance_.items()
            if v < self.threshold
        }
        self.selected_features_ = [c for c in X.columns if c not in self.removed_features_]
        logger.info(
            "SHAPSelector: removed %d / %d features",
            len(self.removed_features_),
            len(X.columns),
        )
        return self


class RFESelector(BaseSelector):
    """Recursive Feature Elimination with cross-validation (RFECV).

    A thin wrapper around ``sklearn.feature_selection.RFECV``.  At each
    iteration the estimator is re-fitted and the least important feature(s)
    are pruned until cross-validated performance stops improving.  Missing
    values in ``X`` are filled with zero before fitting (``X.fillna(0)``).
    Side-effects: populates ``removed_features_``, ``selected_features_``,
    and ``rfecv_`` on the instance.

    Args:
        estimator: sklearn-compatible estimator used to assess feature
            importance.  Defaults to ``LogisticRegression(max_iter=200,
            random_state=42)`` when ``None``.
        step: Number of features to eliminate at each RFE iteration.
            Defaults to ``1``.
        cv: Number of stratified cross-validation folds.  Defaults to ``5``.
        scoring: Metric string passed to ``RFECV`` for fold evaluation.
            Defaults to ``"roc_auc"``.
        min_features_to_select: Lower bound on the number of features that
            ``RFECV`` will retain.  Defaults to ``1``.

    Attributes:
        removed_features_: Dict mapping feature name to the string
            ``"RFE: not selected"``.
        selected_features_: List of column names that RFECV selected.
        rfecv_: The fitted ``sklearn.feature_selection.RFECV`` object,
            exposing ``support_``, ``ranking_``, and ``cv_results_``.
    """

    def __init__(
        self,
        estimator=None,
        step: int = 1,
        cv: int = 5,
        scoring: str = "roc_auc",
        min_features_to_select: int = 1,
    ) -> None:
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RFESelector":
        """Fit RFECV on ``X`` and ``y`` to determine the optimal feature subset.

        Missing values in ``X`` are replaced with zero prior to fitting.  The
        fitted ``RFECV`` object is stored in ``rfecv_`` for further inspection.
        Logs a summary at INFO level.

        Args:
            X: Numeric training feature DataFrame.  Shape
                ``(n_samples, n_features)``.  Non-numeric columns will cause
                the underlying estimator to raise.
            y: Target series.  Required; must be the same length as ``X``.

        Returns:
            The fitted ``RFESelector`` instance (``self``).

        Raises:
            ValueError: If ``y`` is ``None``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        if y is None:
            raise ValueError("RFESelector requires y.")

        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LogisticRegression

        est = (
            self.estimator
            if self.estimator is not None
            else LogisticRegression(
                max_iter=settings.rfe_logreg_max_iter, random_state=settings.random_state
            )
        )
        self.rfecv_ = RFECV(
            estimator=est,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.min_features_to_select,
        )
        self.rfecv_.fit(X.fillna(0), y)

        self.removed_features_ = {
            col: "RFE: not selected"
            for col, keep in zip(X.columns, self.rfecv_.support_)
            if not keep
        }
        self.selected_features_ = [c for c in X.columns if c not in self.removed_features_]
        logger.info(
            "RFESelector: retained %d / %d features",
            len(self.selected_features_),
            len(X.columns),
        )
        return self
