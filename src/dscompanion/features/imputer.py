"""Smart imputer with auto-strategy selection and missing indicators."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Unlocks IterativeImputer below — sklearn's required opt-in for this experimental feature.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

__all__ = ["MultivariateImputer", "SmartImputer"]


class SmartImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values column-by-column, automatically selecting numeric
    strategy based on skewness and optionally appending binary missingness
    indicator columns as a side-effect of ``fit``.

    Args:
        numeric_strategy (str): Strategy for numeric columns. One of ``"auto"``,
            ``"median"``, ``"mean"``, or ``"constant"``. When ``"auto"``, the
            imputer selects median by default and switches to mean only when the
            column's absolute skewness is below ``settings.imputer_skew_threshold``.
        categorical_strategy (str): Strategy for non-numeric columns. One of
            ``"most_frequent"`` (default) or ``"constant"``. When
            ``"most_frequent"`` and the mode is empty the fallback value is
            ``"MISSING"``.
        fill_value (Any): Constant used when either strategy is ``"constant"``.
            Defaults to ``0`` for numeric columns and ``"MISSING"`` for
            categorical columns when this is ``None``.
        missing_indicator_threshold (float): Fraction of missing values above
            which a binary ``{col}_was_missing`` column is appended.
            Only effective when ``add_missing_indicator=True``.
        add_missing_indicator (bool): Whether to append binary missing-indicator
            columns for columns whose missingness exceeds the threshold.
        column_strategies (dict[str, str] | None): Per-column strategy
            override, keyed by column name. A column present here uses this
            strategy instead of ``numeric_strategy``/``categorical_strategy``.
            Values must be one of ``"auto"``, ``"mean"``, ``"median"``,
            ``"constant"`` (numeric columns) or ``"most_frequent"``,
            ``"constant"`` (categorical columns) — not validated against the
            column's actual dtype here; an inappropriate value falls through
            to the median/mode branch.
        column_fill_values (dict[str, Any] | None): Per-column constant fill
            value override, keyed by column name. Only consulted for a column
            whose resolved strategy (after ``column_strategies``) is
            ``"constant"``; falls back to ``fill_value`` when absent.

    Attributes:
        imputation_values_ (dict[str, Any]): Mapping from column name to the
            fill value learned during ``fit``.
        indicator_cols_ (list[str]): Column names that will receive a binary
            missing-indicator in ``transform``. Empty list when no column
            exceeds the threshold or ``add_missing_indicator=False``.
    """

    def __init__(
        self,
        numeric_strategy: str = "auto",
        categorical_strategy: str = "most_frequent",
        fill_value: Any = None,
        missing_indicator_threshold: float | None = None,
        add_missing_indicator: bool = True,
        column_strategies: dict[str, str] | None = None,
        column_fill_values: dict[str, Any] | None = None,
    ) -> None:
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.missing_indicator_threshold = missing_indicator_threshold
        self.add_missing_indicator = add_missing_indicator
        self.column_strategies = column_strategies
        self.column_fill_values = column_fill_values

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SmartImputer":
        """Learns per-column imputation values from ``X`` and records which
        columns exceed the missing-indicator threshold; populates
        ``imputation_values_`` and ``indicator_cols_`` as side-effects.

        Args:
            X (pd.DataFrame): Training feature DataFrame. Numeric and
                non-numeric columns are handled separately according to
                ``numeric_strategy`` and ``categorical_strategy``.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            SmartImputer: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SmartImputer.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("SmartImputer.fit() received an empty DataFrame (0 rows)")

        effective_indicator_threshold = (
            self.missing_indicator_threshold
            if self.missing_indicator_threshold is not None
            else settings.imputer_missing_indicator_threshold
        )

        self.imputation_values_: dict[str, Any] = {}
        self.indicator_cols_: list[str] = []
        self._num_cols: list[str] = X.select_dtypes(include="number").columns.tolist()
        self._cat_cols: list[str] = X.select_dtypes(exclude="number").columns.tolist()

        column_strategies = self.column_strategies or {}
        column_fill_values = self.column_fill_values or {}

        for col in self._num_cols:
            missing_rate = X[col].isna().mean()
            if missing_rate > 0:
                if X[col].count() == 0:
                    logger.warning(
                        "SmartImputer: column '%s' is all-NaN — skipping imputation", col
                    )
                    continue

                strategy = column_strategies.get(col, self.numeric_strategy)
                if strategy == "constant":
                    fill = column_fill_values.get(col, self.fill_value)
                    self.imputation_values_[col] = fill if fill is not None else 0
                elif strategy == "mean" or (
                    strategy == "auto" and abs(X[col].skew()) < settings.imputer_skew_threshold
                ):
                    self.imputation_values_[col] = X[col].mean()
                else:
                    self.imputation_values_[col] = X[col].median()

                if self.add_missing_indicator and missing_rate > effective_indicator_threshold:
                    self.indicator_cols_.append(col)

        for col in self._cat_cols:
            if X[col].isna().any():
                strategy = column_strategies.get(col, self.categorical_strategy)
                if strategy == "constant":
                    fill = column_fill_values.get(col, self.fill_value)
                    self.imputation_values_[col] = fill if fill is not None else "MISSING"
                else:
                    mode = X[col].mode()
                    self.imputation_values_[col] = mode[0] if len(mode) > 0 else "MISSING"

        logger.info(
            "SmartImputer fitted — %d cols imputed, %d indicators",
            len(self.imputation_values_),
            len(self.indicator_cols_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values using the imputation map learned during ``fit``
        and appends binary indicator columns for tracked columns as a
        side-effect; the original DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns absent
                from ``imputation_values_`` are passed through unchanged.

        Returns:
            pd.DataFrame: A copy of ``X`` with missing values filled. When
            ``add_missing_indicator=True``, one additional integer column named
            ``{col}_was_missing`` (values 0 or 1) is appended for every column
            in ``indicator_cols_``. Returns a copy with no missing values in
            tracked columns; columns not seen during ``fit`` are left as-is.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["imputation_values_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "SmartImputer.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug(
            "SmartImputer.transform — rows=%d, imputed cols=%d",
            len(X),
            len(self.imputation_values_),
        )

        out = X.copy()
        for col, val in self.imputation_values_.items():
            if col in out.columns:
                out[col] = out[col].fillna(val)

        if self.add_missing_indicator:
            for col in self.indicator_cols_:
                if col in X.columns:
                    out[f"{col}_was_missing"] = X[col].isna().astype(int)

        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the ordered list of column names produced by ``transform``,
        combining original numeric, original categorical, and any appended
        missing-indicator columns.

        Args:
            None

        Returns:
            list[str]: Column names in output order — all numeric columns
            first, then all categorical columns, then one
            ``{col}_was_missing`` name for every entry in ``indicator_cols_``.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["imputation_values_"])
        base = list(self._num_cols) + list(self._cat_cols)
        indicators = [f"{c}_was_missing" for c in self.indicator_cols_]
        return base + indicators


class MultivariateImputer(BaseEstimator, TransformerMixin):
    """Imputes missing numeric values using cross-column (multivariate) estimation.

    Unlike ``SmartImputer`` (independent per-column mean/median/mode), this
    looks at *other* columns' values to estimate a missing one — often more
    accurate when columns are correlated, at the cost of being a genuinely
    joint operation over multiple columns at once.

    Deliberately standalone — not wired into the per-feature transformation
    registry (``dscompanion.features.registry``) or ``FeatureProcessingPipeline``'s
    column-independent stage sequence, for the same reason already applied to
    ``GroupRelativeTransformer``/``PCATransformer``: it needs several columns
    simultaneously, not a single-column slice.

    Args:
        strategy (str): Imputation method. One of:

            - ``"knn"`` (default) — sklearn's ``KNNImputer``: each missing
              value is the mean of its ``n_neighbors`` nearest rows (by
              Euclidean distance over the other numeric columns).
            - ``"iterative"`` — sklearn's ``IterativeImputer``: each column
              with missing values is modelled as a function of the other
              columns, round-robin, for ``max_iter`` passes.

        n_neighbors (int | None): Neighbours averaged for ``strategy="knn"``.
            When ``None``, defaults to ``settings.knn_imputer_n_neighbors``.
        max_iter (int | None): Round-robin passes for
            ``strategy="iterative"``. When ``None``, defaults to
            ``settings.iterative_imputer_max_iter``.
        features (list[str] | None): Explicit list of numeric column names to
            impute jointly. When ``None`` (default), all numeric columns are
            used.

    Attributes:
        features_ (list[str]): Columns the imputer was fitted on — at least
            2, since cross-column estimation needs another column to draw on.
    """

    def __init__(
        self,
        strategy: str = "knn",
        n_neighbors: int | None = None,
        max_iter: int | None = None,
        features: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MultivariateImputer":
        """Fits the chosen sklearn multivariate imputer on the resolved numeric columns.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            MultivariateImputer: The fitted instance (``self``), enabling
            method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, ``strategy`` is not
                recognised, or fewer than 2 numeric columns are resolved
                (cross-column imputation needs at least one other column).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MultivariateImputer.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("MultivariateImputer.fit() received an empty DataFrame (0 rows)")
        if self.strategy not in ("knn", "iterative"):
            raise ValueError(
                f"strategy must be one of {{'knn', 'iterative'}}, got {self.strategy!r}"
            )

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(include="number").columns.tolist()
        )
        self.features_: list[str] = [c for c in cols if c in X.columns]
        if len(self.features_) < 2:
            raise ValueError(
                "MultivariateImputer needs at least 2 numeric columns to impute jointly, "
                f"got {len(self.features_)}"
            )

        if self.strategy == "knn":
            n_neighbors = (
                self.n_neighbors
                if self.n_neighbors is not None
                else settings.knn_imputer_n_neighbors
            )
            self._imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            max_iter = (
                self.max_iter if self.max_iter is not None else settings.iterative_imputer_max_iter
            )
            self._imputer = IterativeImputer(max_iter=max_iter, random_state=settings.random_state)

        self._imputer.fit(X[self.features_])
        logger.info(
            "MultivariateImputer fitted — strategy=%s, %d columns",
            self.strategy,
            len(self.features_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted multivariate imputer to the fitted columns.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Must contain
                every column seen at ``fit`` time — a partial column set
                can't be jointly imputed.

        Returns:
            pd.DataFrame: A copy of ``X`` with the fitted numeric columns'
            missing values filled. Columns not in ``features_`` are
            unchanged.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If any fitted column is missing from ``X``.
        """
        check_is_fitted(self, attributes=["features_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MultivariateImputer.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        missing = [c for c in self.features_ if c not in X.columns]
        if missing:
            raise ValueError(f"MultivariateImputer.transform() missing fitted columns: {missing}")

        logger.debug(
            "MultivariateImputer.transform — rows=%d, cols=%d", len(X), len(self.features_)
        )

        out = X.copy()
        out[self.features_] = self._imputer.transform(out[self.features_])
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the column names in the order they were fitted; this
        transform never changes column names or count.

        Args:
            None

        Returns:
            list[str]: Imputed column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["features_"])
        return list(self.features_)
