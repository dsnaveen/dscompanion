"""ColumnArithmeticTransformer: two-column arithmetic feature combination."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

__all__ = ["CategoryCombinerTransformer", "ColumnArithmeticTransformer"]

_OPERATIONS = {"sum", "difference", "product", "ratio"}


class ColumnArithmeticTransformer(BaseEstimator, TransformerMixin):
    """Combines two numeric columns into one new feature via a named arithmetic operation.

    Adds a single new column (never replaces the originals) computed from
    ``col_a`` and ``col_b`` — e.g. ``"available_credit" = "credit_limit" -
    "balance"``. No training statistics are learned (the operation is a pure
    row-wise function of the two inputs), but the class still follows this
    codebase's fit/transform convention so it composes with the rest of the
    pipeline: ``fit()`` only validates that both columns are present and
    numeric. Standalone — not wired into ``FeatureProcessingPipeline`` or
    ``TRANSFORMER_REGISTRY``, since it needs two named columns simultaneously
    rather than operating on a single-column slice, matching the existing
    exclusion of ``GroupRelativeTransformer``/``PCATransformer``.

    Args:
        col_a (str): First operand column name.
        col_b (str): Second operand column name.
        operation (str): One of ``"sum"`` (``col_a + col_b``),
            ``"difference"`` (``col_a - col_b``), ``"product"``
            (``col_a * col_b``), or ``"ratio"`` (``col_a / col_b``, a zero
            denominator produces ``NaN`` rather than ``inf``). Defaults to
            ``"difference"``.
        output_name (str | None): Name of the new column. When ``None``
            (default), generated as ``f"{col_a}_{operation}_{col_b}"``.

    Attributes:
        output_name_ (str): Resolved output column name.
    """

    def __init__(
        self,
        col_a: str,
        col_b: str,
        operation: str = "difference",
        output_name: str | None = None,
    ) -> None:
        self.col_a = col_a
        self.col_b = col_b
        self.operation = operation
        self.output_name = output_name

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnArithmeticTransformer":
        """Validates both operand columns are present and numeric, and
        resolves the output column name.

        Args:
            X (pd.DataFrame): Training feature DataFrame, must contain
                ``col_a`` and ``col_b``.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            ColumnArithmeticTransformer: The fitted instance (``self``).

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, ``col_a``/``col_b`` is
                missing or non-numeric, or ``operation`` is not recognised.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ColumnArithmeticTransformer.fit() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError(
                "ColumnArithmeticTransformer.fit() received an empty DataFrame (0 rows)"
            )
        for col in (self.col_a, self.col_b):
            if col not in X.columns:
                raise ValueError(f"column {col!r} not found in X")
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"column {col!r} must be numeric, got dtype {X[col].dtype}")
        if self.operation not in _OPERATIONS:
            raise ValueError(f"operation must be one of {_OPERATIONS}, got {self.operation!r}")

        self.output_name_: str = (
            self.output_name
            if self.output_name is not None
            else f"{self.col_a}_{self.operation}_{self.col_b}"
        )

        logger.info(
            "ColumnArithmeticTransformer fitted — %s(%s, %s) -> %s",
            self.operation,
            self.col_a,
            self.col_b,
            self.output_name_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Adds the computed column to a copy of ``X``; the original
        DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform, must contain
                ``col_a`` and ``col_b``.

        Returns:
            pd.DataFrame: A copy of ``X`` with ``output_name_`` appended.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``col_a``/``col_b`` is missing from ``X``.
        """
        check_is_fitted(self, attributes=["output_name_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ColumnArithmeticTransformer.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        for col in (self.col_a, self.col_b):
            if col not in X.columns:
                raise ValueError(f"column {col!r} not found in X")

        logger.debug(
            "ColumnArithmeticTransformer.transform — rows=%d, operation=%s",
            len(X),
            self.operation,
        )

        out = X.copy()
        a, b = out[self.col_a], out[self.col_b]
        if self.operation == "sum":
            out[self.output_name_] = a + b
        elif self.operation == "difference":
            out[self.output_name_] = a - b
        elif self.operation == "product":
            out[self.output_name_] = a * b
        elif self.operation == "ratio":
            out[self.output_name_] = a / b.replace(0, np.nan)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the single new column name this transformer adds.

        Args:
            None

        Returns:
            list[str]: A one-element list containing ``output_name_``.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["output_name_"])
        return [self.output_name_]


class CategoryCombinerTransformer(BaseEstimator, TransformerMixin):
    """Concatenates two categorical columns' string values into one new categorical feature.

    Adds a single new column (never replaces the originals), e.g.
    ``"city_store_type" = "city" + "_" + "store_type"`` -> ``"NYC_Grocery"``.
    Cardinality of the combined column can reach ``card(col_a) * card(col_b)``
    in the worst case — this is exactly the red flag ``feature_transformation.md``
    calls out for category combination, so a combined cardinality above
    ``settings.category_combiner_cardinality_warn_threshold`` is logged as a
    warning (not raised) so callers can catch it before feeding the result
    into a downstream high-cardinality encoder. Standalone — not wired into
    ``FeatureProcessingPipeline`` or ``TRANSFORMER_REGISTRY``, for the same
    two-column reason as ``ColumnArithmeticTransformer``.

    Args:
        col_a (str): First categorical column name.
        col_b (str): Second categorical column name.
        separator (str): String joining the two values. Defaults to ``"_"``.
        output_name (str | None): Name of the new column. When ``None``
            (default), generated as ``f"{col_a}_{col_b}"``.

    Attributes:
        output_name_ (str): Resolved output column name.
        combined_cardinality_ (int): Number of distinct combined values
            observed in the training data.
    """

    def __init__(
        self,
        col_a: str,
        col_b: str,
        separator: str = "_",
        output_name: str | None = None,
    ) -> None:
        self.col_a = col_a
        self.col_b = col_b
        self.separator = separator
        self.output_name = output_name

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoryCombinerTransformer":
        """Validates both columns are present and computes the resulting
        combined cardinality for the training-set warning check.

        Args:
            X (pd.DataFrame): Training feature DataFrame, must contain
                ``col_a`` and ``col_b``.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            CategoryCombinerTransformer: The fitted instance (``self``).

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows or ``col_a``/``col_b`` is
                missing from ``X``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "CategoryCombinerTransformer.fit() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError(
                "CategoryCombinerTransformer.fit() received an empty DataFrame (0 rows)"
            )
        for col in (self.col_a, self.col_b):
            if col not in X.columns:
                raise ValueError(f"column {col!r} not found in X")

        self.output_name_: str = (
            self.output_name if self.output_name is not None else f"{self.col_a}_{self.col_b}"
        )
        combined = X[self.col_a].astype(str) + self.separator + X[self.col_b].astype(str)
        self.combined_cardinality_: int = combined.nunique()

        if self.combined_cardinality_ > settings.category_combiner_cardinality_warn_threshold:
            logger.warning(
                "CategoryCombinerTransformer: combining '%s' and '%s' produces %d distinct "
                "values (threshold=%d) — consider a high-cardinality encoder downstream",
                self.col_a,
                self.col_b,
                self.combined_cardinality_,
                settings.category_combiner_cardinality_warn_threshold,
            )

        logger.info(
            "CategoryCombinerTransformer fitted — %s + %s -> %s (%d distinct values)",
            self.col_a,
            self.col_b,
            self.output_name_,
            self.combined_cardinality_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Adds the concatenated column to a copy of ``X``; the original
        DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform, must contain
                ``col_a`` and ``col_b``.

        Returns:
            pd.DataFrame: A copy of ``X`` with ``output_name_`` appended.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``col_a``/``col_b`` is missing from ``X``.
        """
        check_is_fitted(self, attributes=["output_name_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "CategoryCombinerTransformer.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        for col in (self.col_a, self.col_b):
            if col not in X.columns:
                raise ValueError(f"column {col!r} not found in X")

        logger.debug("CategoryCombinerTransformer.transform — rows=%d", len(X))

        out = X.copy()
        out[self.output_name_] = (
            out[self.col_a].astype(str) + self.separator + out[self.col_b].astype(str)
        )
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the single new column name this transformer adds.

        Args:
            None

        Returns:
            list[str]: A one-element list containing ``output_name_``.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["output_name_"])
        return [self.output_name_]
