"""PercentileRankTransformer: monotonic outlier-neutralising rank transform."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

__all__ = ["PercentileRankTransformer"]


class PercentileRankTransformer(BaseEstimator, TransformerMixin):
    """Replaces numeric values with their percentile rank in the training distribution.

    A monotonic transform mapping each value to the fraction of training rows
    strictly less than it (via ``np.searchsorted`` against the sorted training
    values) — neutralises the influence of extreme outliers without discarding
    their relative ordering, unlike winsorization (which caps them) or
    standard scaling (which is skewed by them). Good fit for tree models per
    ``feature_transformation.md``'s own decision framework.

    Standalone — not wired into ``FeatureProcessingPipeline``, matching the
    precedent already set by ``AutoBinner``/``RareCategoryGrouper``: defined,
    tested, importable, but not part of the automatic column-detection
    pipeline, since rank transformation is an optional, model-specific choice
    rather than a universal stage.

    Args:
        features (list[str] | None): Explicit list of column names to
            transform. When ``None`` (default), all numeric columns are used.
        method (str): Ranking method. One of ``"percentile"`` (default) —
            fraction of training rows strictly less than the value, in
            ``[0, 1]``; ``"dense"`` — index of the value among *unique*
            training values, ties share the same rank with no gaps;
            ``"global"`` — count of training rows less than or equal to the
            value, raw ordinal position including ties (``[0, n]``, not
            normalised).

    Attributes:
        features_ (list[str]): Columns the transform was fitted on.
    """

    def __init__(self, features: list[str] | None = None, method: str = "percentile") -> None:
        self.features = features
        self.method = method

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PercentileRankTransformer":
        """Stores the sorted training values per column for percentile lookup at
        transform time.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            PercentileRankTransformer: The fitted instance (``self``), enabling
            method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "PercentileRankTransformer.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("PercentileRankTransformer.fit() received an empty DataFrame (0 rows)")
        if self.method not in ("percentile", "dense", "global"):
            raise ValueError(
                f"method must be one of {{'percentile', 'dense', 'global'}}, got {self.method!r}"
            )

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(include="number").columns.tolist()
        )
        self.features_: list[str] = [c for c in cols if c in X.columns]

        self._sorted_values: dict[str, np.ndarray] = {}
        self._unique_sorted_values: dict[str, np.ndarray] = {}
        for col in self.features_:
            sorted_vals = np.sort(X[col].dropna().to_numpy(dtype=float))
            self._sorted_values[col] = sorted_vals
            self._unique_sorted_values[col] = np.unique(sorted_vals)

        logger.info("PercentileRankTransformer fitted — %d columns", len(self.features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces each fitted column with its percentile rank against the
        training distribution learned during ``fit``; the original DataFrame
        is not modified.

        A value below every training value maps to ``0.0``; a value above
        every training value maps to ``1.0`` — no clipping needed, this falls
        naturally out of ``np.searchsorted``. ``NaN`` values remain ``NaN``.
        A column that was empty (all-NaN) at fit time maps every value to
        ``NaN`` at transform time (no training distribution to rank against).

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns absent
                from ``features_`` are passed through unchanged.

        Returns:
            pd.DataFrame: A copy of ``X`` with fitted columns replaced by
            their percentile rank (float, in ``[0, 1]``) against the training
            distribution.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["features_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "PercentileRankTransformer.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )

        logger.debug(
            "PercentileRankTransformer.transform — rows=%d, cols=%d",
            len(X),
            len(self.features_),
        )

        out = X.copy()
        for col in self.features_:
            if col not in out.columns:
                continue
            sorted_vals = self._sorted_values[col]
            n = len(sorted_vals)
            values = out[col].to_numpy(dtype=float)
            if n == 0:
                out[col] = np.nan
                continue
            nan_mask = np.isnan(values)
            if self.method == "percentile":
                ranks = np.searchsorted(sorted_vals, values, side="left") / n
            elif self.method == "global":
                ranks = np.searchsorted(sorted_vals, values, side="right").astype(float)
            else:  # "dense"
                unique_vals = self._unique_sorted_values[col]
                ranks = np.searchsorted(unique_vals, values, side="left").astype(float)
            out[col] = np.where(nan_mask, np.nan, ranks)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the column names in the order they were fitted; this
        transform never changes column names or count.

        Args:
            None

        Returns:
            list[str]: Transformed column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["features_"])
        return list(self.features_)
