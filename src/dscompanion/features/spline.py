"""SplineFeatureTransformer: B-spline basis feature generation for numeric columns."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

logger = logging.getLogger(__name__)

__all__ = ["SplineFeatureTransformer"]


class SplineFeatureTransformer(BaseEstimator, TransformerMixin):
    """Expands numeric columns into a B-spline basis, capturing smooth non-linear effects.

    Wraps sklearn's ``SplineTransformer`` as a DataFrame-in/DataFrame-out,
    fit-on-train/apply-on-holdout transformer, matching ``PCATransformer``'s
    established wrapping pattern. Each input column expands into several
    spline-basis columns (sklearn's own ``n_knots``/``degree`` convention) —
    column-count-changing per source column, like ``OneHotEncoder``.

    Args:
        n_knots (int | None): Number of knots per column's B-spline basis.
            When ``None`` (default), uses ``settings.spline_transformer_n_knots``.
        degree (int | None): B-spline polynomial degree. When ``None``
            (default), uses ``settings.spline_transformer_degree``.
        features (list[str] | None): Explicit list of numeric column names
            to expand. When ``None`` (default), all numeric columns are used.

    Attributes:
        features_ (list[str]): Columns the transform was fitted on.
        feature_names_out_ (list[str]): Ordered list of output column names
            (sklearn's own generated names, prefixed by source column).
    """

    def __init__(
        self,
        n_knots: int | None = None,
        degree: int | None = None,
        features: list[str] | None = None,
    ) -> None:
        self.n_knots = n_knots
        self.degree = degree
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SplineFeatureTransformer":
        """Fits sklearn's ``SplineTransformer`` on the resolved numeric columns.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            SplineFeatureTransformer: The fitted instance (``self``),
            enabling method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows or no numeric columns resolve.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "SplineFeatureTransformer.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("SplineFeatureTransformer.fit() received an empty DataFrame (0 rows)")

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(include="number").columns.tolist()
        )
        self.features_: list[str] = [c for c in cols if c in X.columns]
        if len(self.features_) < 1:
            raise ValueError("SplineFeatureTransformer requires at least 1 numeric column, got 0")

        effective_n_knots = (
            self.n_knots if self.n_knots is not None else settings.spline_transformer_n_knots
        )
        effective_degree = (
            self.degree if self.degree is not None else settings.spline_transformer_degree
        )

        self._spline = SplineTransformer(
            n_knots=effective_n_knots, degree=effective_degree, include_bias=False
        )
        self._spline.fit(X[self.features_])
        self.feature_names_out_: list[str] = list(
            self._spline.get_feature_names_out(self.features_)
        )

        logger.info(
            "SplineFeatureTransformer fitted — n_knots=%d, degree=%d, %d cols → %d cols",
            effective_n_knots,
            effective_degree,
            len(self.features_),
            len(self.feature_names_out_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted B-spline expansion to ``X``.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Must contain
                every column seen at ``fit`` time — a partial column set
                can't be jointly expanded.

        Returns:
            pd.DataFrame: A new DataFrame with ``feature_names_out_`` columns,
            indexed identically to ``X``.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If any fitted column is missing from ``X``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "SplineFeatureTransformer.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        missing = [c for c in self.features_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"SplineFeatureTransformer.transform() missing fitted columns: {missing}"
            )

        logger.debug(
            "SplineFeatureTransformer.transform — rows=%d, output cols=%d",
            len(X),
            len(self.feature_names_out_),
        )

        arr = self._spline.transform(X[self.features_])
        return pd.DataFrame(arr, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self) -> list[str]:
        """Returns the B-spline output column names.

        Args:
            None

        Returns:
            list[str]: Output column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        return list(self.feature_names_out_)
