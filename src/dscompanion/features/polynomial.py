"""PolynomialFeaturesTransformer: polynomial/interaction feature generation for numeric columns."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

logger = logging.getLogger(__name__)

__all__ = ["PolynomialFeaturesTransformer"]


class PolynomialFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Generates polynomial and pairwise-interaction features from numeric columns.

    Wraps sklearn's ``PolynomialFeatures`` as a DataFrame-in/DataFrame-out,
    fit-on-train/apply-on-holdout transformer, matching ``PCATransformer``'s
    established wrapping pattern. For ``degree=2`` and input columns ``x1``,
    ``x2``, produces ``x1``, ``x2``, ``x1_pow2``, ``x1_x_x2``, ``x2_pow2``
    (no bias/intercept column — models fit their own).

    Column-count-changing (like ``OneHotEncoder``) — callers composing this
    into a larger pipeline must concatenate the result rather than assign it
    back into the original column positions.

    Args:
        degree (int | None): Max polynomial degree. When ``None`` (default),
            uses ``settings.polynomial_features_degree``.
        interaction_only (bool): When ``True``, only interaction terms
            (``x1*x2``) are produced — no pure powers (``x1^2``). Defaults
            to ``False``.
        features (list[str] | None): Explicit list of numeric column names
            to expand. When ``None`` (default), all numeric columns are used.

    Attributes:
        features_ (list[str]): Columns the transform was fitted on.
        feature_names_out_ (list[str]): Ordered list of output column names —
            sklearn's own generated names (``"x1 x2"``, ``"x1^2"``) remapped
            to column-safe forms (``"x1_x_x2"``, ``"x1_pow2"``).
    """

    def __init__(
        self,
        degree: int | None = None,
        interaction_only: bool = False,
        features: list[str] | None = None,
    ) -> None:
        self.degree = degree
        self.interaction_only = interaction_only
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PolynomialFeaturesTransformer":
        """Fits sklearn's ``PolynomialFeatures`` on the resolved numeric columns.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            PolynomialFeaturesTransformer: The fitted instance (``self``),
            enabling method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows or no numeric columns resolve.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "PolynomialFeaturesTransformer.fit() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError(
                "PolynomialFeaturesTransformer.fit() received an empty DataFrame (0 rows)"
            )

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(include="number").columns.tolist()
        )
        self.features_: list[str] = [c for c in cols if c in X.columns]
        if len(self.features_) < 1:
            raise ValueError(
                "PolynomialFeaturesTransformer requires at least 1 numeric column, got 0"
            )

        effective_degree = (
            self.degree if self.degree is not None else settings.polynomial_features_degree
        )
        self._poly = PolynomialFeatures(
            degree=effective_degree, interaction_only=self.interaction_only, include_bias=False
        )
        self._poly.fit(X[self.features_])

        raw_names = self._poly.get_feature_names_out(self.features_)
        self.feature_names_out_: list[str] = [
            n.replace(" ", "_x_").replace("^", "_pow") for n in raw_names
        ]

        logger.info(
            "PolynomialFeaturesTransformer fitted — degree=%d, %d cols → %d cols",
            effective_degree,
            len(self.features_),
            len(self.feature_names_out_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted polynomial expansion to ``X``.

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
                "PolynomialFeaturesTransformer.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        missing = [c for c in self.features_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"PolynomialFeaturesTransformer.transform() missing fitted columns: {missing}"
            )

        logger.debug(
            "PolynomialFeaturesTransformer.transform — rows=%d, output cols=%d",
            len(X),
            len(self.feature_names_out_),
        )

        arr = self._poly.transform(X[self.features_])
        return pd.DataFrame(arr, columns=self.feature_names_out_, index=X.index)

    def get_feature_names_out(self) -> list[str]:
        """Returns the polynomial/interaction output column names.

        Args:
            None

        Returns:
            list[str]: Output column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        return list(self.feature_names_out_)
