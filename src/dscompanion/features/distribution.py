"""DistributionTransformer: monotonic distribution-normalising transforms for skewed numerics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

logger = logging.getLogger(__name__)

__all__ = ["DistributionTransformer"]

_STRATEGIES = {
    "none",
    "log",
    "log1p",
    "sqrt",
    "cbrt",
    "reciprocal",
    "yeo_johnson",
    "quantile_uniform",
    "quantile_normal",
}


class DistributionTransformer(BaseEstimator, TransformerMixin):
    """Applies a monotonic distribution-normalising transform to skewed numeric columns.

    Run this before ``SmartScaler`` in the feature pipeline — normalising skew is most
    effective on already-imputed data, and scaling should see the transformed
    distribution rather than the raw skewed one.

    Args:
        strategy (str): Default transform applied to every column in
            ``features_`` that has no ``column_overrides`` entry for
            ``"strategy"``. One of:

            - ``"none"`` (default) — no transform; passthrough.
            - ``"log"`` — natural log. Requires every value to be strictly positive;
              raises at ``fit()`` otherwise. Use ``"log1p"`` or ``"yeo_johnson"``
              for columns that can be zero or negative (e.g. account balances).
            - ``"log1p"`` — ``log(1 + x)``. Requires every value to be greater than
              ``-1``; raises at ``fit()`` otherwise.
            - ``"sqrt"`` — square root. Requires every value to be
              non-negative; raises at ``fit()`` otherwise.
            - ``"cbrt"`` — cube root (``np.cbrt``, correctly handles negative
              inputs unlike ``x ** (1/3)``). No domain restriction.
            - ``"reciprocal"`` — ``1 / x``. Requires every value to be
              nonzero; raises at ``fit()`` otherwise.
            - ``"yeo_johnson"`` — sklearn's ``PowerTransformer``, a per-column
              MLE-fitted power transform that handles zero and negative values
              natively. The right default for skewed banking numerics (balance,
              income) that Box-Cox can't handle.
            - ``"quantile_uniform"`` — sklearn's ``QuantileTransformer``
              (``output_distribution="uniform"``), mapping the column to a
              uniform ``[0, 1]`` distribution via its training-data rank.
            - ``"quantile_normal"`` — same, but ``output_distribution="normal"``
              (standard Gaussian).

        features (list[str] | None): Explicit list of column names to transform.
            When ``None`` (default), all numeric columns are transformed.
        column_overrides (dict[str, dict[str, str]] | None): Per-column
            override of ``strategy``, keyed by column name (e.g.
            ``{"balance": {"strategy": "yeo_johnson"}}``) — lets one instance
            apply different distribution-normalising strategies to different
            columns. A column absent from this dict uses the
            constructor-level ``strategy``.

    Attributes:
        features_ (list[str]): Columns the transform was fitted on.
        column_strategies_ (dict[str, str]): Strategy actually resolved and
            used per column (constructor default, overridden per column).
    """

    def __init__(
        self,
        strategy: str = "none",
        features: list[str] | None = None,
        column_overrides: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.strategy = strategy
        self.features = features
        self.column_overrides = column_overrides

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DistributionTransformer":
        """Validates the resolved strategy's domain requirements against the training
        data and, for ``"yeo_johnson"``/``"quantile_uniform"``/``"quantile_normal"``,
        fits the per-column stateful transformer.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline compatibility.

        Returns:
            DistributionTransformer: The fitted instance (``self``), enabling
            method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, a resolved strategy is not
                recognised, or a strategy's domain requirement is violated by
                the data (e.g. ``"log"`` with a non-positive value present).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DistributionTransformer.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("DistributionTransformer.fit() received an empty DataFrame (0 rows)")

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(include="number").columns.tolist()
        )
        self.features_: list[str] = [c for c in cols if c in X.columns]
        column_overrides = self.column_overrides or {}

        self.column_strategies_: dict[str, str] = {}
        self._fitted_transformers: dict[str, Any] = {}

        for col in self.features_:
            col_strategy = column_overrides.get(col, {}).get("strategy", self.strategy)
            if col_strategy not in _STRATEGIES:
                raise ValueError(
                    f"strategy must be one of {_STRATEGIES}, got {col_strategy!r} "
                    f"(column {col!r})"
                )
            self.column_strategies_[col] = col_strategy

            if col_strategy == "log":
                if (X[col] <= 0).any():
                    raise ValueError(
                        f"strategy='log' requires strictly positive values; column {col!r} "
                        "contains values <= 0 — use 'log1p' or 'yeo_johnson' instead"
                    )
            elif col_strategy == "log1p":
                if (X[col] <= -1).any():
                    raise ValueError(
                        f"strategy='log1p' requires values > -1; column {col!r} violates "
                        "this — use 'yeo_johnson' instead"
                    )
            elif col_strategy == "sqrt":
                if (X[col] < 0).any():
                    raise ValueError(
                        f"strategy='sqrt' requires non-negative values; column {col!r} "
                        "contains negative values — use 'yeo_johnson' instead"
                    )
            elif col_strategy == "reciprocal":
                if (X[col] == 0).any():
                    raise ValueError(
                        f"strategy='reciprocal' requires nonzero values; column {col!r} "
                        "contains zero values"
                    )
            elif col_strategy == "yeo_johnson":
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                pt.fit(X[[col]])
                self._fitted_transformers[col] = pt
            elif col_strategy in ("quantile_uniform", "quantile_normal"):
                output_distribution = "uniform" if col_strategy == "quantile_uniform" else "normal"
                n_quantiles = min(settings.quantile_transformer_n_quantiles, len(X))
                qt = QuantileTransformer(
                    output_distribution=output_distribution,
                    n_quantiles=n_quantiles,
                    random_state=settings.random_state,
                )
                qt.fit(X[[col]])
                self._fitted_transformers[col] = qt

        logger.info(
            "DistributionTransformer fitted — %d columns, strategies=%s",
            len(self.features_),
            sorted(set(self.column_strategies_.values())),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies each column's resolved strategy to the training-time columns
        present in ``X``; the original DataFrame is not modified.

        Values outside a strategy's valid domain at transform time (e.g. a
        negative value under ``"log"``) produce ``NaN`` via the underlying numpy
        call rather than raising — the same tolerance this codebase already
        applies to other numerically-sensitive transforms.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns absent from
                ``features_`` are passed through unchanged.

        Returns:
            pd.DataFrame: A copy of ``X`` with fitted columns transformed
            according to their resolved strategy. Columns resolved to
            ``"none"`` are returned unchanged.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["column_strategies_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DistributionTransformer.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )

        logger.debug("DistributionTransformer.transform — rows=%d", len(X))

        out = X.copy()
        for col, col_strategy in self.column_strategies_.items():
            if col not in out.columns or col_strategy == "none":
                continue
            if col_strategy == "log":
                out[col] = np.log(out[col])
            elif col_strategy == "log1p":
                out[col] = np.log1p(out[col])
            elif col_strategy == "sqrt":
                out[col] = np.sqrt(out[col])
            elif col_strategy == "cbrt":
                out[col] = np.cbrt(out[col])
            elif col_strategy == "reciprocal":
                out[col] = 1.0 / out[col]
            elif col in self._fitted_transformers:
                out[col] = self._fitted_transformers[col].transform(out[[col]]).ravel()
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the column names in the order they were fitted; this transform
        never changes column names or count.

        Args:
            None

        Returns:
            list[str]: Transformed column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["features_"])
        return list(self.features_)
