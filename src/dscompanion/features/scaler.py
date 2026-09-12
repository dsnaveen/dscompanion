"""Smart scaler with configurable strategy."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

__all__ = ["SmartScaler", "StatisticalOutlierCapper", "WinsorizationTransformer"]


class SmartScaler(BaseEstimator, TransformerMixin):
    """Scales all numeric columns in a DataFrame using a configurable sklearn
    scaler strategy, leaving non-numeric columns unchanged; ``NaN`` values are
    filled with ``0`` before scaling as a side-effect.

    Args:
        strategy (str): Scaling algorithm to apply. One of:

            - ``"robust"`` (default) — ``RobustScaler``, resistant to outliers.
            - ``"standard"`` — ``StandardScaler``, zero mean and unit variance.
            - ``"minmax"`` — ``MinMaxScaler``, scales to the ``[0, 1]`` range.
            - ``"max_abs"`` — ``MaxAbsScaler``, scales each column by its own
              maximum absolute value (preserves sparsity, keeps zero at zero).
            - ``"l2_norm"`` — ``Normalizer(norm="l2")``, normalises each
              *row* (not column) to unit L2 length — for when the direction
              of the feature vector matters more than its magnitude (e.g.
              embedding similarity). Not invertible: ``inverse_transform()``
              raises ``NotImplementedError`` for this strategy.
            - ``"none"`` — no scaling is applied; the transformer becomes a
              passthrough for numeric columns.

        with_centering (bool): Centre the data before scaling. Applied to
            ``RobustScaler`` and ``StandardScaler``; ignored for
            ``MinMaxScaler`` and ``"none"``. Defaults to ``True``.
        with_scaling (bool): Scale the data after centring. Applied to
            ``RobustScaler`` and ``StandardScaler``; ignored for
            ``MinMaxScaler`` and ``"none"``. Defaults to ``True``.
    """

    def __init__(
        self,
        strategy: str = "robust",
        with_centering: bool = True,
        with_scaling: bool = True,
    ) -> None:
        self.strategy = strategy
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SmartScaler":
        """Identifies numeric columns and fits the chosen sklearn scaler on
        them, replacing ``NaN`` values with ``0`` for the purpose of fitting
        only; stores the fitted scaler internally.

        Args:
            X (pd.DataFrame): Training feature DataFrame. Only columns
                selected by ``select_dtypes(include="number")`` are fitted;
                all other columns are ignored.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            SmartScaler: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SmartScaler.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("SmartScaler.fit() received an empty DataFrame (0 rows)")

        self._num_cols: list[str] = X.select_dtypes(include="number").columns.tolist()
        self._scaler = self._build_scaler()
        if self._scaler is not None and self._num_cols:
            self._scaler.fit(X[self._num_cols].fillna(0))

        logger.info(
            "SmartScaler fitted — strategy=%s, %d numeric cols",
            self.strategy,
            len(self._num_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted scaler to numeric columns in ``X``; ``NaN``
        values in numeric columns are filled with ``0`` before scaling; the
        original DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Only columns
                that were present and numeric at fit time are scaled; newly
                added or non-numeric columns are passed through unchanged.
                Any numeric column from fit time that is absent in ``X`` is
                silently skipped.

        Returns:
            pd.DataFrame: A copy of ``X`` with fitted numeric columns replaced
            by their scaled values. Non-numeric columns are unchanged. When
            ``strategy="none"`` or no numeric columns were found at fit time,
            the returned DataFrame is identical in content to the input.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["_num_cols"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "SmartScaler.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug("SmartScaler.transform — rows=%d, strategy=%s", len(X), self.strategy)

        out = X.copy()
        if self._scaler is not None and self._num_cols:
            cols_present = [c for c in self._num_cols if c in out.columns]
            out[cols_present] = self._scaler.transform(out[cols_present].fillna(0))
        return out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverses scaling on numeric columns in ``X``, restoring their original scale.

        Delegates to the underlying sklearn scaler's own
        ``inverse_transform`` — no new math. Useful for displaying SHAP
        scatter plots or other diagnostics in the feature's original units
        rather than its scaled representation.

        Args:
            X (pd.DataFrame): Feature DataFrame previously produced by
                ``transform()`` (or any DataFrame with the same scaled
                numeric columns). Only columns that were present and
                numeric at fit time are inverse-transformed; newly added
                or non-numeric columns are passed through unchanged. Any
                numeric column from fit time that is absent in ``X`` is
                silently skipped.

        Returns:
            pd.DataFrame: A copy of ``X`` with fitted numeric columns
            restored to their original (pre-scaling) values. Non-numeric
            columns are unchanged. When ``strategy="none"`` or no numeric
            columns were found at fit time, the returned DataFrame is
            identical in content to the input.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            NotImplementedError: If ``strategy="l2_norm"`` — L2 row
                normalisation discards magnitude information and cannot be
                reversed.
        """
        check_is_fitted(self, attributes=["_num_cols"])
        if self.strategy == "l2_norm":
            raise NotImplementedError(
                "SmartScaler.inverse_transform() is not supported for strategy='l2_norm' — "
                "L2 normalization discards magnitude information and cannot be reversed"
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "SmartScaler.inverse_transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug("SmartScaler.inverse_transform — rows=%d, strategy=%s", len(X), self.strategy)

        out = X.copy()
        if self._scaler is not None and self._num_cols:
            cols_present = [c for c in self._num_cols if c in out.columns]
            out[cols_present] = self._scaler.inverse_transform(out[cols_present])
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the names of the numeric columns that were scaled during
        ``fit``; column names are not changed by scaling.

        Args:
            None

        Returns:
            list[str]: Names of numeric columns in the order they appeared in
            ``X`` at fit time.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["_num_cols"])
        return list(self._num_cols)

    def _build_scaler(self):
        if self.strategy == "robust":
            return RobustScaler(with_centering=self.with_centering, with_scaling=self.with_scaling)
        if self.strategy == "standard":
            return StandardScaler(with_mean=self.with_centering, with_std=self.with_scaling)
        if self.strategy == "minmax":
            return MinMaxScaler()
        if self.strategy == "max_abs":
            return MaxAbsScaler()
        if self.strategy == "l2_norm":
            return Normalizer(norm="l2")
        return None  # "none"


class WinsorizationTransformer(BaseEstimator, TransformerMixin):
    """Caps numeric column values at training-set percentile boundaries.

    Fits clip boundaries from the training data using ``pd.Series.quantile``
    and applies them at transform time via ``pd.Series.clip`` — no recomputation
    on test/OOT data, no scipy dependency.  Run this before ``SmartScaler``
    in the feature pipeline.

    Args:
        lower (float | None): Fraction of the lower tail to clip. When
            ``None``, defaults to ``settings.winsorizer_lower_tail``.
        upper (float | None): Fraction of the upper tail to clip; the
            upper boundary is the ``(1 - upper)`` quantile. When ``None``,
            defaults to ``settings.winsorizer_upper_tail``.
        features (list[str] | None): Explicit list of column names to
            winsorize. When ``None``, all numeric columns are winsorized
            automatically.
        column_overrides (dict[str, dict[str, float]] | None): Per-column
            override of ``lower``/``upper``, keyed by column name (e.g.
            ``{"balance": {"lower": 0.05, "upper": 0.02}}``). A column
            present here uses its own override for any key it supplies,
            falling back to ``lower``/``upper`` (then settings defaults)
            for keys it omits. Columns absent from this dict use
            ``lower``/``upper`` unchanged.

    Attributes:
        features_ (list[str]): Columns that were winsorized (set at fit time).
        clip_values_ (dict): Mapping of column name to ``(lower_cap, upper_cap)``
            computed from training data.
    """

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        features: list[str] | None = None,
        column_overrides: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.features = features
        self.column_overrides = column_overrides

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "WinsorizationTransformer":
        """Computes per-column clip boundaries from training data and stores
        them internally; columns that are all-NaN are skipped with a warning.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            WinsorizationTransformer: The fitted instance (``self``).

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "WinsorizationTransformer.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("WinsorizationTransformer.fit() received an empty DataFrame (0 rows)")

        _lower = self.lower if self.lower is not None else settings.winsorizer_lower_tail
        _upper = self.upper if self.upper is not None else settings.winsorizer_upper_tail
        column_overrides = self.column_overrides or {}

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(include="number").columns.tolist()
        )
        self.features_: list[str] = [c for c in cols if c in X.columns]
        self.clip_values_: dict[str, tuple[float, float]] = {}

        for col in self.features_:
            if X[col].count() == 0:
                logger.warning("WinsorizationTransformer: column '%s' is all-NaN — skipping", col)
                continue
            col_override = column_overrides.get(col, {})
            col_lower = col_override.get("lower", _lower)
            col_upper = col_override.get("upper", _upper)
            lo = X[col].quantile(col_lower)
            hi = X[col].quantile(1 - col_upper)
            self.clip_values_[col] = (lo, hi)

        logger.info(
            "WinsorizationTransformer fitted — %d columns (lower=%.3f, upper=%.3f)",
            len(self.clip_values_),
            _lower,
            _upper,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clips values to the training-set boundaries learned during ``fit``;
        the original DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns absent
                from ``clip_values_`` are passed through unchanged.

        Returns:
            pd.DataFrame: Copy of ``X`` with winsorized columns clipped to
            their training-set boundaries.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["clip_values_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "WinsorizationTransformer.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )

        logger.debug(
            "WinsorizationTransformer.transform — rows=%d, clipped cols=%d",
            len(X),
            len(self.clip_values_),
        )

        out = X.copy()
        for col, (lo, hi) in self.clip_values_.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lo, upper=hi)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the column names in the order they were fitted.

        Args:
            None

        Returns:
            list[str]: Winsorized column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["clip_values_"])
        return list(self.features_)


class StatisticalOutlierCapper(BaseEstimator, TransformerMixin):
    """Caps numeric column values using a statistical outlier bound (IQR, Z-score, or MAD).

    Complements ``WinsorizationTransformer`` (which caps at fixed training-set
    percentiles) with bounds derived from the column's own spread — useful
    when the "right" percentile to clip at isn't known up front but a
    standard statistical outlier rule (IQR fences, Z-score, or robust MAD) is
    a better fit for the distribution shape. Run this before ``SmartScaler``
    in the feature pipeline, same as ``WinsorizationTransformer``.

    Args:
        method (str): Statistical rule used to derive the cap. One of:

            - ``"iqr"`` (default) — ``[Q1 - k*IQR, Q3 + k*IQR]``, robust to
              non-Gaussian and heavy-tailed distributions.
            - ``"zscore"`` — ``[mean - k*std, mean + k*std]``, assumes
              roughly Gaussian data; sensitive to the very outliers it's
              capping since ``mean``/``std`` are computed on the same data.
            - ``"mad"`` — ``[median - k*1.4826*MAD, median + k*1.4826*MAD]``,
              the most robust option for extremely dirty data with multiple
              extreme outliers (``1.4826`` is the standard consistency
              constant making MAD comparable to a normal-distribution std).

        multiplier (float | None): The ``k`` multiplier above. When ``None``,
            resolves per-method to ``settings.outlier_capper_iqr_multiplier``
            / ``outlier_capper_zscore_threshold`` /
            ``outlier_capper_mad_threshold``.
        features (list[str] | None): Explicit list of column names to cap.
            When ``None`` (default), all numeric columns are used.
        column_overrides (dict[str, dict[str, Any]] | None): Per-column
            override of ``method``/``multiplier``, keyed by column name
            (e.g. ``{"balance": {"method": "zscore", "multiplier": 2.5}}``).
            A column present here uses its own override for any key it
            supplies, falling back to the constructor-level ``method``/
            ``multiplier`` for keys it omits.

    Attributes:
        features_ (list[str]): Columns the transform was fitted on.
        clip_values_ (dict[str, tuple[float, float]]): Mapping of column name
            to ``(lower_cap, upper_cap)`` computed from training data. A
            column with zero spread under its resolved method (e.g. a
            zero-variance column under ``"zscore"``) or that is all-NaN is
            omitted here — nothing to cap — and logged at WARNING.
    """

    def __init__(
        self,
        method: str = "iqr",
        multiplier: float | None = None,
        features: list[str] | None = None,
        column_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.method = method
        self.multiplier = multiplier
        self.features = features
        self.column_overrides = column_overrides

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "StatisticalOutlierCapper":
        """Computes per-column clip boundaries from training data using the
        resolved statistical method; stores them internally. Columns that are
        all-NaN or have zero spread under their resolved method are skipped
        with a warning.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            StatisticalOutlierCapper: The fitted instance (``self``).

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, or a resolved ``method``
                (constructor-level or per-column override) is not recognised.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "StatisticalOutlierCapper.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("StatisticalOutlierCapper.fit() received an empty DataFrame (0 rows)")

        column_overrides = self.column_overrides or {}
        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(include="number").columns.tolist()
        )
        self.features_: list[str] = [c for c in cols if c in X.columns]
        self.clip_values_: dict[str, tuple[float, float]] = {}

        for col in self.features_:
            if X[col].count() == 0:
                logger.warning("StatisticalOutlierCapper: column '%s' is all-NaN — skipping", col)
                continue

            col_override = column_overrides.get(col, {})
            col_method = col_override.get("method", self.method)
            if col_method not in ("iqr", "zscore", "mad"):
                raise ValueError(
                    f"method must be one of {{'iqr', 'zscore', 'mad'}}, got {col_method!r} "
                    f"(column {col!r})"
                )
            default_multiplier = {
                "iqr": settings.outlier_capper_iqr_multiplier,
                "zscore": settings.outlier_capper_zscore_threshold,
                "mad": settings.outlier_capper_mad_threshold,
            }[col_method]
            col_multiplier = col_override.get("multiplier", self.multiplier)
            m = col_multiplier if col_multiplier is not None else default_multiplier

            series = X[col]
            if col_method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                spread = q3 - q1
                lo_base, hi_base = q1, q3
            elif col_method == "zscore":
                spread = series.std()
                lo_base = hi_base = series.mean()
            else:  # "mad"
                median = series.median()
                spread = (series - median).abs().median() * 1.4826
                lo_base = hi_base = median

            if pd.isna(spread) or spread == 0:
                logger.warning(
                    "StatisticalOutlierCapper: column '%s' has zero spread under method=%s "
                    "— skipping",
                    col,
                    col_method,
                )
                continue

            self.clip_values_[col] = (lo_base - m * spread, hi_base + m * spread)

        logger.info(
            "StatisticalOutlierCapper fitted — %d columns (method=%s)",
            len(self.clip_values_),
            self.method,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clips values to the training-set statistical boundaries learned
        during ``fit``; the original DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns absent
                from ``clip_values_`` are passed through unchanged.

        Returns:
            pd.DataFrame: Copy of ``X`` with capped columns clipped to their
            training-set boundaries.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["clip_values_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "StatisticalOutlierCapper.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )

        logger.debug(
            "StatisticalOutlierCapper.transform — rows=%d, capped cols=%d",
            len(X),
            len(self.clip_values_),
        )

        out = X.copy()
        for col, (lo, hi) in self.clip_values_.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lo, upper=hi)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the column names in the order they were fitted.

        Args:
            None

        Returns:
            list[str]: Capped column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["clip_values_"])
        return list(self.features_)
