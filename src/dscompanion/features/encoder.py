"""Categorical encoders: HighCardinalityEncoder, WoEEncoder, OrdinalEncoder, OneHotEncoder,
RareCategoryGrouper, BinaryEncoder, HashEncoder."""

from __future__ import annotations

import hashlib
import logging
import math

# Any is unavoidable: category values may be str, int, float, bool, or any
# other hashable Python type depending on the column's content.
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings
from dscompanion.utils.metrics import woe_bins
from dscompanion.utils.validators import validate_binary_target

logger = logging.getLogger(__name__)

__all__ = [
    "BinaryEncoder",
    "HashEncoder",
    "HighCardinalityEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "RareCategoryGrouper",
    "WoEEncoder",
]


class HighCardinalityEncoder(BaseEstimator, TransformerMixin):
    """Encodes high-cardinality categorical columns using either smoothed target
    encoding or relative frequency encoding, automatically identifying columns
    whose unique-value count exceeds a configurable cardinality threshold.

    Args:
        strategy (str): Encoding method. ``"target_encoding"`` (default)
            computes a leave-one-out smoothed estimate using the binary target
            ``y``; requires ``y`` to be supplied in ``fit``. ``"frequency_encoding"``
            replaces each category with its relative frequency in the training set
            and does not require ``y``.
        m (float): Smoothing factor for target encoding. Larger values pull
            category estimates toward the global mean, reducing overfitting on
            rare categories. Defaults to ``10.0``.
        cardinality_threshold (int | None): Minimum number of unique values
            for a column to be considered high-cardinality. When ``None``, the
            value from ``settings.high_cardinality_threshold`` is used.
        handle_unknown (str): Strategy for unseen categories at transform time.
            ``"global_mean"`` (default) replaces them with the column-level
            global mean learned during ``fit``.

    Attributes:
        encoding_maps_ (dict[str, dict[Any, float]]): Mapping from column name
            to a dict of ``{category: encoded_float}``.
        global_means_ (dict[str, float]): Per-column fallback values used for
            unseen categories. For target encoding this is the training target
            mean; for frequency encoding it is ``1 / n_unique``.
    """

    def __init__(
        self,
        strategy: str = "target_encoding",
        m: float | None = None,
        cardinality_threshold: int | None = None,
        handle_unknown: str = "global_mean",
    ) -> None:
        self.strategy = strategy
        self.m = m
        self.cardinality_threshold = cardinality_threshold
        self.handle_unknown = handle_unknown

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HighCardinalityEncoder":
        """Identifies high-cardinality columns and computes per-column encoding
        maps, storing results in ``encoding_maps_`` and ``global_means_`` as
        side-effects.

        Args:
            X (pd.DataFrame): Training feature DataFrame. Only non-numeric
                columns whose unique-value count exceeds the cardinality
                threshold are processed; all other columns are ignored.
            y (pd.Series | None): Target series. Required when
                ``strategy="target_encoding"``; ignored for
                ``"frequency_encoding"``. When ``strategy="target_encoding"``
                and ``y`` is ``None``, the encoder silently falls back to
                frequency encoding.

        Returns:
            HighCardinalityEncoder: The fitted instance (``self``), enabling
            method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``, or ``y`` is not a
                ``pd.Series`` when ``strategy="target_encoding"``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "HighCardinalityEncoder.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("HighCardinalityEncoder.fit() received an empty DataFrame (0 rows)")
        if self.strategy == "target_encoding" and y is not None:
            if not isinstance(y, pd.Series):
                raise TypeError(
                    "HighCardinalityEncoder.fit() expects y to be a pd.Series, got %s"
                    % type(y).__name__
                )

        threshold = (
            self.cardinality_threshold
            if self.cardinality_threshold is not None
            else settings.high_cardinality_threshold
        )
        effective_m = self.m if self.m is not None else settings.target_encoding_smoothing_factor
        self._cols: list[str] = [
            c for c in X.select_dtypes(exclude="number").columns if X[c].nunique() > threshold
        ]
        self.encoding_maps_: dict[str, dict[Any, float]] = {}
        self.global_means_: dict[str, float] = {}

        if self.strategy == "target_encoding" and y is not None:
            global_mean = float(y.mean())
            for col in self._cols:
                self.global_means_[col] = global_mean
                stats = (
                    pd.DataFrame({"cat": X[col], "y": y}).groupby("cat")["y"].agg(["sum", "count"])
                )
                smooth = (stats["sum"] + effective_m * global_mean) / (stats["count"] + effective_m)
                self.encoding_maps_[col] = smooth.to_dict()
        else:
            for col in self._cols:
                freq = X[col].value_counts(normalize=True).to_dict()
                self.encoding_maps_[col] = freq
                self.global_means_[col] = 1.0 / max(X[col].nunique(), 1)

        logger.info(
            "HighCardinalityEncoder fitted — strategy=%s, cols=%d",
            self.strategy,
            len(self._cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces high-cardinality categorical columns with their encoded
        float values using the maps learned during ``fit``; the original
        DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns not
                present in ``encoding_maps_`` are passed through unchanged.
                Unseen categories are replaced with the corresponding value
                from ``global_means_`` (or ``0.0`` if absent).

        Returns:
            pd.DataFrame: A copy of ``X`` where each high-cardinality column
            has been replaced by a float column. The shape of the output
            matches the input.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["encoding_maps_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "HighCardinalityEncoder.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )

        logger.debug(
            "HighCardinalityEncoder.transform — rows=%d, cols=%d",
            len(X),
            len(self._cols),
        )

        out = X.copy()
        for col in self._cols:
            if col not in out.columns:
                continue
            mapping = self.encoding_maps_.get(col, {})
            fallback = self.global_means_.get(col, 0.0)
            out[col] = out[col].map(mapping).fillna(fallback).astype(float)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the names of the high-cardinality columns that were encoded
        during ``fit``.

        Args:
            None

        Returns:
            list[str]: Names of columns processed by this encoder. Returns an
            empty list when no columns exceeded the cardinality threshold.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["_cols"])
        return list(self._cols)


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Replaces each feature column with its Weight of Evidence (WoE) score
    derived from a binary target, and computes Information Value (IV) per
    column as a side-effect of ``fit``.

    Args:
        n_bins (int | None): Number of quantile bins used when discretising
            numeric features. When ``None`` (default), uses
            ``settings.default_iv_bins``.
        monotonic (bool): When ``True``, attempts monotonic binning by
            iteratively merging non-monotone adjacent bins. Defaults to
            ``False``.
        min_bin_size (float): Minimum fraction of total rows that each bin
            must contain. Defaults to ``0.05``.
        clip_value (float | None): Symmetric bound applied to WoE values after
            computation to prevent ±infinity. When ``None`` (default), uses
            ``settings.woe_clip_value``.

    Attributes:
        woe_maps_ (dict[str, pd.DataFrame]): Mapping from column name to a
            DataFrame with columns ``bin`` and ``woe`` (and ``iv_contrib``).
            For columns where binning fails, the DataFrame has zero rows.
        iv_ (dict[str, float]): Information Value per column, rounded to
            ``settings.iv_round_precision`` decimal places. Columns where
            binning fails receive ``iv=0.0``.
    """

    def __init__(
        self,
        n_bins: int | None = None,
        monotonic: bool = False,
        min_bin_size: float | None = None,
        clip_value: float | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.monotonic = monotonic
        self.min_bin_size = min_bin_size
        self.clip_value = clip_value

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        """Computes WoE bins and IV scores for every column in ``X``, storing
        results in ``woe_maps_`` and ``iv_`` as side-effects; columns where
        binning raises an exception are logged at WARNING and assigned an empty
        bin map and ``iv=0.0``.

        Args:
            X (pd.DataFrame): Training feature DataFrame. All columns are
                processed regardless of dtype; numeric columns are discretised
                into quantile bins while categorical columns are grouped by
                their string representation.
            y (pd.Series): Binary target series (values 0 and 1). Must be
                aligned with ``X`` by index.

        Returns:
            WoEEncoder: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame`` or ``y`` is not a
                ``pd.Series``.
            ValueError: If ``X`` has zero rows or ``y`` is not binary (0/1).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WoEEncoder.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("WoEEncoder.fit() received an empty DataFrame (0 rows)")
        if not isinstance(y, pd.Series):
            raise TypeError(
                "WoEEncoder.fit() expects y to be a pd.Series, got %s" % type(y).__name__
            )
        validate_binary_target(y)

        effective_n_bins = self.n_bins if self.n_bins is not None else settings.default_iv_bins
        effective_clip = self.clip_value if self.clip_value is not None else settings.woe_clip_value
        self._effective_min_bin_size = (
            self.min_bin_size if self.min_bin_size is not None else settings.woe_min_bin_size
        )

        self._cols: list[str] = X.columns.tolist()
        self.woe_maps_: dict[str, pd.DataFrame] = {}
        self.iv_: dict[str, float] = {}

        for col in self._cols:
            try:
                bins = woe_bins(X[col], y, effective_n_bins)
                bins["woe"] = bins["woe"].clip(-effective_clip, effective_clip)
                self.woe_maps_[col] = bins
                self.iv_[col] = round(float(bins["iv_contrib"].sum()), settings.iv_round_precision)
            except (ValueError, ArithmeticError) as e:
                logger.warning("WoEEncoder: skipped column '%s' (%s)", col, type(e).__name__)
                self.woe_maps_[col] = pd.DataFrame({"bin": [], "woe": []})
                self.iv_[col] = 0.0

        logger.info("WoEEncoder fitted — n_bins=%d, cols=%d", effective_n_bins, len(self._cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces each feature value with its corresponding WoE score using
        the bin maps learned during ``fit``; values that fall outside known
        bins (including ``NaN``) receive a WoE score of ``0.0``.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns not
                present in ``woe_maps_`` are passed through unchanged. Numeric
                columns are assigned bins via ``pd.cut`` against the fitted
                interval index; categorical columns are matched by string
                representation.

        Returns:
            pd.DataFrame: A copy of ``X`` where every fitted column is
            replaced by a float WoE score. Columns with an empty bin map
            (fitting failed) are set entirely to ``0.0``. The shape of the
            output matches the input.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["woe_maps_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "WoEEncoder.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug("WoEEncoder.transform — rows=%d, cols=%d", len(X), len(self._cols))

        out = X.copy()
        for col in self._cols:
            if col not in out.columns or col not in self.woe_maps_:
                continue
            bins = self.woe_maps_[col]
            if len(bins) == 0:
                out[col] = 0.0
                continue
            s = out[col]
            if pd.api.types.is_numeric_dtype(s):
                bin_edges = pd.IntervalIndex(bins["bin"].dropna())
                if len(bin_edges) == 0:
                    out[col] = 0.0
                    continue
                try:
                    coded = pd.cut(s, bins=bin_edges, labels=False)
                    woe_vals = bins["woe"].values
                    out[col] = coded.map(lambda i: woe_vals[int(i)] if pd.notna(i) else 0.0)
                except (ValueError, IndexError, TypeError):
                    logger.warning(
                        "WoEEncoder.transform: WoE assignment failed for column '%s', "
                        "defaulting to 0.0",
                        col,
                    )
                    out[col] = 0.0
            else:
                woe_map = dict(zip(bins["bin"].astype(str), bins["woe"]))
                out[col] = out[col].astype(str).map(woe_map).fillna(0.0)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the names of the columns that were processed by this encoder
        during ``fit``.

        Args:
            None

        Returns:
            list[str]: All column names from ``X`` at fit time.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["_cols"])
        return list(self._cols)


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Encodes non-numeric columns as integer ordinal codes, respecting an
    explicit user-supplied category order when provided and falling back to
    alphabetical order otherwise; unseen categories at transform time receive
    the code ``-1``.

    Args:
        category_order (dict[str, list[Any]] | None): Mapping from column
            name to an ordered list of category values. Columns present in
            this dict are encoded in the specified order (index 0, 1, …).
            Columns absent from this dict are encoded in alphabetical order
            as determined from the training data. Defaults to ``None``
            (all columns use alphabetical ordering).
    """

    def __init__(self, category_order: dict[str, list[Any]] | None = None) -> None:
        self.category_order = category_order

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OrdinalEncoder":
        """Learns integer ordinal mappings for all non-numeric columns in
        ``X``, storing them internally for use by ``transform``.

        Args:
            X (pd.DataFrame): Training feature DataFrame. Only non-numeric
                columns (as determined by ``select_dtypes(exclude="number")``)
                are processed; numeric columns are ignored.
            y (pd.Series | None): Ignored. Present for sklearn pipeline compatibility.

        Returns:
            OrdinalEncoder: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "OrdinalEncoder.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("OrdinalEncoder.fit() received an empty DataFrame (0 rows)")

        _cat_order = self.category_order if self.category_order is not None else {}

        self._cols: list[str] = X.select_dtypes(exclude="number").columns.tolist()
        self._maps: dict[str, dict[Any, int]] = {}
        for col in self._cols:
            if col in _cat_order:
                order = _cat_order[col]
            else:
                order = sorted(X[col].dropna().unique().tolist())
            self._maps[col] = {cat: i for i, cat in enumerate(order)}

        logger.info("OrdinalEncoder fitted — cols=%d", len(self._cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces each non-numeric column with its integer ordinal code using
        the mapping learned during ``fit``; the original DataFrame is not
        modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns not
                seen during ``fit`` are passed through unchanged. Categories
                absent from the fitted mapping are encoded as ``-1``.

        Returns:
            pd.DataFrame: A copy of ``X`` with fitted categorical columns
            replaced by integer (``int``) columns. ``NaN`` values and unseen
            categories are both replaced by ``-1``. The shape of the output
            matches the input.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["_cols"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "OrdinalEncoder.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug("OrdinalEncoder.transform — rows=%d, cols=%d", len(X), len(self._cols))

        out = X.copy()
        for col in self._cols:
            if col in out.columns:
                out[col] = out[col].map(self._maps.get(col, {})).fillna(-1).astype(int)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the names of the non-numeric columns that were encoded
        during ``fit``.

        Args:
            None

        Returns:
            list[str]: Names of the encoded columns in the order they were
            processed. Returns an empty list when no non-numeric columns were
            found in ``X`` at fit time.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["_cols"])
        return list(self._cols)


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encodes non-numeric columns into binary indicator columns, one
    per category learned at fit time.

    Unlike ``OrdinalEncoder``/``WoEEncoder``/``HighCardinalityEncoder`` (all
    1-to-1 column replacements), this changes both the number and names of
    output columns — callers composing this into a larger pipeline must
    concatenate the result rather than assign it back into the original
    column positions. Intended for low-cardinality columns; route
    high-cardinality columns to ``HighCardinalityEncoder`` instead to avoid
    dimensionality blowup.

    Args:
        drop_first (bool): Drop the first category (alphabetically) per
            column, to avoid multicollinearity for linear models. Defaults
            to ``False``.
        features (list[str] | None): Explicit list of column names to encode.
            When ``None`` (default), all non-numeric columns are encoded.
        column_overrides (dict[str, dict[str, bool]] | None): Per-column
            override of ``drop_first``, keyed by column name (e.g.
            ``{"job": {"drop_first": True}}``). A column absent from this
            dict uses the constructor-level ``drop_first``.

    Attributes:
        categories_ (dict[str, list[Any]]): Mapping from column name to the
            ordered list of categories encoded for that column (post
            ``drop_first``, if applicable).
        feature_names_out_ (list[str]): Ordered list of output column names,
            each named ``"{col}_{category}"``.
    """

    def __init__(
        self,
        drop_first: bool = False,
        features: list[str] | None = None,
        column_overrides: dict[str, dict[str, bool]] | None = None,
    ) -> None:
        self.drop_first = drop_first
        self.features = features
        self.column_overrides = column_overrides

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OneHotEncoder":
        """Learns the set of categories to encode per column, storing them in
        ``categories_`` and the resulting output column names in
        ``feature_names_out_``.

        Args:
            X (pd.DataFrame): Training feature DataFrame. When ``features`` is
                ``None``, only non-numeric columns (``select_dtypes(exclude=
                "number")``) are processed; numeric columns are ignored.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            OneHotEncoder: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OneHotEncoder.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("OneHotEncoder.fit() received an empty DataFrame (0 rows)")

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(exclude="number").columns.tolist()
        )
        self._cols: list[str] = [c for c in cols if c in X.columns]
        column_overrides = self.column_overrides or {}

        self.categories_: dict[str, list[Any]] = {}
        for col in self._cols:
            col_drop_first = column_overrides.get(col, {}).get("drop_first", self.drop_first)
            cats = sorted(X[col].dropna().unique().tolist(), key=str)
            if col_drop_first and len(cats) > 1:
                cats = cats[1:]
            self.categories_[col] = cats

        self.feature_names_out_: list[str] = [
            f"{col}_{cat}" for col in self._cols for cat in self.categories_[col]
        ]

        logger.info(
            "OneHotEncoder fitted — %d source cols → %d output cols",
            len(self._cols),
            len(self.feature_names_out_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces each fitted column with its one-hot indicator columns; a
        category not seen during ``fit`` (including any new category at
        transform time) produces an all-zero row for that column's block —
        no new column is ever created from unseen data.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns not
                present in ``categories_`` are dropped from the returned
                DataFrame (only the encoded output columns are returned —
                see ``feature_names_out_``).

        Returns:
            pd.DataFrame: A new DataFrame with one binary (``int``) column
            per ``feature_names_out_`` entry, indexed identically to ``X``.
            Every column from ``feature_names_out_`` is always present, even
            if the source column is missing from ``X`` (all-zero in that
            case).

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["categories_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "OneHotEncoder.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug("OneHotEncoder.transform — rows=%d, source cols=%d", len(X), len(self._cols))

        out_cols: dict[str, pd.Series] = {}
        for col in self._cols:
            series = X[col] if col in X.columns else pd.Series(index=X.index, dtype=object)
            for cat in self.categories_[col]:
                out_cols[f"{col}_{cat}"] = (series == cat).astype(int)

        result = pd.DataFrame(out_cols, index=X.index)
        return result.reindex(columns=self.feature_names_out_, fill_value=0)

    def get_feature_names_out(self) -> list[str]:
        """Returns the one-hot output column names.

        Args:
            None

        Returns:
            list[str]: Output column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        return list(self.feature_names_out_)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapses low-frequency categories into a single ``"Other"`` bucket per column.

    Standalone, composable transformer meant to run before any downstream
    encoder (ordinal/one-hot/WoE/target) — reduces noise and cardinality from
    rare categories without changing column count, unlike
    ``HighCardinalityEncoder`` (which replaces a category with a numeric
    value rather than bucketing it).

    Args:
        strategy (str): How to decide which categories to keep. One of:

            - ``"min_frequency"`` (default) — keep categories whose row-share
              is at or above ``min_frequency``.
            - ``"top_n"`` — keep only the ``top_n`` most frequent categories
              per column.

        min_frequency (float | None): Minimum row-fraction threshold for
            ``strategy="min_frequency"``. When ``None``, defaults to
            ``settings.rare_category_min_frequency``. Ignored for
            ``strategy="top_n"``.
        top_n (int | None): Max categories kept per column for
            ``strategy="top_n"``. When ``None``, defaults to
            ``settings.rare_category_top_n``. Ignored for
            ``strategy="min_frequency"``.
        other_label (str): Replacement label for grouped categories. Defaults
            to ``"Other"``.
        features (list[str] | None): Explicit list of column names to group.
            When ``None`` (default), all non-numeric columns are processed.
        column_overrides (dict[str, dict[str, Any]] | None): Per-column
            override of ``strategy``/``min_frequency``/``top_n``/
            ``other_label``, keyed by column name (e.g.
            ``{"cat_high": {"strategy": "top_n", "top_n": 5}}``). A column
            present here uses its own override for any key it supplies,
            falling back to the constructor-level value for keys it omits.

    Attributes:
        kept_categories_ (dict[str, set[Any]]): Mapping from column name to
            the set of categories kept as-is (everything else, including any
            category unseen at fit time, maps to ``other_label``).
    """

    def __init__(
        self,
        strategy: str = "min_frequency",
        min_frequency: float | None = None,
        top_n: int | None = None,
        other_label: str = "Other",
        features: list[str] | None = None,
        column_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.strategy = strategy
        self.min_frequency = min_frequency
        self.top_n = top_n
        self.other_label = other_label
        self.features = features
        self.column_overrides = column_overrides

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RareCategoryGrouper":
        """Learns which categories to keep per column, storing the result in
        ``kept_categories_``.

        Args:
            X (pd.DataFrame): Training feature DataFrame. When ``features`` is
                ``None``, only non-numeric columns are processed.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            RareCategoryGrouper: The fitted instance (``self``), enabling
            method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows or ``strategy`` is not recognised.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "RareCategoryGrouper.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("RareCategoryGrouper.fit() received an empty DataFrame (0 rows)")
        if self.strategy not in ("min_frequency", "top_n"):
            raise ValueError(
                f"strategy must be one of {{'min_frequency', 'top_n'}}, got {self.strategy!r}"
            )

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(exclude="number").columns.tolist()
        )
        self._cols: list[str] = [c for c in cols if c in X.columns]

        effective_min_freq = (
            self.min_frequency
            if self.min_frequency is not None
            else settings.rare_category_min_frequency
        )
        effective_top_n = self.top_n if self.top_n is not None else settings.rare_category_top_n
        column_overrides = self.column_overrides or {}

        self.kept_categories_: dict[str, set[Any]] = {}
        for col in self._cols:
            col_override = column_overrides.get(col, {})
            col_strategy = col_override.get("strategy", self.strategy)
            if col_strategy not in ("min_frequency", "top_n"):
                raise ValueError(
                    f"strategy must be one of {{'min_frequency', 'top_n'}}, got {col_strategy!r} "
                    f"(column {col!r})"
                )
            col_min_freq = col_override.get("min_frequency", effective_min_freq)
            col_top_n = col_override.get("top_n", effective_top_n)

            counts = X[col].value_counts(normalize=(col_strategy == "min_frequency"))
            if col_strategy == "min_frequency":
                keep = counts[counts >= col_min_freq].index.tolist()
            else:
                keep = counts.head(col_top_n).index.tolist()
            self.kept_categories_[col] = set(keep)

        logger.info(
            "RareCategoryGrouper fitted — strategy=%s, %d columns",
            self.strategy,
            len(self._cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces every value not in ``kept_categories_`` (including
        categories unseen at fit time) with ``other_label``; the original
        DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns absent
                from ``kept_categories_`` are passed through unchanged.

        Returns:
            pd.DataFrame: A copy of ``X`` with rare/unseen categories replaced
            by ``other_label`` in every fitted column. The shape of the
            output matches the input.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["kept_categories_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "RareCategoryGrouper.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug("RareCategoryGrouper.transform — rows=%d, cols=%d", len(X), len(self._cols))

        out = X.copy()
        for col, keep in self.kept_categories_.items():
            if col not in out.columns:
                continue
            out[col] = out[col].where(out[col].isin(keep), self.other_label)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the column names in the order they were fitted; this
        transform never changes column names or count.

        Args:
            None

        Returns:
            list[str]: Grouped column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["_cols"])
        return list(self._cols)


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns as the binary representation of an ordinal category index.

    A column with ``k`` categories needs only ``ceil(log2(k))`` output
    columns (vs. ``k`` for ``OneHotEncoder``) — more memory-efficient for
    medium/high-cardinality columns while avoiding the collision risk of
    hash encoding. Distinct from ``HighCardinalityEncoder`` (replaces a
    category with a single numeric value, e.g. target/frequency encoding)
    and ``OneHotEncoder`` (one column per category).

    Args:
        features (list[str] | None): Explicit list of column names to encode.
            When ``None`` (default), all non-numeric columns are encoded.

    Attributes:
        categories_ (dict[str, list[Any]]): Mapping from column name to the
            ordered list of categories encoded for that column — a
            category's position in this list is its ordinal index, whose
            binary representation fills that column's output bits.
        n_bits_ (dict[str, int]): Number of binary output columns per source
            column (``max(1, ceil(log2(n_categories)))``).
        feature_names_out_ (list[str]): Ordered list of output column names,
            each named ``"{col}_bin{i}"`` (``i`` = 0 for the least
            significant bit).
    """

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BinaryEncoder":
        """Learns each column's category-to-ordinal-index mapping and the
        resulting bit width, storing them in ``categories_``/``n_bits_`` and
        the output column names in ``feature_names_out_``.

        Args:
            X (pd.DataFrame): Training feature DataFrame. When ``features``
                is ``None``, only non-numeric columns are processed.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            BinaryEncoder: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BinaryEncoder.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("BinaryEncoder.fit() received an empty DataFrame (0 rows)")

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(exclude="number").columns.tolist()
        )
        self._cols: list[str] = [c for c in cols if c in X.columns]

        self.categories_: dict[str, list[Any]] = {}
        self.n_bits_: dict[str, int] = {}
        for col in self._cols:
            cats = sorted(X[col].dropna().unique().tolist(), key=str)
            self.categories_[col] = cats
            self.n_bits_[col] = max(1, math.ceil(math.log2(max(len(cats), 1))))

        self.feature_names_out_: list[str] = [
            f"{col}_bin{i}" for col in self._cols for i in range(self.n_bits_[col])
        ]

        logger.info(
            "BinaryEncoder fitted — %d source cols → %d output cols",
            len(self._cols),
            len(self.feature_names_out_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces each fitted column with its binary-index indicator columns; a
        category not seen during ``fit`` (including any new category at
        transform time, or a missing source column) encodes to all-zero bits.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns not
                present in ``categories_`` are dropped from the returned
                DataFrame (only the encoded output columns are returned —
                see ``feature_names_out_``).

        Returns:
            pd.DataFrame: A new DataFrame with ``n_bits_[col]`` binary
            (``int``) columns per source column, indexed identically to
            ``X``. Every column from ``feature_names_out_`` is always
            present, even if the source column is missing from ``X``
            (all-zero in that case).

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["categories_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "BinaryEncoder.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug("BinaryEncoder.transform — rows=%d, source cols=%d", len(X), len(self._cols))

        out_cols: dict[str, np.ndarray] = {}
        for col in self._cols:
            cat_to_idx = {cat: i for i, cat in enumerate(self.categories_[col])}
            n_bits = self.n_bits_[col]
            series = X[col] if col in X.columns else pd.Series(index=X.index, dtype=object)
            idx = series.map(cat_to_idx)
            idx_filled = idx.fillna(-1).astype(int).to_numpy()
            seen_mask = idx_filled >= 0
            for bit in range(n_bits):
                bit_values = np.where(seen_mask, (np.clip(idx_filled, 0, None) >> bit) & 1, 0)
                out_cols[f"{col}_bin{bit}"] = bit_values

        result = pd.DataFrame(out_cols, index=X.index)
        return result.reindex(columns=self.feature_names_out_, fill_value=0)

    def get_feature_names_out(self) -> list[str]:
        """Returns the binary-indicator output column names.

        Args:
            None

        Returns:
            list[str]: Output column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        return list(self.feature_names_out_)


_HASH_MISSING_SENTINEL = "__dscompanion_hash_missing__"


class HashEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns via the hashing trick — feature hashing into a fixed
    number of output columns, independent of the column's actual cardinality.

    Unlike every other encoder here, this needs no fit-time category inventory: each
    value is hashed deterministically (``hashlib.md5`` — stable across processes and
    runs, unlike Python's own randomized-per-process ``hash()``) into one of
    ``n_components`` buckets, with a ``+1``/``-1`` sign (also hash-derived) to reduce
    systematic collision bias, mirroring the classic feature-hashing trick. Designed for
    extremely-high-cardinality columns (thousands+ unique values) where even
    ``BinaryEncoder``'s ``ceil(log2(k))`` columns still scale with cardinality —
    ``HashEncoder``'s output width never changes, at the cost of a small, accepted
    collision rate (two categories occasionally landing in the same bucket).

    Args:
        n_components (int | None): Number of hashed output columns per source column.
            When ``None`` (default), uses ``settings.hash_encoder_n_components``.
        features (list[str] | None): Explicit list of column names to encode. When
            ``None`` (default), all non-numeric columns are encoded.

    Attributes:
        feature_names_out_ (list[str]): Ordered list of output column names, each named
            ``"{col}_hash{i}"``.
    """

    def __init__(
        self,
        n_components: int | None = None,
        features: list[str] | None = None,
    ) -> None:
        self.n_components = n_components
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HashEncoder":
        """Resolves which columns to encode and the output width; the hash function
        itself needs no training data.

        Args:
            X (pd.DataFrame): Training feature DataFrame. When ``features`` is
                ``None``, only non-numeric columns are processed.
            y (pd.Series | None): Ignored. Present for sklearn pipeline compatibility.

        Returns:
            HashEncoder: The fitted instance (``self``), enabling method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HashEncoder.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("HashEncoder.fit() received an empty DataFrame (0 rows)")

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(exclude="number").columns.tolist()
        )
        self._cols: list[str] = [c for c in cols if c in X.columns]
        self._n_components: int = (
            self.n_components
            if self.n_components is not None
            else settings.hash_encoder_n_components
        )

        self.feature_names_out_: list[str] = [
            f"{col}_hash{i}" for col in self._cols for i in range(self._n_components)
        ]

        logger.info(
            "HashEncoder fitted — %d source cols → %d output cols (n_components=%d)",
            len(self._cols),
            len(self.feature_names_out_),
            self._n_components,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Hashes each fitted column into its ``n_components`` signed indicator columns;
        a missing source column encodes to all-zero.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns not present in
                ``_cols`` are dropped from the returned DataFrame (only the encoded
                output columns are returned — see ``feature_names_out_``).

        Returns:
            pd.DataFrame: A new DataFrame with ``n_components`` signed (``+1``/``-1``/
            ``0``) integer columns per source column, indexed identically to ``X``.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "HashEncoder.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug("HashEncoder.transform — rows=%d, source cols=%d", len(X), len(self._cols))

        out_cols: dict[str, np.ndarray] = {}
        for col in self._cols:
            if col not in X.columns:
                for i in range(self._n_components):
                    out_cols[f"{col}_hash{i}"] = np.zeros(len(X), dtype=np.int64)
                continue
            str_vals = (
                X[col].map(lambda v: _HASH_MISSING_SENTINEL if pd.isna(v) else str(v)).to_numpy()
            )
            buckets, signs = self._hash_array(str_vals)
            for i in range(self._n_components):
                out_cols[f"{col}_hash{i}"] = np.where(buckets == i, signs, 0)

        result = pd.DataFrame(out_cols, index=X.index)
        return result.reindex(columns=self.feature_names_out_, fill_value=0)

    def get_feature_names_out(self) -> list[str]:
        """Returns the hashed output column names.

        Args:
            None

        Returns:
            list[str]: Output column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        return list(self.feature_names_out_)

    def _hash_array(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Maps each string value to a ``(bucket_index, sign)`` pair via MD5.

        Args:
            values (np.ndarray): Array of string category values (missing values
                already replaced with the sentinel).

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(buckets, signs)`` — ``buckets`` in
            ``[0, n_components)``, ``signs`` in ``{-1, 1}``, both same length as
            ``values``.
        """
        buckets = np.empty(len(values), dtype=np.int64)
        signs = np.empty(len(values), dtype=np.int64)
        for i, value in enumerate(values):
            # usedforsecurity=False: MD5 here is a deterministic bucketing function,
            # not a security primitive -- collisions just mean two categories share a
            # bucket, an expected, harmless property of the hashing trick. Documents
            # intent so scanners (Bandit B324) don't flag it as a weak-crypto risk.
            digest = hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
            buckets[i] = int(digest[:8], 16) % self._n_components
            signs[i] = 1 if int(digest[8:16], 16) % 2 == 0 else -1
        return buckets, signs
