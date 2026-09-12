"""Date feature extractor: age, cyclical encoding, and calendar components."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

__all__ = ["DateFeatureExtractor"]

_SUFFIXES = [
    "age_days",
    "age_months",
    "year",
    "month",
    "day_of_week",
    "quarter",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]
_HOUR_SUFFIXES = ["hour_sin", "hour_cos"]


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts eleven numeric calendar and age features from each date or
    datetime column (thirteen when ``include_hour_cyclical=True``),
    optionally dropping the original date column as a side-effect of
    ``transform``.

    For each processed column the following features are generated:
    ``{col}_age_days``, ``{col}_age_months``, ``{col}_year``,
    ``{col}_month``, ``{col}_day_of_week``, ``{col}_quarter``,
    ``{col}_month_sin``, ``{col}_month_cos``, ``{col}_dow_sin``,
    ``{col}_dow_cos``, ``{col}_is_weekend``, and (when
    ``include_hour_cyclical=True``) ``{col}_hour_sin``, ``{col}_hour_cos``.

    Args:
        date_cols (list[str] | None): Explicit list of column names to
            process. When ``None``, columns are auto-detected by matching
            ``datetime64`` dtype or by finding the patterns ``"date"``,
            ``"_dt"``, or ``"_ts"`` (case-insensitive) in the column name.
        reference_date (str | pd.Timestamp | None): Anchor date used to
            compute age features (``age_days``, ``age_months``). When
            ``None``, the per-column maximum value observed in the training
            data is used as the reference. Accepts any value accepted by
            ``pd.to_datetime``.
        drop_original (bool): When ``True`` (default), the original date
            column is removed from the output after features are extracted.
            When ``False``, it is retained alongside the new feature columns.
        include_hour_cyclical (bool): When ``True``, also emits
            ``{col}_hour_sin``/``{col}_hour_cos`` (``sin``/``cos`` of
            ``2*pi*hour/24``). Defaults to ``False`` — most banking date
            columns carry no meaningful time-of-day component, so this is
            opt-in rather than auto-detected.
    """

    def __init__(
        self,
        date_cols: list[str] | None = None,
        reference_date: str | pd.Timestamp | None = None,
        drop_original: bool = True,
        include_hour_cyclical: bool = False,
    ) -> None:
        self.date_cols = date_cols
        self.reference_date = reference_date
        self.drop_original = drop_original
        self.include_hour_cyclical = include_hour_cyclical

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DateFeatureExtractor":
        """Identifies date columns to process and records the per-column
        reference date used for age calculations; stores results internally
        as a side-effect.

        Args:
            X (pd.DataFrame): Training feature DataFrame. When ``date_cols``
                was supplied at construction, those columns are used directly;
                otherwise columns are detected by dtype or name pattern.
                Only columns with a ``datetime64`` dtype contribute to the
                per-column reference date lookup; string-typed date columns
                use the fit-time snapshot stored in ``_default_ref_date``.
            y (pd.Series | None): Ignored. Present for sklearn pipeline compatibility.

        Returns:
            DateFeatureExtractor: The fitted instance (``self``), enabling
            method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DateFeatureExtractor.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("DateFeatureExtractor.fit() received an empty DataFrame (0 rows)")

        if self.date_cols is not None:
            self._cols: list[str] = self.date_cols
        else:
            self._cols = [
                c
                for c in X.columns
                if pd.api.types.is_datetime64_any_dtype(X[c])
                or any(pat in c.lower() for pat in ["date", "_dt", "_ts"])
            ]

        # Capture a deterministic fallback reference date at fit time so
        # transform() never calls pd.Timestamp.now() — a live-clock call
        # would produce different age values between train and OOT runs.
        self._default_ref_date: pd.Timestamp = pd.Timestamp.now()

        self._ref_dates: dict[str, pd.Timestamp] = {}
        for col in self._cols:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                ref = self.reference_date if self.reference_date is not None else X[col].max()
                self._ref_dates[col] = pd.to_datetime(ref)

        logger.info(
            "DateFeatureExtractor fitted — %d date cols: %s",
            len(self._cols),
            self._cols,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Expands each fitted date column into nine numeric feature columns
        and optionally removes the original date column; the input DataFrame
        is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Date columns
                absent from ``X`` at transform time are silently skipped.
                Non-parseable date values are coerced to ``NaT`` by
                ``pd.to_datetime(errors="coerce")``, producing ``NaN``
                in all derived columns for those rows. For columns without a
                reference date recorded during ``fit``, the fit-time snapshot
                ``_default_ref_date`` is used as a deterministic fallback.

        Returns:
            pd.DataFrame: A copy of ``X`` with nine new columns appended per
            processed date column. When ``drop_original=True`` the original
            date column is removed; when ``False`` it is retained. If no
            date columns were identified during ``fit``, the returned DataFrame
            is identical in content to the input.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["_cols"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DateFeatureExtractor.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug(
            "DateFeatureExtractor.transform — rows=%d, date cols=%d",
            len(X),
            len(self._cols),
        )

        out = X.copy()
        for col in self._cols:
            if col not in out.columns:
                continue
            s = pd.to_datetime(out[col], errors="coerce")
            ref = self._ref_dates.get(col, self._default_ref_date)

            delta = (ref - s).dt
            out[f"{col}_age_days"] = delta.days
            out[f"{col}_age_months"] = (delta.days / 30.44).round(0)
            out[f"{col}_year"] = s.dt.year
            out[f"{col}_month"] = s.dt.month
            out[f"{col}_day_of_week"] = s.dt.dayofweek
            out[f"{col}_quarter"] = s.dt.quarter
            out[f"{col}_month_sin"] = np.sin(2 * np.pi * s.dt.month / 12)
            out[f"{col}_month_cos"] = np.cos(2 * np.pi * s.dt.month / 12)
            out[f"{col}_dow_sin"] = np.sin(2 * np.pi * s.dt.dayofweek / 7)
            out[f"{col}_dow_cos"] = np.cos(2 * np.pi * s.dt.dayofweek / 7)
            out[f"{col}_is_weekend"] = (s.dt.dayofweek >= 5).astype(int)
            if self.include_hour_cyclical:
                out[f"{col}_hour_sin"] = np.sin(2 * np.pi * s.dt.hour / 24)
                out[f"{col}_hour_cos"] = np.cos(2 * np.pi * s.dt.hour / 24)

            if self.drop_original:
                out = out.drop(columns=[col])

        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the names of all date-derived feature columns that
        ``transform`` will produce, in the order they are generated.

        Args:
            None

        Returns:
            list[str]: Flat list of ``{col}_{suffix}`` names for every
            combination of fitted date column and the eleven fixed suffixes
            (thirteen when ``include_hour_cyclical=True``). Returns an empty
            list when no date columns were detected or when the extractor
            has not been fitted.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["_cols"])
        suffixes = _SUFFIXES + _HOUR_SUFFIXES if self.include_hour_cyclical else _SUFFIXES
        return [f"{col}_{s}" for col in self._cols for s in suffixes]
