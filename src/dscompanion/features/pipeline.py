"""FeatureProcessingPipeline: chains all feature transformers end-to-end."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings
from dscompanion.features.date_features import DateFeatureExtractor
from dscompanion.features.distribution import DistributionTransformer
from dscompanion.features.encoder import (
    HighCardinalityEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    WoEEncoder,
)
from dscompanion.features.imputer import SmartImputer
from dscompanion.features.leakage_guard import LeakageGuard, LeakageReport
from dscompanion.features.scaler import SmartScaler, WinsorizationTransformer

logger = logging.getLogger(__name__)

__all__ = ["FeatureProcessingPipeline"]

_DATE_PATTERNS = ("date", "_dt", "_ts")
_AUDIT_COLS = ["original_col", "original_dtype", "transformations_applied", "output_col"]


def _auto_detect_cols(X: pd.DataFrame, high_card_threshold: int):
    """Classifies every column in ``X`` into numeric, categorical, date, and
    high-cardinality categorical lists by inspecting dtypes and column name
    patterns; does not modify ``X``.

    Args:
        X (pd.DataFrame): DataFrame whose columns are to be classified.
            Columns whose dtype is ``datetime64`` or whose name contains one
            of the patterns ``"date"``, ``"_dt"``, or ``"_ts"`` (case-
            insensitive) are placed in the date bucket; remaining numeric
            dtypes go to the numeric bucket; everything else goes to the
            categorical bucket.
        high_card_threshold (int): Minimum number of unique values for a
            categorical column to be placed in the high-cardinality list.
            Columns at or below this threshold are not included.

    Returns:
        tuple: A four-tuple ``(numeric, categorical, date, high_card)`` where
        each element is a ``list[str]`` of column names. ``high_card`` is a
        subset of ``categorical``. Any of the four lists may be empty if no
        columns fall into that category.
    """
    numeric, categorical, date = [], [], []
    for col in X.columns:
        dtype = X[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype) or any(
            pat in col.lower() for pat in _DATE_PATTERNS
        ):
            date.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric.append(col)
        else:
            categorical.append(col)

    high_card = [c for c in categorical if X[c].nunique() > high_card_threshold]
    return numeric, categorical, date, high_card


class FeatureProcessingPipeline(BaseEstimator, TransformerMixin):
    """Chains date-feature extraction, imputation, optional leakage checking,
    categorical encoding, and numeric scaling into a single sklearn-compatible
    transformer; column-type detection is automatic when the corresponding
    ``*_cols`` arguments are left as ``None``.

    The fit-time processing order is: (1) date extraction, (2) optional
    winsorization, (3) imputation, (4) leakage check, (5) categorical
    encoding (low- and high-cardinality columns handled separately),
    (6) optional distribution transform, (7) numeric scaling. Winsorization
    runs before imputation so percentile boundaries are computed from the
    true raw distribution; the distribution transform runs after encoding
    (which only touches categorical columns) and before scaling, so the
    scaler sees the normalised numeric distribution rather than the raw
    skewed one.

    Args:
        numeric_cols (list[str] | None): Explicit list of numeric column
            names. When ``None``, numeric columns are auto-detected from the
            DataFrame dtypes after date columns have been removed.
        categorical_cols (list[str] | None): Explicit list of categorical
            column names. When ``None``, columns are auto-detected as those
            that are neither numeric nor date.
        date_cols (list[str] | None): Explicit list of date or datetime
            column names. When ``None``, columns are detected by
            ``datetime64`` dtype or by ``"date"``, ``"_dt"``, ``"_ts"``
            name patterns.
        high_cardinality_cols (list[str] | None): Explicit list of
            high-cardinality categorical columns routed to
            ``HighCardinalityEncoder``. When ``None``, columns are identified
            by comparing ``nunique()`` against
            ``settings.high_cardinality_threshold``.
        imputer (SmartImputer | None): A pre-configured ``SmartImputer``
            instance. When ``None``, a default ``SmartImputer()`` is
            constructed automatically.
        encoder (str): Encoding strategy for low-cardinality categorical
            columns. One of ``"woe"`` (default, requires ``y``),
            ``"target_encoding"`` (requires ``y``), ``"onehot"``, or
            ``"ordinal"``. ``"onehot"`` changes the output column count/names
            (see ``OneHotEncoder``) — handled specially in
            ``_apply_cat_transform``.
        scaler (str): Scaling strategy forwarded to ``SmartScaler``. One of
            ``"robust"`` (default), ``"standard"``, ``"minmax"``,
            or ``"none"``.
        distribution (str): Distribution-normalising transform applied to
            numeric columns before scaling, forwarded to
            ``DistributionTransformer``. One of ``"none"`` (default),
            ``"log"``, ``"log1p"``, or ``"yeo_johnson"``.
        run_leakage_check (bool): When ``True`` (default), runs
            ``LeakageGuard`` after imputation and raises ``LeakageError`` if
            critical correlation is detected.
        winsorize (bool): When ``True``, caps numeric column outliers via
            ``WinsorizationTransformer`` before imputation. Defaults to
            ``False``.
        winsorize_lower (float): Lower-tail fraction forwarded to
            ``WinsorizationTransformer``. Defaults to ``0.01``. Ignored
            when ``winsorize=False``.
        winsorize_upper (float): Upper-tail fraction forwarded to
            ``WinsorizationTransformer``. Defaults to ``0.01``. Ignored
            when ``winsorize=False``.

    Attributes:
        leakage_report_ (LeakageReport | None): The ``LeakageReport``
            produced during the most recent ``fit`` call. ``None`` when
            ``run_leakage_check=False`` or when ``y`` was not provided.
        feature_names_out_ (list[str]): Ordered list of output column names
            populated after ``fit`` completes.
    """

    def __init__(
        self,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        date_cols: list[str] | None = None,
        high_cardinality_cols: list[str] | None = None,
        imputer: SmartImputer | None = None,
        encoder: str = "woe",
        scaler: str = "robust",
        distribution: str = "none",
        run_leakage_check: bool = True,
        winsorize: bool = False,
        winsorize_lower: float = 0.01,
        winsorize_upper: float = 0.01,
    ) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.date_cols = date_cols
        self.high_cardinality_cols = high_cardinality_cols
        self.imputer = imputer
        self.encoder = encoder
        self.scaler = scaler
        self.distribution = distribution
        self.run_leakage_check = run_leakage_check
        self.winsorize = winsorize
        self.winsorize_lower = winsorize_lower
        self.winsorize_upper = winsorize_upper

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureProcessingPipeline":
        """Fits every stage of the pipeline on the training data in sequence,
        populating ``leakage_report_``, ``feature_names_out_``, and an
        internal audit table as side-effects; ``X`` is not modified in-place.

        The fit order is: date feature extraction, optional winsorization,
        imputation, leakage check (if enabled), categorical encoding,
        optional distribution transform, and numeric scaling. After each
        step the working copy of ``X`` is updated so that downstream steps
        see the transformed representation.

        Args:
            X (pd.DataFrame): Training feature DataFrame containing raw
                numeric, categorical, and date columns. The DataFrame is not
                required to be sorted or deduplicated before calling this
                method.
            y (pd.Series | None): Target series aligned with ``X`` by
                index. Required for ``encoder="woe"`` and
                ``encoder="target_encoding"``; required for the leakage check
                when ``run_leakage_check=True``. When ``None``, WoE/target
                encoding falls back to ``OrdinalEncoder`` and the leakage
                check is skipped. Not required for ``encoder="onehot"``.

        Returns:
            FeatureProcessingPipeline: The fitted instance (``self``),
            enabling method chaining inside sklearn ``Pipeline`` objects.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
            LeakageError: When ``run_leakage_check=True``, ``y`` is not
                ``None``, and at least one numeric feature exceeds the
                critical correlation threshold configured in
                ``settings.target_leakage_correlation_threshold``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureProcessingPipeline.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("FeatureProcessingPipeline.fit() received an empty DataFrame (0 rows)")

        threshold = settings.high_cardinality_threshold
        num, cat, date, high_card = _auto_detect_cols(X, threshold)

        self._numeric_cols = self.numeric_cols if self.numeric_cols is not None else num
        self._categorical_cols = self.categorical_cols if self.categorical_cols is not None else cat
        self._date_cols = self.date_cols if self.date_cols is not None else date
        self._high_card_cols = (
            self.high_cardinality_cols if self.high_cardinality_cols is not None else high_card
        )

        # Step 1: date features (before imputer so downstream sees numeric age cols)
        self._date_extractor: DateFeatureExtractor | None = None
        if self._date_cols:
            self._date_extractor = DateFeatureExtractor(
                date_cols=self._date_cols, drop_original=True
            )
            self._date_extractor.fit(X)
            X = self._date_extractor.transform(X)
            # Re-detect numeric after date expansion
            self._numeric_cols = [
                c
                for c in X.select_dtypes(include="number").columns
                if c not in self._categorical_cols
            ]

        # Step 1.5: winsorization (optional, before imputer so percentile
        # boundaries are computed from the raw distribution, not imputed fill values)
        self._winsorizer: WinsorizationTransformer | None = None
        if self.winsorize:
            self._winsorizer = WinsorizationTransformer(
                lower=self.winsorize_lower, upper=self.winsorize_upper
            )
            self._winsorizer.fit(X)
            X = self._winsorizer.transform(X)

        # Step 2: imputation
        self._imputer = self.imputer if self.imputer is not None else SmartImputer()
        self._imputer.fit(X)
        X = self._imputer.transform(X)

        # Step 3: leakage check (post-impute, pre-encode)
        self.leakage_report_: LeakageReport | None = None
        if self.run_leakage_check and y is not None:
            guard = LeakageGuard(raise_on_critical=True)
            target_col = y.name or "__target__"
            self.leakage_report_ = guard.check(X, y, target_col=str(target_col))
            logger.info(
                "Leakage check — %d critical, %d warnings",
                self.leakage_report_.n_critical,
                self.leakage_report_.n_warnings,
            )

        # Step 4: categorical encoding
        self._low_card_cols = [c for c in self._categorical_cols if c not in self._high_card_cols]
        self._encoder: BaseEstimator | None = None
        self._high_card_encoder: HighCardinalityEncoder | None = None

        if self._high_card_cols:
            self._high_card_encoder = HighCardinalityEncoder(
                strategy=(
                    "target_encoding"
                    if self.encoder in ("woe", "target_encoding")
                    else "frequency_encoding"
                )
            )
            self._high_card_encoder.fit(X[self._high_card_cols], y)

        if self._low_card_cols:
            if self.encoder == "woe" and y is not None:
                self._encoder = WoEEncoder()
                self._encoder.fit(X[self._low_card_cols], y)
            elif self.encoder == "target_encoding" and y is not None:
                self._encoder = HighCardinalityEncoder(strategy="target_encoding")
                self._encoder.fit(X[self._low_card_cols], y)
            elif self.encoder == "onehot":
                self._encoder = OneHotEncoder()
                self._encoder.fit(X[self._low_card_cols])
            else:
                self._encoder = OrdinalEncoder()
                self._encoder.fit(X[self._low_card_cols])

        if self._encoder is not None:
            X = self._apply_cat_transform(X)

        if self._high_card_encoder is not None:
            X = self._apply_high_card_transform(X)

        # Step 4.5: distribution transform (optional, after encoding so it
        # only ever sees numeric columns, before scaling so the scaler sees
        # the normalised distribution)
        self._distribution: DistributionTransformer | None = None
        if self.distribution != "none":
            self._distribution = DistributionTransformer(strategy=self.distribution)
            self._distribution.fit(X)
            X = self._distribution.transform(X)

        # Step 5: scaling
        self._scaler = SmartScaler(strategy=self.scaler)
        self._scaler.fit(X)
        X = self._scaler.transform(X)

        self.feature_names_out_: list[str] = X.columns.tolist()
        self._audit: pd.DataFrame = self._build_audit()

        logger.info(
            "FeatureProcessingPipeline fitted — %d output features",
            len(self.feature_names_out_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies all fitted transformers in sequence to produce the final
        numeric feature matrix; the input DataFrame is not modified in-place
        and no refitting occurs.

        The transform order mirrors the fit order: date extraction, imputation,
        categorical encoding (low- then high-cardinality), distribution
        transform, and scaling. The leakage check is skipped during transform.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. May be the
                training set, a validation set, or an out-of-time (OOT) sample.
                The DataFrame must contain the same column names as were
                present at fit time; additional columns are passed through
                unchanged; columns that were present at fit time but are
                missing here are silently skipped by each sub-transformer.

        Returns:
            pd.DataFrame: A transformed DataFrame whose column set matches
            ``feature_names_out_``. All values are numeric floats suitable
            for model input. The shape is ``(n_rows, len(feature_names_out_))``.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureProcessingPipeline.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )

        logger.debug(
            "FeatureProcessingPipeline.transform — rows=%d, output cols=%d",
            len(X),
            len(self.feature_names_out_),
        )

        if self._date_extractor is not None:
            X = self._date_extractor.transform(X)

        if self._winsorizer is not None:
            X = self._winsorizer.transform(X)

        X = self._imputer.transform(X)

        if self._encoder is not None:
            X = self._apply_cat_transform(X)

        if self._high_card_encoder is not None:
            X = self._apply_high_card_transform(X)

        if self._distribution is not None:
            X = self._distribution.transform(X)

        X = self._scaler.transform(X)
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverses the scaling step only, restoring numeric columns to their original scale.

        Date extraction, categorical encoding, and imputation are one-way
        transformations and are intentionally **not** reversed — only the
        ``SmartScaler`` step is inverted, via ``SmartScaler.inverse_transform``.
        Intended for showing original-scale feature values in diagnostics
        (e.g. SHAP scatter plots) built on top of the pipeline's transformed
        output.

        Args:
            X (pd.DataFrame): A DataFrame in the pipeline's transformed
                (post-``transform()``) representation — i.e. already
                date-extracted, imputed, encoded, and scaled.

        Returns:
            pd.DataFrame: A copy of ``X`` with the fitted numeric columns
            restored to their pre-scaling values. Encoded/imputed columns
            are returned unchanged.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureProcessingPipeline.inverse_transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        return self._scaler.inverse_transform(X)

    def get_feature_names_out(self) -> list[str]:
        """Returns the ordered list of column names that the pipeline outputs
        after a successful ``fit`` call.

        Args:
            None

        Returns:
            list[str]: Column names of the transformed DataFrame in the order
            they appear after all pipeline stages complete. The list reflects
            all date-derived expansion columns, missing-indicator columns,
            encoded categorical columns, and scaled numeric columns.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        return list(self.feature_names_out_)

    def audit_report(self) -> pd.DataFrame:
        """Returns a per-column record of every transformation applied during
        ``fit``, enabling traceability from input columns to output columns.

        Args:
            None

        Returns:
            pd.DataFrame: A copy of the internal audit table with four string
            columns — ``original_col`` (source column name), ``original_dtype``
            (one of ``"numeric"``, ``"date"``, ``"categorical_high_card"``,
            ``"categorical_low_card"``), ``transformations_applied`` (a
            comma-separated string listing each stage, e.g.
            ``"date_extract,impute,scale(robust)"``), and ``output_col``
            (the column name in the transformed DataFrame). Date columns
            produce nine rows each (one per derived feature). Returns an
            empty DataFrame with the correct column schema if the pipeline
            has not produced any output columns.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["_audit"])
        return self._audit.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_cat_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self._low_card_cols if c in X.columns]
        if not cols:
            return X
        transformed = self._encoder.transform(X[cols])
        if isinstance(self._encoder, OneHotEncoder):
            # Unlike every other encoder here, one-hot changes both the
            # column count and names — replace-in-place (out[cols] = ...)
            # doesn't apply; drop the originals and concat the new columns.
            out = X.drop(columns=cols)
            return pd.concat([out, transformed], axis=1)
        out = X.copy()
        out[cols] = transformed[cols] if isinstance(transformed, pd.DataFrame) else transformed
        return out

    def _apply_high_card_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self._high_card_cols if c in X.columns]
        if not cols:
            return X
        out = X.copy()
        transformed = self._high_card_encoder.transform(X[cols])
        out[cols] = transformed[cols] if isinstance(transformed, pd.DataFrame) else transformed
        return out

    def _build_audit(self) -> pd.DataFrame:
        numeric_stages = "impute"
        if self.distribution != "none":
            numeric_stages += ",distribution(%s)" % self.distribution
        numeric_stages += ",scale(%s)" % self.scaler

        rows = []
        for col in self._numeric_cols:
            rows.append(
                {
                    "original_col": col,
                    "original_dtype": "numeric",
                    "transformations_applied": numeric_stages,
                    "output_col": col,
                }
            )
        for col in self._date_cols:
            for suffix in [
                "age_days",
                "age_months",
                "year",
                "month",
                "day_of_week",
                "quarter",
                "month_sin",
                "month_cos",
                "is_weekend",
            ]:
                rows.append(
                    {
                        "original_col": col,
                        "original_dtype": "date",
                        "transformations_applied": "date_extract," + numeric_stages,
                        "output_col": "%s_%s" % (col, suffix),
                    }
                )
        for col in self._high_card_cols:
            rows.append(
                {
                    "original_col": col,
                    "original_dtype": "categorical_high_card",
                    "transformations_applied": "impute,target_encoding",
                    "output_col": col,
                }
            )
        for col in self._low_card_cols:
            if self.encoder == "onehot" and isinstance(self._encoder, OneHotEncoder):
                for cat in self._encoder.categories_.get(col, []):
                    rows.append(
                        {
                            "original_col": col,
                            "original_dtype": "categorical_low_card",
                            "transformations_applied": "impute,onehot_encoding",
                            "output_col": "%s_%s" % (col, cat),
                        }
                    )
            else:
                rows.append(
                    {
                        "original_col": col,
                        "original_dtype": "categorical_low_card",
                        "transformations_applied": "impute,%s_encoding" % self.encoder,
                        "output_col": col,
                    }
                )
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_AUDIT_COLS)
