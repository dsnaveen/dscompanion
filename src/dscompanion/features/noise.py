"""NoiseInjector: train-time-only noise augmentation for numeric and categorical columns."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

__all__ = ["NoiseInjector"]

_NUMERIC_NOISE_TYPES = {"gaussian", "uniform", "laplace"}
_CATEGORICAL_NOISE_TYPES = {"empirical", "uniform"}


class NoiseInjector(BaseEstimator, TransformerMixin):
    """Injects distribution-aware random noise into numeric and categorical columns —
    train-time data augmentation only, never applied to held-out data.

    Unlike every other transformer in this package, ``fit()`` and ``transform()`` never
    mutate data — ``fit()`` only learns per-column statistics, and ``transform()`` is an
    unconditional passthrough. ``fit_transform()`` is explicitly overridden (not sklearn's
    default ``fit().transform()`` composition) to be the only method that actually injects
    noise. This mirrors real-world data-augmentation practice (e.g. image augmentation):
    noise perturbs training data only, so validation/test/OOT data stays clean and
    deterministic. It also integrates transparently with ``FeatureTransformChain``, whose
    ``fit()`` calls ``fit_transform()`` for the training block and plain ``transform()`` for
    held-out data — no special-casing needed there.

    Numeric noise is additive, scaled by each column's own standard deviation (learned from
    training data only), so magnitude is relative to that column's natural spread rather than
    a fixed absolute value. Categorical noise replaces affected rows' values with another
    category drawn from the column's own observed categories only — never invents an unseen
    one.

    When used as a step inside ``FeatureTransformChain``, call the chain's own
    ``fit_transform()`` to retrieve noised training data — the chain's ``fit()`` alone (and
    any subsequent plain ``transform()``) never surfaces it, matching this class's own
    ``fit()``/``transform()`` contract of never mutating data outside ``fit_transform()``.

    Args:
        numeric_noise_type (str): Noise distribution for numeric columns. One of
            ``"gaussian"`` (default), ``"uniform"``, or ``"laplace"``.
        numeric_noise_scale (float | None): Fraction of each numeric column's own standard
            deviation used as noise magnitude. When ``None`` (default), uses
            ``settings.noise_numeric_scale``.
        categorical_noise_type (str): How replacement categories are drawn for categorical
            columns. ``"empirical"`` (default) draws proportional to the column's own
            observed category frequencies; ``"uniform"`` draws uniformly across observed
            unique values. Both only ever produce categories seen during ``fit()``.
        affected_row_frac (float | None): Fraction of rows perturbed per column,
            independently sampled per column. When ``None`` (default), uses
            ``settings.noise_affected_row_frac``.
        features (list[str] | None): Explicit list of column names to consider. When
            ``None`` (default), all numeric and non-numeric columns are candidates.
        random_state (int | None): Seed for the internal random generator. When ``None``
            (default), uses ``settings.random_state``.

    Attributes:
        num_cols_ (list[str]): Numeric columns eligible for noise (excludes zero-variance/
            all-NaN columns, logged when skipped).
        cat_cols_ (list[str]): Categorical columns eligible for noise (excludes columns with
            fewer than 2 observed categories, logged when skipped).
        col_std_ (dict[str, float]): Per-column standard deviation, numeric columns only.
        col_category_probs_ (dict[str, dict]): Per-column ``{category: probability}`` map
            learned from training data, categorical columns only.

    Raises:
        ValueError: If ``numeric_noise_type`` or ``categorical_noise_type`` is not
            recognised.
    """

    def __init__(
        self,
        numeric_noise_type: str = "gaussian",
        numeric_noise_scale: float | None = None,
        categorical_noise_type: str = "empirical",
        affected_row_frac: float | None = None,
        features: list[str] | None = None,
        random_state: int | None = None,
    ) -> None:
        self.numeric_noise_type = numeric_noise_type
        self.numeric_noise_scale = numeric_noise_scale
        self.categorical_noise_type = categorical_noise_type
        self.affected_row_frac = affected_row_frac
        self.features = features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NoiseInjector":
        """Learns per-column statistics needed to later inject noise; never mutates ``X``.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline compatibility.

        Returns:
            NoiseInjector: The fitted instance (``self``), enabling method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, or ``numeric_noise_type``/
                ``categorical_noise_type`` is not recognised.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NoiseInjector.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("NoiseInjector.fit() received an empty DataFrame (0 rows)")
        if self.numeric_noise_type not in _NUMERIC_NOISE_TYPES:
            raise ValueError(
                f"numeric_noise_type must be one of {_NUMERIC_NOISE_TYPES}, got "
                f"{self.numeric_noise_type!r}"
            )
        if self.categorical_noise_type not in _CATEGORICAL_NOISE_TYPES:
            raise ValueError(
                f"categorical_noise_type must be one of {_CATEGORICAL_NOISE_TYPES}, got "
                f"{self.categorical_noise_type!r}"
            )

        candidate_cols = self.features if self.features is not None else X.columns.tolist()
        candidate_cols = [c for c in candidate_cols if c in X.columns]
        all_num_cols = X.select_dtypes(include="number").columns.tolist()
        num_candidates = [c for c in candidate_cols if c in all_num_cols]
        cat_candidates = [c for c in candidate_cols if c not in all_num_cols]

        self.col_std_: dict[str, float] = {}
        self.num_cols_: list[str] = []
        for col in num_candidates:
            std = X[col].std()
            if pd.isna(std) or std == 0:
                logger.warning(
                    "NoiseInjector: column '%s' has zero variance or is all-NaN — skipping", col
                )
                continue
            self.col_std_[col] = float(std)
            self.num_cols_.append(col)

        self.col_category_probs_: dict[str, dict] = {}
        self.cat_cols_: list[str] = []
        for col in cat_candidates:
            freq = X[col].value_counts(normalize=True, dropna=True)
            if len(freq) < 2:
                logger.debug(
                    "NoiseInjector: column '%s' has fewer than 2 observed categories — skipping",
                    col,
                )
                continue
            self.col_category_probs_[col] = freq.to_dict()
            self.cat_cols_.append(col)

        effective_random_state = (
            self.random_state if self.random_state is not None else settings.random_state
        )
        self._rng = np.random.RandomState(effective_random_state)

        logger.info(
            "NoiseInjector fitted — %d numeric cols, %d categorical cols eligible for noise",
            len(self.num_cols_),
            len(self.cat_cols_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Unconditional passthrough — never injects noise.

        This is the method every held-out/validation/test/OOT call goes through, so that
        data stays clean and deterministic. Only ``fit_transform()`` actually injects noise.

        Args:
            X (pd.DataFrame): Feature DataFrame to pass through unchanged.

        Returns:
            pd.DataFrame: An exact copy of ``X``.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["num_cols_", "cat_cols_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "NoiseInjector.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        return X.copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fits on ``X`` and injects noise into a copy of it — training-time augmentation only.

        Explicitly overrides sklearn's default ``TransformerMixin.fit_transform()``
        (``fit().transform()`` composition), since ``transform()`` alone never injects
        noise. This is the only method on this class that perturbs data.

        Args:
            X (pd.DataFrame): Training feature DataFrame. Statistics are learned from this
                same data via ``fit()`` before noise is injected into it.
            y (pd.Series | None): Ignored. Present for sklearn pipeline compatibility.

        Returns:
            pd.DataFrame: A copy of ``X`` with noise injected into the fitted columns'
            sampled row subsets. Columns skipped during ``fit()`` (zero-variance numeric,
            fewer-than-2-category categorical) are unchanged.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, or ``numeric_noise_type``/
                ``categorical_noise_type`` is not recognised.
        """
        self.fit(X, y)

        effective_scale = (
            self.numeric_noise_scale
            if self.numeric_noise_scale is not None
            else settings.noise_numeric_scale
        )
        effective_affected_frac = (
            self.affected_row_frac
            if self.affected_row_frac is not None
            else settings.noise_affected_row_frac
        )

        out = X.copy()
        n_rows = len(X)

        for col in self.num_cols_:
            mask = self._rng.rand(n_rows) < effective_affected_frac
            n_affected = int(mask.sum())
            if n_affected == 0:
                continue
            scale = self.col_std_[col] * effective_scale
            if self.numeric_noise_type == "gaussian":
                noise = self._rng.normal(0.0, scale, size=n_affected)
            elif self.numeric_noise_type == "uniform":
                noise = self._rng.uniform(-scale, scale, size=n_affected)
            else:  # "laplace"
                noise = self._rng.laplace(0.0, scale, size=n_affected)
            out[col] = out[col].astype(float)
            affected_positions = np.where(mask)[0]
            col_idx = out.columns.get_loc(col)
            out.iloc[affected_positions, col_idx] = (
                out.iloc[affected_positions, col_idx].to_numpy() + noise
            )

        for col in self.cat_cols_:
            mask = self._rng.rand(n_rows) < effective_affected_frac
            n_affected = int(mask.sum())
            if n_affected == 0:
                continue
            categories = list(self.col_category_probs_[col].keys())
            if self.categorical_noise_type == "empirical":
                probs = list(self.col_category_probs_[col].values())
                replacements = self._rng.choice(categories, size=n_affected, p=probs)
            else:  # "uniform"
                replacements = self._rng.choice(categories, size=n_affected)
            affected_positions = np.where(mask)[0]
            col_idx = out.columns.get_loc(col)
            out.iloc[affected_positions, col_idx] = replacements

        logger.info(
            "NoiseInjector.fit_transform — rows=%d, %d numeric + %d categorical cols perturbed",
            n_rows,
            len(self.num_cols_),
            len(self.cat_cols_),
        )
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the fitted column names; this transform never changes names or count.

        Args:
            None

        Returns:
            list[str]: Numeric then categorical fitted column names.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["num_cols_", "cat_cols_"])
        return list(self.num_cols_) + list(self.cat_cols_)
