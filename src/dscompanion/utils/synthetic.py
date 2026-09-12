"""Synthetic data generator for team training and pipeline testing."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

__all__ = ["SyntheticDataGenerator"]


class SyntheticDataGenerator:
    """Generate synthetic DataFrames with controlled data quality characteristics.

    Produces datasets containing specific proportions of high-null, low-variance,
    constant, high-cardinality, and normally-distributed features.  Useful for
    team training sessions and for testing feature selection and preprocessing
    pipelines against known ground-truth properties.

    Args:
        n_rows: Number of rows to generate.
        n_numeric: Total number of numeric features.
        n_categorical: Total number of categorical features.
        random_state: Seed for reproducibility.  Uses a local ``np.random.RandomState``
            so the global numpy seed is never modified.
        numeric_distribution: Explicit counts per numeric feature type.  Keys:
            ``high_null``, ``low_variance``, ``constant``, ``normal``.
            Must sum to ``n_numeric``.  If ``None``, defaults are used and the
            ``normal`` bucket absorbs any remainder.
        categorical_distribution: Explicit counts per categorical feature type.
            Keys: ``single_value``, ``high_cardinality``, ``dominant_category``,
            ``high_null``, ``normal``.  Must sum to ``n_categorical``.  If ``None``,
            defaults are used and the ``normal`` bucket absorbs any remainder.

    Raises:
        ValueError: If a provided distribution dict does not sum to the
            corresponding ``n_numeric`` / ``n_categorical`` total.

    Examples:
        >>> gen = SyntheticDataGenerator(n_rows=1000, n_numeric=100, n_categorical=50)
        >>> df = gen.generate()
        >>> df.shape
        (1000, 150)

        >>> gen = SyntheticDataGenerator(
        ...     n_rows=5000,
        ...     n_numeric=200,
        ...     numeric_distribution={'high_null': 40, 'low_variance': 20,
        ...                           'constant': 10, 'normal': 130},
        ... )
        >>> df = gen.generate()
    """

    def __init__(
        self,
        n_rows: int = 10_000,
        n_numeric: int = 2_000,
        n_categorical: int = 500,
        random_state: int | None = 42,
        numeric_distribution: dict[str, int] | None = None,
        categorical_distribution: dict[str, int] | None = None,
    ) -> None:
        self.n_rows = n_rows
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

        if numeric_distribution is None:
            # ~5% high_null, ~2.5% low_variance, ~1% constant, rest normal
            _hn = round(n_numeric * 0.05)
            _lv = round(n_numeric * 0.025)
            _ct = round(n_numeric * 0.01)
            _no = n_numeric - _hn - _lv - _ct
            if _no < 0:
                _hn = _lv = _ct = 0
                _no = n_numeric
            self.numeric_distribution = {
                "high_null": _hn,
                "low_variance": _lv,
                "constant": _ct,
                "normal": _no,
            }
        else:
            self.numeric_distribution = numeric_distribution

        if categorical_distribution is None:
            # ~4% single_value, ~10% high_card, ~10% dominant, ~16% high_null, rest normal
            _sv = round(n_categorical * 0.04)
            _hc = round(n_categorical * 0.10)
            _dc = round(n_categorical * 0.10)
            _hn = round(n_categorical * 0.16)
            _no = n_categorical - _sv - _hc - _dc - _hn
            if _no < 0:
                _sv = _hc = _dc = _hn = 0
                _no = n_categorical
            self.categorical_distribution = {
                "single_value": _sv,
                "high_cardinality": _hc,
                "dominant_category": _dc,
                "high_null": _hn,
                "normal": _no,
            }
        else:
            self.categorical_distribution = categorical_distribution

        self._validate_distributions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate the synthetic dataset.

        Returns:
            DataFrame with ``n_numeric + n_categorical`` columns and ``n_rows`` rows.
        """
        logger.info(
            "Generating synthetic data — rows=%d, numeric=%d, categorical=%d",
            self.n_rows,
            self.n_numeric,
            self.n_categorical,
        )

        numeric = self._generate_numeric_features()
        categorical = self._generate_categorical_features()

        df = pd.DataFrame({**numeric, **categorical})
        logger.info("Synthetic dataset ready — shape %s", df.shape)
        return df

    def get_metadata(self) -> dict[str, Any]:
        """Return configuration metadata for the generator.

        Returns:
            dict with keys: ``n_rows``, ``n_numeric``, ``n_categorical``,
            ``total_features``, ``random_state``, ``numeric_distribution``,
            ``categorical_distribution``.
        """
        return {
            "n_rows": self.n_rows,
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "total_features": self.n_numeric + self.n_categorical,
            "random_state": self.random_state,
            "numeric_distribution": self.numeric_distribution,
            "categorical_distribution": self.categorical_distribution,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_distributions(self) -> None:
        numeric_sum = sum(self.numeric_distribution.values())
        if numeric_sum != self.n_numeric:
            raise ValueError(
                f"numeric_distribution sums to {numeric_sum}, expected {self.n_numeric}"
            )
        categorical_sum = sum(self.categorical_distribution.values())
        if categorical_sum != self.n_categorical:
            raise ValueError(
                f"categorical_distribution sums to {categorical_sum}, expected {self.n_categorical}"
            )

    def _generate_numeric_features(self) -> dict[str, Any]:
        features: dict[str, np.ndarray] = {}
        idx = 0

        n_high_null = self.numeric_distribution["high_null"]
        logger.debug("Numeric high_null=%d", n_high_null)
        for _ in range(n_high_null):
            null_rate = self._rng.uniform(0.5, 0.9)
            vals = self._rng.randn(self.n_rows) * 10 + 50
            vals[self._rng.rand(self.n_rows) < null_rate] = np.nan
            features[f"num_high_null_{idx}"] = vals
            idx += 1

        n_low_var = self.numeric_distribution["low_variance"]
        logger.debug("Numeric low_variance=%d", n_low_var)
        for _ in range(n_low_var):
            base = self._rng.uniform(0, 100)
            features[f"num_low_var_{idx}"] = base + self._rng.randn(self.n_rows) * 0.01
            idx += 1

        n_constant = self.numeric_distribution["constant"]
        logger.debug("Numeric constant=%d", n_constant)
        for _ in range(n_constant):
            features[f"num_constant_{idx}"] = np.full(self.n_rows, self._rng.uniform(0, 100))
            idx += 1

        n_normal = self.numeric_distribution["normal"]
        logger.debug("Numeric normal=%d", n_normal)
        for _ in range(n_normal):
            mean = self._rng.uniform(-100, 100)
            std = self._rng.uniform(1, 50)
            vals = self._rng.randn(self.n_rows) * std + mean
            if self._rng.rand() < 0.3:
                null_rate = self._rng.uniform(0.01, 0.1)
                vals[self._rng.rand(self.n_rows) < null_rate] = np.nan
            features[f"num_normal_{idx}"] = vals
            idx += 1

        return features

    def _generate_categorical_features(self) -> dict[str, Any]:
        features: dict[str, np.ndarray] = {}
        idx = 0

        n_single = self.categorical_distribution["single_value"]
        logger.debug("Categorical single_value=%d", n_single)
        for i in range(n_single):
            features[f"cat_single_{idx}"] = np.full(self.n_rows, f"constant_{i}")
            idx += 1

        n_high_card = self.categorical_distribution["high_cardinality"]
        logger.debug("Categorical high_cardinality=%d", n_high_card)
        for i in range(n_high_card):
            n_unique = int(self.n_rows * 0.96)
            pool = [f"id_{i}_{j}" for j in range(n_unique)]
            vals = pool + [pool[0]] * (self.n_rows - n_unique)
            self._rng.shuffle(vals)
            features[f"cat_high_card_{idx}"] = vals
            idx += 1

        n_dominant = self.categorical_distribution["dominant_category"]
        logger.debug("Categorical dominant_category=%d", n_dominant)
        for _ in range(n_dominant):
            dominant_rate = self._rng.uniform(0.90, 0.98)
            n_dom = int(self.n_rows * dominant_rate)
            others = list(self._rng.choice([f"cat_{j}" for j in range(5)], self.n_rows - n_dom))
            vals = ["dominant"] * n_dom + others
            self._rng.shuffle(vals)
            features[f"cat_dominant_{idx}"] = vals
            idx += 1

        n_high_null = self.categorical_distribution["high_null"]
        logger.debug("Categorical high_null=%d", n_high_null)
        for _ in range(n_high_null):
            null_rate = self._rng.uniform(0.5, 0.9)
            n_cats = self._rng.randint(5, 20)
            vals = self._rng.choice([f"cat_{j}" for j in range(n_cats)], self.n_rows).astype(object)
            vals[self._rng.rand(self.n_rows) < null_rate] = None
            features[f"cat_high_null_{idx}"] = vals
            idx += 1

        n_normal = self.categorical_distribution["normal"]
        logger.debug("Categorical normal=%d", n_normal)
        for _ in range(n_normal):
            n_cats = self._rng.randint(5, 50)
            cats = [f"cat_{j}" for j in range(n_cats)]
            probs = self._rng.dirichlet(np.ones(n_cats))
            vals = self._rng.choice(cats, self.n_rows, p=probs)
            if self._rng.rand() < 0.2:
                null_rate = self._rng.uniform(0.01, 0.1)
                vals = vals.astype(object)
                vals[self._rng.rand(self.n_rows) < null_rate] = None
            features[f"cat_normal_{idx}"] = vals
            idx += 1

        return features
