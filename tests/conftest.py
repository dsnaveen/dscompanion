"""Shared pytest fixtures for the dscompanion test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscompanion.split import DataSplit, DataSplitter


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """5000-row synthetic financial dataset with known properties.

    Properties:
        - Binary target (70/30 class imbalance)
        - snapshot_date spanning 24 months
        - customer_id (1000 unique, repeated)
        - 10 numeric features (2 with 10% missingness, 1 near-zero variance)
        - 3 categorical features (low/high/ordinal cardinality)
        - 1 date feature (account_open_date)
        - 1 constant column (all zeros)
        - 1 column highly correlated with another (r > 0.9)
        - 1 column highly correlated with target (potential leakage, r > 0.95)
    """
    rng = np.random.RandomState(42)
    n = 5000

    # Temporal column
    start = pd.Timestamp("2022-01-01")
    snapshot_date = pd.date_range(start, periods=n, freq="3h")[:n]

    customer_id = rng.choice(np.arange(1000), size=n)

    # ── Target ───────────────────────────────────────────────────────────────
    target = rng.choice([0, 1], size=n, p=[0.70, 0.30])

    # ── Numeric features ─────────────────────────────────────────────────────
    f1 = rng.randn(n) + target * 0.5  # mildly correlated with target
    f2 = rng.randn(n)
    f3 = rng.randn(n)
    f4 = rng.randn(n)
    f5 = rng.randn(n)
    f6 = rng.randn(n)
    f7 = rng.randn(n)
    f8 = rng.randn(n)

    # Near-zero variance
    f9 = np.full(n, 0.001) + rng.randn(n) * 0.0001

    # Highly correlated with f1 (r > 0.9)
    f10 = f1 * 0.95 + rng.randn(n) * 0.1

    # Leakage column: very highly correlated with target
    leakage_col = target + rng.randn(n) * 0.05

    # Add 10% missingness to f4 and f5
    f4[rng.choice(n, size=int(0.1 * n), replace=False)] = np.nan
    f5[rng.choice(n, size=int(0.1 * n), replace=False)] = np.nan

    # ── Categorical features ─────────────────────────────────────────────────
    cat_low = rng.choice(["A", "B", "C"], size=n)  # 3 levels
    cat_high = rng.choice([f"cat_{i}" for i in range(80)], size=n)  # 80 levels
    cat_ordinal = rng.choice(["low", "medium", "high"], size=n)

    # ── Date feature ─────────────────────────────────────────────────────────
    account_open_date = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        rng.randint(0, 730, size=n), unit="D"
    )

    # ── Constant column ──────────────────────────────────────────────────────
    constant_col = np.zeros(n)

    df = pd.DataFrame(
        {
            "snapshot_date": snapshot_date,
            "customer_id": customer_id,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "f5": f5,
            "f6": f6,
            "f7": f7,
            "f8": f8,
            "f9_near_zero": f9,
            "f10_corr_f1": f10,
            "leakage_col": leakage_col,
            "cat_low": cat_low,
            "cat_high": cat_high,
            "cat_ordinal": cat_ordinal,
            "account_open_date": account_open_date,
            "constant_col": constant_col,
            "target": target,
        }
    )
    return df


@pytest.fixture(scope="session")
def data_split(synthetic_df: pd.DataFrame) -> DataSplit:
    """Temporal DataSplit on the synthetic dataset."""
    splitter = DataSplitter(
        strategy="temporal",
        test_size=0.2,
        oot_cutoff="2023-09-01",
        date_col="snapshot_date",
        group_col="customer_id",
        target_col="target",
    )
    return splitter.fit_split(synthetic_df)


@pytest.fixture
def numeric_split(data_split: DataSplit) -> DataSplit:
    """A numeric-only, NaN-free subset of the shared synthetic split.

    Several app-level tests (interactive_step8's leaderboard worker,
    interactive_step11's calibrator) call ``model.fit``/``model.evaluate``/
    ``model.predict_proba`` directly — they need a split whose feature
    columns are all numeric with no missing values, unlike the full
    ``data_split`` fixture (which includes categoricals, dates, and two
    columns with 10% missingness by design).
    """
    numeric_cols = ["f1", "f2", "f3", "f6", "f7", "f8"]
    return DataSplit(
        train_X=data_split.train_X[numeric_cols],
        train_y=data_split.train_y,
        val_X=data_split.val_X[numeric_cols],
        val_y=data_split.val_y,
        test_X=data_split.test_X[numeric_cols],
        test_y=data_split.test_y,
        oot_X=data_split.oot_X[numeric_cols],
        oot_y=data_split.oot_y,
        metadata=data_split.metadata,
    )
