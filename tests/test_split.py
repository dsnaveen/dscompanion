"""Tests for dscompanion.split."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from dscompanion.split import DataSplit, DataSplitter


class TestGroupedSplitWithOOT:
    """``data_split`` fixture: strategy="grouped" with an independent oot_cutoff.

    Covers generic DataSplit/metadata behavior (date ranges, non-overlap,
    size bookkeeping, summary) that doesn't depend on which strategy
    produced the split — see TestTemporalSnapshotSplit below for the
    snapshot-based temporal-strategy-specific tests.
    """

    def test_all_oot_dates_after_cutoff(self, data_split: DataSplit):
        if len(data_split.oot_X) == 0:
            pytest.skip("No OOT rows in this split")
        # snapshot_date was dropped at split time; check metadata
        assert data_split.metadata.date_ranges is not None
        oot_min = data_split.metadata.date_ranges.get("oot", (None, None))[0]
        assert oot_min is not None
        assert pd.to_datetime(oot_min) >= pd.to_datetime("2023-09-01")

    def test_splits_are_non_overlapping(self, data_split: DataSplit):
        all_idx = []
        for _, (X, _) in data_split.all_splits().items():
            all_idx.extend(X.index.tolist())
        assert len(all_idx) == len(set(all_idx)), "Overlapping indices between splits"

    def test_metadata_split_sizes_match_actual(self, data_split: DataSplit):
        for name, (X, _) in data_split.all_splits().items():
            assert data_split.metadata.split_sizes[name] == len(X)

    def test_summary_returns_nonempty_string(self, data_split: DataSplit):
        s = data_split.summary()
        assert isinstance(s, str)
        assert len(s) > 0


class TestStratifiedSplit:
    def test_class_rate_within_5pct(self, synthetic_df):
        splitter = DataSplitter(
            strategy="stratified",
            test_size=0.2,
            val_size=0.1,
            target_col="target",
        )
        split = splitter.fit_split(synthetic_df)
        train_rate = split.train_y.mean()
        for name, y in [("val", split.val_y), ("test", split.test_y)]:
            assert (
                abs(train_rate - y.mean()) < 0.05
            ), f"Class rate diff in {name}: {abs(train_rate - y.mean()):.3f}"

    def test_no_overlap(self, synthetic_df):
        splitter = DataSplitter(strategy="stratified", target_col="target")
        split = splitter.fit_split(synthetic_df)
        train_idx = set(split.train_X.index)
        val_idx = set(split.val_X.index)
        test_idx = set(split.test_X.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)


class TestStratifiedSplitRegression:
    """strategy='stratified' with a continuous (regression) target.

    Found via real end-to-end regression testing: StratifiedShuffleSplit
    requires every class to have >= 2 members, but a continuous target has
    (essentially) as many "classes" as rows — every regression pipeline
    using the default split.method='stratified' crashed with a cryptic
    sklearn ValueError. Mirrors _grouped_split()'s existing
    classification/regression branch: falls back to a plain random
    permutation split when y doesn't look classification-shaped.
    """

    @pytest.fixture
    def synthetic_regression_df(self) -> pd.DataFrame:
        rng = np.random.RandomState(11)
        n = 500
        f1 = rng.randn(n)
        f2 = rng.randn(n)
        target = 2.0 * f1 - f2 + rng.randn(n) * 0.1
        return pd.DataFrame({"f1": f1, "f2": f2, "target": target})

    def test_does_not_raise_on_continuous_target(self, synthetic_regression_df):
        splitter = DataSplitter(
            strategy="stratified", test_size=0.2, val_size=0.1, target_col="target"
        )
        split = splitter.fit_split(synthetic_regression_df)
        assert len(split.train_X) > 0
        assert len(split.val_X) > 0
        assert len(split.test_X) > 0

    def test_no_overlap(self, synthetic_regression_df):
        splitter = DataSplitter(strategy="stratified", target_col="target")
        split = splitter.fit_split(synthetic_regression_df)
        train_idx = set(split.train_X.index)
        val_idx = set(split.val_X.index)
        test_idx = set(split.test_X.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_split_sizes_approximately_match_config(self, synthetic_regression_df):
        splitter = DataSplitter(
            strategy="stratified", test_size=0.2, val_size=0.1, target_col="target"
        )
        split = splitter.fit_split(synthetic_regression_df)
        n = len(synthetic_regression_df)
        assert abs(len(split.test_X) / n - 0.2) < 0.02
        # val_size applies to the train+val remainder (0.8 * n), not the full n.
        remainder = n - len(split.test_X)
        assert abs(len(split.val_X) / remainder - 0.1) < 0.02

    def test_deterministic_with_fixed_random_state(self, synthetic_regression_df):
        split_a = DataSplitter(
            strategy="stratified", target_col="target", random_state=42
        ).fit_split(synthetic_regression_df)
        split_b = DataSplitter(
            strategy="stratified", target_col="target", random_state=42
        ).fit_split(synthetic_regression_df)
        assert list(split_a.train_X.index) == list(split_b.train_X.index)


class TestGroupedSplit:
    def test_no_customer_leakage(self, synthetic_df):
        splitter = DataSplitter(
            strategy="grouped",
            test_size=0.2,
            val_size=0.1,
            group_col="customer_id",
            target_col="target",
        )
        split = splitter.fit_split(synthetic_df)
        # customer_id column is in X
        if "customer_id" in split.train_X.columns and "customer_id" in split.test_X.columns:
            train_customers = set(split.train_X["customer_id"])
            test_customers = set(split.test_X["customer_id"])
            assert train_customers.isdisjoint(
                test_customers
            ), "Customer IDs appear in both train and test"

    def test_group_test_values_none_matches_omitted(self, synthetic_df):
        """Explicitly passing group_test_values=None must behave identically to
        omitting it — the additive param must not change today's default behavior."""
        kwargs = dict(
            strategy="grouped",
            test_size=0.2,
            val_size=0.1,
            group_col="customer_id",
            target_col="target",
        )
        split_omitted = DataSplitter(**kwargs).fit_split(synthetic_df)
        split_explicit_none = DataSplitter(**kwargs, group_test_values=None).fit_split(synthetic_df)
        assert list(split_omitted.train_X.index) == list(split_explicit_none.train_X.index)
        assert list(split_omitted.test_X.index) == list(split_explicit_none.test_X.index)

    def test_deliberate_split_by_category_values(self, synthetic_df):
        splitter = DataSplitter(
            strategy="grouped",
            val_size=0.2,
            group_col="cat_low",
            group_test_values={"A"},
            target_col="target",
        )
        split = splitter.fit_split(synthetic_df)
        assert (split.test_X["cat_low"] == "A").all()
        assert not (split.train_X["cat_low"] == "A").any()
        assert not (split.val_X["cat_low"] == "A").any()

    def test_deliberate_split_carves_val_via_stratified(self, synthetic_df):
        splitter = DataSplitter(
            strategy="grouped",
            val_size=0.2,
            group_col="cat_low",
            group_test_values={"A"},
            target_col="target",
        )
        split = splitter.fit_split(synthetic_df)
        train_val_rate = pd.concat([split.train_y, split.val_y]).mean()
        assert abs(train_val_rate - split.val_y.mean()) < 0.05

    def test_unknown_group_test_values_raises_without_leaking_values(self, synthetic_df):
        splitter = DataSplitter(
            strategy="grouped",
            group_col="cat_low",
            group_test_values={"not_a_real_category"},
            target_col="target",
        )
        with pytest.raises(ValueError) as exc_info:
            splitter.fit_split(synthetic_df)
        assert "not_a_real_category" not in str(exc_info.value)

    def test_group_test_values_matching_every_row_raises(self, synthetic_df):
        splitter = DataSplitter(
            strategy="grouped",
            group_col="cat_low",
            group_test_values={"A", "B", "C"},
            target_col="target",
        )
        with pytest.raises(ValueError, match="no rows left for"):
            splitter.fit_split(synthetic_df)


class TestValidation:
    def test_string_target_does_not_crash(self):
        # Regression: _check_class_ratio() and the class_rates dict both called
        # .mean() on the target Series, which raises TypeError for non-numeric
        # dtypes. A string-labeled binary target must split without error.
        rng = np.random.RandomState(0)
        n = 200
        df = pd.DataFrame(
            {
                "f1": rng.randn(n),
                "f2": rng.randn(n),
                "label": rng.choice(["yes", "no"], size=n),
            }
        )
        splitter = DataSplitter(strategy="stratified", target_col="label")
        split = splitter.fit_split(df)
        assert len(split.train_X) > 0
        assert split.metadata.class_rates is None

    def test_missing_target_raises(self, synthetic_df):
        splitter = DataSplitter(strategy="stratified", target_col="nonexistent")
        with pytest.raises(ValueError, match="Target column"):
            splitter.fit_split(synthetic_df)

    def test_temporal_without_date_col_raises(self, synthetic_df):
        splitter = DataSplitter(strategy="temporal")
        with pytest.raises(ValueError, match="date_col"):
            splitter.fit_split(synthetic_df)

    def test_tiny_oot_raises(self):
        # 2 distinct snapshot values; newest has only 3 rows (< splitter_oot_min_rows)
        rng = np.random.RandomState(0)
        n_old, n_new = 200, 3
        df = pd.DataFrame(
            {
                "period": ["2024-01"] * n_old + ["2024-02"] * n_new,
                "f1": rng.randn(n_old + n_new),
                "target": rng.choice([0, 1], size=n_old + n_new),
            }
        )
        splitter = DataSplitter(strategy="temporal", date_col="period", target_col="target")
        with pytest.raises(ValueError, match="OOT"):
            splitter.fit_split(df)


class TestTemporalSnapshotSplit:
    """Snapshot-based temporal strategy: date_col holds a small number of

    discrete values (e.g. monthly/quarterly periods), not a per-row
    continuous timestamp. Newest value is always OOT; automatic assignment
    covers 2-4 distinct values, ``date_value_roles`` covers any count
    (required for 5+).
    """

    @staticmethod
    def _make_df(period_counts: dict, seed: int = 0) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        periods, targets, f1 = [], [], []
        for period, n in period_counts.items():
            periods.extend([period] * n)
            targets.extend(rng.choice([0, 1], size=n).tolist())
            f1.extend(rng.randn(n).tolist())
        return pd.DataFrame({"period": periods, "f1": f1, "target": targets})

    def test_one_distinct_value_raises(self):
        df = self._make_df({"2024-01": 100})
        splitter = DataSplitter(strategy="temporal", date_col="period", target_col="target")
        with pytest.raises(ValueError, match="at least 2 distinct values"):
            splitter.fit_split(df)

    def test_five_distinct_values_without_roles_raises(self):
        df = self._make_df({f"2024-0{i}": 100 for i in range(1, 6)})
        splitter = DataSplitter(strategy="temporal", date_col="period", target_col="target")
        with pytest.raises(ValueError, match="date_value_roles"):
            splitter.fit_split(df)

    def test_two_values_pools_train_test_val_from_oldest(self):
        df = self._make_df({"2024-01": 200, "2024-02": 80})
        splitter = DataSplitter(
            strategy="temporal", date_col="period", target_col="target", test_size=0.2, val_size=0.1
        )
        split = splitter.fit_split(df)

        assert set(split.oot_X.index) == set(df.index[df["period"] == "2024-02"])
        oldest_idx = set(df.index[df["period"] == "2024-01"])
        pooled_idx = set(split.train_X.index) | set(split.val_X.index) | set(split.test_X.index)
        assert pooled_idx == oldest_idx
        assert len(split.test_y) == max(1, int(200 * 0.2))
        assert len(split.val_y) == max(1, int((200 - len(split.test_y)) * 0.1))

    def test_three_values_val_whole_pool_train_test(self):
        df = self._make_df({"2024-01": 200, "2024-02": 60, "2024-03": 80})
        splitter = DataSplitter(strategy="temporal", date_col="period", target_col="target")
        split = splitter.fit_split(df)

        assert set(split.oot_X.index) == set(df.index[df["period"] == "2024-03"])
        assert set(split.val_X.index) == set(df.index[df["period"] == "2024-02"])
        oldest_idx = set(df.index[df["period"] == "2024-01"])
        assert set(split.train_X.index) | set(split.test_X.index) == oldest_idx

    def test_four_values_all_whole_no_randomness(self):
        df = self._make_df({"2024-01": 100, "2024-02": 90, "2024-03": 80, "2024-04": 70})
        splitter = DataSplitter(strategy="temporal", date_col="period", target_col="target")
        split = splitter.fit_split(df)

        assert set(split.train_X.index) == set(df.index[df["period"] == "2024-01"])
        assert set(split.test_X.index) == set(df.index[df["period"] == "2024-02"])
        assert set(split.val_X.index) == set(df.index[df["period"] == "2024-03"])
        assert set(split.oot_X.index) == set(df.index[df["period"] == "2024-04"])

    def test_explicit_date_value_roles_overrides_automatic(self):
        # Only 3 distinct values, but explicitly assign differently than the
        # automatic k=3 table (val whole) would.
        df = self._make_df({"2024-01": 60, "2024-02": 60, "2024-03": 80})
        splitter = DataSplitter(
            strategy="temporal",
            date_col="period",
            target_col="target",
            date_value_roles={
                "train": ["2024-01"],
                "test": ["2024-02"],
                "oot": ["2024-03"],
            },
        )
        split = splitter.fit_split(df)

        assert set(split.train_X.index) == set(df.index[df["period"] == "2024-01"])
        assert set(split.test_X.index) == set(df.index[df["period"] == "2024-02"])
        assert set(split.oot_X.index) == set(df.index[df["period"] == "2024-03"])
        assert len(split.val_X) == 0

    def test_date_value_roles_unknown_value_raises(self):
        df = self._make_df({"2024-01": 60, "2024-02": 80})
        splitter = DataSplitter(
            strategy="temporal",
            date_col="period",
            target_col="target",
            date_value_roles={"train": ["2024-01"], "oot": ["2099-01"]},
        )
        with pytest.raises(ValueError, match="not present"):
            splitter.fit_split(df)

    def test_date_value_roles_missing_oot_raises(self):
        df = self._make_df({"2024-01": 60, "2024-02": 80})
        splitter = DataSplitter(
            strategy="temporal",
            date_col="period",
            target_col="target",
            date_value_roles={"train": ["2024-01"], "test": ["2024-02"]},
        )
        with pytest.raises(ValueError, match="'oot'"):
            splitter.fit_split(df)

    def test_date_value_roles_uncovered_value_raises(self):
        df = self._make_df({"2024-01": 60, "2024-02": 60, "2024-03": 80})
        splitter = DataSplitter(
            strategy="temporal",
            date_col="period",
            target_col="target",
            date_value_roles={"train": ["2024-01"], "oot": ["2024-03"]},
        )
        with pytest.raises(ValueError, match="does not cover"):
            splitter.fit_split(df)

    def test_date_value_roles_duplicate_assignment_raises(self):
        with pytest.raises(ValueError, match="more than one role"):
            DataSplitter(
                strategy="temporal",
                date_col="period",
                target_col="target",
                date_value_roles={"train": ["2024-01"], "test": ["2024-01"], "oot": ["2024-02"]},
            ).fit_split(self._make_df({"2024-01": 60, "2024-02": 80}))


class TestSerialisation:
    def test_datasplit_joblib_roundtrip(self, data_split: DataSplit, tmp_path):
        path = tmp_path / "split.joblib"
        joblib.dump(data_split, path)
        loaded = joblib.load(path)
        pd.testing.assert_frame_equal(loaded.train_X, data_split.train_X)
        assert loaded.metadata.strategy == data_split.metadata.strategy
