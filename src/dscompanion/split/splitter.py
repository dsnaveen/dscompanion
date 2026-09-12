"""Data splitting utilities: DataSplit container and DataSplitter transformer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from dscompanion.config import settings

logger = logging.getLogger(__name__)

__all__ = ["DataSplitMetadata", "DataSplit", "DataSplitter"]

_VERSION = "0.1.0"


# ── Metadata model ───────────────────────────────────────────────────────────


class DataSplitMetadata(BaseModel):
    """Pydantic model capturing the provenance and shape of a completed data split.

    Instances are created automatically by ``DataSplitter.fit_split`` and
    stored on the resulting ``DataSplit``.  They are immutable once created.

    Attributes:
        strategy: Splitting strategy used, e.g. ``"temporal"``,
            ``"stratified"``, ``"grouped"``, or ``"random"``.
        target_col: Name of the target column extracted from the source
            DataFrame.
        n_features: Number of feature columns in each split (excludes target).
        split_sizes: Dict mapping split name (``"train"``, ``"val"``,
            ``"test"``, ``"oot"``) to its row count.  OOT is ``0`` when no
            OOT rows exist.
        class_rates: Dict mapping split name to the binary event rate (mean of
            the target).  ``None`` for regression tasks where the number of
            distinct target values exceeds 20.
        date_ranges: Dict mapping split name to a ``(min_date, max_date)``
            tuple of ISO strings.  ``None`` when ``date_col`` is not
            available or not set.
        created_at: Timestamp at which the split was created.  Defaults to the
            time of ``DataSplitter.fit_split`` call.
        dscompanion_version: Version string of the dscompanion package used to create
            this split.
    """

    strategy: str
    target_col: str
    n_features: int
    split_sizes: dict[str, int]
    class_rates: dict[str, float] | None = None
    date_ranges: dict[str, tuple] | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    dscompanion_version: str = _VERSION


# ── DataSplit container ──────────────────────────────────────────────────────


@dataclass
class DataSplit:
    """Immutable dataclass container holding all train/val/test/OOT data partitions.

    Created exclusively by ``DataSplitter.fit_split`` and consumed downstream
    by model trainers, evaluators, and tuners.  The OOT split is populated
    only when relevant (temporal strategy, or grouped strategy with
    ``oot_cutoff`` set); otherwise ``oot_X`` and ``oot_y`` are empty
    DataFrame/Series instances with zero rows.

    Attributes:
        train_X: Training feature DataFrame, shape ``(n_train, n_features)``.
        train_y: Training label Series, length ``n_train``.
        val_X: Validation feature DataFrame, shape ``(n_val, n_features)``.
        val_y: Validation label Series, length ``n_val``.
        test_X: Test feature DataFrame, shape ``(n_test, n_features)``.
        test_y: Test label Series, length ``n_test``.
        oot_X: Out-of-time feature DataFrame.  Empty DataFrame (zero rows)
            when no OOT split is applicable.
        oot_y: Out-of-time label Series.  Empty Series with ``dtype=float``
            when no OOT split is applicable.
        metadata: ``DataSplitMetadata`` instance describing the strategy,
            sizes, class rates, and date ranges for this split.
    """

    train_X: pd.DataFrame
    train_y: pd.Series
    val_X: pd.DataFrame
    val_y: pd.Series
    test_X: pd.DataFrame
    test_y: pd.Series
    oot_X: pd.DataFrame
    oot_y: pd.Series
    metadata: DataSplitMetadata

    # ── sklearn-style aliases (X_train / y_train) ────────────────────────────

    @property
    def X_train(self) -> pd.DataFrame:
        return self.train_X

    @property
    def y_train(self) -> pd.Series:
        return self.train_y

    @property
    def X_val(self) -> pd.DataFrame:
        return self.val_X

    @property
    def y_val(self) -> pd.Series:
        return self.val_y

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test_X

    @property
    def y_test(self) -> pd.Series:
        return self.test_y

    @property
    def X_oot(self) -> pd.DataFrame:
        return self.oot_X

    @property
    def y_oot(self) -> pd.Series:
        return self.oot_y

    # ── Convenience accessors ────────────────────────────────────────────────

    def all_splits(self) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
        """Return every data partition as a dict keyed by split name.

        Provides a uniform interface for iterating over all partitions
        without referencing individual attributes by name.

        Args:
            None

        Returns:
            Dict with keys ``"train"``, ``"val"``, ``"test"``, and ``"oot"``,
            each mapping to a ``(X, y)`` tuple of ``(pd.DataFrame,
            pd.Series)``.  The ``"oot"`` value is always present; its
            DataFrame and Series are empty when no OOT split was produced.
        """
        return {
            "train": (self.train_X, self.train_y),
            "val": (self.val_X, self.val_y),
            "test": (self.test_X, self.test_y),
            "oot": (self.oot_X, self.oot_y),
        }

    def class_distribution(self) -> pd.DataFrame:
        """Compute per-split class counts and event rates for classification targets.

        Iterates over all splits returned by ``all_splits()``, skipping any
        that have a ``None`` or empty target Series, and aggregates
        ``value_counts`` for each class label.

        Args:
            None

        Returns:
            DataFrame with columns ``split`` (str), ``class`` (target label
            value), ``count`` (int), and ``rate`` (float — count divided by
            split size).  One row per (split, class) combination.  Returns
            an empty DataFrame with those four columns when all target Series
            are empty (e.g. regression task or unpopulated splits).
        """
        rows = []
        for name, (_, y) in self.all_splits().items():
            if y is None or len(y) == 0:
                continue
            for cls, cnt in y.value_counts().items():
                rows.append({"split": name, "class": cls, "count": cnt, "rate": cnt / len(y)})
        return pd.DataFrame(rows)

    def date_ranges(self) -> pd.DataFrame:
        """Return the minimum and maximum date for each split when date metadata is available.

        Reads pre-computed date range information from
        ``self.metadata.date_ranges``.  If that attribute is ``None``
        (e.g. the splitter was run without a ``date_col``), returns an empty
        DataFrame with the correct column schema.

        Args:
            None

        Returns:
            DataFrame with columns ``split`` (str), ``min_date`` (str — ISO
            date), and ``max_date`` (str — ISO date), with one row per split
            that has date information.  Returns an empty DataFrame with those
            three columns when ``metadata.date_ranges`` is ``None``.
        """
        if self.metadata.date_ranges is None:
            return pd.DataFrame(columns=["split", "min_date", "max_date"])
        rows = [
            {"split": k, "min_date": v[0], "max_date": v[1]}
            for k, v in self.metadata.date_ranges.items()
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Build a human-readable multi-line summary of this data split for logging or display.

        Formats strategy, target column, feature count, and per-split row
        counts from ``self.metadata``.  When class rates are available,
        appends the event rate beside each split's row count.

        Args:
            None

        Returns:
            Multi-line string with one header line followed by one line per
            split.  Never returns ``None`` or an empty string; at minimum the
            header line is always present.
        """
        m = self.metadata
        lines = [
            f"DataSplit — strategy={m.strategy!r}  target={m.target_col!r}",
            f"  n_features : {m.n_features}",
        ]
        for name, size in m.split_sizes.items():
            rate = (m.class_rates or {}).get(name, None)
            rate_str = f"  event_rate={rate:.3f}" if rate is not None else ""
            lines.append(f"  {name:<8}: {size:>7,} rows{rate_str}")
        return "\n".join(lines)


# ── DataSplitter ─────────────────────────────────────────────────────────────


class DataSplitter:
    """Flexible data splitter supporting temporal, stratified, grouped, and random
    partitioning strategies.

    Validates inputs on construction and dispatches to the appropriate
    strategy implementation inside ``fit_split``.  For classification tasks
    (target with 20 or fewer distinct values) the class rate is recorded in
    ``DataSplitMetadata``; for regression it is omitted.  Logs a split summary
    at INFO level after each successful split.

    Args:
        strategy: Partitioning strategy.  One of ``"temporal"`` (chronological
            train/val/test with OOT carved out by date), ``"stratified"``
            (class-balanced random split via ``StratifiedShuffleSplit``),
            ``"grouped"`` (splits by ``group_col`` — either a deliberate,
            user-directed population boundary when ``group_test_values`` is
            set, e.g. "train on one segment, test on a different, specifically
            chosen segment", or, when ``group_test_values`` is ``None``, the
            original entity-leakage-prevention behavior via
            ``GroupShuffleSplit`` that randomly assigns whole groups to each
            split without regard to which specific values land where), or
            ``"random"`` (plain random permutation).
        test_size: Fraction of the non-OOT data to allocate to the test set.
            Defaults to ``0.2``.
        val_size: Fraction of the post-test training data to allocate to
            validation.  Defaults to ``0.1``.
        oot_cutoff: ISO date string (e.g. ``"2023-01-01"``).  Rows on or after
            this date form the OOT set.  Required for ``strategy="temporal"``;
            also supported by ``strategy="grouped"`` when ``date_col`` is set.
        date_col: Name of the datetime column in the DataFrame.  Required for
            ``strategy="temporal"``; optional for ``strategy="grouped"``.
        group_col: Name of the group/entity column.  Required for
            ``strategy="grouped"``.
        group_test_values: Set/list of ``group_col`` values that define the
            deliberate test population for ``strategy="grouped"``.  Rows whose
            ``group_col`` value is in this set become the test set; every
            other row becomes the train+val pool, from which validation is
            carved via stratified (classification) or random (otherwise)
            sampling — no group-shuffling for validation, since there is no
            leakage concern within the training population.  ``None`` (the
            default) preserves the original random ``GroupShuffleSplit``
            behavior unchanged, for backward compatibility with existing
            callers (Full Pipeline mode, ``dscompanion/api``) that don't yet pass
            it.  Ignored for every other strategy.
        target_col: Name of the target column to extract from the DataFrame
            before splitting features.  Defaults to ``"target"``.
        random_state: Integer seed for reproducibility of all random
            operations.  Defaults to ``42``.

    Raises:
        ValueError: If required parameters for the chosen strategy are missing
            at construction time, or if validation checks in ``fit_split``
            fail (e.g. missing columns, OOT split too small, unknown or
            all-matching/no-matching ``group_test_values``).
    """

    def __init__(
        self,
        strategy: str = "temporal",
        test_size: float = settings.default_test_size,
        val_size: float = settings.default_val_size,
        oot_cutoff: str | None = None,
        date_col: str | None = None,
        group_col: str | None = None,
        group_test_values: set | list | None = None,
        target_col: str = "target",
        random_state: int = settings.random_state,
    ) -> None:
        self.strategy = strategy
        self.test_size = test_size
        self.val_size = val_size
        self.oot_cutoff = oot_cutoff
        self.date_col = date_col
        self.group_col = group_col
        self.group_test_values = group_test_values
        self.target_col = target_col
        self.random_state = random_state

    # ── Public API ───────────────────────────────────────────────────────────

    def fit_split(self, df: pd.DataFrame) -> DataSplit:
        """Validate ``df``, apply the configured splitting strategy, and return a
        populated ``DataSplit``.

        Separates the target column from the feature columns, dispatches to
        the strategy-specific private method (``_temporal_split``,
        ``_stratified_split``, ``_grouped_split``, or ``_random_split``),
        then builds ``DataSplitMetadata`` and wraps everything in a
        ``DataSplit``.  For stratified splits, logs a warning when the class
        rate differs by more than 5 percentage points between train and
        val/test.  Logs a human-readable summary of the resulting split at
        INFO level.

        Args:
            df: Input DataFrame that must include the ``target_col`` column
                and, depending on the strategy, the ``date_col`` and/or
                ``group_col`` columns.  Shape ``(n_rows, n_cols)``.

        Returns:
            A ``DataSplit`` instance with ``train_X``, ``train_y``, ``val_X``,
            ``val_y``, ``test_X``, ``test_y``, ``oot_X``, ``oot_y``, and
            ``metadata`` fully populated.  ``oot_X`` and ``oot_y`` are empty
            (zero rows) for strategies that do not produce an OOT set.

        Raises:
            ValueError: If ``target_col`` is absent from ``df``, if required
                strategy-specific columns (``date_col``, ``group_col``,
                ``oot_cutoff``) are missing, or if the OOT split contains
                fewer than 50 rows for the temporal strategy.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pd.DataFrame, got %s" % type(df).__name__)
        self._validate(df)
        logger.info("Splitting %s rows  strategy=%r", f"{len(df):,}", self.strategy)

        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        is_classification = y.nunique() <= settings.binner_max_classes

        if self.strategy == "temporal":
            splits = self._temporal_split(df, X, y)
        elif self.strategy == "stratified":
            splits = self._stratified_split(X, y, is_classification)
        elif self.strategy == "grouped":
            splits = self._grouped_split(df, X, y, is_classification)
        else:
            splits = self._random_split(X, y)

        train_X, train_y, val_X, val_y, test_X, test_y, oot_X, oot_y = splits

        # Validate stratification
        if is_classification and self.strategy == "stratified":
            self._check_class_ratio(train_y, val_y, "val")
            self._check_class_ratio(train_y, test_y, "test")

        # Build metadata
        split_sizes = {
            "train": len(train_y),
            "val": len(val_y),
            "test": len(test_y),
            "oot": len(oot_y) if oot_y is not None else 0,
        }
        class_rates = None
        if is_classification and pd.api.types.is_numeric_dtype(train_y):
            class_rates = {
                name: float(s.mean()) if s is not None and len(s) > 0 else 0.0
                for name, s in [
                    ("train", train_y),
                    ("val", val_y),
                    ("test", test_y),
                    ("oot", oot_y),
                ]
            }

        date_ranges = None
        if self.date_col and self.date_col in df.columns:

            def _dr(idx):
                if idx is None or len(idx) == 0:
                    return None
                col = df.loc[idx, self.date_col]
                return (str(col.min()), str(col.max()))

            date_ranges = {
                k: v
                for k, v in {
                    "train": _dr(train_X.index),
                    "val": _dr(val_X.index),
                    "test": _dr(test_X.index),
                    "oot": _dr(oot_X.index) if oot_X is not None else None,
                }.items()
                if v is not None
            }

        meta = DataSplitMetadata(
            strategy=self.strategy,
            target_col=self.target_col,
            n_features=X.shape[1],
            split_sizes=split_sizes,
            class_rates=class_rates,
            date_ranges=date_ranges,
            created_at=datetime.now(),
            dscompanion_version=_VERSION,
        )

        result = DataSplit(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            oot_X=oot_X if oot_X is not None else pd.DataFrame(),
            oot_y=oot_y if oot_y is not None else pd.Series(dtype=float),
            metadata=meta,
        )
        logger.info("\n%s", result.summary())
        return result

    # ── Strategy implementations ─────────────────────────────────────────────

    def _temporal_split(self, df, X, y):
        cutoff = pd.to_datetime(self.oot_cutoff)
        date_col = df[self.date_col]

        oot_mask = date_col >= cutoff
        in_sample_mask = ~oot_mask

        oot_X = X[oot_mask]
        oot_y = y[oot_mask]
        in_X = (
            X[in_sample_mask].sort_values(
                by=self.date_col if self.date_col in X.columns else X.columns[0]
            )
            if self.date_col in X.columns
            else X[in_sample_mask]
        )
        in_y = y[in_sample_mask]

        # Chronological train/val/test
        n = len(in_y)
        test_n = max(1, int(n * self.test_size))
        val_n = max(1, int((n - test_n) * self.val_size))

        test_X, test_y = in_X.iloc[-test_n:], in_y.iloc[-test_n:]
        remaining_X = in_X.iloc[:-test_n]
        remaining_y = in_y.iloc[:-test_n]
        val_X, val_y = remaining_X.iloc[-val_n:], remaining_y.iloc[-val_n:]
        train_X, train_y = remaining_X.iloc[:-val_n], remaining_y.iloc[:-val_n]

        return train_X, train_y, val_X, val_y, test_X, test_y, oot_X, oot_y

    def _stratified_split(self, X, y, is_classification):
        """Split into train/val/test, stratifying on ``y`` only when it's classification-shaped.

        ``StratifiedShuffleSplit`` requires each class to have at least 2
        members — a continuous regression target has (essentially) as many
        "classes" as rows, which crashes with a cryptic sklearn
        ``ValueError``. Mirrors ``_grouped_split()``'s existing
        classification/regression branch: stratified sampling when
        ``is_classification``, else a plain random permutation split.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target.
            is_classification (bool): Whether ``y`` looks
                classification-shaped (``y.nunique() <=
                settings.binner_max_classes``), computed once by the caller.

        Returns:
            tuple: ``(train_X, train_y, val_X, val_y, test_X, test_y, None,
            None)`` — no OOT partition for this strategy.
        """
        if is_classification:
            sss_test = StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.random_state
            )
            train_val_idx, test_idx = next(sss_test.split(X, y))
        else:
            rng = np.random.RandomState(self.random_state)
            idx = rng.permutation(len(X))
            test_n = max(1, int(len(X) * self.test_size))
            test_idx, train_val_idx = idx[:test_n], idx[test_n:]
        train_val_X, train_val_y = X.iloc[train_val_idx], y.iloc[train_val_idx]
        test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

        if is_classification:
            sss_val = StratifiedShuffleSplit(
                n_splits=1, test_size=self.val_size, random_state=self.random_state
            )
            train_idx, val_idx = next(sss_val.split(train_val_X, train_val_y))
        else:
            rng = np.random.RandomState(self.random_state)
            idx = rng.permutation(len(train_val_X))
            val_n = max(1, int(len(train_val_X) * self.val_size))
            val_idx, train_idx = idx[:val_n], idx[val_n:]
        train_X = train_val_X.iloc[train_idx]
        train_y = train_val_y.iloc[train_idx]
        val_X = train_val_X.iloc[val_idx]
        val_y = train_val_y.iloc[val_idx]

        return train_X, train_y, val_X, val_y, test_X, test_y, None, None

    def _grouped_split(self, df, X, y, is_classification):
        if self.group_test_values:
            # Deliberate, user-directed population split: group_col values in
            # group_test_values define the test set exactly (deterministic, no
            # shuffling). Validation is carved from the remaining train+val pool via
            # ordinary stratified/random sampling — there's no leakage concern within
            # the training population, only the train/test boundary is deliberate.
            test_mask = df[self.group_col].isin(set(self.group_test_values))
            test_X, test_y = X[test_mask], y[test_mask]
            train_val_X, train_val_y = X[~test_mask], y[~test_mask]

            if is_classification:
                sss_val = StratifiedShuffleSplit(
                    n_splits=1, test_size=self.val_size, random_state=self.random_state
                )
                train_idx, val_idx = next(sss_val.split(train_val_X, train_val_y))
            else:
                rng = np.random.RandomState(self.random_state)
                idx = rng.permutation(len(train_val_X))
                val_n = max(1, int(len(train_val_X) * self.val_size))
                val_idx, train_idx = idx[:val_n], idx[val_n:]
            train_X = train_val_X.iloc[train_idx]
            train_y = train_val_y.iloc[train_idx]
            val_X = train_val_X.iloc[val_idx]
            val_y = train_val_y.iloc[val_idx]
        else:
            # Original entity-leakage-prevention behavior: random whole-group
            # assignment via GroupShuffleSplit — which specific groups land where
            # doesn't matter, only that no group's rows are split across sets.
            groups = df[self.group_col].values
            gss = GroupShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.random_state
            )
            train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
            train_val_X = X.iloc[train_val_idx]
            train_val_y = y.iloc[train_val_idx]
            test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

            gss_val = GroupShuffleSplit(
                n_splits=1, test_size=self.val_size, random_state=self.random_state
            )
            sub_groups = groups[train_val_idx]
            train_idx, val_idx = next(gss_val.split(train_val_X, train_val_y, groups=sub_groups))
            train_X = train_val_X.iloc[train_idx]
            train_y = train_val_y.iloc[train_idx]
            val_X = train_val_X.iloc[val_idx]
            val_y = train_val_y.iloc[val_idx]

        # OOT by date if oot_cutoff provided
        oot_X, oot_y = None, None
        if self.oot_cutoff and self.date_col and self.date_col in df.columns:
            cutoff = pd.to_datetime(self.oot_cutoff)
            oot_mask = df[self.date_col] >= cutoff
            oot_X = X[oot_mask]
            oot_y = y[oot_mask]

        return train_X, train_y, val_X, val_y, test_X, test_y, oot_X, oot_y

    def _random_split(self, X, y):
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(len(X))
        test_n = max(1, int(len(X) * self.test_size))
        val_n = max(1, int((len(X) - test_n) * self.val_size))
        test_idx = idx[:test_n]
        val_idx = idx[test_n : test_n + val_n]
        train_idx = idx[test_n + val_n :]
        return (
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[val_idx],
            y.iloc[val_idx],
            X.iloc[test_idx],
            y.iloc[test_idx],
            None,
            None,
        )

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        if self.target_col not in df.columns:
            raise ValueError(f"Target column {self.target_col!r} not found in DataFrame.")

        if self.strategy == "temporal":
            if not self.date_col:
                raise ValueError("strategy='temporal' requires date_col to be set.")
            if self.date_col not in df.columns:
                raise ValueError(f"date_col {self.date_col!r} not found in DataFrame.")
            if not self.oot_cutoff:
                raise ValueError("strategy='temporal' requires oot_cutoff to be set.")
            cutoff = pd.to_datetime(self.oot_cutoff)
            oot_n = (pd.to_datetime(df[self.date_col]) >= cutoff).sum()
            if oot_n < settings.splitter_oot_min_rows:
                raise ValueError(
                    "OOT split has only %d rows (< %d). Adjust oot_cutoff."
                    % (oot_n, settings.splitter_oot_min_rows)
                )

        if self.strategy == "grouped" and not self.group_col:
            raise ValueError("strategy='grouped' requires group_col to be set.")

        if self.group_col and self.group_col not in df.columns:
            raise ValueError(f"group_col {self.group_col!r} not found in DataFrame.")

        if self.strategy == "grouped" and self.group_test_values:
            present = set(df[self.group_col].dropna().unique())
            requested = set(self.group_test_values)
            n_unknown = len(requested - present)
            if n_unknown:
                raise ValueError(
                    "%d of the specified group_test_values are not present in %r."
                    % (n_unknown, self.group_col)
                )
            # Every requested value is confirmed present above, so the test mask is
            # guaranteed non-empty — only "matches every row" (no train/val left) is
            # reachable here.
            test_mask = df[self.group_col].isin(requested)
            if (~test_mask).sum() == 0:
                raise ValueError(
                    "group_test_values matches every row in %r — no rows left for "
                    "train/val." % self.group_col
                )

    @staticmethod
    def _check_class_ratio(train_y: pd.Series, other_y: pd.Series, split_name: str) -> None:
        if other_y is None or len(other_y) == 0:
            return
        if not pd.api.types.is_numeric_dtype(train_y):
            return
        train_rate = train_y.mean()
        other_rate = other_y.mean()
        if abs(train_rate - other_rate) > settings.splitter_class_ratio_tolerance:
            logger.warning(
                "Class rate difference train vs %s: %.3f vs %.3f (> %.0f%%)",
                split_name,
                train_rate,
                other_rate,
                settings.splitter_class_ratio_tolerance * 100,
            )
