"""GroupRelativeTransformer: ratio-to-group-mean and z-score-within-group features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

__all__ = ["GroupRelativeTransformer"]


class GroupRelativeTransformer(BaseEstimator, TransformerMixin):
    """Adds features expressing each numeric value relative to its group's statistics.

    For each fitted numeric column, adds new columns (never replaces the
    original) depending on ``strategy``: ratio to group mean, z-score within
    group, difference from group mean, and/or percentage of group total.
    Useful for domain features like "this customer's balance relative to
    their segment" rather than the raw value in isolation.

    Group statistics (mean, std, sum) are learned from training data only,
    keyed by group value — the standard fit-on-train/apply-on-holdout
    pattern already used throughout this codebase. A group value unseen at
    transform time falls back to the global (all-training-rows) statistic.
    Standalone — not wired into ``FeatureProcessingPipeline``, matching the
    precedent already set by ``AutoBinner``/``RareCategoryGrouper``.

    Args:
        group_col (str | None): Column whose values define the group (e.g.
            ``"segment"``, ``"state"``). When ``None``, every row is treated
            as belonging to one implicit group — every relative feature is
            computed against the whole-dataset statistic instead of a
            per-group one, and output column names drop the
            ``_{group_col}`` segment (e.g. ``"{col}_ratio_to_mean"`` instead
            of ``"{col}_ratio_to_{group_col}_mean"``).
        features (list[str] | None): Numeric columns to compute relative
            features for. When ``None`` (default), all numeric columns
            except ``group_col`` are used.
        strategy (str): Which relative feature to add. One of ``"ratio"``
            (ratio to group mean), ``"zscore"`` (z-score within group),
            ``"both"`` (ratio + zscore, default), ``"difference"``
            (value minus group mean), or ``"pct_of_total"`` (value as a
            percentage of the group's sum).

    Attributes:
        group_means_ (dict[str, dict]): Per-feature mapping of group value to
            training-set group mean. Empty when ``group_col is None``.
        group_stds_ (dict[str, dict]): Per-feature mapping of group value to
            training-set group standard deviation. Empty when
            ``group_col is None``.
        group_sums_ (dict[str, dict]): Per-feature mapping of group value to
            training-set group sum. Empty when ``group_col is None``.
        global_means_ (dict[str, float]): Per-feature fallback mean — used
            directly when ``group_col is None``, or as the fallback for a
            group value unseen at transform time.
        global_stds_ (dict[str, float]): Per-feature fallback std, same rule.
        global_sums_ (dict[str, float]): Per-feature fallback sum, same rule.
    """

    def __init__(
        self,
        group_col: str | None = None,
        features: list[str] | None = None,
        strategy: str = "both",
    ) -> None:
        self.group_col = group_col
        self.features = features
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "GroupRelativeTransformer":
        """Computes per-group (or, when ``group_col is None``, whole-dataset)
        mean/std/sum for every fitted column, storing them internally along
        with the resolved output column names.

        Args:
            X (pd.DataFrame): Training feature DataFrame. Must contain
                ``group_col`` when it is not ``None``.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            GroupRelativeTransformer: The fitted instance (``self``),
            enabling method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, ``group_col`` is not
                ``None`` and missing from ``X``, or ``strategy`` is not
                recognised.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "GroupRelativeTransformer.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("GroupRelativeTransformer.fit() received an empty DataFrame (0 rows)")
        if self.group_col is not None and self.group_col not in X.columns:
            raise ValueError(f"group_col {self.group_col!r} not found in X")
        if self.strategy not in ("ratio", "zscore", "both", "difference", "pct_of_total"):
            raise ValueError(
                "strategy must be one of {'ratio', 'zscore', 'both', 'difference', "
                f"'pct_of_total'}}, got {self.strategy!r}"
            )

        cols = (
            self.features
            if self.features is not None
            else [c for c in X.select_dtypes(include="number").columns if c != self.group_col]
        )
        self.features_: list[str] = [c for c in cols if c in X.columns and c != self.group_col]

        self.group_means_: dict[str, dict] = {}
        self.group_stds_: dict[str, dict] = {}
        self.group_sums_: dict[str, dict] = {}
        self.global_means_: dict[str, float] = {}
        self.global_stds_: dict[str, float] = {}
        self.global_sums_: dict[str, float] = {}

        for col in self.features_:
            if self.group_col is not None:
                grouped = X.groupby(self.group_col)[col]
                self.group_means_[col] = grouped.mean().to_dict()
                self.group_stds_[col] = grouped.std().fillna(0.0).to_dict()
                self.group_sums_[col] = grouped.sum().to_dict()
            global_mean = X[col].mean()
            global_std = X[col].std()
            global_sum = X[col].sum()
            self.global_means_[col] = float(global_mean) if pd.notna(global_mean) else 0.0
            self.global_stds_[col] = float(global_std) if pd.notna(global_std) else 0.0
            self.global_sums_[col] = float(global_sum) if pd.notna(global_sum) else 0.0

        group_suffix = f"_{self.group_col}" if self.group_col is not None else ""
        self._output_names: dict[str, dict[str, str]] = {}
        self.feature_names_out_: list[str] = []
        for col in self.features_:
            names = {
                "ratio": (
                    f"{col}_ratio_to{group_suffix}_mean" if group_suffix else f"{col}_ratio_to_mean"
                ),
                "zscore": (
                    f"{col}_zscore_within{group_suffix}" if group_suffix else f"{col}_zscore"
                ),
                "difference": (
                    f"{col}_diff_from{group_suffix}_mean"
                    if group_suffix
                    else f"{col}_diff_from_mean"
                ),
                "pct_of_total": (
                    f"{col}_pct_of{group_suffix}_total" if group_suffix else f"{col}_pct_of_total"
                ),
            }
            self._output_names[col] = names
            if self.strategy in ("ratio", "both"):
                self.feature_names_out_.append(names["ratio"])
            if self.strategy in ("zscore", "both"):
                self.feature_names_out_.append(names["zscore"])
            if self.strategy == "difference":
                self.feature_names_out_.append(names["difference"])
            if self.strategy == "pct_of_total":
                self.feature_names_out_.append(names["pct_of_total"])

        logger.info(
            "GroupRelativeTransformer fitted — group_col=%s, strategy=%s, %d features",
            self.group_col,
            self.strategy,
            len(self.features_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Adds the relative feature columns to a copy of ``X``; the original
        DataFrame is not modified and original columns are kept unchanged.

        A ratio, z-score, or percentage-of-total against a zero mean/std/sum
        produces ``NaN`` rather than raising or dividing by zero — the same
        tolerance this codebase already applies to other numerically-sensitive
        transforms (see ``DistributionTransformer``).

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Must contain
                ``group_col`` when it is not ``None``.

        Returns:
            pd.DataFrame: A copy of ``X`` with one new column per fitted
            feature appended (see ``feature_names_out_``).

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``group_col`` is not ``None`` and missing from ``X``.
        """
        check_is_fitted(self, attributes=["global_means_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "GroupRelativeTransformer.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )
        if self.group_col is not None and self.group_col not in X.columns:
            raise ValueError(f"group_col {self.group_col!r} not found in X")

        logger.debug(
            "GroupRelativeTransformer.transform — rows=%d, features=%d",
            len(X),
            len(self.features_),
        )

        out = X.copy()
        groups = X[self.group_col] if self.group_col is not None else None
        for col in self.features_:
            if col not in out.columns:
                continue

            if groups is not None:
                means = groups.map(self.group_means_[col]).fillna(self.global_means_[col])
                stds = groups.map(self.group_stds_[col]).fillna(self.global_stds_[col])
                sums = groups.map(self.group_sums_[col]).fillna(self.global_sums_[col])
            else:
                means = pd.Series(self.global_means_[col], index=out.index)
                stds = pd.Series(self.global_stds_[col], index=out.index)
                sums = pd.Series(self.global_sums_[col], index=out.index)

            names = self._output_names[col]
            if self.strategy in ("ratio", "both"):
                safe_means = means.replace(0, np.nan)
                out[names["ratio"]] = out[col] / safe_means
            if self.strategy in ("zscore", "both"):
                safe_stds = stds.replace(0, np.nan)
                out[names["zscore"]] = (out[col] - means) / safe_stds
            if self.strategy == "difference":
                out[names["difference"]] = out[col] - means
            if self.strategy == "pct_of_total":
                safe_sums = sums.replace(0, np.nan)
                out[names["pct_of_total"]] = out[col] / safe_sums * 100
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the names of the new relative-feature columns this
        transformer adds (not the full output column set — original columns
        are unchanged and not repeated here).

        Args:
            None

        Returns:
            list[str]: New column names, in fit-time order.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["feature_names_out_"])
        return list(self.feature_names_out_)
