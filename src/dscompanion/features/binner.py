"""AutoBinner: quantile, uniform, and decision-tree binning strategies."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

__all__ = ["AutoBinner"]


class AutoBinner(BaseEstimator, TransformerMixin):
    """Discretises numeric columns into bins using quantile, uniform, or
    decision-tree derived cut-points, and optionally encodes the resulting
    bins as integer codes or human-readable string labels.

    Args:
        strategy (str): Binning algorithm. One of:

            - ``"quantile"`` (default) — equal-frequency bins via
              ``KBinsDiscretizer``.
            - ``"uniform"`` — equal-width bins via ``KBinsDiscretizer``.
            - ``"tree"`` — split points derived from a decision tree fitted
              against the target ``y``; falls back to ``"quantile"`` when
              ``y`` is ``None`` during ``fit``.

        n_bins (int | None): Number of bins to create. When ``None``
            (default), uses ``settings.binner_n_bins``.
        encode (str): Output encoding for bin assignments. One of:

            - ``"ordinal"`` (default) — integer bin index (0-based).
            - ``"labels"`` — string range label, e.g. ``"[0.00, 1.00)"``.
              Bins that cannot be assigned are labelled ``"unknown"``.

        tree_criterion (str): Splitting criterion used when
            ``strategy="tree"`` and the target is a classifier.
            One of ``"gini"`` or ``"entropy"``. Defaults to ``"gini"``.
        min_samples_leaf (float | None): Minimum fraction of total training
            rows required in each leaf node when ``strategy="tree"``.
            When ``None`` (default), uses ``settings.binner_min_samples_leaf``.
    """

    def __init__(
        self,
        strategy: str = "quantile",
        n_bins: int | None = None,
        encode: str = "ordinal",
        tree_criterion: str = "gini",
        min_samples_leaf: float | None = None,
    ) -> None:
        self.strategy = strategy
        self.n_bins = n_bins
        self.encode = encode
        self.tree_criterion = tree_criterion
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AutoBinner":
        """Learns bin edges for every numeric column in ``X`` using the chosen
        strategy, storing them in ``_bin_edges`` and ``_labels`` as side-effects;
        columns with fewer than two non-null values are skipped.

        Args:
            X (pd.DataFrame): Training feature DataFrame. Only columns
                selected by ``select_dtypes(include="number")`` are binned;
                non-numeric columns are ignored. Columns with fewer than
                two non-null values are excluded from ``_bin_edges``.
            y (pd.Series | None): Target series. Required when
                ``strategy="tree"``; when ``None`` with ``strategy="tree"``
                the binner silently falls back to ``"quantile"`` binning.
                Ignored for ``"quantile"`` and ``"uniform"`` strategies.

        Returns:
            AutoBinner: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AutoBinner.fit() expects a pd.DataFrame, got %s" % type(X).__name__)
        if len(X) == 0:
            raise ValueError("AutoBinner.fit() received an empty DataFrame (0 rows)")

        effective_n_bins = self.n_bins if self.n_bins is not None else settings.binner_n_bins
        effective_min_leaf = (
            self.min_samples_leaf
            if self.min_samples_leaf is not None
            else settings.binner_min_samples_leaf
        )

        self._num_cols: list[str] = X.select_dtypes(include="number").columns.tolist()
        self._bin_edges: dict[str, np.ndarray] = {}
        self._labels: dict[str, list[str]] = {}

        if self.strategy == "tree" and y is not None:
            self._fit_tree(X, y, effective_min_leaf)
        else:
            effective_strategy = self.strategy if self.strategy != "tree" else "quantile"
            kbd = KBinsDiscretizer(
                n_bins=effective_n_bins,
                encode="ordinal",
                strategy=effective_strategy,
                subsample=None,
            )
            cols = [c for c in self._num_cols if X[c].notna().sum() > 1]
            if cols:
                kbd.fit(X[cols].fillna(X[cols].median()))
                for i, col in enumerate(cols):
                    edges = kbd.bin_edges_[i]
                    self._bin_edges[col] = edges
                    self._labels[col] = [
                        f"[{edges[j]:.3g}, {edges[j + 1]:.3g})" for j in range(len(edges) - 1)
                    ]

        logger.info(
            "AutoBinner fitted — strategy=%s, n_bins=%d, cols=%d",
            self.strategy,
            effective_n_bins,
            len(self._bin_edges),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Assigns each numeric value to a bin using the edges learned during
        ``fit`` and replaces the column values with integer codes or string
        labels depending on ``encode``; the original DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns not
                present in ``_bin_edges`` (including non-numeric or
                too-sparse columns skipped during ``fit``) are passed
                through unchanged. Values that fall outside the bin edges
                or are ``NaN`` receive code ``-1`` (ordinal) or
                ``"unknown"`` (labels).

        Returns:
            pd.DataFrame: A copy of ``X`` where each binned column has been
            replaced. When ``encode="ordinal"`` the column dtype is ``int``.
            When ``encode="labels"`` the column dtype is ``object`` (string).

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["_bin_edges"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "AutoBinner.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )

        logger.debug(
            "AutoBinner.transform — rows=%d, binning %d cols", len(X), len(self._bin_edges)
        )

        out = X.copy()
        for col, edges in self._bin_edges.items():
            if col not in out.columns:
                continue
            binned = pd.cut(out[col], bins=edges, labels=False, include_lowest=True)
            if self.encode == "labels":
                label_map = {i: lbl for i, lbl in enumerate(self._labels.get(col, []))}
                out[col] = binned.map(label_map).fillna("unknown")
            else:
                out[col] = binned.fillna(-1).astype(int)
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns the names of all numeric columns identified at fit time,
        including those that were skipped due to insufficient data.

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

    def _fit_tree(self, X: pd.DataFrame, y: pd.Series, min_leaf_frac: float) -> None:
        """Derives bin edges per numeric column from a decision tree fitted against ``y``.

        Uses a classifier when ``y`` has at most ``settings.binner_max_classes`` unique
        values; uses a regressor otherwise. All-NaN columns are skipped. Stores results
        into ``_bin_edges`` and ``_labels`` in-place.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series): Target series aligned with ``X``.
            min_leaf_frac (float): Minimum fraction of training rows required per leaf.
        """
        is_clf = y.nunique() <= settings.binner_max_classes
        for col in self._num_cols:
            if X[col].notna().sum() == 0:
                logger.debug("AutoBinner._fit_tree: skipping all-NaN column %s", col)
                continue
            s = X[col].fillna(X[col].median()).values.reshape(-1, 1)
            min_leaf = max(1, int(len(s) * min_leaf_frac))
            if is_clf:
                tree = DecisionTreeClassifier(
                    criterion=self.tree_criterion,
                    min_samples_leaf=min_leaf,
                    random_state=settings.random_state,
                )
            else:
                tree = DecisionTreeRegressor(
                    min_samples_leaf=min_leaf,
                    random_state=settings.random_state,
                )
            tree.fit(s, y)
            thresholds = np.unique(tree.tree_.threshold[tree.tree_.threshold != -2])
            edges = np.concatenate([[s.min() - 1], np.sort(thresholds), [s.max() + 1]])
            self._bin_edges[col] = edges
            self._labels[col] = [
                f"[{edges[j]:.3g}, {edges[j + 1]:.3g})" for j in range(len(edges) - 1)
            ]
