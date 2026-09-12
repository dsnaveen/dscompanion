"""ImbalanceHandler: resampling and class-weight strategies for imbalanced targets."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight

from dscompanion.config import settings

logger = logging.getLogger(__name__)

__all__ = ["ImbalanceHandler"]


class ImbalanceHandler:
    """Correct class imbalance in a training set via resampling or class weighting.

    Supports four strategies: balanced class weights (no resampling), SMOTE
    over-sampling, random under-sampling, and no-op pass-through. SMOTE and
    under-sampling are implemented from scratch on top of scikit-learn's
    ``NearestNeighbors`` (no external resampling dependency) — SMOTE draws a
    synthetic point between each new sample and one of its same-class nearest
    neighbors, exactly following Chawla et al.'s original algorithm; random
    under-sampling drops rows without replacement per class. ``fit_resample``
    must only be applied to training data; applying it to validation or OOT
    splits would introduce data leakage. No side-effects occur at construction
    time.

    Args:
        strategy (str): Imbalance correction strategy. Must be one of
            ``"class_weight"`` (compute sklearn balanced weights, return X/y
            unchanged), ``"smote"`` (synthetic minority over-sampling),
            ``"undersample"`` (random majority under-sampling), or ``"none"``
            (pass X/y through without modification). Defaults to
            ``"class_weight"``.
        sampling_strategy (float or str): ``"auto"`` (bring every non-extreme
            class to exactly match the majority count for ``"smote"``, or the
            minority count for ``"undersample"``) or a float ratio — binary
            classification only, the desired minority:majority count ratio
            after resampling. Has no effect for ``"class_weight"`` and
            ``"none"`` strategies. Defaults to ``"auto"``.
        random_state (int, optional): Integer random seed for reproducible
            resampling. Defaults to ``None`` (resolves to
            ``settings.random_state``).

    Attributes:
        class_weights_ (dict or None): Mapping of class label (int) to its
            balanced weight (float). Set after ``fit`` only when
            ``strategy="class_weight"``; ``None`` for all other strategies.
        resample_audit_ (pd.DataFrame): Before-and-after class distribution
            summary set after ``fit_resample``. Contains columns ``class``,
            ``count_before``, ``pct_before``, ``count_after``, and
            ``pct_after``.
    """

    def __init__(
        self,
        strategy: str = "class_weight",
        sampling_strategy: float | str = "auto",
        random_state: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ImbalanceHandler":
        """Learn the class distribution and, for the class-weight strategy, compute
        balanced weights.

        Records the unique class labels and pre-resampling class frequencies.
        When ``strategy="class_weight"``, also computes sklearn balanced weights
        and stores them in ``self.class_weights_``. Side-effects: sets
        ``self._classes``, ``self._before_dist``, and ``self.class_weights_``;
        emits a DEBUG log message with the strategy and class distribution.

        Args:
            X (pd.DataFrame): Training feature matrix, shape
                ``(n_samples, n_features)``. Not used for weight computation
                but accepted to keep the API consistent with ``fit_resample``.
            y (pd.Series): Binary (or multi-class) target vector, shape
                ``(n_samples,)``. Used to derive class labels, frequencies,
                and balanced weights.

        Returns:
            ImbalanceHandler: ``self``, allowing method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame`` or ``y`` is not a
                ``pd.Series``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame, got %s" % type(X).__name__)
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pd.Series, got %s" % type(y).__name__)

        self._classes = np.sort(np.unique(y))
        self._before_dist = y.value_counts(normalize=True).sort_index()

        if self.strategy == "class_weight":
            weights = compute_class_weight("balanced", classes=self._classes, y=y)
            self.class_weights_: dict[int, float] | None = {
                int(c): float(w) for c, w in zip(self._classes, weights)
            }
        else:
            self.class_weights_ = None

        logger.debug(
            "ImbalanceHandler fitted — strategy=%s, class_dist=%s",
            self.strategy,
            self._before_dist.to_dict(),
        )
        return self

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Fit the handler on training data and return a resampled copy of X and y.

        Calls ``fit`` internally to record class distributions and weights, then
        applies the selected resampling strategy. For ``"class_weight"`` and
        ``"none"``, the original X and y are returned without modification. For
        ``"smote"`` and ``"undersample"``, rows are resampled per-class and
        reconstructed as pandas objects with the original column names and
        series name preserved, but a fresh ``RangeIndex`` (the original index is
        not preserved for either strategy). Side-effects: calls ``fit``, sets
        ``self._after_dist`` and ``self.resample_audit_``, and emits an INFO log
        message for resampling strategies. Must only be called on training data
        to avoid leakage.

        Args:
            X (pd.DataFrame): Training feature matrix, shape
                ``(n_samples, n_features)``. For ``"smote"`` every column must
                be numeric with no missing values. For over-sampling
                strategies, the returned DataFrame may have more rows than the
                input; for under-sampling, fewer rows.
            y (pd.Series): Binary target vector, shape ``(n_samples,)``. Must
                align row-for-row with ``X``.

        Returns:
            tuple: A two-element tuple ``(X_resampled, y_resampled)`` where:

            - ``X_resampled`` (pd.DataFrame): Feature matrix after resampling,
              shape ``(n_resampled, n_features)``. Equal to the input ``X``
              when ``strategy`` is ``"class_weight"`` or ``"none"``.
            - ``y_resampled`` (pd.Series): Target vector after resampling,
              shape ``(n_resampled,)``. Equal to the input ``y`` when
              ``strategy`` is ``"class_weight"`` or ``"none"``.

        Raises:
            ValueError: If ``strategy`` is unrecognized, if
                ``sampling_strategy`` is an unsupported string, if a float
                ``sampling_strategy`` is used with more than two classes, if a
                float ``sampling_strategy`` would require removing rows rather
                than adding/dropping them, if ``"smote"`` is used on
                non-numeric or missing-value data, or if a class has fewer
                rows than ``settings.imbalance_smote_k_neighbors + 1`` for
                ``"smote"``.
        """
        self.fit(X, y)

        if self.strategy in ("class_weight", "none"):
            self._after_dist = self._before_dist.copy()
            self.resample_audit_ = self._build_audit(y, y)
            return X, y

        effective_random_state = (
            self.random_state if self.random_state is not None else settings.random_state
        )
        rng = np.random.RandomState(effective_random_state)

        if self.strategy == "smote":
            X_resampled, y_resampled = self._smote_resample(X, y, rng)
        elif self.strategy == "undersample":
            X_resampled, y_resampled = self._undersample_resample(X, y, rng)
        else:
            raise ValueError("Unknown strategy: %r" % self.strategy)

        self._after_dist = y_resampled.value_counts(normalize=True).sort_index()
        self.resample_audit_ = self._build_audit(y, y_resampled)

        logger.info(
            "Resampled: %d → %d rows | strategy=%s",
            len(y),
            len(y_resampled),
            self.strategy,
        )
        return X_resampled, y_resampled

    @property
    def resample_audit_(self) -> pd.DataFrame:
        """Return the before-and-after class distribution comparison set by ``fit_resample``.

        Provides a tidy summary of how many samples each class had before and
        after resampling, useful for verifying that the chosen strategy achieved
        the expected balance. No side-effects.

        Returns:
            pd.DataFrame: Frame with one row per class and columns ``class``
            (class label), ``count_before`` (int), ``pct_before`` (float,
            rounded to 4 decimal places), ``count_after`` (int), and
            ``pct_after`` (float, rounded to 4 decimal places). Returns an
            empty DataFrame if ``fit_resample`` has not been called yet.
        """
        return self._audit

    @resample_audit_.setter
    def resample_audit_(self, value: pd.DataFrame) -> None:
        self._audit = value

    # ------------------------------------------------------------------
    # Private helpers — target-count resolution
    # ------------------------------------------------------------------

    def _resolve_targets(self, counts: pd.Series, extreme: str) -> dict:
        """Resolve each class's desired row count under ``self.sampling_strategy``.

        Args:
            counts (pd.Series): Per-class row counts, indexed by class label.
            extreme (str): ``"max"`` for over-sampling (SMOTE) — every class
                below the majority count is topped up to it — or ``"min"``
                for under-sampling — every class above the minority count is
                dropped down to it.

        Returns:
            dict: Mapping of class label to target row count, containing only
            classes whose count actually needs to change.

        Raises:
            ValueError: If ``sampling_strategy`` is a string other than
                ``"auto"``, if it's a float with more than two classes, or if
                the implied target would require removing rows for
                ``extreme="max"`` or adding rows for ``extreme="min"``.
        """
        extreme_count = int(counts.max() if extreme == "max" else counts.min())

        if isinstance(self.sampling_strategy, str):
            if self.sampling_strategy != "auto":
                raise ValueError(
                    f"sampling_strategy string must be 'auto', got {self.sampling_strategy!r}"
                )
            return {
                cls: extreme_count
                for cls, count in counts.items()
                if (count < extreme_count if extreme == "max" else count > extreme_count)
            }

        if len(counts) != 2:
            raise ValueError(
                "sampling_strategy as a float ratio is only supported for binary "
                f"classification, got {len(counts)} classes"
            )
        other_cls = counts.idxmin() if extreme == "max" else counts.idxmax()
        other_count = int(counts[other_cls])
        if extreme == "max":
            target = round(self.sampling_strategy * extreme_count)
            if target < other_count:
                raise ValueError(
                    f"sampling_strategy={self.sampling_strategy!r} implies removing rows "
                    "from the minority class for 'smote' — must be >= its current ratio"
                )
        else:
            target = round(extreme_count / self.sampling_strategy)
            if target > other_count:
                raise ValueError(
                    f"sampling_strategy={self.sampling_strategy!r} implies adding rows "
                    "to the majority class for 'undersample' — must be <= its current ratio"
                )
        return {other_cls: target}

    # ------------------------------------------------------------------
    # Private helpers — resampling
    # ------------------------------------------------------------------

    def _smote_resample(
        self, X: pd.DataFrame, y: pd.Series, rng: np.random.RandomState
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Synthetic Minority Over-sampling (Chawla et al.), scikit-learn-only.

        For each class needing more rows, fits ``NearestNeighbors`` on that
        class's own rows, then draws synthetic rows by interpolating between a
        random existing row and one of its ``k`` nearest same-class neighbors.

        Args:
            X (pd.DataFrame): Training feature matrix — must be fully numeric
                with no missing values.
            y (pd.Series): Training target vector.
            rng (np.random.RandomState): Shared random generator — drawn from
                once per class, never re-seeded, so the overall draw sequence
                is reproducible given the same seed.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Resampled features (fresh
            ``RangeIndex``, original column names) and target (original name).

        Raises:
            ValueError: If ``X`` has non-numeric columns or missing values, or
                if a class needing synthetic rows has fewer than
                ``settings.imbalance_smote_k_neighbors + 1`` rows.
        """
        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise ValueError(
                f"'smote' requires fully numeric features — non-numeric columns: {non_numeric}"
            )
        if X.isna().any().any():
            raise ValueError("'smote' requires no missing values in X — impute first")

        counts = y.value_counts()
        targets = self._resolve_targets(counts, extreme="max")
        k = settings.imbalance_smote_k_neighbors

        X_arr = X.to_numpy(dtype=float)
        y_arr = y.to_numpy()
        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls in self._classes:
            target_count = targets.get(cls)
            if target_count is None:
                continue
            class_mask = y_arr == cls
            class_data = X_arr[class_mask]
            n_current = len(class_data)
            n_synthetic = target_count - n_current
            if n_synthetic <= 0:
                continue
            if n_current < k + 1:
                raise ValueError(
                    f"class {cls!r} has only {n_current} row(s), fewer than "
                    f"imbalance_smote_k_neighbors + 1 ({k + 1}) — reduce "
                    "settings.imbalance_smote_k_neighbors or choose a different strategy"
                )

            neighbors = NearestNeighbors(n_neighbors=k + 1).fit(class_data)
            neighbor_idx = neighbors.kneighbors(class_data, return_distance=False)[:, 1:]

            base_idx = rng.randint(0, n_current, n_synthetic)
            neighbor_slot = rng.randint(0, k, n_synthetic)
            gaps = rng.uniform(0, 1, (n_synthetic, 1))
            neighbor_rows = class_data[neighbor_idx[base_idx, neighbor_slot]]
            new_rows = class_data[base_idx] + gaps * (neighbor_rows - class_data[base_idx])

            synthetic_X.append(new_rows)
            synthetic_y.append(np.full(n_synthetic, cls))

        if synthetic_X:
            X_final = np.vstack([X_arr, *synthetic_X])
            y_final = np.concatenate([y_arr, *synthetic_y])
        else:
            X_final, y_final = X_arr, y_arr

        X_resampled = pd.DataFrame(X_final, columns=X.columns)
        y_resampled = pd.Series(y_final, name=y.name)
        return X_resampled, y_resampled

    def _undersample_resample(
        self, X: pd.DataFrame, y: pd.Series, rng: np.random.RandomState
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Random under-sampling — drops rows without replacement per class.

        Args:
            X (pd.DataFrame): Training feature matrix — any dtype mix.
            y (pd.Series): Training target vector.
            rng (np.random.RandomState): Shared random generator for row
                selection.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Resampled features (fresh
            ``RangeIndex``, original column names) and target (original name),
            row order shuffled across classes.
        """
        counts = y.value_counts()
        targets = self._resolve_targets(counts, extreme="min")

        y_arr = y.to_numpy()
        keep_idx: list[np.ndarray] = []
        for cls in self._classes:
            class_idx = np.where(y_arr == cls)[0]
            target_count = targets.get(cls)
            if target_count is None:
                keep_idx.append(class_idx)
            else:
                keep_idx.append(rng.choice(class_idx, size=target_count, replace=False))

        all_idx = np.concatenate(keep_idx)
        rng.shuffle(all_idx)

        X_resampled = X.iloc[all_idx].reset_index(drop=True)
        y_resampled = y.iloc[all_idx].reset_index(drop=True)
        return X_resampled, y_resampled

    def _build_audit(self, y_before: pd.Series, y_after: pd.Series) -> pd.DataFrame:
        before = y_before.value_counts().sort_index()
        after = y_after.value_counts().sort_index()
        df = pd.DataFrame(
            {
                "class": before.index,
                "count_before": before.values,
                "pct_before": (before / before.sum()).round(4).values,
                "count_after": after.reindex(before.index, fill_value=0).values,
                "pct_after": (after.reindex(before.index, fill_value=0) / after.sum())
                .round(4)
                .values,
            }
        )
        return df
