"""TargetBinariser: convert continuous / multi-class targets to binary."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["TargetBinariser"]


class TargetBinariser:
    """Convert a continuous or multi-valued target series into a binary series using a
    learned threshold.

    After ``fit``, the cut-point is stored in ``threshold_`` and can be reused
    across train, validation, and OOT splits via ``transform``.  No side-effects
    occur at construction time.

    Args:
        strategy (str): Determines how the threshold is derived during ``fit``.
            Must be one of ``"threshold"`` (use the explicit value supplied in
            ``threshold``), ``"median"`` (compute the median of the fit series),
            or ``"quantile"`` (compute an arbitrary quantile).  Defaults to
            ``"threshold"``.
        threshold (float, optional): Fixed cut-point used when
            ``strategy="threshold"``.  Values strictly greater than this are
            labelled positive.  Must be provided when ``strategy="threshold"``;
            ignored for other strategies.  Defaults to ``None``.
        quantile (float): Quantile fraction in ``[0, 1]`` used when
            ``strategy="quantile"``.  A value of ``0.5`` is equivalent to the
            median.  Defaults to ``0.5``.
        positive_label (int): Integer label assigned to samples that exceed the
            threshold.  The negative class receives ``1 - positive_label``.
            Defaults to ``1``.

    Attributes:
        threshold_ (float): Learned cut-point set after calling ``fit``.
    """

    def __init__(
        self,
        strategy: str = "threshold",
        threshold: float | None = None,
        quantile: float = 0.5,
        positive_label: int = 1,
    ) -> None:
        self.strategy = strategy
        self.threshold = threshold
        self.quantile = quantile
        self.positive_label = positive_label

    def fit(self, y: pd.Series) -> "TargetBinariser":
        """Derive and store the binarisation threshold from the supplied target series.

        Depending on ``strategy``, the threshold is set to either the fixed
        ``threshold`` value, the sample median, or an arbitrary quantile of
        ``y``.  Side-effects: sets ``self.threshold_`` and stores the series
        name in ``self._original_name`` for downstream use; emits a DEBUG log
        message with the resolved threshold.

        Args:
            y (pd.Series): Target series of continuous or discrete values from
                which the threshold is derived.  The series is not mutated.

        Returns:
            TargetBinariser: ``self``, allowing method chaining (e.g.
            ``binariser.fit(y).transform(y)``).

        Raises:
            TypeError: If ``y`` is not a ``pd.Series``.
            ValueError: If ``strategy="threshold"`` but ``self.threshold`` is
                ``None``, or if an unrecognised ``strategy`` string is supplied.
        """
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pd.Series, got %s" % type(y).__name__)

        if self.strategy == "threshold":
            if self.threshold is None:
                raise ValueError("threshold must be set when strategy='threshold'.")
            self.threshold_ = float(self.threshold)
        elif self.strategy == "median":
            self.threshold_ = float(y.median())
        elif self.strategy == "quantile":
            self.threshold_ = float(y.quantile(self.quantile))
        else:
            raise ValueError("Unknown strategy: %r" % self.strategy)

        self._original_name = y.name
        logger.debug(
            "TargetBinariser fitted — strategy=%s, threshold=%.4f",
            self.strategy,
            self.threshold_,
        )
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """Apply the fitted threshold to produce a binary integer series.

        Samples whose value is strictly greater than ``threshold_`` receive
        ``positive_label``; all others receive ``1 - positive_label``.  The
        returned series preserves the name of the input series.  No side-effects
        beyond returning a new series.

        Args:
            y (pd.Series): Target series to binarise, using the same scale as
                the series passed to ``fit``.  The series is not mutated.

        Returns:
            pd.Series: Integer series of shape ``(n_samples,)`` containing only
            the values ``positive_label`` and ``1 - positive_label``.  The
            series name matches ``y.name``.  Returns a series of all
            ``1 - positive_label`` values if every element of ``y`` is at or
            below ``threshold_``.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if not hasattr(self, "threshold_"):
            raise RuntimeError("TargetBinariser.transform() called before fit()")
        negative_label = 1 - self.positive_label
        result = (y > self.threshold_).astype(int)
        if self.positive_label != 1:
            result = result.map({1: self.positive_label, 0: negative_label})
        return result.rename(y.name)

    def inverse_transform(self, y: pd.Series) -> pd.Series:
        """Map binary labels back to an approximate continuous representation.

        Reverses the binarisation by substituting ``threshold_`` for positive
        labels and ``0.0`` for negative labels.  This is a lossy reconstruction
        that approximates the original scale; it does not recover the original
        values exactly.  No side-effects beyond returning a new series.

        Args:
            y (pd.Series): Binary integer series produced by ``transform``,
                containing only ``positive_label`` and ``1 - positive_label``
                values.  The series is not mutated.

        Returns:
            pd.Series: Float series of shape ``(n_samples,)`` where positive
            class entries are replaced with ``threshold_`` and negative class
            entries are replaced with ``0.0``.  The series name matches
            ``y.name``.  Returns ``NaN`` for any label value not equal to
            ``positive_label`` or ``1 - positive_label``.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if not hasattr(self, "threshold_"):
            raise RuntimeError("TargetBinariser.inverse_transform() called before fit()")
        negative_label = 1 - self.positive_label
        return y.map({self.positive_label: self.threshold_, negative_label: 0.0}).rename(y.name)
