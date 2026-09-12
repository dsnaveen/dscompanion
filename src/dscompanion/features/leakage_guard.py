"""LeakageGuard: detect potential target leakage before training."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import pandas as pd
from pydantic import BaseModel

from dscompanion.config import settings

__all__ = ["LeakageGuard", "LeakageReport", "LeakageError"]


class LeakageError(Exception):
    """Raised by ``LeakageGuard.check`` when one or more features exceed the
    critical correlation threshold and ``raise_on_critical=True``; the
    exception message lists the offending feature names and their
    correlation values.
    """


class LeakageReport(BaseModel):
    """Immutable structured result returned by ``LeakageGuard.check``,
    summarising all critical and warning-level leakage signals found during a
    single check call.

    Attributes:
        critical (list[str]): Human-readable descriptions of features whose
            absolute Pearson correlation with the target exceeds the critical
            threshold, formatted as ``"{col} (|r|={value:.3f})"``. Empty list
            when no critical issues are found.
        warnings (list[str]): Human-readable descriptions of softer leakage
            signals: future dates in datetime columns and features whose name
            contains the target column name. Empty list when no warnings are
            found.
        n_critical (int): Count of entries in ``critical``. Zero when the
            feature set is free of critical leakage.
        n_warnings (int): Count of entries in ``warnings``. Zero when no
            soft leakage signals were detected.
    """

    critical: list[str]
    warnings: list[str]
    n_critical: int
    n_warnings: int

    @property
    def is_clean(self) -> bool:
        """Returns ``True`` when no critical leakage issues were found in
        the last ``LeakageGuard.check`` call, indicating the feature set
        is safe to proceed to model training.

        Args:
            None

        Returns:
            bool: ``True`` if ``n_critical == 0``; ``False`` otherwise.
                Warning-level issues recorded in ``warnings`` do not affect
                this flag — only critical correlation findings count.
        """
        return self.n_critical == 0


class LeakageGuard:
    """Validates a feature set for target leakage by running three checks:
    Pearson correlation, future date detection, and target-name substring
    overlap; does not modify or transform the input DataFrame.

    Args:
        raise_on_critical (bool): When ``True`` (default), raises
            ``LeakageError`` if any numeric feature exceeds the correlation
            threshold. When ``False``, findings are recorded in the returned
            ``LeakageReport`` without raising.
        correlation_threshold (float | None): Absolute Pearson correlation
            value above which a feature is flagged as critically leaking.
            When ``None``, the value from
            ``settings.target_leakage_correlation_threshold`` is used.
    """

    def __init__(
        self,
        raise_on_critical: bool = True,
        correlation_threshold: float | None = None,
    ) -> None:
        self.raise_on_critical = raise_on_critical
        self.correlation_threshold = correlation_threshold

    def check(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_col: str,
        reference_date: pd.Timestamp | None = None,
    ) -> LeakageReport:
        """Runs three leakage detection passes on ``X`` against ``y`` and
        returns a structured report; raises ``LeakageError`` as a side-effect
        when critical issues are found and ``raise_on_critical=True``.

        The three passes are: (1) absolute Pearson correlation of every numeric
        column against ``y`` compared to the threshold, (2) detection of
        datetime columns whose values exceed ``reference_date`` (future dates),
        and (3) a name-substring check that flags any column whose name
        contains the target column name string.

        Args:
            X (pd.DataFrame): Feature DataFrame to audit. Non-numeric columns
                are skipped during the correlation check but included in the
                name and date checks. Correlation computation errors for
                individual columns are logged and skipped.
            y (pd.Series): Binary or continuous target series aligned with
                ``X`` by index.
            target_col (str): Name of the target column used for the
                name-substring check. Columns whose lowercase name contains
                the lowercase ``target_col`` string (and are not themselves
                ``target_col``) are added to warnings.
            reference_date (pd.Timestamp | None): Upper bound for allowable
                dates. When provided, any ``datetime64`` column in ``X`` that
                contains values strictly after this timestamp generates a
                warning entry. When ``None``, the future-date check is skipped.

        Returns:
            LeakageReport: A populated ``LeakageReport`` instance with
            ``critical`` and ``warnings`` lists, along with their respective
            counts. The ``critical`` list is empty when no feature exceeds the
            correlation threshold. The ``warnings`` list is empty when no
            future-date or name-overlap issues are detected.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame`` or ``y`` is not a
                ``pd.Series``.
            ValueError: If ``X`` has zero rows.
            LeakageError: When ``raise_on_critical=True`` and at least one
                feature's absolute Pearson correlation with ``y`` exceeds
                ``correlation_threshold``. The exception message includes the
                count and names of offending features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "LeakageGuard.check() expects X to be a pd.DataFrame, got %s" % type(X).__name__
            )
        if not isinstance(y, pd.Series):
            raise TypeError(
                "LeakageGuard.check() expects y to be a pd.Series, got %s" % type(y).__name__
            )
        if len(X) == 0:
            raise ValueError("LeakageGuard.check() received an empty DataFrame (0 rows)")

        threshold = (
            self.correlation_threshold
            if self.correlation_threshold is not None
            else settings.target_leakage_correlation_threshold
        )

        critical: list[str] = []
        warnings: list[str] = []

        # 1. Correlation check
        num_X = X.select_dtypes(include="number")
        for col in num_X.columns:
            try:
                r = abs(num_X[col].corr(y))
                if r > threshold:
                    critical.append("%s (|r|=%.3f)" % (col, r))
                    logger.warning("CRITICAL leakage — %r |r|=%.3f with target", col, r)
            except (ValueError, ArithmeticError) as e:
                logger.warning(
                    "LeakageGuard: correlation failed for column '%s' — %s",
                    col,
                    type(e).__name__,
                )

        # 2. Future date check
        if reference_date is not None:
            for col in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    if (X[col] > reference_date).any():
                        warnings.append("%s contains dates after reference_date" % col)
                        logger.warning("WARNING — %r has future dates", col)

        # 3. Name overlap check
        for col in X.columns:
            if target_col.lower() in col.lower() and col != target_col:
                warnings.append("%s name contains target_col string" % col)
                logger.warning("WARNING — %r name contains target string", col)

        report = LeakageReport(
            critical=critical,
            warnings=warnings,
            n_critical=len(critical),
            n_warnings=len(warnings),
        )

        if self.raise_on_critical and critical:
            raise LeakageError(
                "Critical leakage detected in %d feature(s): %s" % (len(critical), critical)
            )

        return report
