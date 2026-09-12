"""Theory-driven recommendation engine for per-feature transformation chains.

``recommend_column_recipe()`` is a pure, rule-based v1 recommender: given one
column's EDA summary-statistics row (from
``UnivariateAnalyser.numeric_summary()``/``categorical_summary()``), it
proposes a conservative ``ColumnRecipe`` using the same settings-driven
thresholds ``EDAReport.alerts()`` already uses to flag issues, rather than
introducing new hardcoded cutoffs.

Deliberately conservative: it recommends imputation only when missing values
are present, tail-clipping and distribution-normalisation only when a column
is already flagged ``SKEWED``, and encoding driven only by
``cardinality_flag``. Bucketing is never auto-recommended — it is a
modelling choice, not a universal default.

Designed as a swappable strategy: a future search-based recommender (trying
multiple candidate recipes, scoring by IV/AUC/lift) can replace this
function without changing ``ColumnRecipe``/``FeatureTransformChain``'s
contract.
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

import pandas as pd

from dscompanion.config import settings
from dscompanion.features.transform_chain import ColumnRecipe, TransformStep

__all__ = ["recommend_column_recipe"]


def _recommend_numeric(column: str, stats: pd.Series) -> ColumnRecipe:
    """Builds a conservative recommended recipe for one numeric column.

    Args:
        column (str): Column name.
        stats (pd.Series): One row of
            ``UnivariateAnalyser.numeric_summary()`` output — must expose
            ``missing_pct``, ``skewness``, ``min``, and ``constant_flag``.

    Returns:
        ColumnRecipe: A ``source="recommended"`` recipe with a human-readable
        ``rationale``.
    """
    if bool(stats.get("constant_flag", False)):
        return ColumnRecipe(
            column=column,
            steps=[],
            source="recommended",
            rationale="Constant column — no transform recommended (consider dropping in Step 5).",
        )

    steps: list[TransformStep] = []
    rationale_parts: list[str] = []

    missing_pct = float(stats.get("missing_pct", 0.0) or 0.0)
    if missing_pct > 0:
        steps.append(TransformStep(transformer="impute", params={"numeric_strategy": "auto"}))
        rationale_parts.append(
            f"{missing_pct:.1%} missing -> impute (auto strategy: mean for low skew, "
            "median otherwise)."
        )

    skewness = float(stats.get("skewness", 0.0) or 0.0)
    if abs(skewness) >= settings.eda_skewness_alert_threshold:
        steps.append(TransformStep(transformer="clip_lower"))
        steps.append(TransformStep(transformer="clip_upper"))
        rationale_parts.append(
            f"|skewness|={abs(skewness):.2f} >= {settings.eda_skewness_alert_threshold} "
            "(SKEWED) -> clip extreme tails before normalising."
        )

        min_value = stats.get("min")
        if min_value is not None and min_value > 0:
            steps.append(TransformStep(transformer="log"))
            rationale_parts.append("all values > 0 -> log transform.")
        elif min_value is not None and min_value > -1:
            steps.append(TransformStep(transformer="log1p"))
            rationale_parts.append("values > -1 -> log1p transform.")
        else:
            steps.append(TransformStep(transformer="yeo_johnson"))
            rationale_parts.append("values <= -1 present -> yeo_johnson (handles negatives).")

    if not rationale_parts:
        rationale_parts.append("No missingness or skewness flags — no transform recommended.")

    return ColumnRecipe(
        column=column, steps=steps, source="recommended", rationale=" ".join(rationale_parts)
    )


def _recommend_categorical(column: str, stats: pd.Series) -> ColumnRecipe:
    """Builds a conservative recommended recipe for one categorical column.

    Args:
        column (str): Column name.
        stats (pd.Series): One row of
            ``UnivariateAnalyser.categorical_summary()`` output — must
            expose ``missing_pct`` and ``cardinality_flag``.

    Returns:
        ColumnRecipe: A ``source="recommended"`` recipe with a human-readable
        ``rationale``.
    """
    steps: list[TransformStep] = []
    rationale_parts: list[str] = []

    missing_pct = float(stats.get("missing_pct", 0.0) or 0.0)
    if missing_pct > 0:
        steps.append(
            TransformStep(transformer="impute", params={"categorical_strategy": "most_frequent"})
        )
        rationale_parts.append(f"{missing_pct:.1%} missing -> impute (most frequent category).")

    if bool(stats.get("cardinality_flag", False)):
        steps.append(TransformStep(transformer="rare_group"))
        steps.append(TransformStep(transformer="target_encode"))
        rationale_parts.append(
            f"n_unique > {settings.high_cardinality_threshold} (HIGH_CARDINALITY) -> group rare "
            "categories, then target-encode."
        )
    else:
        steps.append(TransformStep(transformer="onehot_encode"))
        rationale_parts.append("Low cardinality -> one-hot encode.")

    return ColumnRecipe(
        column=column, steps=steps, source="recommended", rationale=" ".join(rationale_parts)
    )


def recommend_column_recipe(
    column: str, stats: pd.Series, dtype: Literal["numeric", "categorical"]
) -> ColumnRecipe:
    """Recommend a conservative, theory-driven transformation recipe for one column.

    Args:
        column (str): Column name.
        stats (pd.Series): One row from
            ``UnivariateAnalyser.numeric_summary()`` (when ``dtype="numeric"``)
            or ``UnivariateAnalyser.categorical_summary()`` (when
            ``dtype="categorical"``).
        dtype (Literal["numeric", "categorical"]): Which summary table
            ``stats`` came from — selects the recommendation rules applied.

    Returns:
        ColumnRecipe: ``source="recommended"``, with a non-empty
        ``rationale`` explaining every step chosen (or why none were).

    Raises:
        ValueError: If ``dtype`` is not ``"numeric"`` or ``"categorical"``.
    """
    if dtype == "numeric":
        recipe = _recommend_numeric(column, stats)
    elif dtype == "categorical":
        recipe = _recommend_categorical(column, stats)
    else:
        raise ValueError(f"dtype must be 'numeric' or 'categorical', got {dtype!r}")

    logger.info(
        "recommend_column_recipe — column=%s, dtype=%s, steps=%d",
        column,
        dtype,
        len(recipe.steps),
    )
    return recipe
