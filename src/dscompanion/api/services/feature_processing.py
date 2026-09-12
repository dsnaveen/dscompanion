"""Step 5 (Feature Processing / ColumnRecipe decisions) service functions — the REST
equivalent of ``dscompanion/app/interactive_step_feature_processing.py``, with every
``streamlit`` call stripped out. Every feature column gets a theory-driven recommended
``ColumnRecipe``; the frontend applies the same per-column Accept/Reject/Drop mechanism
Streamlit does, but all recipe editing (add/remove/reorder steps) happens client-side —
only the recommendation, the live before/after preview, and the final confirm need a
server round-trip.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from dscompanion.api.schemas import (
    FeatureProcessingCandidate,
    FeatureProcessingConfirmRequest,
    FeatureProcessingConfirmResponse,
    FeatureProcessingPreviewResponse,
    FeatureProcessingTransformPreviewRequest,
    FeatureProcessingTransformPreviewResponse,
)
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id
from dscompanion.eda.univariate import UnivariateAnalyser
from dscompanion.features import ColumnRecipe, FeatureTransformChain, recommend_column_recipe
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["preview_feature_processing", "preview_transform", "confirm_feature_processing"]


def _excluded_columns(run: RunState) -> set[str]:
    """Columns never eligible for a recipe decision — date/identifier/bool/datetime.

    Args:
        run (RunState): The run holding Step 2/3's confirmed choices.

    Returns:
        set[str]: Column names to exclude from both candidate groups.
    """
    split: DataSplit = run.artifacts["split"]
    split_data = run.step_data.get("split", {})
    date_col = split_data.get("date_col")
    identifier_columns = run.artifacts.get("identifier_columns") or []
    bool_cols = split.train_X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    # Any datetime-dtype column, not just date_col — a second incidental date column
    # has no row in either UnivariateAnalyser summary table (both explicitly exclude
    # datetime_cols), so recommend_column_recipe() has nothing to look up for it.
    datetime_cols = split.train_X.select_dtypes(include="datetime").columns.tolist()
    return {c for c in [date_col, *identifier_columns, *bool_cols, *datetime_cols] if c}


def _univariate_summaries(run: RunState) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (numeric_summary, categorical_summary), reusing Step 4's EDAReport if present.

    Args:
        run (RunState): The run holding Step 3's confirmed split and, if Step 4's preview
            was called, a cached ``EDAReport`` in ``run.artifacts["eda_report"]``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``UnivariateAnalyser.numeric_summary()`` and
        ``.categorical_summary()`` output — the input ``recommend_column_recipe()`` needs.
    """
    eda_report = run.artifacts.get("eda_report")
    if eda_report is not None:
        return eda_report.numeric_summary(), eda_report.categorical_summary()
    split: DataSplit = run.artifacts["split"]
    analyser = UnivariateAnalyser().fit(split)
    return analyser.numeric_summary(), analyser.categorical_summary()


def _stats_row(
    col: str, is_numeric: bool, numeric_summary: pd.DataFrame, categorical_summary: pd.DataFrame
) -> pd.Series:
    """Looks up one column's summary-statistics row for ``recommend_column_recipe()``."""
    table = numeric_summary if is_numeric else categorical_summary
    return table.loc[table["feature"] == col].iloc[0]


def _recommended_recipe(
    col: str, is_numeric: bool, numeric_summary: pd.DataFrame, categorical_summary: pd.DataFrame
) -> ColumnRecipe:
    """Computes the recommended ``ColumnRecipe`` for one column."""
    stats = _stats_row(col, is_numeric, numeric_summary, categorical_summary)
    dtype = "numeric" if is_numeric else "categorical"
    return recommend_column_recipe(col, stats, dtype)


def _build_candidates(run: RunState) -> list[FeatureProcessingCandidate]:
    """Builds the full ordered candidate list — "Needing Attention" first, then "Rest".

    Args:
        run (RunState): The run holding Step 3's confirmed split.

    Returns:
        list[FeatureProcessingCandidate]: One entry per eligible feature column.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed for this run.
    """
    split: DataSplit = run.artifacts.get("split")
    if split is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 5.")

    excluded_cols = _excluded_columns(run)
    train_X = split.train_X
    candidate_cols = [c for c in train_X.columns if c not in excluded_cols]

    eda_report = run.artifacts.get("eda_report")
    suggested_reasons = (
        eda_report.recommended_drops_with_reasons() if eda_report is not None else {}
    )

    missing_counts = (
        train_X[candidate_cols].isna().sum() if candidate_cols else pd.Series(dtype=int)
    )
    numeric_summary, categorical_summary = _univariate_summaries(run)

    attention_cols = [
        c for c in candidate_cols if missing_counts[c] > 0 or suggested_reasons.get(c)
    ]
    rest_cols = [c for c in candidate_cols if c not in attention_cols]

    candidates: list[FeatureProcessingCandidate] = []
    for needs_attention, cols in ((True, attention_cols), (False, rest_cols)):
        for col in cols:
            is_numeric = pd.api.types.is_numeric_dtype(train_X[col])
            n_missing = int(missing_counts[col])
            recommended = _recommended_recipe(col, is_numeric, numeric_summary, categorical_summary)
            candidates.append(
                FeatureProcessingCandidate(
                    column=col,
                    is_numeric=is_numeric,
                    missing_rows=n_missing,
                    missing_pct=round(100 * n_missing / len(train_X), 2) if len(train_X) else 0.0,
                    suggested_reasons=suggested_reasons.get(col, []),
                    needs_attention=needs_attention,
                    recommended=recommended,
                )
            )
    return candidates


def preview_feature_processing(run: RunState) -> FeatureProcessingPreviewResponse:
    """Computes every feature column's recommended recipe, without persisting anything.

    Args:
        run (RunState): The run holding Step 3's confirmed split.

    Returns:
        FeatureProcessingPreviewResponse: The full candidate list.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed for this run.
    """
    candidates = _build_candidates(run)
    logger.info(
        "run_id=%s previewed Step 5 feature processing: %d candidates", run.run_id, len(candidates)
    )
    return FeatureProcessingPreviewResponse(candidates=candidates)


def _sample_records(df: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    """JSON-safe records for the first ``n`` rows, via pandas' own JSON encoder."""
    return json.loads(df.head(n).to_json(orient="records", date_format="iso"))


def preview_transform(
    run: RunState, request: FeatureProcessingTransformPreviewRequest
) -> FeatureProcessingTransformPreviewResponse:
    """Fits a real ``FeatureTransformChain`` on the client's current accepted recipes and
    returns a before/after sample — mirrors ``_render_live_preview()``.

    Args:
        run (RunState): The run holding Step 3's confirmed split.
        request (FeatureProcessingTransformPreviewRequest): The client's current
            accepted-columns recipe dict.

    Returns:
        FeatureProcessingTransformPreviewResponse: First 5 training rows, before and
        after the chain — empty before/after when ``request.recipes`` is empty.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed, or the chain fails to fit
            with the given recipes.
    """
    split: DataSplit = run.artifacts.get("split")
    if split is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 5.")

    if not request.recipes:
        return FeatureProcessingTransformPreviewResponse(
            before=[], after=[], columns_before=[], columns_after=[]
        )

    accepted_cols = list(request.recipes)
    before_sample = split.train_X[accepted_cols].head(5)
    try:
        chain = FeatureTransformChain(recipes=request.recipes)
        after = chain.fit_transform(split.train_X)
    except Exception as exc:
        logger.warning(
            "run_id=%s Step 5 transform preview failed: %s", run.run_id, type(exc).__name__
        )
        raise ValueError(f"Could not fit the current recipe selections: {exc}")
    after_sample = after.loc[before_sample.index]

    return FeatureProcessingTransformPreviewResponse(
        before=_sample_records(before_sample.reset_index(drop=True), 5),
        after=_sample_records(after_sample.reset_index(drop=True), 5),
        columns_before=list(before_sample.columns),
        columns_after=list(after_sample.columns),
    )


def confirm_feature_processing(
    run: RunState, request: FeatureProcessingConfirmRequest
) -> FeatureProcessingConfirmResponse:
    """Persists the final recipe/drop decisions for this run.

    Args:
        run (RunState): The run to persist into.
        request (FeatureProcessingConfirmRequest): Final accepted recipes and dropped
            columns.

    Returns:
        FeatureProcessingConfirmResponse: Confirmation and accepted/dropped/rejected
        counts, for the audit-trail summary.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed for this run.
    """
    candidates = _build_candidates(run)
    total = len(candidates)
    n_dropped = len(request.dropped_columns)
    n_accepted = sum(1 for r in request.recipes.values() if r.steps)
    n_rejected = total - n_accepted - n_dropped

    run.step_data["feature_processing"] = {
        "recipes": {col: r.model_dump(mode="json") for col, r in request.recipes.items()},
        "dropped_columns": request.dropped_columns,
    }
    run.artifacts["feature_processing_recipes"] = request.recipes
    run.artifacts["feature_processing_dropped_columns"] = request.dropped_columns
    run.step_confirmed["feature_processing"] = True
    run.step_status["feature_processing"] = "done"
    nxt = next_step_id("feature_processing")
    if nxt is not None:
        run.step_status[nxt] = "current"

    parts = []
    if request.dropped_columns:
        parts.append(f"dropped {n_dropped} column(s) ({', '.join(request.dropped_columns)})")
    if n_accepted:
        parts.append(f"accepted a recipe for {n_accepted} column(s)")
    if n_rejected:
        parts.append(f"left {n_rejected} column(s) as-is")
    run.audit_trail.append(
        (
            "feature_processing",
            (
                "Feature processing: " + "; ".join(parts) + "."
                if parts
                else "Feature processing: no columns to configure."
            ),
        )
    )
    logger.info(
        "run_id=%s confirmed Step 5: accepted=%d dropped=%d rejected=%d",
        run.run_id,
        n_accepted,
        n_dropped,
        n_rejected,
    )
    return FeatureProcessingConfirmResponse(
        confirmed=True, n_accepted=n_accepted, n_dropped=n_dropped, n_rejected=n_rejected
    )
