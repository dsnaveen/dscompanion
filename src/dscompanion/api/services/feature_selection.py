"""Step 6 (Feature Selection) service functions — the REST equivalent of
``dscompanion/app/interactive_step_feature_selection.py``, with every ``streamlit`` call
stripped out. Runs dscompanion's real ``FeatureSelectionPipeline`` (the same selector chain
``PipelineRunner._build_selection_pipeline()`` uses) on the output of a
``FeatureTransformChain`` built from Step 5's confirmed recipes — selection metrics are
computed post-encoding, matching how ``PipelineRunner`` runs selection after (not
before) feature processing in a real batch run.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dscompanion.api.schemas import (
    FeatureSelectionAuditRow,
    FeatureSelectionConfirmRequest,
    FeatureSelectionConfirmResponse,
    FeatureSelectionPreviewResponse,
)
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id
from dscompanion.features import ColumnRecipe, FeatureTransformChain
from dscompanion.selection import (
    CardinalitySelector,
    ConstantSelector,
    CorrelationSelector,
    FeatureSelectionPipeline,
    IVSelector,
    NullRateSelector,
)
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["preview_feature_selection", "confirm_feature_selection"]

# Selector class name -> plain-language noun, for the audit-trail summary — mirrors
# interactive_step_feature_selection.py's _STAGE_AUDIT_NOUNS.
_STAGE_AUDIT_NOUNS = {
    "NullRateSelector": "column(s) with too many missing values",
    "ConstantSelector": "constant/near-constant column(s)",
    "CardinalitySelector": "high-cardinality column(s)",
    "CorrelationSelector": "highly-correlated column(s)",
    "IVSelector": "low-predictive-value column(s)",
}


def _build_selectors(task: str) -> list:
    """Mirrors ``PipelineRunner._build_selection_pipeline()``'s default selector chain.

    Args:
        task (str): Confirmed task — ``IVSelector`` only applies to classification.

    Returns:
        list: Ordered ``BaseSelector`` instances built from dscompanion's settings-driven
        defaults.
    """
    selectors = [
        NullRateSelector(),
        ConstantSelector(),
        CardinalitySelector(),
        CorrelationSelector(),
    ]
    if task == "classification":
        selectors.append(IVSelector())
    return selectors


def _fit_selection_pipeline(run: RunState) -> tuple[FeatureSelectionPipeline, int]:
    """Builds the Step-5-transformed candidate set and fits selection on it.

    Args:
        run (RunState): The run holding Step 3's split and Step 5's confirmed recipes.

    Returns:
        tuple[FeatureSelectionPipeline, int]: The fitted pipeline and the transformed
        candidate set's column count.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed for this run.
    """
    split: DataSplit = run.artifacts.get("split")
    if split is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 6.")

    split_data = run.step_data.get("split", {})
    date_col = split_data.get("date_col")
    identifier_columns = run.artifacts.get("identifier_columns") or []
    dropped_at_step5 = run.artifacts.get("feature_processing_dropped_columns") or []
    excluded_cols = {c for c in [date_col, *identifier_columns, *dropped_at_step5] if c}
    candidate_cols = [c for c in split.train_X.columns if c not in excluded_cols]
    task = run.artifacts.get("task", "classification")

    recipes: dict[str, ColumnRecipe] = run.artifacts.get("feature_processing_recipes") or {}
    recipes_for_candidates = {c: r for c, r in recipes.items() if c in candidate_cols}

    chain = FeatureTransformChain(recipes=recipes_for_candidates)
    transformed_X = chain.fit_transform(split.train_X[candidate_cols])
    sel_pipeline = FeatureSelectionPipeline(selectors=_build_selectors(task))
    sel_pipeline.fit(transformed_X, split.train_y)
    return sel_pipeline, transformed_X.shape[1]


def preview_feature_selection(run: RunState) -> FeatureSelectionPreviewResponse:
    """Runs feature selection read-only and reports the proposed drop-list.

    Args:
        run (RunState): The run holding Step 3's split and Step 5's confirmed recipes.

    Returns:
        FeatureSelectionPreviewResponse: The audit table and removal waterfall chart.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed, or selection itself fails
            (chain fit error, selector crash) — the caller decides whether to offer a
            skip path.
    """
    try:
        sel_pipeline, total_features = _fit_selection_pipeline(run)
    except ValueError:
        raise
    except Exception as exc:
        logger.warning("run_id=%s Step 6 selection failed: %s", run.run_id, type(exc).__name__)
        raise ValueError(f"Couldn't run feature selection: {exc}")

    audit = sel_pipeline.audit_report()
    audit_rows = [
        FeatureSelectionAuditRow(
            feature=row.feature,
            removed_by=row.removed_by,
            reason=row.reason,
            original_position=row.original_position,
        )
        for row in audit.itertuples()
    ]
    waterfall_chart: dict[str, Any] | None = None
    if not audit.empty:
        waterfall_chart = json.loads(sel_pipeline.plot_removal_waterfall().to_json())

    logger.info(
        "run_id=%s previewed Step 6 feature selection: %d flagged of %d",
        run.run_id,
        len(audit_rows),
        total_features,
    )
    return FeatureSelectionPreviewResponse(
        audit=audit_rows, waterfall_chart=waterfall_chart, total_features=total_features
    )


def confirm_feature_selection(
    run: RunState, request: FeatureSelectionConfirmRequest
) -> FeatureSelectionConfirmResponse:
    """Persists the final removed-columns decision for this run.

    Args:
        run (RunState): The run to persist into.
        request (FeatureSelectionConfirmRequest): Final removed columns, or
            ``skipped=True`` if selection itself failed.

    Returns:
        FeatureSelectionConfirmResponse: Confirmation, status, and the persisted
        removed-columns list.

    Raises:
        ValueError: If ``request.removed_columns`` would remove every transformed
            feature (nothing left to train on).
    """
    if request.skipped:
        removed_columns: list[str] = []
        status = "flagged"
        run.audit_trail.append(("feature_selection", "Feature selection failed — skipped."))
    else:
        sel_pipeline, total_features = _fit_selection_pipeline(run)
        if request.removed_columns and len(request.removed_columns) >= total_features:
            raise ValueError("Cannot remove every feature — keep at least one.")
        removed_columns = request.removed_columns
        status = "done"

        if removed_columns:
            audit = sel_pipeline.audit_report()
            by_stage: dict[str, int] = {}
            for row in audit.itertuples():
                if row.feature in removed_columns:
                    by_stage[row.removed_by] = by_stage.get(row.removed_by, 0) + 1
            parts = [
                f"removed {n} {_STAGE_AUDIT_NOUNS.get(stage, stage)}"
                for stage, n in by_stage.items()
            ]
            run.audit_trail.append(
                ("feature_selection", "Feature selection: " + "; ".join(parts) + ".")
            )
        else:
            run.audit_trail.append(
                ("feature_selection", "Feature selection: no columns removed (all kept).")
            )

    run.step_data["feature_selection"] = {"removed_columns": removed_columns}
    run.artifacts["feature_selection_removed_columns"] = removed_columns
    run.step_confirmed["feature_selection"] = True
    run.step_status["feature_selection"] = status
    nxt = next_step_id("feature_selection")
    if nxt is not None:
        run.step_status[nxt] = "current"

    logger.info(
        "run_id=%s confirmed Step 6: status=%s removed=%d",
        run.run_id,
        status,
        len(removed_columns),
    )
    return FeatureSelectionConfirmResponse(
        confirmed=True, status=status, removed_columns=removed_columns
    )
