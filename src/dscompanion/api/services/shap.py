"""Step 12 (Explainability / SHAP) service functions — the REST equivalent of
``dscompanion/app/interactive_step_shap.py``, with every ``streamlit`` call stripped out.

Read-only, but unlike Steps 10/12's usual shape, this step's real work
(``run_shap``) is an explicit opt-in the caller must trigger — SHAP stays off
by default here too, consistent with ``explain.shap_enabled``'s default.
``preview_shap`` never computes SHAP — only ``run_shap`` does, mirroring
Streamlit's off-by-default toggle rather than a value the frontend could
accidentally trigger via a routine preview call.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import plotly.graph_objects as go

from dscompanion.api.schemas import (
    ShapConfirmResponse,
    ShapFeatureRow,
    ShapPreviewResponse,
    ShapRunResponse,
)
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id
from dscompanion.config import settings
from dscompanion.explain.shap_explainer import SHAPExplainer
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["preview_shap", "run_shap", "confirm_shap"]


def _fig_to_json(fig: go.Figure) -> dict[str, Any]:
    """JSON-safe Plotly figure dict via Plotly's own encoder (handles numpy arrays and
    datetimes correctly — more robust than ``fig.to_dict()``). Defined locally rather
    than imported, matching this codebase's existing per-service convention (see
    ``services/eda.py``/``services/tuning.py``'s identical private helper).
    """
    return json.loads(fig.to_json())


def _shap_data(processed_split: DataSplit) -> tuple[Any, str] | None:
    """Select a sample from the best available held-out split for SHAP.

    Prefers val, then test, then train — mirrors
    ``interactive_step_shap.py``'s ``_shap_data`` exactly, with the sample
    cap and seed routed through ``settings`` instead of literals.

    Args:
        processed_split (DataSplit): Step 8's processed split.

    Returns:
        tuple[Any, str] | None: ``(X_sample, split_name)``, or ``None`` if
        every split is empty.
    """
    for attr_x, name in (("val_X", "val"), ("test_X", "test"), ("train_X", "train")):
        X = getattr(processed_split, attr_x, None)
        if X is not None and len(X) > 0:
            if len(X) > settings.shap_interactive_max_rows:
                X = X.sample(settings.shap_interactive_max_rows, random_state=settings.random_state)
            return X, name
    return None


# ── Preview ───────────────────────────────────────────────────────────────────


def preview_shap(run: RunState) -> ShapPreviewResponse:
    """Reports whether SHAP is available for this run, without computing anything.

    Args:
        run (RunState): The run holding Step 11's calibrated model and Step 8's
            processed split.

    Returns:
        ShapPreviewResponse: Availability — always ``True`` today.

    Raises:
        ValueError: If Step 11 has not been confirmed for this run.
    """
    model = run.artifacts.get("calibrated_model")
    processed_split = run.artifacts.get("processed_split")
    if model is None or processed_split is None:
        raise ValueError("Step 11 must be confirmed before Step 12.")

    logger.info("run_id=%s previewed Step 12", run.run_id)
    return ShapPreviewResponse(available=True)


# ── Run (synchronous — see module docstring for why this isn't backgrounded) ──


def run_shap(run: RunState) -> ShapRunResponse:
    """Fits ``SHAPExplainer`` on a sample of the held-out split and returns the summary.

    Caches the result in ``run.artifacts["shap_result"]`` so a repeated call
    within the same run doesn't recompute, mirroring
    ``interactive_step_shap.py``'s ``int.shap.explainer``/``int.shap.fig``
    session-state cache.

    Args:
        run (RunState): The run holding Step 11's calibrated model and Step 8's
            processed split.

    Returns:
        ShapRunResponse: The summary chart, top-10 feature table, split used,
        and row count.

    Raises:
        ValueError: If Step 11 has not been confirmed, or no held-out data
            is available, for this run.
    """
    cached = run.artifacts.get("shap_result")
    if cached is not None:
        return cached

    model = run.artifacts.get("calibrated_model")
    processed_split: DataSplit | None = run.artifacts.get("processed_split")
    if model is None or processed_split is None:
        raise ValueError("Step 11 must be confirmed before Step 12.")

    data = _shap_data(processed_split)
    if data is None:
        raise ValueError("No held-out data available to compute SHAP.")
    X, split_name = data

    explainer = SHAPExplainer(model)
    explainer.fit(X)
    fig = explainer.summary_plot()
    importance_df = explainer.mean_abs_shap().head(10)

    response = ShapRunResponse(
        figure=_fig_to_json(fig),
        top_features=[
            ShapFeatureRow(
                rank=i + 1, feature=row["feature"], mean_abs_shap=float(row["mean_abs_shap"])
            )
            for i, row in importance_df.reset_index(drop=True).iterrows()
        ],
        split_used=split_name,
        n_rows=len(X),
    )
    run.artifacts["shap_result"] = response
    run.artifacts["_shap_top_feature"] = (
        importance_df["feature"].iloc[0] if len(importance_df) > 0 else None
    )

    logger.info(
        "run_id=%s ran Step 12: split=%s n_rows=%d n_features=%d",
        run.run_id,
        split_name,
        len(X),
        len(explainer.feature_names_),
    )
    return response


# ── Confirm ───────────────────────────────────────────────────────────────────


def confirm_shap(run: RunState, shap_run: bool) -> ShapConfirmResponse:
    """Records Step 12 completion.

    Args:
        run (RunState): The run to persist into.
        shap_run (bool): Whether SHAP was actually computed (``True``) or
            skipped by the user (``False``).

    Returns:
        ShapConfirmResponse: Confirmation.
    """
    top_feature = run.artifacts.get("_shap_top_feature") if shap_run else None
    note = (
        f"SHAP run — top feature: {top_feature}."
        if shap_run and top_feature
        else ("SHAP run — no features available." if shap_run else "SHAP skipped by user.")
    )

    run.step_confirmed["shap"] = True
    run.step_status["shap"] = "done"
    nxt = next_step_id("shap")
    if nxt is not None:
        run.step_status[nxt] = "current"
    run.audit_trail.append(("shap", note))

    logger.info("run_id=%s confirmed Step 12: %s", run.run_id, note)
    return ShapConfirmResponse(confirmed=True)
