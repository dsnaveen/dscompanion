"""Step 9 (Hyperparameter Tuning) service functions — the REST equivalent of
``dscompanion/app/interactive_step_tuning.py``, with every ``streamlit`` call stripped out.

Streamlit's ``Tuner.run()`` call is synchronous (blocks the script thread, tolerable
there since the browser tab just shows a spinner) — a REST client can't hold an HTTP
connection open through a multi-minute Optuna run, so this wraps the same call in the
job/poll pattern already used for Step 8's leaderboard.
This is genuinely new design, not a straight port: the Streamlit UI never needed
backgrounding for tuning.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from dscompanion.api.schemas import (
    TuningConfirmRequest,
    TuningConfirmResponse,
    TuningPreviewResponse,
    TuningStartResponse,
)
from dscompanion.api.state import RunState, RunStateStore
from dscompanion.api.steps import next_step_id
from dscompanion.split import DataSplit
from dscompanion.tuning.search_spaces import PREDEFINED_PARAMS
from dscompanion.tuning.tuner import Tuner

logger = logging.getLogger(__name__)

__all__ = ["preview_tuning", "start_tuning_job", "confirm_tuning"]

# Algorithms that have no classification search space in search_spaces.py — mirrors
# interactive_step_tuning.py's _NO_SEARCH_SPACE exactly (no shared module between
# dscompanion/app/ and dscompanion/api/ per this project's convention).
_NO_SEARCH_SPACE: frozenset[str] = frozenset({"gradient_boosting", "lightgbm"})

_DEFAULT_N_TRIALS = 20
_MIN_N_TRIALS = 5
_MAX_N_TRIALS = 100


def _fig_to_json(fig: go.Figure) -> dict[str, Any]:
    """JSON-safe Plotly figure dict via Plotly's own encoder (handles numpy arrays and
    datetimes correctly — more robust than ``fig.to_dict()``), same helper as
    ``services/eda.py``'s ``_fig_to_json``.
    """
    return json.loads(fig.to_json())


def _optimization_curve_figure(oc_df: pd.DataFrame, metric: str) -> dict[str, Any]:
    """Builds the trial-score-vs-best-so-far Plotly figure, mirroring
    ``interactive_step_tuning.py``'s ``_render_optimization_curve`` exactly (minus
    ``dark_fig()`` — the frontend's ``ThemedPlot`` re-themes client-side from CSS
    variables, same pattern already used for every other chart this API returns).

    Args:
        oc_df (pd.DataFrame): Columns ``trial_number``, ``metric_value``,
            ``best_so_far`` — from ``Tuner.optimization_curve_``.
        metric (str): Metric name used for the y-axis label.

    Returns:
        dict[str, Any]: JSON-safe Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=oc_df["trial_number"],
            y=oc_df["metric_value"],
            mode="markers",
            name="Trial Score",
            marker=dict(color="#3987e5", size=6, opacity=0.55),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=oc_df["trial_number"],
            y=oc_df["best_so_far"],
            mode="lines",
            name="Best So Far",
            line=dict(color="#e66767", width=2),
        )
    )
    fig.update_layout(
        xaxis_title="Trial",
        yaxis_title=metric.upper().replace("_", " "),
        height=420,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _fig_to_json(fig)


def _baseline_auc(model: object, processed_split: DataSplit) -> float | None:
    """Evaluates the untuned Step 8 model to get the "before tuning" AUC, mirroring
    ``interactive_step_tuning.py``'s ``_baseline_auc`` (val split, falling back to
    test). Always re-evaluated fresh here rather than reusing Step 8's own
    ``TrainConfirmResponse.metrics`` — those aren't persisted anywhere in
    ``run.artifacts``, and re-evaluating is cheap relative to the tuning run itself.

    Args:
        model: Fitted ``BaseDSCompanionModel`` from Step 8 (untuned).
        processed_split (DataSplit): The same processed split Step 8 trained on.

    Returns:
        float | None: Validation (or test) AUC-ROC, or ``None`` if unavailable.
    """
    metrics_df = model.evaluate(processed_split)
    for split_name in ("val", "test"):
        sub = metrics_df[(metrics_df["split"] == split_name) & (metrics_df["metric"] == "roc_auc")]
        if not sub.empty:
            return float(sub["value"].iloc[0])
    return None


def _params_comparison(algorithm: str, best_params: dict) -> list[dict[str, Any]]:
    """Builds the default-vs-tuned parameter comparison rows, mirroring
    ``interactive_step_tuning.py``'s ``_params_comparison_table``.

    Args:
        algorithm (str): Algorithm name (e.g. ``"xgboost"``).
        best_params (dict): Best parameters found by ``Tuner``.

    Returns:
        list[dict[str, Any]]: One ``{"parameter", "default", "tuned", "changed"}``
        dict per parameter in ``best_params``.
    """
    defaults = PREDEFINED_PARAMS.get(f"{algorithm}_classification", {})
    rows = []
    for param, tuned_val in best_params.items():
        default_val = defaults.get(param, "—")
        rows.append(
            {
                "parameter": param,
                "default": str(default_val),
                "tuned": str(tuned_val),
                "changed": str(tuned_val) != str(default_val),
            }
        )
    return rows


# ── Preview ───────────────────────────────────────────────────────────────────


def preview_tuning(run: RunState) -> TuningPreviewResponse:
    """Reports whether tuning is available for Step 8's confirmed algorithm.

    Args:
        run (RunState): The run holding Step 8's confirmed model.

    Returns:
        TuningPreviewResponse: Availability and slider bounds.

    Raises:
        ValueError: If Step 8 has not been confirmed for this run.
    """
    algorithm = run.step_data.get("train", {}).get("algorithm")
    if algorithm is None:
        raise ValueError("Step 8 must be confirmed before Step 9.")

    logger.info("run_id=%s previewed Step 9: algorithm=%s", run.run_id, algorithm)
    return TuningPreviewResponse(
        algorithm=algorithm,
        available=algorithm not in _NO_SEARCH_SPACE,
        default_n_trials=_DEFAULT_N_TRIALS,
        min_n_trials=_MIN_N_TRIALS,
        max_n_trials=_MAX_N_TRIALS,
    )


# ── Tuning run (backgrounded) ───────────────────────────────────────────────────


def _tuning_worker(
    run_id: str,
    artifacts: dict,
    model: object,
    n_trials: int,
    store: RunStateStore,
    job_id: str,
) -> None:
    """Runs ``Tuner.run()`` in a background thread and writes the result.

    Stashes the fitted tuned model directly into the shared ``artifacts`` dict
    (a single ``dict.__setitem__`` is atomic under the GIL — job results must
    stay JSON-safe, so the model itself can't live in ``job.result``, same
    reasoning as Step 8's leaderboard winner).

    Args:
        run_id (str): The run this job belongs to (for logging only).
        artifacts (dict): The run's shared ``artifacts`` dict — mutated in
            place once tuning succeeds.
        model: Fitted ``BaseDSCompanionModel`` from Step 8, to tune.
        n_trials (int): Number of Optuna trials to run.
        store (RunStateStore): Used to fetch the live ``JobState`` to write into.
        job_id (str): This job's identifier.

    Returns:
        None
    """
    job = store.get_job(run_id, job_id)
    processed_split: DataSplit = artifacts["processed_split"]
    algorithm = artifacts.get("_tuning_algorithm", "")
    try:
        baseline_auc = _baseline_auc(model, processed_split)
        tuner = Tuner(
            model=model,
            backend="optuna",
            n_trials=n_trials,
            cv=processed_split,
            metric="roc_auc",
            direction="maximize",
        )
        tuned_model = tuner.run()

        oc_df = getattr(tuner, "optimization_curve_", pd.DataFrame())
        optimization_curve = (
            _optimization_curve_figure(oc_df, tuner.metric) if not oc_df.empty else None
        )
        trials_df = tuner.trials_dataframe_
        trials = (
            [
                {
                    "trial_number": int(r["trial_number"]),
                    "metric_value": round(float(r["metric_value"]), 4),
                }
                for _, r in trials_df.iterrows()
            ]
            if not trials_df.empty
            else []
        )

        best_score = None if math.isnan(tuner.best_score_) else round(tuner.best_score_, 4)
        artifacts["tuning_tuned_model"] = tuned_model
        artifacts["_tuning_best_score"] = best_score
        with job.lock:
            job.status = "done"
            job.result = {
                "best_score": best_score,
                "baseline_auc": baseline_auc,
                "params_comparison": _params_comparison(algorithm, tuner.best_params_),
                "trials": trials,
                "optimization_curve": optimization_curve,
            }
        logger.info(
            "run_id=%s tuning done: algorithm=%s best_score=%s",
            run_id,
            algorithm,
            tuner.best_score_,
        )
    except ValueError as exc:
        # Raised when the search space cannot be inferred for this algorithm.
        logger.warning("run_id=%s Step 9 tuning failed for %s: %s", run_id, algorithm, exc)
        with job.lock:
            job.status = "error"
            job.error = (
                f"Could not tune {algorithm}: {exc} This algorithm may not have a search "
                "space defined. You can skip tuning and continue with the default model."
            )
    except Exception as exc:
        logger.exception("run_id=%s Step 9 tuning failed unexpectedly", run_id)
        with job.lock:
            job.status = "error"
            job.error = f"Tuning failed unexpectedly: {exc}"


def start_tuning_job(run: RunState, store: RunStateStore, n_trials: int) -> TuningStartResponse:
    """Starts a background Optuna tuning run for Step 8's confirmed model.

    Args:
        run (RunState): The run holding Step 8's confirmed model and
            processed split.
        store (RunStateStore): Used to create the job and fetch it from
            within the background thread.
        n_trials (int): Number of Optuna trials to run.

    Returns:
        TuningStartResponse: The new job's id, for polling.

    Raises:
        ValueError: If Step 8 has not been confirmed for this run.
    """
    model = run.artifacts.get("trained_model")
    algorithm = run.step_data.get("train", {}).get("algorithm")
    if model is None or algorithm is None:
        raise ValueError("Step 8 must be confirmed before Step 9.")

    run.artifacts["_tuning_algorithm"] = algorithm
    job = store.create_job(run.run_id, "tuning")
    thread = threading.Thread(
        target=_tuning_worker,
        args=(run.run_id, run.artifacts, model, n_trials, store, job.job_id),
        daemon=True,
    )
    thread.start()

    logger.info(
        "run_id=%s started Step 9 tuning job_id=%s: algorithm=%s n_trials=%d",
        run.run_id,
        job.job_id,
        algorithm,
        n_trials,
    )
    return TuningStartResponse(job_id=job.job_id)


# ── Confirm ───────────────────────────────────────────────────────────────────


def confirm_tuning(run: RunState, request: TuningConfirmRequest) -> TuningConfirmResponse:
    """Persists the final model choice (tuned or Step 8's default) for this run.

    Args:
        run (RunState): The run to persist into.
        request (TuningConfirmRequest): Whether to accept the tuned model.

    Returns:
        TuningConfirmResponse: Confirmation, whether tuned, and the best score
        (only when ``use_tuned=True`` and a completed tuning job produced one).

    Raises:
        ValueError: If Step 8 has not been confirmed, or ``use_tuned=True``
            but no tuned model is available.
    """
    algorithm = run.step_data.get("train", {}).get("algorithm")
    if algorithm is None:
        raise ValueError("Step 8 must be confirmed before Step 9.")

    best_score: float | None = None
    if request.use_tuned:
        tuned_model = run.artifacts.get("tuning_tuned_model")
        if tuned_model is None:
            raise ValueError("No completed tuning run found — start tuning before accepting it.")
        run.artifacts["final_model"] = tuned_model
        best_score = run.artifacts.get("_tuning_best_score")
    else:
        model = run.artifacts.get("trained_model")
        if model is None:
            raise ValueError("Step 8 must be confirmed before Step 9.")
        run.artifacts["final_model"] = model

    run.step_data["tuning"] = {"tuned": request.use_tuned}
    run.step_confirmed["tuning"] = True
    run.step_status["tuning"] = "done"
    nxt = next_step_id("tuning")
    if nxt is not None:
        run.step_status[nxt] = "current"

    if request.use_tuned and best_score is not None:
        run.audit_trail.append(
            ("tuning", f"Hyperparameter tuning: accepted tuned {algorithm} (AUC {best_score:.4f}).")
        )
    elif request.use_tuned:
        run.audit_trail.append(("tuning", f"Hyperparameter tuning: accepted tuned {algorithm}."))
    else:
        run.audit_trail.append(
            ("tuning", f"Hyperparameter tuning: skipped. Kept default {algorithm}.")
        )

    logger.info(
        "run_id=%s confirmed Step 9: algorithm=%s tuned=%s",
        run.run_id,
        algorithm,
        request.use_tuned,
    )
    return TuningConfirmResponse(confirmed=True, tuned=request.use_tuned, best_score=best_score)
