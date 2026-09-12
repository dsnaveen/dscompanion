"""Step 8 (Train a Model) service functions — the REST equivalent of
``dscompanion/app/interactive_step_train.py``, with every ``streamlit`` call stripped out.

Pre-processing applied here (so it isn't duplicated in Step 9+), mirrors
``interactive_step_train.py``'s ``_prepare_split`` exactly:
- Feature column restriction (identifier drops, Step 5 drops).
- Feature processing: a ``FeatureTransformChain`` built from Step 5's confirmed
  ``ColumnRecipe`` per-column decisions — fit on train only, transform on val/test/oot.
- Feature selection (Step 6 drops), subtracted from the chain's *output* column names
  (not the raw input names — column-expanding recipe steps like one-hot encoding
  change them).
- Imbalance handling: ``ImbalanceHandler`` using the Step 7 strategy, applied to train
  only (val/test are never resampled — that would be data leakage).

Two training paths, matching the Streamlit UI: single-algorithm (synchronous, handled
entirely inside ``confirm_train``) and leaderboard (backgrounded via
``start_leaderboard_job`` — trains every algorithm sequentially in a
``threading.Thread``, polled through the generic ``GET /api/runs/{run_id}/jobs/{job_id}``
endpoint already built for this in ``dscompanion/api/routers/jobs.py``). ``confirm_train``
distinguishes the two implicitly: if a leaderboard job already produced a winner
matching the requested algorithm, its already-fitted model is reused rather than
retrained — no separate "which path" field needed on the request.
"""

from __future__ import annotations

import logging
import threading
import time

import pandas as pd

from dscompanion.api.schemas import (
    LeaderboardStartResponse,
    TrainAlgorithmOption,
    TrainConfirmRequest,
    TrainConfirmResponse,
    TrainMetricRow,
    TrainPreviewResponse,
)
from dscompanion.api.state import RunState, RunStateStore
from dscompanion.api.steps import next_step_id
from dscompanion.features import ColumnRecipe, FeatureTransformChain
from dscompanion.models import ModelFactory
from dscompanion.split import DataSplit
from dscompanion.targets import ImbalanceHandler

logger = logging.getLogger(__name__)

__all__ = ["preview_train", "confirm_train", "start_leaderboard_job"]

# Plain-language one-liners shown next to each algorithm in the picker — verbatim
# from interactive_step_train.py's _ALGO_NOTES, kept in sync by hand (no shared
# module between dscompanion/app/ and dscompanion/api/ per this project's convention).
_ALGO_NOTES: dict[str, str] = {
    "xgboost": "XGBoost: strong default choice for most tabular classification problems.",
    "lightgbm": "LightGBM: similar to XGBoost but often faster on larger datasets.",
    "logistic": "Logistic Regression: simple, fast, and easy to explain; works well when "
    "relationships are roughly linear.",
    "random_forest": "Random Forest: robust ensemble, good baseline with minimal tuning.",
    "gradient_boosting": "Gradient Boosting: sklearn's built-in boosting; slower than XGBoost "
    "but no extra dependency.",
    "svm": "SVM: can work well on small datasets; very slow to train; not recommended "
    "for large datasets.",
    "knn": "k-Nearest Neighbours: simple algorithm; slow on large datasets.",
    "decision_tree": "Decision Tree: highly interpretable; tends to overfit without pruning.",
    "extra_trees": "Extra Trees: like Random Forest but uses more randomness; often fast.",
    "adaboost": "AdaBoost: classic boosting method; sensitive to noisy data.",
    "naive_bayes": "Naïve Bayes: very fast and surprisingly good on text-like sparse data.",
}

# Algorithms excluded from the leaderboard comparison (available for single-model
# selection but too slow to include in an automated comparison run).
_LEADERBOARD_EXCLUDE: frozenset[str] = frozenset({"svm"})


# ── Data preparation ──────────────────────────────────────────────────────────


def _feature_cols(run: RunState) -> list[str]:
    """Derive the raw input columns that feed the Step 5 feature-processing chain.

    Args:
        run (RunState): The run holding Steps 1-6's confirmed artifacts.

    Returns:
        list[str]: Original ``train_X`` columns minus the identifier columns
        (Step 2), date column (Step 3), and columns dropped at Step 5. Columns
        removed at Step 6 are subtracted later, inside ``_prepare_split``,
        after the ``FeatureTransformChain`` runs — Step 6 selection operates
        on the chain's output column names, which differ from these raw
        input names for column-expanding recipe steps (e.g. one-hot encoding).
    """
    split: DataSplit = run.artifacts["split"]
    all_cols = list(split.train_X.columns)
    date_col = run.step_data.get("split", {}).get("date_col")
    identifier_cols = run.artifacts.get("identifier_columns") or []
    step5_drops = run.artifacts.get("feature_processing_dropped_columns") or []
    excluded = {c for c in [date_col, *identifier_cols, *step5_drops] if c}
    return [c for c in all_cols if c not in excluded]


def _prepare_split(
    run: RunState, feat_cols: list[str]
) -> tuple[DataSplit, ImbalanceHandler | None]:
    """Apply feature processing (ColumnRecipe chain) + selection + imbalance handling.

    Feature processing is fit on train only via a ``FeatureTransformChain`` built
    from Step 5's confirmed recipes, and applied by ``.transform()`` to val/test/oot.
    Step 6's removed columns are subtracted from the *transformed* output. Imbalance
    handling is applied to train only — applying it to val/test would be leakage.

    Args:
        run (RunState): The run holding Steps 1-7's confirmed artifacts.
        feat_cols (list[str]): Raw input feature columns from ``_feature_cols``.

    Returns:
        tuple[DataSplit, ImbalanceHandler | None]: Processed split ready for
        ``model.fit``, and the fitted ``ImbalanceHandler`` (``None`` when the
        strategy is ``"class_weight"`` or ``"none"``).
    """
    split: DataSplit = run.artifacts["split"]

    recipes: dict[str, ColumnRecipe] = run.artifacts.get("feature_processing_recipes") or {}
    recipes_for_feat_cols = {c: r for c, r in recipes.items() if c in feat_cols}
    chain = FeatureTransformChain(recipes=recipes_for_feat_cols)

    train_X_proc = chain.fit_transform(split.train_X[feat_cols])

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        return chain.transform(df[feat_cols]) if len(df) > 0 else df[[]]

    val_X_proc = _transform(split.val_X)
    test_X_proc = _transform(split.test_X)
    oot_X_proc = _transform(split.oot_X)

    step6_drops = run.artifacts.get("feature_selection_removed_columns") or []
    keep_cols = [c for c in train_X_proc.columns if c not in step6_drops]
    train_X_proc = train_X_proc[keep_cols]
    val_X_proc = val_X_proc[[c for c in keep_cols if c in val_X_proc.columns]]
    test_X_proc = test_X_proc[[c for c in keep_cols if c in test_X_proc.columns]]
    oot_X_proc = oot_X_proc[[c for c in keep_cols if c in oot_X_proc.columns]]

    train_y = split.train_y
    strategy = run.artifacts.get("imbalance_strategy", "class_weight")
    imbalance_handler = None
    if strategy != "none":
        handler = ImbalanceHandler(strategy=strategy)
        train_X_proc, train_y = handler.fit_resample(train_X_proc, train_y)
        imbalance_handler = handler

    processed_split = DataSplit(
        train_X=train_X_proc,
        train_y=train_y,
        val_X=val_X_proc,
        val_y=split.val_y,
        test_X=test_X_proc,
        test_y=split.test_y,
        oot_X=oot_X_proc,
        oot_y=split.oot_y,
        metadata=split.metadata,
    )
    return processed_split, imbalance_handler


def _apply_class_weight(model: object, imbalance_handler: ImbalanceHandler | None) -> None:
    """Best-effort ``class_weight`` application — silently no-ops for estimators
    that don't accept it (e.g. XGBoost). Preserves this exact, silent asymmetry
    from ``interactive_step_train.py`` rather than "fixing" it, so results match
    Streamlit's for the same inputs.

    Args:
        model: Freshly built, unfitted ``BaseDSCompanionModel`` — mutated in place.
        imbalance_handler (ImbalanceHandler | None): Fitted handler from
            ``_prepare_split``, or ``None``.

    Returns:
        None
    """
    if (
        imbalance_handler is not None
        and hasattr(imbalance_handler, "class_weights_")
        and imbalance_handler.class_weights_ is not None
    ):
        try:
            model.estimator.set_params(class_weight=imbalance_handler.class_weights_)
        except Exception:
            pass


def _metrics_rows(metrics_df: pd.DataFrame) -> list[TrainMetricRow]:
    """Convert a ``model.evaluate()`` result into the brief per-split summary rows.

    Args:
        metrics_df (pd.DataFrame): Columns ``split``, ``metric``, ``value``.

    Returns:
        list[TrainMetricRow]: One row per split present in ``metrics_df``.
    """
    rows = []
    for split_name in ("train", "val", "test", "oot"):
        split_data = metrics_df[metrics_df["split"] == split_name]
        if split_data.empty:
            continue

        def _val(metric_key: str) -> float | None:
            vals = split_data[split_data["metric"] == metric_key]["value"]
            return float(vals.iloc[0]) if len(vals) > 0 else None

        rows.append(
            TrainMetricRow(
                split=split_name,
                auc_roc=_val("roc_auc"),
                gini=_val("gini"),
                ks_statistic=_val("ks_statistic"),
            )
        )
    return rows


# ── Preview ───────────────────────────────────────────────────────────────────


def preview_train(run: RunState) -> TrainPreviewResponse:
    """Reports the available algorithms and feature-column count, without training.

    Args:
        run (RunState): The run holding Steps 1-7's confirmed artifacts.

    Returns:
        TrainPreviewResponse: Algorithm picker options and feature count.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed for this run.
    """
    if run.artifacts.get("split") is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 8.")

    task = run.artifacts.get("task", "classification")
    algorithms = ModelFactory.SUPPORTED_ALGORITHMS.get(task, [])
    leaderboard_algorithms = [a for a in algorithms if a not in _LEADERBOARD_EXCLUDE]
    feat_cols = _feature_cols(run)

    logger.info("run_id=%s previewed Step 8: n_features=%d", run.run_id, len(feat_cols))
    return TrainPreviewResponse(
        algorithms=[TrainAlgorithmOption(name=a, note=_ALGO_NOTES.get(a, "")) for a in algorithms],
        leaderboard_algorithms=leaderboard_algorithms,
        n_features=len(feat_cols),
    )


# ── Confirm (single-algorithm path, and leaderboard-winner acceptance) ────────


def confirm_train(run: RunState, request: TrainConfirmRequest) -> TrainConfirmResponse:
    """Trains (or reuses an already-trained leaderboard winner) and persists the result.

    If a completed leaderboard job produced a winner matching
    ``request.algorithm``, its already-fitted model is reused — no retraining.
    Otherwise, trains ``request.algorithm`` synchronously against a freshly
    prepared split (the single-algorithm path; also the fallback if the
    leaderboard was never run).

    Args:
        run (RunState): The run to persist into.
        request (TrainConfirmRequest): The algorithm to confirm.

    Returns:
        TrainConfirmResponse: Confirmation and a brief metrics summary.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed, or if training
            fails (message includes the underlying exception).
    """
    if run.artifacts.get("split") is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 8.")

    task = run.artifacts.get("task", "classification")
    algorithm = request.algorithm

    leaderboard_result = run.artifacts.get("train_leaderboard_result")
    if leaderboard_result is not None and leaderboard_result.get("winner") == algorithm:
        model = leaderboard_result["winner_model"]
        processed_split: DataSplit = run.artifacts["processed_split"]
        if model is None:
            raise ValueError(f"Leaderboard winner {algorithm!r} has no fitted model available.")
        metrics_df = model.evaluate(processed_split)
    else:
        feat_cols = _feature_cols(run)
        if not feat_cols:
            raise ValueError(
                "No feature columns remain after applying your Step 5 and Step 6 "
                "decisions. Go back and un-drop at least one column."
            )
        try:
            processed_split, imbalance_handler = _prepare_split(run, feat_cols)
            model = ModelFactory.build(task=task, algorithm=algorithm)
            _apply_class_weight(model, imbalance_handler)
            eval_set = None
            if len(processed_split.val_X) > 0:
                eval_set = [(processed_split.val_X, processed_split.val_y)]
            model.fit(processed_split.train_X, processed_split.train_y, eval_set=eval_set)
            metrics_df = model.evaluate(processed_split)
        except Exception as exc:
            logger.exception("run_id=%s Step 8 training failed: %s", run.run_id, algorithm)
            raise ValueError(f"Training failed: {exc}") from exc
        run.artifacts["processed_split"] = processed_split

    run.artifacts["trained_model"] = model
    run.step_data["train"] = {"algorithm": algorithm}
    run.step_confirmed["train"] = True
    run.step_status["train"] = "done"
    nxt = next_step_id("train")
    if nxt is not None:
        run.step_status[nxt] = "current"
    run.audit_trail.append(("train", f"Trained model: {algorithm}."))

    logger.info("run_id=%s confirmed Step 8: algorithm=%s", run.run_id, algorithm)
    return TrainConfirmResponse(
        confirmed=True, algorithm=algorithm, metrics=_metrics_rows(metrics_df)
    )


# ── Leaderboard (backgrounded) ─────────────────────────────────────────────────


def _leaderboard_worker(
    run_id: str,
    artifacts: dict,
    processed_split: DataSplit,
    imbalance_handler: ImbalanceHandler | None,
    task: str,
    algorithms: list[str],
    store: RunStateStore,
    job_id: str,
) -> None:
    """Trains each algorithm sequentially in a background thread.

    Mutates ``job.progress``/``job.result`` (fetched fresh via
    ``store.get_job()`` under its own lock) and, once a winner is found,
    writes the fitted model directly into the shared ``artifacts`` dict — a
    single ``dict.__setitem__`` is atomic under the GIL, matching how
    ``interactive_step_train.py``'s own ``result: dict`` was written
    lock-free (only the incrementally-polled ``progress`` list needs a lock).

    Args:
        run_id (str): The run this job belongs to (for logging only).
        artifacts (dict): The run's shared ``artifacts`` dict — mutated
            in place once a winner is found.
        processed_split (DataSplit): Pre-processed split, ready for
            ``model.fit``.
        imbalance_handler (ImbalanceHandler | None): Fitted handler from
            ``_prepare_split``, or ``None``.
        task (str): Model task — always ``"classification"`` at this point.
        algorithms (list[str]): Algorithms to compare, in training order.
        store (RunStateStore): Used to fetch the live ``JobState`` to write
            into (avoids holding a stale reference across the run).
        job_id (str): This job's identifier.

    Returns:
        None
    """
    job = store.get_job(run_id, job_id)
    try:
        eval_set = None
        if len(processed_split.val_X) > 0:
            eval_set = [(processed_split.val_X, processed_split.val_y)]

        rows: list[dict] = []
        fitted_models: dict[str, object] = {}

        for algo in algorithms:
            with job.lock:
                job.progress.append({"algorithm": algo, "status": "running", "detail": None})
            t0 = time.perf_counter()
            try:
                model = ModelFactory.build(task=task, algorithm=algo)
                _apply_class_weight(model, imbalance_handler)
                model.fit(processed_split.train_X, processed_split.train_y, eval_set=eval_set)
                elapsed = time.perf_counter() - t0
                metrics_df = model.evaluate(processed_split)
                eval_split_name = "val" if len(processed_split.val_X) > 0 else "test"
                split_row = metrics_df[metrics_df["split"] == eval_split_name]
                roc_auc = split_row[split_row["metric"] == "roc_auc"]["value"]
                auc_val = float(roc_auc.iloc[0]) if len(roc_auc) > 0 else float("nan")
                row = {
                    "algorithm": algo,
                    "status": "ok",
                    "fit_time_s": round(elapsed, 1),
                    "roc_auc": round(auc_val, 4),
                    "error": None,
                }
                fitted_models[algo] = model
                detail = f"AUC {auc_val:.3f} ({elapsed:.1f}s)"
                entry_status = "ok"
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.warning("run_id=%s leaderboard: %s failed: %s", run_id, algo, exc)
                row = {
                    "algorithm": algo,
                    "status": "failed",
                    "fit_time_s": round(elapsed, 1),
                    "roc_auc": float("nan"),
                    "error": str(exc),
                }
                entry_status, detail = "failed", str(exc)
            rows.append(row)
            with job.lock:
                job.progress[-1] = {"algorithm": algo, "status": entry_status, "detail": detail}

        lb_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False, na_position="last")
        lb_df = lb_df.reset_index(drop=True)
        ok_rows = lb_df[lb_df["status"] == "ok"]

        if ok_rows.empty:
            with job.lock:
                job.status = "error"
                job.error = (
                    "Every algorithm failed to train. Check that your data has no "
                    "remaining missing values or constant columns, then go back and "
                    "adjust Steps 5-6."
                )
                job.result = {"rows": rows, "winner": None}
            artifacts["train_leaderboard_result"] = {"winner": None, "winner_model": None}
            return

        winner = ok_rows.iloc[0]["algorithm"]
        winner_model = fitted_models.get(winner)
        artifacts["train_leaderboard_result"] = {"winner": winner, "winner_model": winner_model}
        with job.lock:
            job.status = "done"
            job.result = {"rows": rows, "winner": winner}
        logger.info("run_id=%s leaderboard done: winner=%s", run_id, winner)
    except Exception as exc:
        # Belt-and-braces: guarantee the job always reaches a terminal state, even
        # on a failure this function didn't anticipate — the poller reads
        # job.status/.result/.error unconditionally once status != "running".
        logger.exception("run_id=%s leaderboard worker crashed unexpectedly", run_id)
        with job.lock:
            job.status = "error"
            job.error = f"Unexpected error during comparison: {exc}"
            job.result = {"rows": [], "winner": None}
        artifacts["train_leaderboard_result"] = {"winner": None, "winner_model": None}


def start_leaderboard_job(run: RunState, store: RunStateStore) -> LeaderboardStartResponse:
    """Prepares training data synchronously, then hands training off to a background job.

    Data preparation (feature processing, imbalance handling) is comparatively
    fast and stays on the request thread. Only the per-algorithm training loop
    — the dominant cost, potentially minutes — runs in a background
    ``threading.Thread``, mirroring ``interactive_step_train.py``'s
    ``_start_leaderboard_thread``/``_leaderboard_worker`` split.

    Args:
        run (RunState): The run to prepare data for and start the job under.
        store (RunStateStore): Used to create the job and fetch it from within
            the background thread.

    Returns:
        LeaderboardStartResponse: The new job's id, for polling.

    Raises:
        ValueError: If Steps 1-3 have not all been confirmed, or no feature
            columns remain.
    """
    if run.artifacts.get("split") is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 8.")

    task = run.artifacts.get("task", "classification")
    algorithms = [
        a for a in ModelFactory.SUPPORTED_ALGORITHMS.get(task, []) if a not in _LEADERBOARD_EXCLUDE
    ]
    feat_cols = _feature_cols(run)
    if not feat_cols:
        raise ValueError(
            "No feature columns remain after applying your Step 5 and Step 6 decisions. "
            "Go back and un-drop at least one column."
        )

    processed_split, imbalance_handler = _prepare_split(run, feat_cols)
    run.artifacts["processed_split"] = processed_split

    job = store.create_job(run.run_id, "train")
    thread = threading.Thread(
        target=_leaderboard_worker,
        args=(
            run.run_id,
            run.artifacts,
            processed_split,
            imbalance_handler,
            task,
            algorithms,
            store,
            job.job_id,
        ),
        daemon=True,
    )
    thread.start()

    logger.info(
        "run_id=%s started Step 8 leaderboard job_id=%s: %d algorithms",
        run.run_id,
        job.job_id,
        len(algorithms),
    )
    return LeaderboardStartResponse(job_id=job.job_id)
