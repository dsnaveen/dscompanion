"""Interactive mode — Step 8: Train a Model (or Compare Several).

Asks the user to either pick a single algorithm or run a leaderboard
comparison, pre-processes the raw Step 3 split using decisions from Steps 5-7,
trains the chosen model(s), and confirms the winner before continuing.

Pre-processing applied here (so it isn't duplicated in Steps 9+):
- Feature column restriction (identifier drops, Step 5 drops)
- Feature processing: a ``FeatureTransformChain`` built from Step 5's
  confirmed ``ColumnRecipe`` per-column decisions — fit on train only,
  transform on val/test/oot.
- Feature selection (Step 6 drops), subtracted from the chain's *output*
  column names (not the raw input names — column-expanding recipe steps
  like one-hot encoding change them).
- Imbalance handling: ``ImbalanceHandler`` using the Step 7 strategy, applied
  to train only (val/test are never resampled — that would be data leakage).

``"oversample"`` is excluded from ``ImbalanceConfig`` choices and will never
reach here from Interactive mode (see ``interactive_step7.py``'s docstring).

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging
import threading
import time

import pandas as pd
import streamlit as st
from interactive_state import STEP_NAMES, add_audit_entry, set_step_confirmed, set_step_status

from dscompanion.features import ColumnRecipe, FeatureTransformChain
from dscompanion.models import ModelFactory
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["render_step_train"]

# Plain-language one-liners shown next to each algorithm in the selectbox.
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
    "lda": "Linear Discriminant Analysis: fast, interpretable, works well when classes are "
    "roughly Gaussian with similar covariance.",
    "qda": "Quadratic Discriminant Analysis: like LDA but allows each class its own "
    "covariance; more flexible, needs more data per class.",
    "mlp": "Multi-Layer Perceptron: small neural network; can capture non-linear patterns "
    "but slower to train and less interpretable.",
}


# Algorithms excluded from the leaderboard comparison (available for single-model
# selection but too slow to include in an automated comparison run).
_LEADERBOARD_EXCLUDE: frozenset[str] = frozenset({"svm"})


# ── Data preparation ──────────────────────────────────────────────────────────


def _feature_cols(split: DataSplit) -> list[str]:
    """Derive the raw input columns that feed the Step 5 feature-processing chain.

    Args:
        split (DataSplit): Raw split from Step 3.

    Returns:
        list[str]: Original train_X columns minus the identifier columns
        (Step 2), date column (Step 3), and columns dropped at Step 5.
        Columns removed at Step 6 are subtracted later, inside
        ``_prepare_split``, *after* the ``FeatureTransformChain`` runs —
        Step 6 selection operates on the chain's output column names, which
        for column-expanding recipe steps (e.g. one-hot encoding) differ
        from these raw input names.
    """
    all_cols = list(split.train_X.columns)
    date_col = st.session_state.get("int.split.date_col")
    identifier_cols = st.session_state.get("interactive.identifier_columns") or []
    step5_drops = st.session_state.get("interactive.feature_processing_dropped_columns") or []
    excluded = {c for c in [date_col, *identifier_cols, *step5_drops] if c}
    return [c for c in all_cols if c not in excluded]


def _prepare_split(split: DataSplit, feat_cols: list[str]) -> tuple[DataSplit, object | None]:
    """Apply feature processing (ColumnRecipe chain) + selection + imbalance handling.

    Feature processing is fit on train only via a ``FeatureTransformChain``
    built from Step 5's confirmed ``interactive.feature_processing_recipes``,
    and applied by ``.transform()`` to val/test/oot — mirrors the
    train-only/held-out-clean contract every dscompanion transformer follows (see
    ``NoiseInjector``). Step 6's removed columns are subtracted from the
    *transformed* output, since column-expanding recipe steps (e.g. one-hot
    encoding) mean those names differ from the raw ``feat_cols`` input names.
    Imbalance handling is applied to train only — applying it to val/test
    would constitute data leakage.

    Args:
        split (DataSplit): Raw split from Step 3.
        feat_cols (list[str]): Raw input feature columns from ``_feature_cols``.

    Returns:
        tuple[DataSplit, ImbalanceHandler | None]: Processed split ready for
        ``model.fit``/``Leaderboard.run``, and the fitted
        ``ImbalanceHandler`` (``None`` when strategy is ``"class_weight"`` or
        ``"none"``).
    """
    # --- Feature processing (Step 5's ColumnRecipe chain) ---
    recipes: dict[str, ColumnRecipe] = (
        st.session_state.get("interactive.feature_processing_recipes") or {}
    )
    recipes_for_feat_cols = {c: r for c, r in recipes.items() if c in feat_cols}
    chain = FeatureTransformChain(recipes=recipes_for_feat_cols)

    train_X_proc = chain.fit_transform(split.train_X[feat_cols])

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        return chain.transform(df[feat_cols]) if len(df) > 0 else df[[]]

    val_X_proc = _transform(split.val_X)
    test_X_proc = _transform(split.test_X)
    oot_X_proc = _transform(split.oot_X)

    # --- Feature selection (Step 6, post-transform column names) ---
    step6_drops = st.session_state.get("interactive.feature_selection_removed_columns") or []
    keep_cols = [c for c in train_X_proc.columns if c not in step6_drops]
    train_X_proc = train_X_proc[keep_cols]
    val_X_proc = val_X_proc[[c for c in keep_cols if c in val_X_proc.columns]]
    test_X_proc = test_X_proc[[c for c in keep_cols if c in test_X_proc.columns]]
    oot_X_proc = oot_X_proc[[c for c in keep_cols if c in oot_X_proc.columns]]

    train_y = split.train_y

    # --- Imbalance handling (train only) ---
    strategy = st.session_state.get("interactive.imbalance_strategy", "class_weight")
    imbalance_handler = None
    if strategy != "none":
        from dscompanion.targets import ImbalanceHandler

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


# ── Single-algorithm path ─────────────────────────────────────────────────────


def _render_single_path(split: DataSplit, feat_cols: list[str]) -> None:
    """Render the 'I'll choose' single-algorithm training path.

    Args:
        split (DataSplit): Raw split from Step 3.
        feat_cols (list[str]): Feature columns derived from Steps 5-7.

    Returns:
        None
    """
    task = st.session_state.get("interactive.task", "classification")
    algorithms = ModelFactory.SUPPORTED_ALGORITHMS.get(task, [])

    already_trained = "int.train.trained_model" in st.session_state

    if not already_trained:
        chosen = st.selectbox(
            "Choose an algorithm",
            options=algorithms,
            format_func=lambda a: f"{a}: {_ALGO_NOTES.get(a, '')}",
            index=0,
            key="int.train.algorithm_select",
        )

        if st.button("Train and continue", key="int.train.train_single"):
            st.session_state["int.train.algorithm"] = chosen
            with st.spinner(f"Training {chosen}..."):
                try:
                    processed_split, imbalance_handler = _prepare_split(split, feat_cols)
                    model = ModelFactory.build(task=task, algorithm=chosen)

                    # Apply class weights if strategy=class_weight
                    if (
                        imbalance_handler is not None
                        and hasattr(imbalance_handler, "class_weights_")
                        and imbalance_handler.class_weights_ is not None
                    ):
                        try:
                            model.estimator.set_params(
                                class_weight=imbalance_handler.class_weights_
                            )
                        except Exception as exc:
                            logger.debug("Estimator does not support class_weight: %s", exc)

                    eval_set = None
                    if len(processed_split.val_X) > 0:
                        eval_set = [(processed_split.val_X, processed_split.val_y)]
                    model.fit(processed_split.train_X, processed_split.train_y, eval_set=eval_set)
                    metrics_df = model.evaluate(processed_split)
                    st.session_state["int.train.trained_model"] = model
                    st.session_state["int.train.metrics_df"] = metrics_df
                    st.session_state["int.train.processed_split"] = processed_split
                except Exception as exc:
                    logger.exception("Interactive mode — Step 8 training failed: %s", exc)
                    st.error(
                        f"Training failed: {exc}\n\n" "You can try a different algorithm below."
                    )
                    if st.button("Try a different algorithm", key="int.train.retry_single"):
                        st.session_state.pop("int.train.algorithm", None)
                        st.rerun()
                    return
            st.rerun()

    if already_trained:
        algorithm = st.session_state.get("int.train.algorithm", "?")
        metrics_df: pd.DataFrame = st.session_state["int.train.metrics_df"]
        st.success(f"**{algorithm}** trained successfully.")
        _render_metrics_summary(metrics_df)

        col_use, col_retrain = st.columns([2, 1])
        with col_use:
            if st.button(
                f"Accept {algorithm} and continue", key="int.train.confirm_single", width="stretch"
            ):
                _confirm_step8(algorithm)
        with col_retrain:
            if st.button(
                "Train a different algorithm", key="int.train.retrain_single", width="stretch"
            ):
                _reset_step8_state()
                st.rerun()


# ── Leaderboard path ──────────────────────────────────────────────────────────


def _render_leaderboard_path(split: DataSplit, feat_cols: list[str]) -> None:
    """Render the 'Compare for me' leaderboard path.

    Trains every supported algorithm in a background thread so the UI stays
    responsive, showing live per-algorithm progress via a polling fragment,
    then presents the ranked table and lets the user pick which
    successfully-trained algorithm to use (defaulting to the highest-AUC
    winner) before continuing.

    Args:
        split (DataSplit): Raw split from Step 3.
        feat_cols (list[str]): Feature columns derived from Steps 5-7.

    Returns:
        None
    """
    task = st.session_state.get("interactive.task", "classification")
    algorithms = [
        a for a in ModelFactory.SUPPORTED_ALGORITHMS.get(task, []) if a not in _LEADERBOARD_EXCLUDE
    ]
    already_run = "int.train.leaderboard_df" in st.session_state
    running = st.session_state.get("int.train.lb_running", False)

    if not already_run and not running:
        excluded_note = (
            " (SVM excluded, too slow for automated comparison)"
            if _LEADERBOARD_EXCLUDE & set(ModelFactory.SUPPORTED_ALGORITHMS.get(task, []))
            else ""
        )
        st.caption(
            f"Will compare {len(algorithms)} algorithms{excluded_note}. This may take a few "
            "minutes depending on dataset size. Algorithms that fail to train are skipped. "
            "The rest of the app stays usable while this runs."
        )
        if st.button("Start comparison", key="int.train.start_leaderboard"):
            _start_leaderboard_thread(split, feat_cols, task, algorithms)
            st.rerun()

        prep_error = st.session_state.pop("int.train.lb_prep_error", None)
        if prep_error:
            st.error(f"Could not prepare training data: {prep_error}")

    if running:
        _render_leaderboard_progress(algorithms)

    if already_run:
        lb_df: pd.DataFrame = st.session_state["int.train.leaderboard_df"]
        winner: str = st.session_state["int.train.winner"]
        fitted_models: dict = st.session_state.get("int.train.fitted_models", {})

        st.markdown(f"**Recommended winner: `{winner}`** (highest AUC)")
        _render_leaderboard_table(lb_df)

        ok_algorithms = lb_df[lb_df["status"] == "ok"]["algorithm"].tolist()
        default_index = ok_algorithms.index(winner) if winner in ok_algorithms else 0
        selected = st.selectbox(
            "Choose which model to use",
            ok_algorithms,
            index=default_index,
            key="int.train.selected_algorithm",
            help="Defaults to the recommended winner, but you can pick any algorithm that "
            "trained successfully above.",
        )

        selected_model = fitted_models.get(selected)
        if selected_model is not None:
            metrics_df = selected_model.evaluate(st.session_state["int.train.processed_split"])
            _render_metrics_summary(metrics_df)

        st.session_state["int.train.algorithm"] = selected
        st.session_state["int.train.winner_model"] = selected_model

        col_use, col_retrain = st.columns([2, 1])
        with col_use:
            if st.button(
                f"Use {selected} and continue",
                key="int.train.confirm_leaderboard",
                width="stretch",
            ):
                _confirm_step8(selected)
        with col_retrain:
            if st.button("Compare again", key="int.train.rerun_leaderboard", width="stretch"):
                _reset_step8_state()
                st.rerun()


def _start_leaderboard_thread(
    split: DataSplit, feat_cols: list[str], task: str, algorithms: list[str]
) -> None:
    """Prepare training data synchronously, then hand training off to a background thread.

    Data preparation (imputation, encoding, imbalance handling) reads
    ``st.session_state`` directly via ``_prepare_split`` and is comparatively
    fast, so it stays on the main script thread where that's safe. Only the
    actual per-algorithm training loop — the dominant cost, potentially
    minutes — moves to a background ``threading.Thread``, which is given
    plain arguments and shared mutable containers instead of any
    Streamlit dependency (see ``_leaderboard_worker``).

    Args:
        split (DataSplit): Raw split from Step 3.
        feat_cols (list[str]): Feature columns derived from Steps 5-7.
        task (str): Model task — always ``"classification"`` at this point.
        algorithms (list[str]): Algorithms to compare.

    Returns:
        None
    """
    try:
        processed_split, imbalance_handler = _prepare_split(split, feat_cols)
    except Exception as exc:
        st.session_state["int.train.lb_prep_error"] = str(exc)
        return

    st.session_state["int.train.processed_split"] = processed_split

    progress: list[dict] = []
    lock = threading.Lock()
    result: dict = {}
    thread = threading.Thread(
        target=_leaderboard_worker,
        args=(processed_split, imbalance_handler, task, algorithms, progress, lock, result),
        daemon=True,
    )
    st.session_state["int.train.lb_progress"] = progress
    st.session_state["int.train.lb_progress_lock"] = lock
    st.session_state["int.train.lb_result"] = result
    st.session_state["int.train.lb_thread"] = thread
    st.session_state["int.train.lb_running"] = True
    thread.start()


def _leaderboard_worker(
    processed_split: DataSplit,
    imbalance_handler: object | None,
    task: str,
    algorithms: list[str],
    progress: list[dict],
    progress_lock: threading.Lock,
    result: dict,
) -> None:
    """Train each algorithm sequentially in a background thread.

    Runs off the Streamlit script-run thread — must never call any
    Streamlit rendering function (``st.*``) directly, since Streamlit ties
    rendering to the script-run thread and a foreign thread calling ``st.*``
    raises "missing ScriptRunContext" errors. This function only mutates
    ``progress`` (under ``progress_lock``) and ``result``; the caller
    (running on the main thread, via an ``st.fragment`` poll in
    ``_render_leaderboard_progress``) is responsible for all rendering.

    Args:
        processed_split (DataSplit): Pre-processed split (feature processing
            and imbalance handling already applied) ready for ``model.fit``.
        imbalance_handler (object | None): Fitted ``ImbalanceHandler``, or
            ``None`` when the strategy was ``"class_weight"``/``"none"``.
        task (str): Model task — always ``"classification"`` at this point.
        algorithms (list[str]): Algorithms to compare, in training order.
        progress (list[dict]): Shared list this function appends/updates one
            entry per algorithm on: ``{"algorithm", "status" ("running" |
            "ok" | "failed"), "detail"}``. Mutated only under
            ``progress_lock``.
        progress_lock (threading.Lock): Lock guarding ``progress`` against
            concurrent read (by the polling fragment) and write (by this
            thread).
        result (dict): Shared dict this function fills exactly once, after
            every algorithm has been attempted (or immediately, on an
            unexpected crash): ``{"lb_df": pd.DataFrame, "winner": str |
            None, "winner_model": object | None, "fitted_models":
            dict[str, object], "error": str | None}``. ``fitted_models``
            holds every successfully-trained algorithm's fitted model, not
            just the winner's, so the UI can offer a choice without
            retraining.

    Returns:
        None
    """
    try:
        eval_set = None
        if len(processed_split.val_X) > 0:
            eval_set = [(processed_split.val_X, processed_split.val_y)]

        rows: list[dict] = []
        fitted_models: dict[str, object] = {}

        for algo in algorithms:
            with progress_lock:
                progress.append({"algorithm": algo, "status": "running", "detail": None})
            t0 = time.perf_counter()
            try:
                model = ModelFactory.build(task=task, algorithm=algo)
                if (
                    imbalance_handler is not None
                    and hasattr(imbalance_handler, "class_weights_")
                    and imbalance_handler.class_weights_ is not None
                ):
                    try:
                        model.estimator.set_params(class_weight=imbalance_handler.class_weights_)
                    except Exception as exc:
                        logger.debug("Estimator does not support class_weight: %s", exc)

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
                entry_status, detail = "ok", f"AUC {auc_val:.3f} ({elapsed:.1f}s)"
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.warning("Leaderboard: %s failed: %s", algo, exc)
                row = {
                    "algorithm": algo,
                    "status": "failed",
                    "fit_time_s": round(elapsed, 1),
                    "roc_auc": float("nan"),
                    "error": str(exc),
                }
                entry_status, detail = "failed", str(exc)
            rows.append(row)
            with progress_lock:
                progress[-1] = {"algorithm": algo, "status": entry_status, "detail": detail}

        lb_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False, na_position="last")
        lb_df = lb_df.reset_index(drop=True)

        ok_rows = lb_df[lb_df["status"] == "ok"]
        if ok_rows.empty:
            result["error"] = (
                "Every algorithm failed to train. Check that your data has no remaining "
                "missing values or constant columns, then go back and adjust Steps 5-6."
            )
            result["lb_df"] = lb_df
            result["winner"] = None
            result["winner_model"] = None
            result["fitted_models"] = {}
            return

        winner = ok_rows.iloc[0]["algorithm"]
        result["lb_df"] = lb_df
        result["winner"] = winner
        result["winner_model"] = fitted_models.get(winner)
        # Every successfully-fitted model, not just the winner — lets the user pick a
        # different algorithm from the leaderboard without retraining from scratch.
        result["fitted_models"] = fitted_models
        result["error"] = None
    except Exception as exc:
        # Belt-and-braces: guarantee `result` is always populated once the
        # thread finishes, even on a failure this function didn't
        # anticipate — the polling fragment reads result["error"]/["lb_df"]
        # unconditionally once thread.is_alive() is False.
        logger.exception("Leaderboard worker crashed unexpectedly")
        result["error"] = f"Unexpected error during comparison: {exc}"
        result["lb_df"] = pd.DataFrame()
        result["winner"] = None
        result["winner_model"] = None
        result["fitted_models"] = {}


@st.fragment(run_every=1)
def _render_leaderboard_progress(algorithms: list[str]) -> None:
    """Poll and render live per-algorithm leaderboard progress.

    Reruns on its own every second via ``st.fragment`` instead of blocking
    or rerunning the whole page, so the rest of the app stays usable while
    the background training thread (started by
    ``_start_leaderboard_thread``) runs. Once the thread finishes, transfers
    its result into the stable ``int.train.leaderboard_df`` / ``winner`` /
    ``winner_model`` keys the rest of Step 8 already expects — the
    finished-state rendering in ``_render_leaderboard_path`` needs no
    changes — then triggers one full-page rerun to move out of the
    "in progress" branch.

    Args:
        algorithms (list[str]): Full list of algorithms being compared,
            used only to show an overall progress count.

    Returns:
        None
    """
    progress: list[dict] = st.session_state.get("int.train.lb_progress", [])
    lock: threading.Lock = st.session_state["int.train.lb_progress_lock"]
    with lock:
        snapshot = list(progress)

    with st.status(
        f"Comparing Algorithms... ({len(snapshot)}/{len(algorithms)})", expanded=True
    ) as status:
        icons = {"running": "⏳", "ok": "✅", "failed": "❌"}
        for entry in snapshot:
            detail = f": {entry['detail']}" if entry["detail"] else "..."
            status.write(f"{icons[entry['status']]} `{entry['algorithm']}`{detail}")

        thread: threading.Thread = st.session_state["int.train.lb_thread"]
        if not thread.is_alive():
            result: dict = st.session_state["int.train.lb_result"]
            st.session_state["int.train.lb_running"] = False
            if result.get("error"):
                status.update(
                    label="All Algorithms Failed: No Winner", state="error", expanded=True
                )
                st.error(result["error"])
            else:
                st.session_state["int.train.leaderboard_df"] = result["lb_df"]
                st.session_state["int.train.winner"] = result["winner"]
                st.session_state["int.train.winner_model"] = result["winner_model"]
                st.session_state["int.train.fitted_models"] = result.get("fitted_models", {})
                n_ok = int((result["lb_df"]["status"] == "ok").sum())
                status.update(
                    label=f"Done: {n_ok}/{len(algorithms)} Succeeded. "
                    f"Winner: {result['winner']}",
                    state="complete",
                    expanded=False,
                )
            st.rerun()


def _render_leaderboard_table(lb_df: pd.DataFrame) -> None:
    """Render the leaderboard results as a styled dataframe.

    Args:
        lb_df (pd.DataFrame): Leaderboard result with columns
            ``algorithm``, ``status``, ``fit_time_s``, ``roc_auc``, ``error``.

    Returns:
        None
    """
    display = lb_df[["algorithm", "status", "roc_auc", "fit_time_s", "error"]].copy()
    display.columns = ["Algorithm", "Status", "AUC (Val/Test)", "Time (s)", "Error"]
    display["Status"] = display["Status"].map({"ok": "✅ ok", "failed": "❌ failed"})
    display["AUC (Val/Test)"] = display["AUC (Val/Test)"].apply(
        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
    )
    st.dataframe(display, hide_index=True, width="stretch")


# ── Metrics summary ───────────────────────────────────────────────────────────


def _render_metrics_summary(metrics_df: pd.DataFrame) -> None:
    """Show a compact AUC/Gini summary for each evaluated split.

    Full metric breakdowns with plain-language glosses belong to Step 10 —
    this summary is intentionally brief: just enough to confirm the model
    trained successfully and give a sense of performance.

    Args:
        metrics_df (pd.DataFrame): DataFrame returned by
            ``model.evaluate(split)`` with columns ``split``, ``metric``,
            ``value``.

    Returns:
        None
    """
    key_metrics = {"roc_auc": "AUC-ROC", "gini": "Gini", "ks_statistic": "KS"}
    rows = []
    for split_name in ("train", "val", "test", "oot"):
        split_data = metrics_df[metrics_df["split"] == split_name]
        if split_data.empty:
            continue
        row = {"Split": split_name}
        for metric_key, label in key_metrics.items():
            vals = split_data[split_data["metric"] == metric_key]["value"]
            row[label] = f"{vals.iloc[0]:.4f}" if len(vals) > 0 else "—"
        rows.append(row)

    if rows:
        st.caption(
            "Quick performance summary. Full metrics with plain-language explanations "
            f"are shown in the {STEP_NAMES['evaluate']} step."
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="content")


# ── Confirm ───────────────────────────────────────────────────────────────────


def _confirm_step8(algorithm: str) -> None:
    """Commit the algorithm choice and trained model to session state.

    Args:
        algorithm (str): Confirmed algorithm name.

    Returns:
        None
    """
    st.session_state["interactive.train_algorithm"] = algorithm
    set_step_confirmed("train", True)
    set_step_status("train", "done")
    add_audit_entry("train", f"Trained model: {algorithm}.")
    logger.info("Interactive mode — Step 8 confirmed: algorithm=%s", algorithm)


def _reset_step8_state() -> None:
    """Clear every Step 8 result so the user can rebuild from scratch.

    Used by both paths' "train a different algorithm"/"compare again"
    buttons. Going back from Step 9 via the wizard's Back button only
    invalidates ``interactive.step_confirmed``/``step_status`` — it never
    touches this step's own cached results (``int.train.*``), so without
    this reset the user would land back on Step 8 and see nothing but the
    same frozen leaderboard or single-algorithm result with no way to
    actually retrain.

    Args:
        None

    Returns:
        None
    """
    for key in (
        "int.train.path",
        "int.train.algorithm",
        "int.train.algorithm_select",
        "int.train.trained_model",
        "int.train.metrics_df",
        "int.train.processed_split",
        "int.train.leaderboard_df",
        "int.train.winner",
        "int.train.winner_model",
        "int.train.fitted_models",
        "int.train.selected_algorithm",
        "int.train.lb_running",
        "int.train.lb_progress",
        "int.train.lb_progress_lock",
        "int.train.lb_result",
        "int.train.lb_thread",
        "int.train.lb_prep_error",
    ):
        st.session_state.pop(key, None)


# ── Main entrypoint ───────────────────────────────────────────────────────────


def render_step_train(split: DataSplit) -> bool:
    """Render Step 8 (Train a Model or Compare Several) and report confirmation.

    Asks the user to choose between picking one algorithm themselves or
    running a leaderboard comparison to find the best one.  Processes the
    raw split from Step 3 using feature decisions from Steps 5-6 and the
    imbalance strategy from Step 7, then trains accordingly.  The full
    metric breakdown belongs to Step 10 — this step shows only a brief
    AUC/Gini summary so the user can confirm the run succeeded.

    Args:
        split (DataSplit): Raw split confirmed at Step 3 — feature
            processing is applied here from within this function.

    Returns:
        bool: ``True`` once the user clicks the confirm button.  ``False``
        while still in progress.
    """
    already_confirmed = st.session_state["interactive.step_confirmed"]["train"]
    if already_confirmed:
        return True

    st.subheader("Step 8: Train a model")
    st.caption(
        "dscompanion will apply the feature processing and imbalance decisions from the "
        "previous steps, then train the chosen algorithm on your training data."
    )

    feat_cols = _feature_cols(split)
    if not feat_cols:
        st.error(
            "No feature columns remain after applying your Step 5 and Step 6 decisions. "
            "Go back and un-drop at least one column."
        )
        return False

    st.caption(f"Features available for training: {len(feat_cols)}")

    path = st.session_state.get("int.train.path")

    if path is None:
        st.markdown("**How Would You Like to Choose an Algorithm?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "I'll choose one",
                key="int.train.pick_single",
                width="stretch",
                help="Pick a specific algorithm from a list with plain-language descriptions.",
            ):
                st.session_state["int.train.path"] = "single"
                st.rerun()
        with col2:
            if st.button(
                "Compare several for me (leaderboard)",
                key="int.train.pick_leaderboard",
                width="stretch",
                help="Train all supported algorithms and automatically pick the best one.",
            ):
                st.session_state["int.train.path"] = "leaderboard"
                st.rerun()
        return False

    if path == "single":
        _render_single_path(split, feat_cols)
    elif path == "leaderboard":
        _render_leaderboard_path(split, feat_cols)

    return st.session_state["interactive.step_confirmed"]["train"]
