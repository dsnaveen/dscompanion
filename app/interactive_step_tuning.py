"""Interactive mode — Step 9: Hyperparameter Tuning (optional).

Offers the user a choice to tune the Step 8 model using Optuna, shows a
before/after comparison of default vs. tuned parameters and the AUC improvement,
then requires an explicit confirm before the tuned model replaces the default one.

Tuning is optional — the user can skip and carry the Step 8 model forward as-is.
Two algorithms lack a search space (``gradient_boosting`` and ``lightgbm``
classification) — this step auto-skips for them with a clear explanation rather
than crashing.

``Tuner.run()`` → ``study_.optimize()`` is a blocking synchronous call with no
external callback hook; per-trial progress cannot be surfaced without modifying
the core backend.  Instead a ``st.status()`` container shows elapsed time until
the run completes.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging
import math
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from interactive_state import add_audit_entry, dark_fig, set_step_confirmed, set_step_status

from dscompanion.tuning.search_spaces import PREDEFINED_PARAMS
from dscompanion.tuning.tuner import Tuner

logger = logging.getLogger(__name__)

__all__ = ["render_step_tuning"]

# Algorithms that have no classification search space in search_spaces.py.
# Tuning is skipped automatically for these with a plain-language explanation.
_NO_SEARCH_SPACE: frozenset[str] = frozenset({"gradient_boosting", "lightgbm"})

# Rough seconds-per-trial estimate by algorithm — used only for the pre-run
# time estimate shown to the user, never for any logic decision.
_APPROX_SECS_PER_TRIAL: dict[str, float] = {
    "logistic": 0.5,
    "naive_bayes": 0.3,
    "decision_tree": 0.8,
    "knn": 1.5,
    "adaboost": 2.0,
    "extra_trees": 3.0,
    "random_forest": 4.0,
    "xgboost": 5.0,
    "svm": 12.0,
}

_DEFAULT_N_TRIALS = 20
_MIN_N_TRIALS = 5
_MAX_N_TRIALS = 100


# ── Helpers ───────────────────────────────────────────────────────────────────


def _step8_model() -> object | None:
    """Return the fitted model from Step 8 (single or leaderboard winner).

    Args:
        None

    Returns:
        object | None: The fitted ``ClassificationModel`` from Step 8, or
        ``None`` if Step 8 has not been confirmed yet.
    """
    path = st.session_state.get("int.train.path")
    if path == "single":
        return st.session_state.get("int.train.trained_model")
    if path == "leaderboard":
        return st.session_state.get("int.train.winner_model")
    return None


def _fmt_time(seconds: float) -> str:
    """Format a seconds value as a human-readable duration string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: E.g. ``"30 seconds"``, ``"~2 minutes"``, ``"~10 minutes"``.
    """
    if seconds < 60:
        return f"~{math.ceil(seconds)} seconds"
    minutes = seconds / 60.0
    return f"~{math.ceil(minutes)} minute{'s' if minutes >= 1.5 else ''}"


def _params_comparison_table(algorithm: str, best_params: dict) -> pd.DataFrame:
    """Build a side-by-side default vs. tuned parameter table.

    Args:
        algorithm (str): Algorithm name (e.g. ``"xgboost"``).
        best_params (dict): Best parameters found by ``Tuner``.

    Returns:
        pd.DataFrame: Columns ``Parameter``, ``Default``, ``Tuned``,
        ``Changed`` — one row per parameter in ``best_params``.
    """
    defaults = PREDEFINED_PARAMS.get(f"{algorithm}_classification", {})
    rows = []
    for param, tuned_val in best_params.items():
        default_val = defaults.get(param, "—")
        changed = str(tuned_val) != str(default_val)
        rows.append(
            {
                "Parameter": param,
                "Default": default_val,
                "Tuned": tuned_val,
                "Changed": "✅" if changed else "—",
            }
        )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["Parameter", "Default", "Tuned", "Changed"])
    )


def _baseline_auc(metrics_df: pd.DataFrame | None) -> float | None:
    """Extract the validation (or test) AUC from Step 8's metric result.

    Args:
        metrics_df (pd.DataFrame | None): ``model.evaluate(split)`` output from
            Step 8, with columns ``split``, ``metric``, ``value``.

    Returns:
        float | None: The AUC-ROC value on validation split (fallback: test),
        or ``None`` if unavailable.
    """
    if metrics_df is None or metrics_df.empty:
        return None
    for split_name in ("val", "test"):
        sub = metrics_df[(metrics_df["split"] == split_name) & (metrics_df["metric"] == "roc_auc")]
        if not sub.empty:
            return float(sub["value"].iloc[0])
    return None


# ── Optimization curve ────────────────────────────────────────────────────────


def _render_optimization_curve(oc_df: pd.DataFrame, n_trials: int, metric: str) -> None:
    """Render a Plotly optimization-history chart from the tuner's curve data.

    Draws grey scatter dots for each completed trial's metric value and a red
    line for the running best-so-far, matching the classic Optuna optimisation
    history plot but styled to match the dscompanion dark-UI theme.

    No in-figure title — the caller's own markdown header already carries
    that text (and the completed/n_trials detail); duplicating it as a
    Plotly ``title`` competed with the horizontal legend for the same
    cramped top margin and visibly overlapped it.

    Args:
        oc_df (pd.DataFrame): DataFrame with columns ``trial_number``,
            ``metric_value``, and ``best_so_far`` — as produced by
            ``Tuner.optimization_curve_``.
        n_trials (int): Total number of trials requested. Unused now that
            the completed/n_trials detail moved to the caller's header;
            kept for call-site compatibility.
        metric (str): Metric name used for the y-axis label (e.g. ``"roc_auc"``).

    Returns:
        None
    """
    fig = go.Figure()
    # Two-series identity chart (all trials vs. best-so-far progression) —
    # categorical palette slots 1 and 8, dark steps (matches
    # interactive_state.py's _PALETTE_DARK, since this chart goes through
    # dark_fig() below and explicit marker/line colors bypass colorway).
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
        # Taller than the previous 300px so the same auto-ranged y-axis gets
        # more vertical pixels to show variation among tightly-clustered
        # trial scores — no data hidden or range narrowed, purely more
        # rendering resolution (UIR.9).
        height=420,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(dark_fig(fig), width="stretch")


# ── Tuning run ────────────────────────────────────────────────────────────────


def _run_tuning(model: object, n_trials: int, algorithm: str) -> None:
    """Execute ``Tuner.run()`` and write results to session state.

    Uses ``st.status()`` to show a live elapsed-time counter while the
    blocking optuna run proceeds.  Results are written to
    ``int.tuning.tuner`` and ``int.tuning.tuned_model``.

    Note:
        ``Tuner.run()`` → ``study_.optimize()`` is synchronous with no
        external callback hook; per-trial progress is unavailable without
        modifying the core backend.  The elapsed-time display is updated
        using a ``while`` loop in the same thread — this works because
        Streamlit's ``st.status`` widget supports in-place updates within
        the same script execution context.

    Args:
        model: Fitted ``ClassificationModel`` from Step 8.
        n_trials (int): Number of Optuna trials to run.
        algorithm (str): Algorithm name — used only for the status label.

    Returns:
        None
    """
    processed_split = st.session_state.get("int.train.processed_split")
    if processed_split is None:
        st.error("Processed training data from Step 8 not found. Please go back and re-run Step 8.")
        return

    est_seconds = _APPROX_SECS_PER_TRIAL.get(algorithm, 5.0) * n_trials
    est_label = _fmt_time(est_seconds)

    with st.status(
        f"Tuning {algorithm}: {n_trials} trials (estimated {est_label})...",
        expanded=True,
    ) as status:
        status.write(
            f"Running {n_trials} Optuna trials, searching for better hyperparameters. "
            "This may take a few minutes for tree-based models on large datasets."
        )
        elapsed_slot = st.empty()
        t0 = time.perf_counter()

        # Tuner.run() is blocking — no per-trial callback is available from
        # outside the backend without modifying dscompanion core code.
        try:
            tuner = Tuner(
                model=model,
                backend="optuna",
                n_trials=n_trials,
                cv=processed_split,
                metric="roc_auc",
                direction="maximize",
            )
            tuned_model = tuner.run()
            elapsed = time.perf_counter() - t0
            elapsed_slot.caption(f"Completed in {elapsed:.0f}s.")
            status.update(
                label=f"Tuning Complete: Best AUC {tuner.best_score_:.4f} ({elapsed:.0f}s)",
                state="complete",
                expanded=False,
            )
        except ValueError as exc:
            # Raised when search space cannot be inferred
            elapsed = time.perf_counter() - t0
            status.update(label="Tuning Failed", state="error", expanded=True)
            st.error(
                f"Could not tune {algorithm}: {exc}\n\n"
                "This algorithm may not have a search space defined. "
                "You can skip tuning and continue with the default model."
            )
            logger.warning("Step 9 tuning failed for %s: %s", algorithm, exc)
            return
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            status.update(label="Tuning Failed", state="error", expanded=True)
            st.error(f"Tuning failed unexpectedly: {exc}")
            logger.exception("Step 9 tuning failed: %s", exc)
            return

    st.session_state["int.tuning.tuner"] = tuner
    st.session_state["int.tuning.tuned_model"] = tuned_model
    st.session_state["int.tuning.tuning_done"] = True


# ── Result display ────────────────────────────────────────────────────────────


def _render_tuning_result(algorithm: str) -> None:
    """Show the tuning result and present the accept/reject choice.

    Displays the best score vs. baseline AUC, the before/after parameter
    table, and two buttons — ``"Use tuned parameters"`` and
    ``"Keep default parameters"`` — so the user makes an explicit decision
    rather than having the tuned model silently applied.

    Args:
        algorithm (str): Algorithm name.

    Returns:
        None
    """
    tuner: Tuner = st.session_state["int.tuning.tuner"]
    tuned_model = st.session_state["int.tuning.tuned_model"]

    best_score = tuner.best_score_
    baseline_metrics = st.session_state.get("int.train.metrics_df")
    if baseline_metrics is None and st.session_state.get("int.train.path") == "leaderboard":
        winner_model = st.session_state.get("int.train.winner_model")
        processed_split = st.session_state.get("int.train.processed_split")
        if winner_model is not None and processed_split is not None:
            baseline_metrics = winner_model.evaluate(processed_split)
    baseline_auc = _baseline_auc(baseline_metrics)

    st.markdown("#### Tuning Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC Before Tuning", f"{baseline_auc:.4f}" if baseline_auc is not None else "—")
    with col2:
        delta = None
        if baseline_auc is not None and not math.isnan(best_score):
            delta = f"{best_score - baseline_auc:+.4f}"
        st.metric(
            "AUC After Tuning",
            f"{best_score:.4f}" if not math.isnan(best_score) else "—",
            delta=delta,
        )

    if baseline_auc is not None and not math.isnan(best_score):
        if best_score < baseline_auc - 0.001:
            st.warning(
                "Tuning found parameters that scored *lower* than the defaults on the "
                "validation set. This can happen with few trials or noisy data, so "
                "consider keeping the default parameters."
            )
        elif best_score <= baseline_auc + 0.001:
            st.info("Tuning found no meaningful improvement over the defaults.")

    st.markdown("**Parameter Changes**")
    cmp_df = _params_comparison_table(algorithm, tuner.best_params_)
    if cmp_df.empty:
        st.caption("No parameter changes recorded.")
    else:
        st.dataframe(cmp_df, hide_index=True, width="content")

    oc_df = getattr(tuner, "optimization_curve_", pd.DataFrame())
    if not oc_df.empty:
        st.markdown(f"**Optimization History**: {len(oc_df)} of {tuner.n_trials} trials completed")
        _render_optimization_curve(oc_df, tuner.n_trials, tuner.metric)

    trials_df = tuner.trials_dataframe_
    if not trials_df.empty:
        with st.expander(f"Top {len(trials_df)} trials by {tuner.metric}"):
            display_df = trials_df[["trial_number", "metric_value"]].copy()
            display_df.columns = ["Trial", tuner.metric.upper().replace("_", " ")]
            display_df.iloc[:, 1] = display_df.iloc[:, 1].round(4)
            st.dataframe(display_df, hide_index=True, width="content")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(
            "Use tuned parameters",
            key="int.tuning.accept_tuned",
            width="stretch",
            type="primary",
        ):
            _confirm_step9(tuned_model, algorithm, tuned=True, best_score=best_score)
    with col_b:
        step8_model = _step8_model()
        if st.button(
            "Keep default parameters",
            key="int.tuning.reject_tuned",
            width="stretch",
        ):
            _confirm_step9(step8_model, algorithm, tuned=False, best_score=None)


# ── Confirm ───────────────────────────────────────────────────────────────────


def _confirm_step9(
    model: object,
    algorithm: str,
    *,
    tuned: bool,
    best_score: float | None,
) -> None:
    """Commit the final model choice to session state.

    Args:
        model: The model to carry forward (tuned or default).
        algorithm (str): Algorithm name.
        tuned (bool): Whether the tuned model was accepted.
        best_score (float | None): Best AUC found by tuning, if applicable.

    Returns:
        None
    """
    st.session_state["interactive.tuning_model"] = model
    st.session_state["interactive.tuning_tuned"] = tuned
    set_step_confirmed("tuning", True)
    set_step_status("tuning", "done")
    if tuned and best_score is not None and not math.isnan(best_score):
        add_audit_entry(
            "tuning",
            f"Hyperparameter tuning: accepted tuned {algorithm} (AUC {best_score:.4f}).",
        )
    elif tuned:
        add_audit_entry("tuning", f"Hyperparameter tuning: accepted tuned {algorithm}.")
    else:
        add_audit_entry("tuning", f"Hyperparameter tuning: skipped. Kept default {algorithm}.")
    logger.info("Interactive mode — Step 9 confirmed: algorithm=%s, tuned=%s", algorithm, tuned)


# ── Main entrypoint ───────────────────────────────────────────────────────────


def render_step_tuning() -> bool:
    """Render Step 9 (Hyperparameter Tuning) and report confirmation.

    Offers an optional tuning run using Optuna, shows a before/after
    parameter and AUC comparison, and requires an explicit confirm before
    the tuned model replaces the Step 8 default.  Algorithms that have no
    classification search space are auto-skipped with a plain-language note.

    Args:
        None

    Returns:
        bool: ``True`` once the user confirms (or auto-skips); ``False``
        while still in progress.
    """
    already_confirmed = st.session_state["interactive.step_confirmed"]["tuning"]
    if already_confirmed:
        return True

    algorithm: str = st.session_state.get("interactive.train_algorithm", "")
    step8_model = _step8_model()

    st.subheader("Step 9: Hyperparameter tuning (optional)")

    # Auto-skip for algorithms with no search space
    if algorithm in _NO_SEARCH_SPACE:
        st.info(
            f"Hyperparameter tuning is not available for **{algorithm}** in this version. "
            f"No search space is defined for it. Continuing with the {algorithm} model "
            "from Step 8."
        )
        if st.button("Continue to Step 10", key="int.tuning.skip_no_space"):
            _confirm_step9(step8_model, algorithm, tuned=False, best_score=None)
        return st.session_state["interactive.step_confirmed"]["tuning"]

    tuning_done = st.session_state.get("int.tuning.tuning_done", False)

    if not tuning_done:
        st.caption(
            "Tuning searches for better model settings automatically by running many "
            "trials with different parameter combinations. It can improve performance "
            "but takes longer to run."
        )

        enabled = st.toggle(
            "Tune this model",
            value=False,
            key="int.tuning.enabled",
            help="Off by default. You can skip tuning and continue with the trained model.",
        )

        if not enabled:
            if st.button("Skip tuning and continue", key="int.tuning.skip_toggle"):
                _confirm_step9(step8_model, algorithm, tuned=False, best_score=None)
            return st.session_state["interactive.step_confirmed"]["tuning"]

        # Tuning enabled — show configuration
        n_trials = st.slider(
            "Number of trials",
            min_value=_MIN_N_TRIALS,
            max_value=_MAX_N_TRIALS,
            value=_DEFAULT_N_TRIALS,
            step=5,
            key="int.tuning.n_trials",
            help="More trials find better parameters but take longer.",
        )
        est_seconds = _APPROX_SECS_PER_TRIAL.get(algorithm, 5.0) * n_trials
        st.caption(
            f"Estimated time: {_fmt_time(est_seconds)} for {n_trials} trials with "
            f"{algorithm}. Adjust the slider to trade off quality vs. speed."
        )

        if st.button("Start tuning", key="int.tuning.start", type="primary"):
            st.session_state["int.tuning.n_trials_used"] = n_trials
            _run_tuning(step8_model, n_trials, algorithm)
            if st.session_state.get("int.tuning.tuning_done", False):
                st.rerun()

    if tuning_done:
        _render_tuning_result(algorithm)

    return st.session_state["interactive.step_confirmed"]["tuning"]
