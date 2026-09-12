"""Interactive mode — Step 7: Class Imbalance Handling.

Shows the training-set class balance from the confirmed Step 3 split, lets
the user pick an imbalance-correction strategy, and records the choice as
``target.imbalance.strategy`` in config.  Classification only, consistent
with this slice's scope (``streamlit.md`` §Step 7).

No resampling is actually run here — the handler fires at training time
(Step 8).  For strategies that change row counts (SMOTE/undersample) an
estimated before/after row count is shown as a preview before the user
confirms, per the confirm-before-apply rule.

``"oversample"`` is accepted by ``ImbalanceConfig`` but not handled by
``ImbalanceHandler._build_sampler()`` — it would raise at training time.
It is intentionally excluded from the UI options here to prevent the user
from choosing a broken path.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from interactive_state import add_audit_entry, set_step_confirmed, set_step_status

from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["render_step_imbalance"]

# Event-rate range considered "roughly balanced" — outside this band the
# imbalance warning fires.  Chosen as ±15 pp from 50/50 (i.e. 35–65%).
_BALANCED_MIN_EVENT_RATE = 0.35

# Strategies that don't change row counts — no row-count preview needed.
_NO_RESAMPLE_STRATEGIES = {"class_weight", "none"}

_STRATEGY_OPTIONS = ["class_weight", "smote", "undersample", "none"]

_STRATEGY_LABELS = {
    "class_weight": (
        "Class weight (recommended, fastest): tells the model to pay more attention to "
        "the minority class, without changing your actual data."
    ),
    "smote": ("SMOTE: creates new synthetic minority-class rows; " "training set size increases."),
    "undersample": (
        "Undersample: removes random majority-class rows; " "training set size decreases."
    ),
    "none": ("None: no adjustment; use only if you are certain imbalance " "won't hurt the model."),
}

_AUDIT_MSGS = {
    "class_weight": "Imbalance handling: class weight (model-internal, no data change).",
    "smote": "Imbalance handling: SMOTE. Training set will be oversampled at training time.",
    "undersample": (
        "Imbalance handling: undersample. Training set will be undersampled at training time."
    ),
    "none": "Imbalance handling: none. Training data unchanged.",
}


def _preview_row_counts(train_y: pd.Series, strategy: str) -> tuple[int, int]:
    """Estimate before/after training-set row counts for a resampling strategy.

    Uses the ``sampling_strategy="auto"`` assumption — ``ImbalanceHandler``'s
    default — which balances both classes to the same count.

    Args:
        train_y (pd.Series): Training target series (binary 0/1).
        strategy (str): One of ``"smote"`` or ``"undersample"``.

    Returns:
        tuple[int, int]: ``(rows_before, rows_after)`` where ``rows_before``
        is ``len(train_y)`` and ``rows_after`` is the estimated post-resample
        size.
    """
    rows_before = len(train_y)
    counts = train_y.value_counts()
    minority_count = int(counts.min())
    majority_count = int(counts.max())
    if strategy == "smote":
        # Minority upsampled to match majority → total = 2 × majority_count
        rows_after = 2 * majority_count
    else:
        # Majority downsampled to match minority → total = 2 × minority_count
        rows_after = 2 * minority_count
    return rows_before, rows_after


def _confirm_step7(strategy: str) -> None:
    """Commit the strategy choice to session state and advance the stepper.

    Args:
        strategy (str): Confirmed imbalance strategy string — one of
            ``"class_weight"``, ``"smote"``, ``"undersample"``, ``"none"``.

    Returns:
        None
    """
    st.session_state["interactive.imbalance_strategy"] = strategy
    set_step_confirmed("imbalance", True)
    set_step_status("imbalance", "done")
    add_audit_entry("imbalance", _AUDIT_MSGS.get(strategy, f"Imbalance handling: {strategy}."))
    logger.info("Interactive mode — Step 7 confirmed: strategy=%s", strategy)


def render_step_imbalance(split: DataSplit) -> bool:
    """Render Step 7 (Class Imbalance Handling) and report whether it's confirmed.

    Shows the current training-set class balance, asks the user to pick an
    imbalance-correction strategy, and gates confirmation on an explicit
    button click.  For strategies that change row counts (SMOTE/undersample)
    a row-count preview is shown before the confirm button appears, consistent
    with the "preview before deciding" design rule.

    Args:
        split (DataSplit): Split confirmed at Step 3.  The training target
            ``split.train_y`` is used to compute class balance; it is never
            mutated here.

    Returns:
        bool: ``True`` once the user clicks the confirm button.  ``False``
        while still reviewing.
    """
    st.subheader("Step 7: Class imbalance handling")
    st.caption(
        "Imbalanced data, where one class is much rarer than the other, can cause "
        "a model to mostly ignore the minority class. Choosing a strategy here corrects "
        "for that. The actual adjustment happens at training time (Step 8), not now."
    )

    task = st.session_state.get("interactive.task", "classification")
    if task != "classification":
        st.info("Imbalance handling applies to classification tasks only. Skipped.")
        if st.button("Continue", key="int.imbalance.continue_skip"):
            _confirm_step7("none")
        return st.session_state["interactive.step_confirmed"]["imbalance"]

    train_y = split.train_y
    n_train = len(train_y)
    event_rate = float(train_y.mean())
    neg_rate = 1.0 - event_rate

    # --- Class balance display ---
    is_balanced = _BALANCED_MIN_EVENT_RATE <= event_rate <= (1.0 - _BALANCED_MIN_EVENT_RATE)
    minority_pct = min(event_rate, neg_rate)
    majority_pct = max(event_rate, neg_rate)
    minority_label = "positive (1)" if event_rate <= neg_rate else "negative (0)"
    majority_label = "negative (0)" if minority_label.startswith("positive") else "positive (1)"

    if is_balanced:
        st.success(
            f"Your training data is roughly balanced: {event_rate:.1%} positive class, "
            f"{neg_rate:.1%} negative. No adjustment is required, but you can still pick "
            "a strategy below if you prefer."
        )
    else:
        st.warning(
            f"Your training data is imbalanced: {minority_pct:.1%} {minority_label}, "
            f"{majority_pct:.1%} {majority_label}. Models trained without correction tend "
            "to mostly predict the majority class and miss the minority class."
        )

    counts = train_y.value_counts().sort_index()
    balance_df = pd.DataFrame(
        {
            "Class": [f"{c} ({'positive' if c == 1 else 'negative'})" for c in counts.index],
            "Rows": counts.values,
            "Share": [f"{v:.1%}" for v in (counts / n_train).values],
        }
    )
    st.dataframe(balance_df, hide_index=True, width="content")
    st.caption(f"Total training rows: {n_train:,}")

    st.divider()

    # --- Strategy selector ---
    st.markdown("**Choose How to Handle Imbalance:**")
    chosen = st.radio(
        "Strategy",
        options=_STRATEGY_OPTIONS,
        format_func=lambda s: _STRATEGY_LABELS[s],
        index=0,
        key="int.imbalance.strategy",
        label_visibility="collapsed",
    )

    st.divider()

    # --- Confirm gate ---
    already_confirmed = st.session_state["interactive.step_confirmed"]["imbalance"]
    if already_confirmed:
        return True

    if chosen in _NO_RESAMPLE_STRATEGIES:
        if chosen == "class_weight":
            st.info(
                "Class weight adjusts the model internally at training time. "
                "No rows are added or removed from your training data."
            )
        else:
            st.info("No adjustment will be made to your training data.")

        if st.button("Continue", key="int.imbalance.continue_no_resample"):
            _confirm_step7(chosen)
    else:
        rows_before, rows_after = _preview_row_counts(train_y, chosen)
        change = rows_after - rows_before
        direction = "increase" if change > 0 else "decrease"
        st.info(
            f"Training set will **{direction}** from {rows_before:,} "
            f"to {rows_after:,} rows "
            f"({abs(change):,} rows {'added' if change > 0 else 'removed'})."
        )

        if st.button(f"Apply {chosen} and continue", key="int.imbalance.continue_resample"):
            _confirm_step7(chosen)

    return st.session_state["interactive.step_confirmed"]["imbalance"]
