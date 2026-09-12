"""Interactive mode — Step 11: Calibration (optional).

Runs post-hoc probability calibration on the final model from Step 9/10,
shows the Expected Calibration Error (ECE) before and after, and lets the
user decide whether to apply the calibrated version or keep the raw scores.

The step is always shown for classification tasks — the user always confirms
one of two choices (Apply / Skip), so the step is never silently bypassed.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from interactive_state import add_audit_entry, reset_from_step, set_step_confirmed, set_step_status

logger = logging.getLogger(__name__)

__all__ = ["render_step_calibration"]

# ── Helpers ───────────────────────────────────────────────────────────────────

_ECE_IMPROVEMENT_THRESHOLD = 0.001  # delta below which improvement is considered negligible


def _final_model():
    """Return the model to calibrate — Step 9's final fitted model.

    Args:
        None

    Returns:
        object | None: The ``ClassificationModel`` stored at
        ``interactive.tuning_model``, or ``None`` if unavailable.
    """
    return st.session_state.get("interactive.tuning_model")


def _cal_data() -> tuple[pd.DataFrame, pd.Series, str] | None:
    """Select the best available held-out split to use as the calibration set.

    Prefers val (cleanest held-out data), then test, then train as a last
    resort.  Returns the feature matrix, label series, and the split name
    used.

    Args:
        None

    Returns:
        tuple[pd.DataFrame, pd.Series, str] | None: ``(X, y, split_name)``
        or ``None`` if no processed split is available.
    """
    split = st.session_state.get("int.train.processed_split")
    if split is None:
        return None

    for attr_x, attr_y, name in (
        ("val_X", "val_y", "val"),
        ("test_X", "test_y", "test"),
        ("train_X", "train_y", "train"),
    ):
        X = getattr(split, attr_x, None)
        y = getattr(split, attr_y, None)
        if X is not None and len(X) > 0:
            return X, y, name
    return None


def _fit_calibrator():
    """Fit (or return cached) the Calibrator on the best available held-out split.

    Caches the fitted Calibrator in ``int.calibration.calibrator`` and the split
    name in ``int.calibration.cal_split`` so subsequent renders don't re-fit.

    Args:
        None

    Returns:
        tuple[Calibrator, str] | None: ``(fitted_calibrator, split_name)`` or
        ``None`` if the model, split, or fit fails.
    """
    cached = st.session_state.get("int.calibration.calibrator")
    if cached is not None:
        return cached, st.session_state.get("int.calibration.cal_split", "val")

    model = _final_model()
    cal = _cal_data()
    if model is None or cal is None:
        return None

    X_cal, y_cal, split_name = cal

    try:
        from dscompanion.calibration.calibrator import Calibrator

        calibrator = Calibrator(method="isotonic")
        calibrator.fit(model, X_cal, y_cal)
        st.session_state["int.calibration.calibrator"] = calibrator
        st.session_state["int.calibration.cal_split"] = split_name
        logger.info(
            "Step 11: calibrator fitted on %s split, ECE %.4f → %.4f",
            split_name,
            calibrator.ece_before_,
            calibrator.ece_after_,
        )
        return calibrator, split_name
    except Exception as exc:
        logger.warning("Step 11 calibration fit failed: %s", exc)
        return None


def _ece_delta(calibrator) -> float:
    """Return ece_after - ece_before (negative means improvement).

    Args:
        calibrator: A fitted ``Calibrator`` instance.

    Returns:
        float: ``ece_after_ - ece_before_`` (lower ECE is better, so
        negative means calibration helped).
    """
    return calibrator.ece_after_ - calibrator.ece_before_


def _recommend_apply(calibrator) -> bool:
    """Return True if calibration meaningfully improved ECE.

    Args:
        calibrator: A fitted ``Calibrator`` instance.

    Returns:
        bool: ``True`` if ECE improved by more than the negligible threshold.
    """
    return _ece_delta(calibrator) < -_ECE_IMPROVEMENT_THRESHOLD


# ── Confirm ───────────────────────────────────────────────────────────────────


def _confirm_step11(calibrator, applied: bool) -> None:
    """Record Step 11 completion, store the final model, and write an audit entry.

    When calibration is applied, wraps the Step 9 model with the calibrated
    predict_proba and stores it at ``int.calibration.model``.  When skipped, stores
    the unwrapped Step 9 model instead.  Either way, downstream steps
    (Steps 12-13) should read from ``int.calibration.model``.

    Args:
        calibrator: A fitted ``Calibrator`` instance.
        applied (bool): ``True`` if the user chose to apply calibration.

    Returns:
        None
    """
    model = _final_model()
    if applied:
        try:
            final_model = calibrator.wrap(model)
        except Exception as exc:
            logger.warning("Step 11: calibrator.wrap() failed (%s), using raw model", exc)
            final_model = model
    else:
        final_model = model

    st.session_state["int.calibration.model"] = final_model
    st.session_state["interactive.calibration_applied"] = applied
    set_step_confirmed("calibration", True)
    set_step_status("calibration", "done")

    delta = _ece_delta(calibrator)
    action = "applied" if applied else "skipped"
    add_audit_entry(
        "calibration",
        f"Calibration {action} (isotonic). "
        f"ECE before: {calibrator.ece_before_:.4f}, after: {calibrator.ece_after_:.4f} "
        f"(delta {delta:+.4f}).",
    )
    logger.info("Interactive mode — Step 11 confirmed: calibration %s", action)


# ── Main entrypoint ───────────────────────────────────────────────────────────


def render_step_calibration() -> bool:
    """Render Step 11 (Calibration — optional) and report confirmation.

    Fits an isotonic calibrator on the best available held-out split, shows
    the ECE before and after, recommends Apply or Skip based on whether
    calibration actually improved ECE, and records the user's choice.  Always
    presents both choices — the step is never auto-confirmed.

    For non-classification tasks this step is a no-op and auto-confirms with
    a skip.

    Args:
        None

    Returns:
        bool: ``True`` once the user clicks Apply or Skip; ``False`` while
        reviewing.
    """
    if st.session_state["interactive.step_confirmed"]["calibration"]:
        return True

    task = st.session_state.get("interactive.task", "classification")

    st.subheader("Step 11: Calibration (optional)")

    # Non-classification: auto-skip silently
    if task != "classification":
        st.info("Calibration is only available for classification tasks. This step is skipped.")
        if st.button(
            "Continue to Step 12", key="int.calibration.skip_nonclassification", type="primary"
        ):
            model = _final_model()
            st.session_state["int.calibration.model"] = model
            st.session_state["interactive.calibration_applied"] = False
            set_step_confirmed("calibration", True)
            set_step_status("calibration", "done")
            add_audit_entry("calibration", "Calibration skipped. Non-classification task.")
        return st.session_state["interactive.step_confirmed"]["calibration"]

    # ── Plain-language framing ──────────────────────────────────────────────
    st.caption(
        "Calibration adjusts the model's confidence scores so they better reflect "
        "real probabilities. For example, when the model says '70% likely,' it should "
        "be right about 70% of the time. This doesn't change *which* prediction is made, "
        "only *how confident* it looks. It is optional: if your use case doesn't depend "
        "on well-calibrated probabilities (e.g. you only need a rank order), skipping is fine."
    )

    # ── Fit calibrator ─────────────────────────────────────────────────────
    with st.spinner("Fitting calibration on held-out data…"):
        result = _fit_calibrator()

    if result is None:
        st.error(
            "Could not fit calibration. The model or processed split from Steps 8–9 is "
            "missing. Please go back and re-run Steps 8–9."
        )
        if st.button("Go back to Step 8 and retrain", key="int.calibration.back_to_train"):
            reset_from_step("train")
            st.rerun()
        return False

    calibrator, split_name = result

    # ── ECE before / after ─────────────────────────────────────────────────
    ece_before = calibrator.ece_before_
    ece_after = calibrator.ece_after_
    delta = _ece_delta(calibrator)
    improved = _recommend_apply(calibrator)

    st.markdown("**Expected Calibration Error (ECE): Lower Is Better**")
    st.caption(
        f"Measured on the **{split_name}** split. "
        "ECE = average gap between predicted probabilities and actual outcome rates. "
        "0 is perfect; values above 0.10 are typically noticeable."
    )

    col_before, col_after, col_delta = st.columns(3)
    col_before.metric("Before Calibration", f"{ece_before:.4f}")
    col_after.metric(
        "After Calibration (Isotonic)",
        f"{ece_after:.4f}",
        delta=f"{delta:+.4f}",
        delta_color="inverse",  # negative delta (improvement) shows green
    )
    col_delta.metric(
        "Change",
        f"{abs(delta):.4f} {'better' if delta < 0 else 'worse' if delta > 0 else 'unchanged'}",
    )

    st.divider()

    # ── Recommendation ─────────────────────────────────────────────────────
    if improved:
        st.success(
            f"Calibration reduced ECE by {abs(delta):.4f}. "
            "**Recommendation: apply calibration.** The scores will better reflect real "
            "probabilities."
        )
    elif abs(delta) <= _ECE_IMPROVEMENT_THRESHOLD:
        st.info(
            f"Calibration changed ECE by only {abs(delta):.4f} (negligible). "
            "The scores are already well-calibrated. "
            "**Recommendation: skip.** Applying calibration here adds overhead without benefit."
        )
    else:
        st.warning(
            f"Calibration made ECE **worse** by {abs(delta):.4f}. "
            "This can happen with very small validation sets or unusual score distributions. "
            "**Recommendation: skip.** Keep the raw scores."
        )

    st.divider()

    # ── Action buttons ─────────────────────────────────────────────────────
    st.markdown("**Your Choice**")
    col_apply, col_skip = st.columns(2)

    with col_apply:
        if st.button(
            "Apply calibration",
            key="int.calibration.apply",
            type="primary" if improved else "secondary",
            help="Replace raw scores with calibrated probabilities for Steps 12-13.",
        ):
            _confirm_step11(calibrator, applied=True)

    with col_skip:
        if st.button(
            "Skip: Keep raw scores",
            key="int.calibration.skip",
            type="primary" if not improved else "secondary",
            help="Keep the model's original probability output unchanged.",
        ):
            _confirm_step11(calibrator, applied=False)

    return st.session_state["interactive.step_confirmed"]["calibration"]
