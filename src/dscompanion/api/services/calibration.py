"""Step 11 (Calibration) service functions — the REST equivalent of
``dscompanion/app/interactive_step_calibration.py``, with every ``streamlit`` call stripped out.

Always presents a real choice for classification tasks (apply/skip); auto-skips
silently for non-classification tasks, matching the Streamlit reference. Every path —
including the auto-skip — writes ``run.artifacts["calibrated_model"]``, since Step 12
(SHAP) and Step 13 read from that key exclusively, never from Step 9's ``final_model``
directly.
"""

from __future__ import annotations

import logging

import pandas as pd

from dscompanion.api.schemas import CalibrationConfirmResponse, CalibrationPreviewResponse
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id
from dscompanion.calibration.calibrator import Calibrator
from dscompanion.config import settings
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["preview_calibration", "confirm_calibration"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cal_data(processed_split: DataSplit) -> tuple[pd.DataFrame, pd.Series, str] | None:
    """Select the best available held-out split to use as the calibration set.

    Prefers val (cleanest held-out data), then test, then train as a last
    resort — mirrors ``interactive_step_calibration.py``'s ``_cal_data``
    exactly.

    Args:
        processed_split (DataSplit): Step 8's processed split.

    Returns:
        tuple[pd.DataFrame, pd.Series, str] | None: ``(X, y, split_name)``,
        or ``None`` if every split is empty.
    """
    for attr_x, attr_y, name in (
        ("val_X", "val_y", "val"),
        ("test_X", "test_y", "test"),
        ("train_X", "train_y", "train"),
    ):
        X = getattr(processed_split, attr_x, None)
        y = getattr(processed_split, attr_y, None)
        if X is not None and len(X) > 0:
            return X, y, name
    return None


def _fit_calibrator(model: object, processed_split: DataSplit) -> tuple[Calibrator, str] | None:
    """Fits an isotonic ``Calibrator`` on the best available held-out split.

    Args:
        model: Fitted ``BaseDSCompanionModel`` to calibrate.
        processed_split (DataSplit): Step 8's processed split.

    Returns:
        tuple[Calibrator, str] | None: ``(fitted_calibrator, split_name)``,
        or ``None`` if no non-empty split is available.
    """
    cal = _cal_data(processed_split)
    if cal is None:
        return None
    X_cal, y_cal, split_name = cal
    calibrator = Calibrator(method="isotonic")
    calibrator.fit(model, X_cal, y_cal)
    return calibrator, split_name


def _ece_delta(calibrator: Calibrator) -> float:
    """Return ``ece_after_ - ece_before_`` — negative means improvement.

    Args:
        calibrator (Calibrator): A fitted ``Calibrator``.

    Returns:
        float: The ECE delta.
    """
    return calibrator.ece_after_ - calibrator.ece_before_


def _recommend_apply(calibrator: Calibrator) -> bool:
    """Return ``True`` if calibration meaningfully improved ECE.

    Args:
        calibrator (Calibrator): A fitted ``Calibrator``.

    Returns:
        bool: ``True`` if ECE improved by more than
        ``settings.calibration_ece_improvement_threshold``.
    """
    return _ece_delta(calibrator) < -settings.calibration_ece_improvement_threshold


# ── Preview ───────────────────────────────────────────────────────────────────


def preview_calibration(run: RunState) -> CalibrationPreviewResponse:
    """Fits calibration on the final model and reports ECE before/after.

    Non-classification tasks return ``available=False`` immediately, with no
    calibrator fit — mirrors the Streamlit reference's silent auto-skip.

    Args:
        run (RunState): The run holding Step 9's final model and Step 8's
            processed split.

    Returns:
        CalibrationPreviewResponse: Availability, ECE before/after, and the
        apply/skip recommendation.

    Raises:
        ValueError: If Step 9 has not been confirmed, or no held-out split
            is available, for this run.
    """
    task = run.artifacts.get("task", "classification")
    if task != "classification":
        logger.info("run_id=%s previewed Step 11: non-classification, auto-skip", run.run_id)
        return CalibrationPreviewResponse(available=False)

    model = run.artifacts.get("final_model")
    processed_split: DataSplit | None = run.artifacts.get("processed_split")
    if model is None or processed_split is None:
        raise ValueError("Step 9 must be confirmed before Step 11.")

    result = _fit_calibrator(model, processed_split)
    if result is None:
        raise ValueError("No held-out data available to fit calibration.")
    calibrator, split_name = result

    logger.info(
        "run_id=%s previewed Step 11: split=%s ece %.4f -> %.4f",
        run.run_id,
        split_name,
        calibrator.ece_before_,
        calibrator.ece_after_,
    )
    return CalibrationPreviewResponse(
        available=True,
        split_used=split_name,
        ece_before=calibrator.ece_before_,
        ece_after=calibrator.ece_after_,
        recommend_apply=_recommend_apply(calibrator),
    )


# ── Confirm ───────────────────────────────────────────────────────────────────


def confirm_calibration(run: RunState, apply: bool) -> CalibrationConfirmResponse:
    """Applies or skips calibration and persists ``calibrated_model``.

    Writes ``run.artifacts["calibrated_model"]`` on every path — Step 12
    (SHAP) and Step 13 read exclusively from this key, never from
    ``final_model`` directly. Non-classification tasks always store the raw
    model here and ignore ``apply``.

    Args:
        run (RunState): The run to persist into.
        apply (bool): Whether to wrap the model with calibrated
            probabilities. Ignored (treated as ``False``) for
            non-classification tasks.

    Returns:
        CalibrationConfirmResponse: Confirmation and whether calibration was
        actually applied.

    Raises:
        ValueError: If Step 9 has not been confirmed, or no held-out split
            is available, for this run.
    """
    task = run.artifacts.get("task", "classification")
    model = run.artifacts.get("final_model")
    if model is None:
        raise ValueError("Step 9 must be confirmed before Step 11.")

    if task != "classification":
        run.artifacts["calibrated_model"] = model
        run.step_confirmed["calibration"] = True
        run.step_status["calibration"] = "done"
        nxt = next_step_id("calibration")
        if nxt is not None:
            run.step_status[nxt] = "current"
        run.audit_trail.append(("calibration", "Calibration skipped. Non-classification task."))
        logger.info("run_id=%s confirmed Step 11: non-classification, skipped", run.run_id)
        return CalibrationConfirmResponse(confirmed=True, applied=False)

    processed_split: DataSplit | None = run.artifacts.get("processed_split")
    if processed_split is None:
        raise ValueError("Step 9 must be confirmed before Step 11.")

    result = _fit_calibrator(model, processed_split)
    if result is None:
        raise ValueError("No held-out data available to fit calibration.")
    calibrator, _split_name = result

    if apply:
        try:
            final_model = calibrator.wrap(model)
        except Exception as exc:
            logger.warning(
                "run_id=%s Step 11: calibrator.wrap() failed (%s), using raw model",
                run.run_id,
                exc,
            )
            final_model = model
    else:
        final_model = model

    run.artifacts["calibrated_model"] = final_model
    run.step_confirmed["calibration"] = True
    run.step_status["calibration"] = "done"
    nxt = next_step_id("calibration")
    if nxt is not None:
        run.step_status[nxt] = "current"

    delta = _ece_delta(calibrator)
    action = "applied" if apply else "skipped"
    run.audit_trail.append(
        (
            "calibration",
            f"Calibration {action} (isotonic). ECE before: {calibrator.ece_before_:.4f}, "
            f"after: {calibrator.ece_after_:.4f} (delta {delta:+.4f}).",
        )
    )
    logger.info("run_id=%s confirmed Step 11: calibration %s", run.run_id, action)
    return CalibrationConfirmResponse(confirmed=True, applied=apply)
