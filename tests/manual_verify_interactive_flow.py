"""Manual verification script — drives the Interactive-mode wizard end-to-end.

Not a pytest-collected test. Written to verify that the interactive_step*.py
rename (interactive_step1.py -> interactive_step_load_data.py, etc.) did not
break the real step-by-step user flow. Starts from a completely fresh
session (no pre-seeded session_state) and drives every widget exactly as a
real user would via streamlit.testing.v1.AppTest, following STEP_ORDER from
dscompanion/app/interactive_state.py: load_data, task_target, split, eda,
feature_processing, feature_selection, imbalance, train, tuning, evaluate,
calibration, shap, report.

Run with (from dscompanion/ as cwd)::

    source ~/.zshrc 2>/dev/null; conda activate dscompanion312 && \
        python tests/manual_verify_interactive_flow.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

# Dataset choice: the 29,332-row/87-col android-permissions dataset used in
# the first attempt made multivariate EDA alone take ~3.5 minutes, timing out
# the next AppTest.run() call (default_timeout=180) — not a rename bug, just
# too slow a dataset for this harness. bank_marketing_clean.parquet is the
# same dataset test_streamlit_app.py already uses for its target —
# 45,211 rows x 17 cols, clean binary target 'subscribed' (0/1).
DATA_RELATIVE_PATH = "bank_marketing_clean.parquet"
TARGET_COLUMN = "subscribed"
ALGORITHM = "logistic"


def settle(at: AppTest) -> None:
    """Run the script twice: once to process pending interactions, once to
    materialize whatever an internal ``st.rerun()`` call left pending.

    Necessary because a manual ``st.rerun()`` inside a step's button handler
    (e.g. interactive_step_eda.py's "Continue to X" buttons, or
    interactive_step_train.py's post-training rerun) aborts the *current*
    ``AppTest.run()`` call immediately — the state mutation lands, but any
    content that would render *after* the rerun call only appears on a
    subsequent, bare ``run()`` with no new interaction queued.

    Args:
        at (AppTest): The running app-under-test.

    Returns:
        None
    """
    at.run()
    at.run()
    if at.exception:
        for e in at.exception:
            print("EXCEPTION:", e)
        raise AssertionError(f"Unexpected exception: {[str(e) for e in at.exception]}")


def click(at: AppTest, key: str) -> None:
    """Click a button by key, then settle any internal rerun.

    Args:
        at (AppTest): The running app-under-test.
        key (str): The button's Streamlit widget key.

    Returns:
        None
    """
    at.button(key=key).click()
    settle(at)


def has_button(at: AppTest, key: str) -> bool:
    """Return whether a button with the given key is currently rendered.

    Args:
        at (AppTest): The running app-under-test.
        key (str): The button's Streamlit widget key.

    Returns:
        bool: True if present on the current render.
    """
    try:
        at.button(key=key)
        return True
    except KeyError:
        return False


def confirmed(at: AppTest, step: str) -> bool:
    """Return whether ``step`` is marked confirmed in session state.

    Args:
        at (AppTest): The running app-under-test.
        step (str): Step slug.

    Returns:
        bool: The step's confirmation flag.
    """
    return bool(at.session_state["interactive.step_confirmed"][step])


def main() -> None:
    """Drive the full 13-step Interactive-mode wizard from a fresh session.

    Args:
        None

    Returns:
        None
    """
    t_start = time.perf_counter()
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=600)
    at.session_state["run_mode"] = "Interactive"
    at.run()
    assert not at.exception, f"Initial run failed: {at.exception}"
    print("Initial render OK.")

    # ---- Step 1: Load Data ----
    at.selectbox(key="int.load_data.browse_choice").select(DATA_RELATIVE_PATH)
    settle(at)
    click(at, "int.load_data.load_button")
    assert st_ok(at), at.exception
    assert not confirmed(at, "load_data")
    click(at, "int.load_data.confirm")
    assert confirmed(at, "load_data"), "Step 1 (load_data) did not confirm."
    print("Step 1 (load_data): CONFIRMED.")

    # ---- Step 2: Task & Target ----
    at.selectbox(key="int.task_target.target").select(TARGET_COLUMN)
    settle(at)
    assert has_button(
        at, "int.task_target.confirm"
    ), "Step 2 confirm button missing — target validation likely failed."
    click(at, "int.task_target.confirm")
    assert confirmed(at, "task_target"), "Step 2 (task_target) did not confirm."
    print("Step 2 (task_target): CONFIRMED.")

    # ---- Step 3: Split ----
    # Defaults: method=stratified (first option), test_size=0.2, val_size=0.1.
    click(at, "int.split.confirm")
    assert confirmed(at, "split"), "Step 3 (split) did not confirm."
    print("Step 3 (split): CONFIRMED.")

    # ---- Step 4: EDA ----
    # Default toggle int.eda.run_eda=True. Walk the 4 gated sub-checks.
    click(at, "int.eda.continue_to_univariate")
    click(at, "int.eda.continue_to_categorical")
    click(at, "int.eda.continue_to_bivariate")
    click(at, "int.eda.continue_to_multivariate")
    click(at, "int.eda.continue")
    assert confirmed(at, "eda"), "Step 4 (eda) did not confirm."
    print("Step 4 (eda): CONFIRMED.")

    # ---- Step 5: Feature Processing ----
    if has_button(at, "int.feature_processing.continue_empty"):
        click(at, "int.feature_processing.continue_empty")
    else:
        # Every column defaults to "Accept recommended" already — no widget
        # overrides needed for a normal run. The "Rest" group renders inside
        # a collapsed expander; its bulk-accept button is exercised here
        # only to prove it works, not because Accept isn't already the
        # per-column default.
        if has_button(at, "int.feature_processing.accept_all_rest"):
            click(at, "int.feature_processing.accept_all_rest")
        click(at, "int.feature_processing.continue")
    assert confirmed(at, "feature_processing"), "Step 5 (feature_processing) did not confirm."
    print("Step 5 (feature_processing): CONFIRMED.")

    # ---- Step 6: Feature Selection ----
    if has_button(at, "int.feature_selection.continue_empty"):
        click(at, "int.feature_selection.continue_empty")
    else:
        # Same rationale as Step 5 — uncheck every proposed removal so the
        # feature set survives for Step 8 to train on.
        remove_boxes = [w for w in at.checkbox if w.key.startswith("int.feature_selection.remove.")]
        print(f"  Step 6: unchecking {len(remove_boxes)} proposed removal(s).")
        for w in remove_boxes:
            w.uncheck()
        settle(at)
        continue_buttons = at.button(key="int.feature_selection.continue")
        continue_buttons.click()
        settle(at)
        if at.error and any("Cannot remove every feature" in e.value for e in at.error):
            raise AssertionError(
                "Feature selection flagged every candidate feature for removal — "
                "script would need to uncheck some rows. See printed error."
            )
    assert confirmed(at, "feature_selection"), "Step 6 (feature_selection) did not confirm."
    print("Step 6 (feature_selection): CONFIRMED.")

    # ---- Step 7: Imbalance ----
    # Default radio = "class_weight" (first option) -> no-resample path.
    click(at, "int.imbalance.continue_no_resample")
    assert confirmed(at, "imbalance"), "Step 7 (imbalance) did not confirm."
    print("Step 7 (imbalance): CONFIRMED.")

    # ---- Step 8: Train ----
    click(at, "int.train.pick_single")
    at.selectbox(key="int.train.algorithm_select").select(ALGORITHM)
    settle(at)
    t0 = time.perf_counter()
    click(at, "int.train.train_single")
    print(f"  training took {time.perf_counter() - t0:.1f}s")
    assert has_button(
        at, "int.train.confirm_single"
    ), "Training did not complete — 'Accept and continue' button missing."
    click(at, "int.train.confirm_single")
    assert confirmed(at, "train"), "Step 8 (train) did not confirm."
    print("Step 8 (train): CONFIRMED.")

    # ---- Step 9: Tuning (skip — optional, toggle defaults to False) ----
    click(at, "int.tuning.skip_toggle")
    assert confirmed(at, "tuning"), "Step 9 (tuning) did not confirm."
    print("Step 9 (tuning): CONFIRMED (skipped).")

    # ---- Step 10: Evaluate ----
    click(at, "int.evaluate.continue")
    assert confirmed(at, "evaluate"), "Step 10 (evaluate) did not confirm."
    print("Step 10 (evaluate): CONFIRMED.")

    # ---- Step 11: Calibration ----
    click(at, "int.calibration.apply")
    assert confirmed(at, "calibration"), "Step 11 (calibration) did not confirm."
    print("Step 11 (calibration): CONFIRMED.")

    # ---- Step 12: SHAP ----
    t0 = time.perf_counter()
    click(at, "int.shap.continue")
    print(f"  SHAP took {time.perf_counter() - t0:.1f}s")
    assert confirmed(at, "shap"), "Step 12 (shap) did not confirm."
    print("Step 12 (shap): CONFIRMED.")

    # ---- Step 13: Report ----
    at.text_input(key="int.report.author").set_value("Verification Script")
    at.text_area(key="int.report.use_case").set_value("Interactive-mode rename verification run.")
    settle(at)
    click(at, "int.report.generate")
    assert confirmed(at, "report"), "Step 13 (report) did not confirm."
    print("Step 13 (report): CONFIRMED.")

    success_text = " ".join(s.value for s in at.success)
    assert "All 13 steps complete" in success_text, "Terminal summary message missing."
    assert not at.exception, at.exception

    print(f"\nALL 13 STEPS PASSED. Total time: {time.perf_counter() - t_start:.1f}s")


def st_ok(at: AppTest) -> bool:
    """Return whether the app-under-test has no active exception.

    Args:
        at (AppTest): The running app-under-test.

    Returns:
        bool: True when ``at.exception`` is empty.
    """
    return not at.exception


if __name__ == "__main__":
    main()
