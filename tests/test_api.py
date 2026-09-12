"""Tests for dscompanion/api/ — the Steps 1-4 vertical slice proving the FastAPI skeleton
end-to-end. Steps 5-13 are only checked for their 501 stub, except Steps 5-9
which are also ported.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import dscompanion.api as dscompanion_api_pkg
import dscompanion.api.state as api_state_module
from dscompanion.api.auth import get_auth_backend
from dscompanion.api.config import api_settings
from dscompanion.api.main import create_app
from dscompanion.api.steps import STEP_ORDER

_OPENAPI_SNAPSHOT_PATH = Path(dscompanion_api_pkg.__file__).resolve().parent / "openapi.json"


@pytest.fixture
def client() -> TestClient:
    """A fresh FastAPI TestClient, built from ``create_app()`` per test."""
    return TestClient(create_app())


@pytest.fixture
def csv_path(synthetic_df: pd.DataFrame, tmp_path: Path) -> Path:
    """A small CSV of the shared synthetic dataset's binary-target columns, written to
    a temp file for Step 1's load-by-path flow.
    """
    subset = synthetic_df[["f1", "f2", "cat_low", "target"]]
    path = tmp_path / "synthetic.csv"
    subset.to_csv(path, index=False)
    return path


@pytest.fixture
def feature_csv_path(synthetic_df: pd.DataFrame, tmp_path: Path) -> Path:
    """A richer CSV for Steps 5-7 — includes a column with missingness (f4), a
    near-zero-variance column, a constant column, and both low/high cardinality
    categoricals, so recipe recommendation and feature selection have something real
    to flag.
    """
    subset = synthetic_df[
        ["f1", "f2", "f4", "f9_near_zero", "constant_col", "cat_low", "cat_high", "target"]
    ]
    path = tmp_path / "synthetic_features.csv"
    subset.to_csv(path, index=False)
    return path


def _create_run(client: TestClient) -> str:
    """Creates a run via the API and returns its ``run_id``."""
    resp = client.post("/api/runs")
    assert resp.status_code == 200
    return resp.json()["run_id"]


def _confirm_step1(client: TestClient, run_id: str, csv_path: Path) -> None:
    resp = client.post(
        f"/api/runs/{run_id}/steps/load_data/confirm",
        json={"path": str(csv_path), "format": "csv"},
    )
    assert resp.status_code == 200, resp.text


def _confirm_step2(client: TestClient, run_id: str) -> None:
    resp = client.post(
        f"/api/runs/{run_id}/steps/task_target/confirm",
        json={"task": "classification", "target": "target", "identifier_columns": []},
    )
    assert resp.status_code == 200, resp.text


def test_data_files_returns_list(client: TestClient) -> None:
    resp = client.get("/api/data-files")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["files"], list)
    assert all(f.endswith((".csv", ".parquet", ".xlsx")) for f in body["files"])


def test_create_run_and_get_state(client: TestClient) -> None:
    run_id = _create_run(client)
    resp = client.get(f"/api/runs/{run_id}/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert all(v is False for v in body["step_confirmed"].values())
    # The first step starts "current", not "pending" — mirrors interactive_state.py's
    # init_interactive_state(), which promotes the first step to "current" immediately.
    assert body["step_status"][STEP_ORDER[0]] == "current"
    assert all(body["step_status"][s] == "pending" for s in STEP_ORDER[1:])


def test_get_state_unknown_run_id_404(client: TestClient) -> None:
    resp = client.get("/api/runs/does-not-exist/state")
    assert resp.status_code == 404


def test_step1_preview_does_not_persist(client: TestClient, csv_path: Path) -> None:
    run_id = _create_run(client)
    resp = client.post(
        f"/api/runs/{run_id}/steps/load_data/preview",
        json={"path": str(csv_path), "format": "csv"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["columns"]) == {"f1", "f2", "cat_low", "target"}
    assert body["duplicate_columns"] == []

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["load_data"] is False


def test_step1_confirm_persists_and_advances_state(client: TestClient, csv_path: Path) -> None:
    run_id = _create_run(client)
    _confirm_step1(client, run_id, csv_path)

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["load_data"] is True
    assert state["step_status"]["load_data"] == "done"
    assert state["step_status"]["task_target"] == "current"
    assert len(state["audit_trail"]) == 1


def test_step1_duplicate_columns_flagged_and_resolved(client: TestClient, tmp_path: Path) -> None:
    dup_path = tmp_path / "dupes.csv"
    dup_path.write_text("a,a,b\n1,2,3\n4,5,6\n")

    run_id = _create_run(client)
    preview = client.post(
        f"/api/runs/{run_id}/steps/load_data/preview",
        json={"path": str(dup_path), "format": "csv"},
    ).json()
    assert preview["duplicate_columns"] == ["a"]

    resp = client.post(
        f"/api/runs/{run_id}/steps/load_data/confirm",
        json={
            "path": str(dup_path),
            "format": "csv",
            "duplicate_resolutions": {"a__0": "rename:a_first", "a__1": "drop"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert sorted(body["columns"]) == ["a_first", "b"]


def test_step2_valid_target_confirms(client: TestClient, csv_path: Path) -> None:
    run_id = _create_run(client)
    _confirm_step1(client, run_id, csv_path)
    _confirm_step2(client, run_id)

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["task_target"] is True
    assert state["step_status"]["split"] == "current"


def test_step2_before_step1_rejected(client: TestClient) -> None:
    run_id = _create_run(client)
    resp = client.post(
        f"/api/runs/{run_id}/steps/task_target/preview",
        json={"task": "classification", "target": "target", "identifier_columns": []},
    )
    assert resp.status_code == 400


def test_step2_non_binary_target_rejected(client: TestClient, tmp_path: Path) -> None:
    df = pd.DataFrame({"f1": [1, 2, 3, 4], "target": [0, 1, 2, 0]})
    path = tmp_path / "multiclass.csv"
    df.to_csv(path, index=False)

    run_id = _create_run(client)
    _confirm_step1(client, run_id, path)
    resp = client.post(
        f"/api/runs/{run_id}/steps/task_target/confirm",
        json={"task": "classification", "target": "target", "identifier_columns": []},
    )
    assert resp.status_code == 400


def _confirm_step3(client: TestClient, run_id: str) -> None:
    resp = client.post(
        f"/api/runs/{run_id}/steps/split/confirm",
        json={"method": "stratified", "test_size": 0.2, "val_size": 0.1},
    )
    assert resp.status_code == 200, resp.text


def _confirm_through_split(client: TestClient, feature_csv_path: Path) -> str:
    """Creates a run and confirms Steps 1-3 against ``feature_csv_path``, returning
    ``run_id`` — the shared setup every Step 5/6/7 test needs.
    """
    run_id = _create_run(client)
    _confirm_step1(client, run_id, feature_csv_path)
    client.post(
        f"/api/runs/{run_id}/steps/task_target/confirm",
        json={"task": "classification", "target": "target", "identifier_columns": []},
    )
    _confirm_step3(client, run_id)
    return run_id


def test_step3_split_full_vertical_slice(client: TestClient, csv_path: Path) -> None:
    run_id = _create_run(client)
    _confirm_step1(client, run_id, csv_path)
    _confirm_step2(client, run_id)

    resp = client.post(
        f"/api/runs/{run_id}/steps/split/confirm",
        json={"method": "stratified", "test_size": 0.2, "val_size": 0.1},
    )
    assert resp.status_code == 200, resp.text
    row_counts = resp.json()["row_counts"]
    assert row_counts["train"] > 0
    assert row_counts["test"] > 0

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["split"] is True
    assert state["step_status"]["eda"] == "current"
    assert len(state["audit_trail"]) == 3
    assert set(state["config_preview"]) == {"load_data", "task_target", "split"}


def test_back_requires_two_phase_confirmation(client: TestClient, csv_path: Path) -> None:
    run_id = _create_run(client)
    _confirm_step1(client, run_id, csv_path)
    _confirm_step2(client, run_id)

    resp = client.post(
        f"/api/runs/{run_id}/back", json={"to_step": "task_target", "confirmed": False}
    )
    assert resp.status_code == 200
    assert resp.json()["requires_confirmation"] is True

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["task_target"] is True

    resp = client.post(
        f"/api/runs/{run_id}/back", json={"to_step": "task_target", "confirmed": True}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["requires_confirmation"] is False
    assert body["state"]["step_confirmed"]["task_target"] is False
    assert body["state"]["step_confirmed"]["load_data"] is True


def test_transformer_registry_returns_names(client: TestClient) -> None:
    resp = client.get("/api/transformer-registry")
    assert resp.status_code == 200
    names = resp.json()["names"]
    assert isinstance(names, list)
    assert "log" in names
    assert "clip_lower" in names


def test_feature_processing_preview_groups_candidates(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_split(client, feature_csv_path)

    resp = client.post(f"/api/runs/{run_id}/steps/feature_processing/preview")
    assert resp.status_code == 200, resp.text
    candidates = resp.json()["candidates"]
    by_col = {c["column"]: c for c in candidates}

    # f4 has real missingness — must appear, flagged "needing attention".
    assert by_col["f4"]["needs_attention"] is True
    assert by_col["f4"]["missing_rows"] > 0
    assert by_col["f4"]["recommended"][
        "steps"
    ], "a column with missing values should recommend at least an impute step"

    # Clean numeric column — present, in the "rest" group.
    assert by_col["f1"]["needs_attention"] is False
    assert by_col["f1"]["missing_rows"] == 0

    # target/identifier columns never appear as candidates.
    assert "target" not in by_col


def test_feature_processing_transform_preview(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_split(client, feature_csv_path)
    candidates = client.post(f"/api/runs/{run_id}/steps/feature_processing/preview").json()[
        "candidates"
    ]
    f1_recipe = next(c["recommended"] for c in candidates if c["column"] == "f1")

    resp = client.post(
        f"/api/runs/{run_id}/steps/feature_processing/transform-preview",
        json={"recipes": {"f1": f1_recipe}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["before"]) == len(body["after"]) <= 5
    assert body["columns_before"] == ["f1"]


def test_feature_processing_confirm_persists_and_advances_state(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_split(client, feature_csv_path)
    candidates = client.post(f"/api/runs/{run_id}/steps/feature_processing/preview").json()[
        "candidates"
    ]
    recipes = {c["column"]: c["recommended"] for c in candidates if c["column"] != "constant_col"}

    resp = client.post(
        f"/api/runs/{run_id}/steps/feature_processing/confirm",
        json={"recipes": recipes, "dropped_columns": ["constant_col"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_dropped"] == 1

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["feature_processing"] is True
    assert state["step_status"]["feature_selection"] == "current"
    assert state["config_preview"]["feature_processing"]["dropped_columns"] == ["constant_col"]


def _confirm_step5_accept_all(client: TestClient, run_id: str) -> None:
    candidates = client.post(f"/api/runs/{run_id}/steps/feature_processing/preview").json()[
        "candidates"
    ]
    recipes = {c["column"]: c["recommended"] for c in candidates}
    resp = client.post(
        f"/api/runs/{run_id}/steps/feature_processing/confirm",
        json={"recipes": recipes, "dropped_columns": []},
    )
    assert resp.status_code == 200, resp.text


def test_feature_selection_preview_flags_something(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_split(client, feature_csv_path)
    _confirm_step5_accept_all(client, run_id)

    resp = client.post(f"/api/runs/{run_id}/steps/feature_selection/preview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # f9_near_zero should be flagged by ConstantSelector/CardinalitySelector-style checks.
    flagged = {row["feature"] for row in body["audit"]}
    assert flagged, "expected at least one feature flagged given a near-zero-variance column"
    assert body["waterfall_chart"] is not None
    assert body["total_features"] > 0


def test_feature_selection_confirm_persists_and_advances_state(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_split(client, feature_csv_path)
    _confirm_step5_accept_all(client, run_id)
    audit = client.post(f"/api/runs/{run_id}/steps/feature_selection/preview").json()["audit"]
    removed = [row["feature"] for row in audit]

    resp = client.post(
        f"/api/runs/{run_id}/steps/feature_selection/confirm",
        json={"removed_columns": removed},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "done"
    assert set(body["removed_columns"]) == set(removed)

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["feature_selection"] is True
    assert state["step_status"]["imbalance"] == "current"


def test_feature_selection_confirm_cannot_remove_everything(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_split(client, feature_csv_path)
    _confirm_step5_accept_all(client, run_id)

    # Full post-transform column universe — via transform-preview with every accepted
    # recipe — not just the audit's flagged subset, so this reliably exercises the
    # guard regardless of which features feature selection happens to flag.
    candidates = client.post(f"/api/runs/{run_id}/steps/feature_processing/preview").json()[
        "candidates"
    ]
    recipes = {c["column"]: c["recommended"] for c in candidates}
    all_columns = client.post(
        f"/api/runs/{run_id}/steps/feature_processing/transform-preview",
        json={"recipes": recipes},
    ).json()["columns_after"]

    resp = client.post(
        f"/api/runs/{run_id}/steps/feature_selection/confirm",
        json={"removed_columns": all_columns},
    )
    assert resp.status_code == 400


def test_imbalance_preview_reports_class_balance(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_split(client, feature_csv_path)
    _confirm_step5_accept_all(client, run_id)
    client.post(f"/api/runs/{run_id}/steps/feature_selection/confirm", json={"skipped": True})

    resp = client.post(f"/api/runs/{run_id}/steps/imbalance/preview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applicable"] is True
    assert body["n_train"] > 0
    # synthetic_df's target is a 70/30 split — event rate should land near 0.30.
    assert 0.2 < body["event_rate"] < 0.4
    assert body["is_balanced"] is False


def test_imbalance_confirm_persists_and_advances_state(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_split(client, feature_csv_path)
    _confirm_step5_accept_all(client, run_id)
    client.post(f"/api/runs/{run_id}/steps/feature_selection/confirm", json={"skipped": True})

    resp = client.post(f"/api/runs/{run_id}/steps/imbalance/confirm", json={"strategy": "smote"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["strategy"] == "smote"
    assert body["rows_after"] > body["rows_before"]

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["imbalance"] is True
    assert state["step_status"]["train"] == "current"


def _confirm_through_imbalance(
    client: TestClient, feature_csv_path: Path, strategy: str = "class_weight"
) -> str:
    """Creates a run and confirms Steps 1-7 against ``feature_csv_path``, returning
    ``run_id`` — the shared setup every Step 8/9 test needs.
    """
    run_id = _confirm_through_split(client, feature_csv_path)
    _confirm_step5_accept_all(client, run_id)
    client.post(f"/api/runs/{run_id}/steps/feature_selection/confirm", json={"skipped": True})
    resp = client.post(f"/api/runs/{run_id}/steps/imbalance/confirm", json={"strategy": strategy})
    assert resp.status_code == 200, resp.text
    return run_id


def _poll_job(client: TestClient, run_id: str, job_id: str, timeout_s: float = 60.0) -> dict:
    """Polls a background job until it reaches a terminal state, or raises on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/api/runs/{run_id}/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] != "running":
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s")


def test_train_preview_lists_algorithms(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/train/preview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = {a["name"] for a in body["algorithms"]}
    assert "xgboost" in names
    assert "svm" not in body["leaderboard_algorithms"]
    assert body["n_features"] > 0


def test_train_single_algorithm_confirm(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "logistic"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["confirmed"] is True
    assert body["algorithm"] == "logistic"
    splits = {row["split"] for row in body["metrics"]}
    assert "train" in splits

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["train"] is True
    assert state["step_status"]["tuning"] == "current"


def test_train_single_algorithm_class_weight_applied_silently(
    client: TestClient, feature_csv_path: Path
) -> None:
    # class_weight strategy + an estimator (xgboost) that silently ignores class_weight
    # via a bare try/except: pass — this must not raise, matching Streamlit's exact,
    # deliberately-preserved silent asymmetry (see services/train.py's _apply_class_weight).
    run_id = _confirm_through_imbalance(client, feature_csv_path, strategy="class_weight")
    resp = client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "xgboost"})
    assert resp.status_code == 200, resp.text


def test_train_before_split_rejected(client: TestClient) -> None:
    run_id = _create_run(client)
    resp = client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "logistic"})
    assert resp.status_code == 400


def test_train_leaderboard_start_poll_confirm(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/train/leaderboard/start")
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    job = _poll_job(client, run_id, job_id)
    assert job["status"] == "done", job
    winner = job["result"]["winner"]
    assert winner is not None
    rows = job["result"]["rows"]
    assert len(rows) > 0
    assert all(r["algorithm"] != "svm" for r in rows)

    resp = client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": winner})
    assert resp.status_code == 200, resp.text
    assert resp.json()["algorithm"] == winner

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["train"] is True


def test_tuning_preview_auto_skip_for_no_search_space_algorithm(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "gradient_boosting"})

    resp = client.post(f"/api/runs/{run_id}/steps/tuning/preview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["algorithm"] == "gradient_boosting"
    assert body["available"] is False


def test_tuning_preview_before_train_rejected(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/tuning/preview")
    assert resp.status_code == 400


def test_tuning_start_poll_confirm_accept_tuned(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "logistic"})

    resp = client.post(f"/api/runs/{run_id}/steps/tuning/tune/start", json={"n_trials": 5})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    job = _poll_job(client, run_id, job_id)
    assert job["status"] == "done", job
    result = job["result"]
    assert isinstance(result["best_score"], float)
    assert isinstance(result["baseline_auc"], float)
    assert isinstance(result["params_comparison"], list)
    assert isinstance(result["trials"], list)
    assert len(result["trials"]) > 0

    resp = client.post(f"/api/runs/{run_id}/steps/tuning/confirm", json={"use_tuned": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tuned"] is True
    assert body["best_score"] is not None

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["tuning"] is True
    assert state["step_status"]["evaluate"] == "current"


def test_tuning_confirm_reject_keeps_default(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "logistic"})

    resp = client.post(f"/api/runs/{run_id}/steps/tuning/confirm", json={"use_tuned": False})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tuned"] is False
    assert body["best_score"] is None


def test_tuning_confirm_use_tuned_without_a_tuning_run_rejected(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "logistic"})

    resp = client.post(f"/api/runs/{run_id}/steps/tuning/confirm", json={"use_tuned": True})
    assert resp.status_code == 400


def _confirm_through_tuning(client: TestClient, feature_csv_path: Path, tuned: bool = False) -> str:
    """Confirms Steps 1-9 (logistic algorithm) against ``feature_csv_path``, returning
    ``run_id`` — the shared setup every Step 10/11/12 test needs.
    """
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "logistic"})
    if tuned:
        resp = client.post(f"/api/runs/{run_id}/steps/tuning/tune/start", json={"n_trials": 5})
        job_id = resp.json()["job_id"]
        _poll_job(client, run_id, job_id)
    resp = client.post(f"/api/runs/{run_id}/steps/tuning/confirm", json={"use_tuned": tuned})
    assert resp.status_code == 200, resp.text
    return run_id


def test_evaluate_preview_no_baseline_when_not_tuned(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path, tuned=False)
    resp = client.post(f"/api/runs/{run_id}/steps/evaluate/preview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["algorithm"] == "logistic"
    assert body["tuned"] is False
    assert body["baseline"] == []
    assert len(body["metrics"]) > 0
    assert any(m["metric"] == "roc_auc" for m in body["metrics"])


def test_evaluate_preview_includes_baseline_when_tuned(
    client: TestClient, feature_csv_path: Path
) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path, tuned=True)
    resp = client.post(f"/api/runs/{run_id}/steps/evaluate/preview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tuned"] is True
    assert len(body["baseline"]) == 3
    assert {b["metric"] for b in body["baseline"]} == {"roc_auc", "gini", "ks_statistic"}


def test_evaluate_confirm(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/evaluate/confirm")
    assert resp.status_code == 200, resp.text
    assert resp.json()["confirmed"] is True

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["evaluate"] is True
    assert state["step_status"]["calibration"] == "current"


def test_evaluate_before_tuning_rejected(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/train/confirm", json={"algorithm": "logistic"})
    resp = client.post(f"/api/runs/{run_id}/steps/evaluate/preview")
    assert resp.status_code == 400


def test_calibration_preview_classification(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/calibration/preview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is True
    assert body["split_used"] in ("val", "test", "train")
    assert isinstance(body["ece_before"], float)
    assert isinstance(body["ece_after"], float)

    # Non-classification's auto-skip branch (preview_calibration/confirm_calibration's
    # `task != "classification"` check) is deliberately not exercised at this API level —
    # no run ever reaches Step 11 with a non-classification task in this test file, since
    # Steps 5-13 are all classification-only fixtures, consistent with this project's
    # documented Task Scope Priority decision (project.md: prioritise classification,
    # don't proactively invest in regression/clustering-specific hardening).


def test_calibration_confirm_apply(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/calibration/confirm", json={"apply": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["confirmed"] is True
    assert body["applied"] is True

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["calibration"] is True
    assert state["step_status"]["shap"] == "current"


def test_calibration_confirm_skip(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/calibration/confirm", json={"apply": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is False


def test_calibration_before_tuning_rejected(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_imbalance(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/calibration/preview")
    assert resp.status_code == 400


def test_shap_preview(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/calibration/confirm", json={"apply": False})
    resp = client.post(f"/api/runs/{run_id}/steps/shap/preview")
    assert resp.status_code == 200, resp.text
    assert resp.json()["available"] is True


def test_shap_run(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/calibration/confirm", json={"apply": True})

    resp = client.post(f"/api/runs/{run_id}/steps/shap/run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "data" in body["figure"]
    assert len(body["top_features"]) > 0
    assert body["top_features"][0]["rank"] == 1
    assert body["split_used"] in ("val", "test", "train")
    assert body["n_rows"] > 0


def test_shap_confirm_run_true(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/calibration/confirm", json={"apply": False})
    client.post(f"/api/runs/{run_id}/steps/shap/run")

    resp = client.post(f"/api/runs/{run_id}/steps/shap/confirm", json={"shap_run": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["confirmed"] is True

    state = client.get(f"/api/runs/{run_id}/state").json()
    assert state["step_confirmed"]["shap"] is True
    audit_steps = [entry[0] for entry in state["audit_trail"]]
    assert "shap" in audit_steps


def test_shap_confirm_run_false(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    client.post(f"/api/runs/{run_id}/steps/calibration/confirm", json={"apply": False})

    resp = client.post(f"/api/runs/{run_id}/steps/shap/confirm", json={"shap_run": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["confirmed"] is True


def test_shap_before_calibration_rejected(client: TestClient, feature_csv_path: Path) -> None:
    run_id = _confirm_through_tuning(client, feature_csv_path)
    resp = client.post(f"/api/runs/{run_id}/steps/shap/preview")
    assert resp.status_code == 400


def test_unported_step_returns_501(client: TestClient) -> None:
    # report (Step 13) is the only step left unported after G.5 (Steps 10-12).
    run_id = _create_run(client)
    resp = client.post(f"/api/runs/{run_id}/steps/report/preview")
    assert resp.status_code == 501


def test_job_status_unknown_job_id_404(client: TestClient, csv_path: Path) -> None:
    run_id = _create_run(client)
    _confirm_step1(client, run_id, csv_path)
    resp = client.get(f"/api/runs/{run_id}/jobs/does-not-exist")
    assert resp.status_code == 404


def test_auth_rejects_bad_api_key(client: TestClient) -> None:
    original = api_settings.api_key
    api_settings.api_key = "expected-key"
    try:
        resp = client.post("/api/runs", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

        resp = client.post("/api/runs", headers={"X-API-Key": "expected-key"})
        assert resp.status_code == 200
    finally:
        api_settings.api_key = original


def test_openapi_snapshot_matches_live_schema() -> None:
    """A contract change without regenerating ``openapi.json`` should fail CI, not
    silently diverge from what the frontend was built against.
    """
    live_schema = create_app().openapi()
    committed_schema = json.loads(_OPENAPI_SNAPSHOT_PATH.read_text())
    assert live_schema == committed_schema, (
        "dscompanion/api/openapi.json is stale — regenerate it "
        "(see dscompanion/api/main.py's create_app().openapi())."
    )


def test_auth_backend_unknown_value_raises_value_error() -> None:
    original = api_settings.auth_backend
    api_settings.auth_backend = "not_a_real_backend"
    try:
        with pytest.raises(ValueError, match="Unknown auth_backend"):
            get_auth_backend()
    finally:
        api_settings.auth_backend = original


def test_state_backend_unknown_value_raises_value_error() -> None:
    original_backend = api_settings.state_backend
    original_store = api_state_module._store
    api_state_module._store = None
    api_settings.state_backend = "not_a_real_backend"
    try:
        with pytest.raises(ValueError, match="Unknown state_backend"):
            api_state_module.get_run_state_store()
    finally:
        api_settings.state_backend = original_backend
        api_state_module._store = original_store
