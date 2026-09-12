"""One router per wizard step (plus run-level and job endpoints) — ``main.py`` includes
every router listed in ``ALL_ROUTERS`` on the FastAPI app.
"""

from __future__ import annotations

from dscompanion.api.routers import (
    calibration,
    data_files,
    eda,
    evaluate,
    feature_processing,
    feature_selection,
    imbalance,
    jobs,
    load_data,
    runs,
    shap,
    split,
    steps_stub,
    task_target,
    train,
    transformer_registry,
    tuning,
)

__all__ = ["ALL_ROUTERS"]

ALL_ROUTERS = [
    runs.router,
    jobs.router,
    data_files.router,
    transformer_registry.router,
    load_data.router,
    task_target.router,
    split.router,
    eda.router,
    feature_processing.router,
    feature_selection.router,
    imbalance.router,
    train.router,
    tuning.router,
    evaluate.router,
    calibration.router,
    shap.router,
    steps_stub.router,
]
