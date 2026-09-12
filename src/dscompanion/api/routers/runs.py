"""Run-level endpoints: create a run, read its state, and the two-phase "back" action —
the REST equivalent of ``dscompanion/app/interactive_state.py``'s ``go_to_step()``/
``reset_from_step()`` helpers.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from dscompanion.api.deps import get_auth, get_pipeline_service, get_state_store
from dscompanion.api.schemas import BackRequest, BackResponse, RunCreateResponse, RunStateResponse
from dscompanion.api.state import RunState, RunStateStore
from dscompanion.api.steps import STEP_ORDER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs", tags=["runs"], dependencies=[Depends(get_auth)])

__all__ = ["router"]


def _build_state_response(run: RunState) -> RunStateResponse:
    """Assembles the REST equivalent of ``_interactive_flags()`` +
    ``_build_partial_config_dict()`` from a ``RunState``.

    Args:
        run (RunState): The run to summarise.

    Returns:
        RunStateResponse: Step statuses, confirmations, audit trail, and a config
        preview built from confirmed steps' persisted ``step_data``.
    """
    # Explicit STEP_ORDER position, not dict insertion/sort order — a bare sorted()
    # over string slug keys would be alphabetical, not wizard order.
    config_preview = {s: run.step_data[s] for s in STEP_ORDER if s in run.step_data}
    return RunStateResponse(
        run_id=run.run_id,
        step_status=dict(run.step_status),
        step_confirmed=dict(run.step_confirmed),
        audit_trail=list(run.audit_trail),
        config_preview=config_preview,
    )


@router.post("", response_model=RunCreateResponse)
def create_run(store: RunStateStore = Depends(get_state_store)) -> RunCreateResponse:
    """Creates a new run.

    Args:
        store (RunStateStore): Injected via ``get_state_store``.

    Returns:
        RunCreateResponse: The new run's ``run_id``.
    """
    run = store.create_run()
    return RunCreateResponse(run_id=run.run_id)


@router.get("/{run_id}/state", response_model=RunStateResponse)
def get_state(run: RunState = Depends(get_pipeline_service)) -> RunStateResponse:
    """Returns the current state of a run.

    Args:
        run (RunState): Injected via ``get_pipeline_service`` from the ``run_id`` path
            parameter.

    Returns:
        RunStateResponse: Step statuses, confirmations, audit trail, config preview.
    """
    return _build_state_response(run)


@router.post("/{run_id}/back", response_model=BackResponse)
def go_back(body: BackRequest, run: RunState = Depends(get_pipeline_service)) -> BackResponse:
    """Two-phase reset to an earlier step, mirroring the UI's confirm-before-apply back
    button.

    Args:
        body (BackRequest): ``to_step`` and whether this call actually applies the
            reset (``confirmed=True``) or is just asking for confirmation.
        run (RunState): Injected via ``get_pipeline_service``.

    Returns:
        BackResponse: If ``body.confirmed`` is ``False``, only
        ``requires_confirmation=True`` is set. Otherwise the reset is applied and the
        updated state is returned.
    """
    if not body.confirmed:
        return BackResponse(requires_confirmation=True, state=None)

    run.reset_from_step(body.to_step)
    return BackResponse(requires_confirmation=False, state=_build_state_response(run))
