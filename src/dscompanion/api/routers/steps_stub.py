"""Scaffolded endpoints for the not-yet-ported steps.

Registered now so the full step surface is visible in the OpenAPI schema from day
one (frontend/enterprise/ can be built against a stable contract before every step's
real logic lands), but every call returns 501 until its ``services/`` module exists.
Which steps land here is entirely a function of ``PORTED_STEPS`` — porting a step (or
inserting a brand-new one) never requires touching this file.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from dscompanion.api.deps import get_auth, get_pipeline_service
from dscompanion.api.state import RunState
from dscompanion.api.steps import PORTED_STEPS, STEP_ORDER, StepId, step_number

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/runs/{run_id}/steps", tags=["steps-unported"], dependencies=[Depends(get_auth)]
)

__all__ = ["router"]

_UNPORTED_STEPS: list[StepId] = [s for s in STEP_ORDER if s not in PORTED_STEPS]


def _not_implemented(step: StepId) -> None:
    """Raises the standard 501 for an unported step.

    Args:
        step (StepId): The wizard step slug.

    Returns:
        None

    Raises:
        HTTPException: 501, always.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            f"Step {step_number(step)} ({step}) is not ported to the API yet — "
            "not yet implemented."
        ),
    )


for _step in _UNPORTED_STEPS:

    def _make_handlers(step: StepId):
        def preview(run: RunState = Depends(get_pipeline_service)) -> None:
            _not_implemented(step)

        def confirm(run: RunState = Depends(get_pipeline_service)) -> None:
            _not_implemented(step)

        return preview, confirm

    _preview, _confirm = _make_handlers(_step)
    router.add_api_route(f"/{_step}/preview", _preview, methods=["POST"])
    router.add_api_route(f"/{_step}/confirm", _confirm, methods=["POST"])
