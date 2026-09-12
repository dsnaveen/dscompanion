"""FastAPI Depends() wiring — routers depend only on these functions, never on a
concrete auth/state backend or a channel conditional directly.
"""

from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException

from dscompanion.api.auth import get_auth_backend
from dscompanion.api.state import RunState, RunStateStore, get_run_state_store

logger = logging.getLogger(__name__)

__all__ = ["get_state_store", "get_auth", "get_pipeline_service"]


def get_state_store() -> RunStateStore:
    """FastAPI dependency returning the process-wide ``RunStateStore``.

    Args:
        None

    Returns:
        RunStateStore: The active store instance (see
        ``dscompanion.api.state.get_run_state_store``).
    """
    return get_run_state_store()


def get_auth(x_api_key: str | None = Header(default=None)):
    """FastAPI dependency enforcing authentication on every protected route.

    Dispatches to whichever backend ``api_settings.auth_backend`` selects — the
    route itself never knows which backend ran.

    Args:
        x_api_key (str | None): The ``X-API-Key`` request header, injected by
            FastAPI. Only meaningful for ``auth_backend="api_key"``; other
            backends read their own credential source inside their module.

    Returns:
        AuthResult: The authenticated principal.

    Raises:
        HTTPException: 401, if the configured backend rejects the credential.
    """
    backend = get_auth_backend()
    result = backend(x_api_key)
    if not result.authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return result


def get_pipeline_service(run_id: str, store: RunStateStore = Depends(get_state_store)) -> RunState:
    """FastAPI dependency giving a router the per-run handle it needs to read/write
    state and invoke ``dscompanion/api/services/`` functions for this request.

    ``run_id`` is bound automatically from the route's own path parameter of the
    same name — every step router declares ``run_id: str`` in its path
    (``/api/runs/{run_id}/steps/{n}/...``), and FastAPI resolves this dependency's
    ``run_id`` argument against it. Named ``get_pipeline_service`` for the
    conceptual role it plays — in practice it returns the ``RunState``
    object itself, since `services/` functions are plain functions taking
    ``RunState`` + a request body, not methods on a separate service class.

    Args:
        run_id (str): Bound from the request path.
        store (RunStateStore): Injected via ``get_state_store``.

    Returns:
        RunState: The state for this run.

    Raises:
        HTTPException: 404, if ``run_id`` is not known to the store.
    """
    try:
        return store.get_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
