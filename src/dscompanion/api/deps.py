"""FastAPI Depends() wiring — routers depend only on these functions, never on a
concrete auth/state backend or a channel conditional directly.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from dscompanion.api.auth import get_auth_backend
from dscompanion.api.state import RunState, RunStateStore, get_run_state_store

logger = logging.getLogger(__name__)

__all__ = ["get_state_store", "get_auth", "get_pipeline_service"]

# A proper FastAPI security scheme (not a plain Header() parameter) so the
# generated OpenAPI spec actually declares this as an auth requirement
# (components.securitySchemes + per-operation security) -- a plain Header()
# parameter is indistinguishable from an ordinary header to FastAPI's OpenAPI
# generator, which is why the previous version left security/securitySchemes
# empty (checkov CKV_OPENAPI_4/CKV_OPENAPI_5, confirmed as a real gap, not a
# false positive). auto_error=False preserves the existing behavior of
# passing through to get_auth() below even when the header is absent, since
# check_api_key() already handles the "no key configured" local-dev case
# itself.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_state_store() -> RunStateStore:
    """FastAPI dependency returning the process-wide ``RunStateStore``.

    Args:
        None

    Returns:
        RunStateStore: The active store instance (see
        ``dscompanion.api.state.get_run_state_store``).
    """
    return get_run_state_store()


def get_auth(x_api_key: str | None = Security(_api_key_header)):
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
