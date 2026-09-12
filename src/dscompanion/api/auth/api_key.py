"""api_key auth backend — local-dev stub, checked against APISettings.api_key."""

from __future__ import annotations

import logging

from dscompanion.api.auth import AuthResult

logger = logging.getLogger(__name__)

__all__ = ["check_api_key"]


def check_api_key(credential: str | None) -> AuthResult:
    """Checks ``credential`` against ``api_settings.api_key``.

    Local-dev only — needed regardless of eventual deployment target, easy to
    upgrade later. Real auth (SSO/Entra, Databricks OAuth) is deferred until
    that deployment channel is actually pursued.

    Args:
        credential (str | None): The raw ``X-API-Key`` header value, or ``None``
            if the header was absent.

    Returns:
        AuthResult: ``authenticated=True`` when ``api_settings.api_key`` is
        ``None`` (auth disabled — local dev only) or when ``credential`` matches
        it exactly; ``authenticated=False`` otherwise.
    """
    from dscompanion.api.config import api_settings

    if api_settings.api_key is None:
        logger.debug("api_key auth: no key configured, allowing all requests (local dev only)")
        return AuthResult(authenticated=True, principal="local-dev")

    if credential == api_settings.api_key:
        return AuthResult(authenticated=True, principal="api-key-user")

    logger.warning("api_key auth: rejected request with invalid or missing credential")
    return AuthResult(authenticated=False, principal=None)
