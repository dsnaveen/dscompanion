"""dscompanion.api.auth — pluggable auth backends, dispatched by APISettings.auth_backend."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["get_auth_backend", "AuthResult"]


class AuthResult:
    """Outcome of an auth check.

    Args:
        authenticated (bool): Whether the request is allowed through.
        principal (str | None): Identifier for who/what made the request (e.g. the
            API key's label, or an OAuth subject claim). ``None`` when
            ``authenticated=False``.

    Returns:
        AuthResult: A simple, backend-agnostic result every auth module returns,
        so ``deps.py`` never has to know which backend produced it.
    """

    def __init__(self, authenticated: bool, principal: str | None = None) -> None:
        self.authenticated = authenticated
        self.principal = principal


def get_auth_backend():
    """Returns the auth-check callable for the currently configured backend,
    dispatched by ``api_settings.auth_backend`` exactly like ``Tuner.backend``
    dispatches to ``OptunaBackend``/``HyperoptBackend`` — a plain string setting
    plus a lazy import per branch, no abstract base class.

    Args:
        None

    Returns:
        Callable[[str | None], AuthResult]: A function taking the raw credential
        (e.g. the ``X-API-Key`` header value) and returning an ``AuthResult``.

    Raises:
        ValueError: If ``api_settings.auth_backend`` is not a recognised value.
    """
    from dscompanion.api.config import api_settings

    if api_settings.auth_backend == "api_key":
        from dscompanion.api.auth.api_key import check_api_key

        return check_api_key
    elif api_settings.auth_backend == "databricks_oauth":
        from dscompanion.api.auth.databricks_oauth import check_databricks_oauth

        return check_databricks_oauth
    elif api_settings.auth_backend == "azure_entra":
        from dscompanion.api.auth.azure_entra import check_azure_entra

        return check_azure_entra
    else:
        raise ValueError(
            f"Unknown auth_backend: {api_settings.auth_backend!r}. "
            "Choose api_key/databricks_oauth/azure_entra."
        )
