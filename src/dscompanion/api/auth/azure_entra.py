"""azure_entra auth backend — stub only, not built.

Same treatment as databricks_oauth.py: this file exists so the seam is ready,
not implemented until the Azure channel is
actually pursued. Would lazy-import azure-identity
inside the function body when implemented, never at module level.
"""

from __future__ import annotations

from dscompanion.api.auth import AuthResult

__all__ = ["check_azure_entra"]


def check_azure_entra(credential: str | None) -> AuthResult:
    """Not implemented — see module docstring.

    Args:
        credential (str | None): The raw bearer token, once implemented.

    Returns:
        AuthResult: Never returns.

    Raises:
        NotImplementedError: Always — this backend has no implementation yet.
    """
    raise NotImplementedError(
        "auth_backend='azure_entra' is not built yet. "
        "Use 'api_key' (the default) until the Azure channel is actually pursued."
    )
