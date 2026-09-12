"""databricks_oauth auth backend — stub only, not built.

OAuth2 machine-to-machine via WorkspaceClient is Databricks Apps' native auth
pattern (confirmed in frontend.md's Databricks Apps research). Not implemented
until that deployment channel is actually pursued — this
file exists so the seam is ready, not so this
resolution requires touching deps.py/routers later.

Lazy-imports databricks-sdk inside the function body (never at module level), per
this codebase's "lazy imports for optional dependencies" convention — zero new
required dependency until this file has real content.
"""

from __future__ import annotations

from dscompanion.api.auth import AuthResult

__all__ = ["check_databricks_oauth"]


def check_databricks_oauth(credential: str | None) -> AuthResult:
    """Not implemented — see module docstring.

    Args:
        credential (str | None): The raw bearer token, once implemented.

    Returns:
        AuthResult: Never returns.

    Raises:
        NotImplementedError: Always — this backend has no implementation yet.
    """
    raise NotImplementedError(
        "auth_backend='databricks_oauth' is not built yet. "
        "Use 'api_key' (the default) until the Databricks Apps channel is actually pursued."
    )
