"""APISettings: single source of truth for which channel-specific backend is active.

Mirrors dscompanion/dscompanion/config.py's pydantic-settings convention. This is the one place
that says which auth backend / run-state store is active — routers and services never
branch on a channel condition directly, they depend on deps.py, which reads this.
"""

from __future__ import annotations

from typing import Literal

import pydantic as _pydantic

if int(_pydantic.VERSION.split(".")[0]) >= 2:
    try:
        from pydantic_settings import BaseSettings
    except ImportError:
        from pydantic.v1 import BaseSettings  # pydantic v2 bundles v1 compat layer
else:
    from pydantic import BaseSettings  # pydantic v1.x
from pydantic import Field

__all__ = ["APISettings", "api_settings"]


class APISettings(BaseSettings):
    """Holds all `dscompanion/api/`-specific configuration, overridable via environment
    variables prefixed ``DSCOMPANION_API_`` (e.g. ``DSCOMPANION_API_AUTH_BACKEND=databricks_oauth``)
    or a ``.env`` file in the current working directory.

    Args:
        auth_backend (str): Which auth module (`dscompanion/api/auth/`) handles incoming
            request authentication. One of ``"api_key"`` (default — local-dev stub),
            ``"databricks_oauth"``, or ``"azure_entra"``. The latter two are stubs only
            until their respective deployment channel is actually pursued.
        state_backend (str): Which run-state store (`dscompanion/api/state.py`) holds
            per-``run_id`` pipeline state. One of ``"memory"`` (default — a plain dict,
            lost on restart) or ``"redis"`` (stub only, not built — see
            `RunStateStore`'s own docstring for the audit-durability decision this is
            gated on).
        api_key (str | None): The single accepted key when ``auth_backend="api_key"``.
            ``None`` means auth is disabled entirely (local dev only — never set this
            to ``None`` outside a personal machine).
        cors_origins (list[str]): Origins allowed to call this API from a browser.
            Defaults to the Vite dev server's default port only.
        preview_sample_rows (int): Number of sample rows Step 1's preview endpoint
            returns alongside the column/dtype summary — mirrors
            ``interactive_step1.py``'s ``df.head(10)`` default.
        eda_high_missing_threshold (float): Missing-value rate above which Step 4's
            EDA preview raises a HIGH_MISSING alert — mirrors
            ``interactive_step4.py``'s ``_HIGH_MISSING_DEFAULT`` (a deliberate
            UI-level override of ``dscompanion.config.settings.high_missing_threshold``,
            not the core library's own default).

    Returns:
        APISettings: A fully validated settings instance.
    """

    auth_backend: Literal["api_key", "databricks_oauth", "azure_entra"] = "api_key"
    state_backend: Literal["memory", "redis"] = "memory"
    api_key: str | None = Field(
        default=None, description="Accepted API key for auth_backend='api_key'"
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    preview_sample_rows: int = 10
    eda_high_missing_threshold: float = 0.30

    model_config = {"env_prefix": "DSCOMPANION_API_", "env_file": ".env", "extra": "ignore"}


api_settings = APISettings()
