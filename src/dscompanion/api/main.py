"""FastAPI app factory for dscompanion/api/.

Run locally with:

    conda run -n dscompanion312 uvicorn dscompanion.api.main:app --reload

``create_app()`` exists separately from the module-level ``app`` so tests can build a
fresh app (and thus a fresh dependency graph) per test without import-time side effects.

**Local, single-user use only -- never deploy this as a shared or publicly-hosted
service without revisiting its security posture first.** In particular, Step 1's
load-data endpoints (``dscompanion.api.services.load_data``) deliberately allow an
absolute file path from the caller with no access-root restriction, on the
assumption that the person calling this API and the person running it are the same
person on the same machine (they already have that same file access directly,
with or without this API). That assumption breaks -- turning this into a real
arbitrary-file-read vulnerability -- the moment this service is exposed to any
caller who isn't also the machine's own trusted user.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from dscompanion.api.config import api_settings
from dscompanion.api.routers import ALL_ROUTERS

logger = logging.getLogger(__name__)

__all__ = ["create_app", "app"]


def create_app() -> FastAPI:
    """Builds and returns a FastAPI application with every router mounted.

    Args:
        None

    Returns:
        FastAPI: The assembled application — CORS configured from
        ``api_settings.cors_origins``, every router in ``ALL_ROUTERS`` included.
    """
    application = FastAPI(
        title="dscompanion API",
        description="REST surface over dscompanion's 13-step interactive pipeline wizard.",
        version="0.1.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in ALL_ROUTERS:
        application.include_router(router)

    def custom_openapi() -> dict:
        """Adds a top-level default ``security`` requirement to the generated
        schema. Every route already declares this per-operation (via
        ``deps.get_auth``'s ``APIKeyHeader`` dependency); this only adds the
        global default OpenAPI itself doesn't infer automatically (checkov
        ``CKV_OPENAPI_4``, confirmed as a real gap, not a false positive).
        """
        if application.openapi_schema:
            return application.openapi_schema
        schema = get_openapi(
            title=application.title,
            version=application.version,
            description=application.description,
            routes=application.routes,
        )
        schema["security"] = [{"APIKeyHeader": []}]
        application.openapi_schema = schema
        return application.openapi_schema

    application.openapi = custom_openapi

    logger.info(
        "dscompanion API app created: auth_backend=%s state_backend=%s",
        api_settings.auth_backend,
        api_settings.state_backend,
    )
    return application


app = create_app()
