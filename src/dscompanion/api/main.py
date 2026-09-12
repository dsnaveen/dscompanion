"""FastAPI app factory for dscompanion/api/.

Run locally with:

    conda run -n dscompanion312 uvicorn dscompanion.api.main:app --reload

``create_app()`` exists separately from the module-level ``app`` so tests can build a
fresh app (and thus a fresh dependency graph) per test without import-time side effects.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    logger.info(
        "dscompanion API app created: auth_backend=%s state_backend=%s",
        api_settings.auth_backend,
        api_settings.state_backend,
    )
    return application


app = create_app()
