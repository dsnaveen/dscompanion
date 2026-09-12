"""Transformer registry endpoint — not run-scoped, thin wrapper around
``dscompanion.features.registry``. Step 5's recipe editor reads this to build its "add step"
control.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from dscompanion.api.deps import get_auth
from dscompanion.api.schemas import TransformerRegistryResponse
from dscompanion.features import registered_transformer_names

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/transformer-registry",
    tags=["transformer-registry"],
    dependencies=[Depends(get_auth)],
)

__all__ = ["router"]


@router.get("", response_model=TransformerRegistryResponse)
def get_transformer_registry() -> TransformerRegistryResponse:
    """Lists every registered transformer key.

    Args:
        None

    Returns:
        TransformerRegistryResponse: Valid values for ``TransformStep.transformer``.
    """
    return TransformerRegistryResponse(names=registered_transformer_names())
