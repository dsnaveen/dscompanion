"""Thin wrapper functions around dscompanion calls, one module per wizard step.

Every function here takes a ``RunState`` (or nothing, for a stateless preview) plus a
``dscompanion.api.schemas`` request model, and returns a ``dscompanion.api.schemas`` response
model — no FastAPI/HTTP concepts leak in here, so these are also directly unit-testable
without a running app (see ``dscompanion/tests/test_api.py``).

Only Steps 1-3 are implemented (the vertical slice proving the architecture end-to-end).
Steps 4-13 are ported over time.
"""

from __future__ import annotations

__all__: list[str] = []
