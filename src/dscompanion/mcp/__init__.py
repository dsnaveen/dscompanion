"""Local MCP (Model Context Protocol) server exposing dscompanion as agent-callable
tools. Runs entirely on the user's own machine via stdio transport -- no data leaves
the local environment. Install with ``pip install dscompanion[mcp]`` and run with
``python -m dscompanion.mcp``.
"""

from __future__ import annotations

__all__: list[str] = []
