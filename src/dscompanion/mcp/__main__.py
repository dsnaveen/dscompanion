"""Entry point for ``python -m dscompanion.mcp`` -- launches the local MCP server
over stdio transport. Point your MCP client's config (e.g. Claude Desktop) at this
command; nothing else needs to run.
"""

from dscompanion.mcp.server import mcp

if __name__ == "__main__":
    mcp.run()
