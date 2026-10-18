from typing import List, Dict, Any
from pathlib import Path
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)

def build_mcp_servers_config() -> Dict[str, Any]:
    """Construct the MCP servers configuration dictionary from settings."""
    settings = get_settings()
    servers: Dict[str, Any] = {}

    # Local stdio arithmetic server (if script exists)
    if settings.MCP_ARITH_SCRIPT_PATH:
        script_path = Path(settings.MCP_ARITH_SCRIPT_PATH)
        if script_path.exists():
            servers["arith"] = {
                "transport": "stdio",
                "command": "python",
                "args": [str(script_path)],
            }
        else:
            logger.warning("Configured MCP_ARITH_SCRIPT_PATH not found: %s", script_path)

    # Remote streamable_http servers
    if settings.MCP_EXPENSE_URL:
        servers["expense"] = {
            "transport": "streamable_http",
            "url": settings.MCP_EXPENSE_URL
        }

    if settings.MCP_CONTEXT7_URL:
        servers["context7"] = {
            "transport": "streamable_http",
            "url": settings.MCP_CONTEXT7_URL
        }

    if settings.MCP_DEEPWIKI_URL:
        servers["deepwiki"] = {
            "transport": "streamable_http",
            "url": settings.MCP_DEEPWIKI_URL
        }

    return servers

async def load_mcp_tools() -> List[BaseTool]:
    """
    Asynchronously connect to configured MCP servers and load available tools.
    Returns an empty list on failure instead of crashing the server.
    """
    server_configs = build_mcp_servers_config()
    if not server_configs:
        logger.info("No MCP servers configured.")
        return []

    import asyncio
    try:
        client = MultiServerMCPClient(server_configs)
        tools = await asyncio.wait_for(client.get_tools(), timeout=4.0)
        logger.info("Successfully loaded %d MCP tools: %s", len(tools), [t.name for t in tools])
        return tools
    except Exception as exc:
        logger.warning("MCP servers connection timed out or failed: %s. Continuing with built-in tools.", exc)
        return []
