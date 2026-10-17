from typing import List, Optional
from langchain_core.tools import BaseTool

from app.tools.search import search_tool, semantic_scholar_tool
from app.tools.stock import get_stock_price
from app.tools.rag import rag_tool
from app.tools.mcp import load_mcp_tools

def get_base_tools() -> List[BaseTool]:
    """Return all standard, built-in tools (excluding external MCP tools)."""
    return [
        search_tool,
        semantic_scholar_tool,
        get_stock_price,
        rag_tool,
    ]

def get_all_tools(mcp_tools: Optional[List[BaseTool]] = None) -> List[BaseTool]:
    """
    Registry function: return the full list of tools for the agent.
    Optionally accepts dynamically loaded MCP tools.
    """
    tools = list(get_base_tools())
    if mcp_tools:
        tools.extend(mcp_tools)
    return tools

__all__ = [
    "search_tool",
    "semantic_scholar_tool",
    "get_stock_price",
    "rag_tool",
    "load_mcp_tools",
    "get_base_tools",
    "get_all_tools",
]
