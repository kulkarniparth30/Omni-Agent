from fastapi import Request, HTTPException, status
from typing import Any, List
from langchain_core.tools import BaseTool

def get_agent(request: Request) -> Any:
    """Dependency provider: extract compiled LangGraph agent from application state."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent graph is not initialized."
        )
    return agent

def get_checkpointer(request: Request) -> Any:
    """Dependency provider: extract AsyncSqliteSaver checkpointer from application state."""
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Checkpointer is not initialized."
        )
    return checkpointer

def get_tools(request: Request) -> List[BaseTool]:
    """Dependency provider: extract the active list of registered tools."""
    return getattr(request.app.state, "tools", [])
