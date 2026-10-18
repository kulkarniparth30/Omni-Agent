from fastapi import APIRouter, Depends
from typing import List
from langchain_core.tools import BaseTool

from app.schemas.health import HealthResponse
from app.config import get_settings
from app.api.deps import get_tools
from app.core.redis import is_redis_available

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check(tools: List[BaseTool] = Depends(get_tools)):
    """Health check endpoint: verifies service is running and reports active tools."""
    settings = get_settings()
    tool_names = [t.name for t in tools]
    return HealthResponse(
        status="ok",
        environment=settings.APP_ENV,
        tools_count=len(tool_names),
        available_tools=tool_names,
        redis_status="online" if is_redis_available() else "offline"
    )
