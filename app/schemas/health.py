from pydantic import BaseModel, Field
from typing import List

class HealthResponse(BaseModel):
    """System health status."""
    status: str = Field(default="ok", description="Overall service status")
    environment: str = Field(..., description="Application runtime environment")
    tools_count: int = Field(..., description="Number of active tools registered in the agent")
    available_tools: List[str] = Field(default_factory=list, description="Names of available tools")
