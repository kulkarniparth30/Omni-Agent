from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    """Incoming chat request payload."""
    message: str = Field(..., min_length=1, description="The user prompt or query.")
    thread_id: str = Field(..., min_length=1, description="Unique conversation session ID.")

class ChatResponse(BaseModel):
    """Final response returned from a non-streaming chat turn."""
    response: str = Field(..., description="The assistant's text response.")
    tools_used: List[str] = Field(default_factory=list, description="Names of all tools invoked during this turn.")
    thread_id: str = Field(..., description="The conversation thread ID.")
    latency_ms: float = Field(..., description="Turn execution time in milliseconds.")

class StreamTokenEvent(BaseModel):
    """Payload for individual streaming token events."""
    token: str

class StreamToolEvent(BaseModel):
    """Payload for tool lifecycle events in stream mode."""
    tool_name: str
    stage: str = Field(..., description="'tool_start' or 'tool_end'")
    details: Optional[str] = None
