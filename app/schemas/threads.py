from pydantic import BaseModel, Field
from typing import List, Optional

class MessageItem(BaseModel):
    """Normalized representation of a single conversation message."""
    role: str = Field(..., description="Message role: 'user', 'assistant', 'system', or 'tool'")
    content: str = Field(..., description="Text content of the message")
    tool_name: Optional[str] = Field(None, description="Name of tool if this is a tool execution result")

class ThreadListResponse(BaseModel):
    """List of all known conversation thread IDs."""
    threads: List[str] = Field(default_factory=list, description="Array of thread IDs")
    total: int = Field(..., description="Count of distinct conversation threads")

class ThreadMessagesResponse(BaseModel):
    """Full message history for a given thread."""
    thread_id: str
    messages: List[MessageItem] = Field(default_factory=list)
