from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.schemas.threads import ThreadListResponse, ThreadMessagesResponse, MessageItem
from app.api.deps import get_agent, get_checkpointer
from app.core.exceptions import ThreadNotFoundError

router = APIRouter(prefix="/threads", tags=["Threads"])

@router.get("", response_model=ThreadListResponse)
async def list_threads(checkpointer: Any = Depends(get_checkpointer)):
    """Retrieve all unique conversation thread IDs stored in the checkpointer."""
    threads = set()
    try:
        async for checkpoint in checkpointer.alist(None):
            cfg = getattr(checkpoint, "config", {})
            thread_id = cfg.get("configurable", {}).get("thread_id")
            if thread_id:
                threads.add(str(thread_id))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query thread checkpoints: {str(exc)}"
        )

    thread_list = sorted(list(threads), reverse=True)
    return ThreadListResponse(threads=thread_list, total=len(thread_list))

@router.get("/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def get_thread_messages(
    thread_id: str,
    agent: Any = Depends(get_agent)
):
    """Fetch the full conversation message history for a given thread."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await agent.aget_state(config)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading thread state: {str(exc)}"
        )

    if not state or not state.values or "messages" not in state.values or not state.values["messages"]:
        raise ThreadNotFoundError(thread_id=thread_id)

    raw_messages = state.values.get("messages", [])
    normalized: List[MessageItem] = []

    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            role = "user"
            tool_name = None
        elif isinstance(msg, AIMessage):
            role = "assistant"
            tool_name = None
        elif isinstance(msg, ToolMessage):
            role = "tool"
            tool_name = getattr(msg, "name", "tool")
        elif isinstance(msg, SystemMessage):
            role = "system"
            tool_name = None
        else:
            role = "unknown"
            tool_name = None

        normalized.append(
            MessageItem(
                role=role,
                content=str(msg.content),
                tool_name=tool_name
            )
        )

    return ThreadMessagesResponse(thread_id=thread_id, messages=normalized)
