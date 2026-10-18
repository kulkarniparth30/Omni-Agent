import time
import json
from typing import Any, List, Set
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.schemas.chat import ChatRequest, ChatResponse
from app.api.deps import get_agent
from app.core.exceptions import LLMExecutionError
from app.core.redis import check_rate_limit

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    agent: Any = Depends(get_agent)
):
    """
    Synchronous / non-streaming chat turn.
    Protected by Redis sliding-window rate limiter.
    """
    # Rate Limiting Guard
    client_ip = request.client.host if request.client else "unknown_client"
    allowed, current_count, retry_after = await check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({current_count} requests/min). Please retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )

    config = {
        "configurable": {"thread_id": payload.thread_id},
        "metadata": {"thread_id": payload.thread_id},
        "run_name": "chat_turn"
    }

    start_time = time.perf_counter()
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=payload.message)]},
            config=config
        )
    except Exception as exc:
        raise LLMExecutionError(message=str(exc), details={"thread_id": payload.thread_id})

    latency_ms = (time.perf_counter() - start_time) * 1000
    messages = result.get("messages", [])

    # Find the last assistant message and collect tools used in this turn
    last_response = ""
    tools_used: List[str] = []
    seen_tools: Set[str] = set()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            if tool_name not in seen_tools:
                seen_tools.add(tool_name)
                tools_used.append(tool_name)
        elif isinstance(msg, AIMessage):
            last_response = str(msg.content)

    return ChatResponse(
        response=last_response,
        tools_used=tools_used,
        thread_id=payload.thread_id,
        latency_ms=round(latency_ms, 2)
    )

@router.post("/stream")
async def chat_stream_endpoint(
    payload: ChatRequest,
    request: Request,
    agent: Any = Depends(get_agent)
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Streams individual tokens and tool lifecycle events (tool_start, tool_end, token, done).
    Protected by Redis sliding-window rate limiter.
    """
    # Rate Limiting Guard
    client_ip = request.client.host if request.client else "unknown_client"
    allowed, current_count, retry_after = await check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({current_count} requests/min). Please retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )
    config = {
        "configurable": {"thread_id": payload.thread_id},
        "metadata": {"thread_id": payload.thread_id},
        "run_name": "chat_stream_turn"
    }

    async def event_generator():
        try:
            active_tools = set()
            async for message_chunk, metadata in agent.astream(
                {"messages": [HumanMessage(content=payload.message)]},
                config=config,
                stream_mode="messages"
            ):
                # Process AIMessage chunks
                if isinstance(message_chunk, AIMessage):
                    # 1. Stream any text content
                    if message_chunk.content:
                        text_token = message_chunk.content
                        if isinstance(text_token, list):
                            text_token = "".join(
                                block.get("text", "") if isinstance(block, dict) else str(block)
                                for block in text_token
                            )
                        if text_token:
                            yield {
                                "event": "token",
                                "data": json.dumps({"token": text_token})
                            }

                    # 2. Emit tool start signal when tool call is detected
                    if getattr(message_chunk, "tool_calls", None):
                        for tc in message_chunk.tool_calls:
                            tname = tc.get("name", "tool")
                            if tname not in active_tools:
                                active_tools.add(tname)
                                yield {
                                    "event": "tool_start",
                                    "data": json.dumps({"tool_name": tname})
                                }

                # Detect tool execution completion
                elif isinstance(message_chunk, ToolMessage):
                    tname = getattr(message_chunk, "name", "tool")
                    yield {
                        "event": "tool_end",
                        "data": json.dumps({"tool_name": tname})
                    }

            # Final completion signal
            yield {
                "event": "done",
                "data": json.dumps({"thread_id": payload.thread_id})
            }

        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc), "status_code": 502})
            }

    return EventSourceResponse(event_generator())
