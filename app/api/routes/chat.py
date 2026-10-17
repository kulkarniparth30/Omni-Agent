import time
import json
from typing import Any, List, Set
from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.schemas.chat import ChatRequest, ChatResponse
from app.api.deps import get_agent
from app.core.exceptions import LLMExecutionError

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    agent: Any = Depends(get_agent)
):
    """
    Synchronous / non-streaming chat turn.
    Invokes the agent graph, records all tools executed, and returns the final response.
    """
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
    agent: Any = Depends(get_agent)
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Streams individual tokens and tool lifecycle events (tool_start, tool_end, token, done).
    """
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
                # Detect tool call intent
                if isinstance(message_chunk, AIMessage) and getattr(message_chunk, "tool_calls", None):
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

                # Stream assistant text tokens
                elif isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield {
                        "event": "token",
                        "data": json.dumps({"token": message_chunk.content})
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
