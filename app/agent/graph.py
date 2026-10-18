from typing import Sequence, Optional, Any
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from app.agent.state import ChatState
from app.agent.llm import get_chat_llm
from app.prompts.system import build_system_prompt
from app.services.pdf_store import has_document, get_document_metadata

def build_graph(
    tools: Sequence[BaseTool],
    checkpointer: Optional[Any] = None,
    llm: Optional[Any] = None
):
    """
    Pure factory function to assemble and compile the LangGraph conversation state machine.
    
    Contains NO FastAPI dependencies, NO top-level execution side effects, and allows
    injecting mock tools, LLMs, or checkpointers for evals and testing.
    """
    if llm is None:
        llm = get_chat_llm()

    # Bind tools to the model if any are provided
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    async def chat_node(state: ChatState, config: Optional[RunnableConfig] = None):
        """Node that prepares context, dynamically injects system prompts, and calls the LLM."""
        thread_id = None
        if config and isinstance(config, dict):
            thread_id = config.get("configurable", {}).get("thread_id")

        # Determine if a PDF is indexed for this thread to tailor system guidance
        doc_exists = has_document(thread_id)
        doc_meta = get_document_metadata(thread_id) if doc_exists else None
        sys_msg = build_system_prompt(has_document=doc_exists, doc_metadata=doc_meta)

        # Exclude previous system messages so they don't pile up
        history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        prompt_messages = [sys_msg, *history]

        # Call the LLM. Exceptions are deliberately NOT swallowed into a fake message here;
        # they propagate to the caller or API layer to return proper HTTP error codes.
        response = await llm_with_tools.ainvoke(prompt_messages, config=config)
        return {"messages": [response]}

    # Construct the state machine
    workflow = StateGraph(ChatState)
    workflow.add_node("chat_node", chat_node)
    workflow.add_edge(START, "chat_node")

    if tools:
        tool_node = ToolNode(tools, handle_tool_errors=True)
        workflow.add_node("tools", tool_node)
        workflow.add_conditional_edges("chat_node", tools_condition)
        workflow.add_edge("tools", "chat_node")
    else:
        workflow.add_edge("chat_node", END)

    return workflow.compile(checkpointer=checkpointer)
