from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """
    Represents the state of the conversation graph.
    `messages` holds the append-only sequence of conversation messages
    (HumanMessage, AIMessage, ToolMessage, SystemMessage).
    The `add_messages` reducer handles appending new messages and merging streaming deltas.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
