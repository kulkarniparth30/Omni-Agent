from langchain_core.messages import SystemMessage

def build_system_prompt(has_document: bool = False) -> SystemMessage:
    """
    Construct the agent's system prompt dynamically based on whether
    a document is currently indexed for the active thread.
    """
    if has_document:
        doc_instruction = (
            "A PDF document has been uploaded and indexed for this conversation. "
            "For any questions regarding the document, invoke the `rag_tool` with your search query. "
            "Base your answers on the retrieved context."
        )
    else:
        doc_instruction = (
            "No PDF document is currently indexed for this conversation. "
            "If the user asks questions about a document, explain that no PDF has been uploaded yet "
            "and invite them to upload one."
        )

    content = (
        "You are ToolBot, an intelligent and helpful AI assistant equipped with specialized tools.\n"
        f"{doc_instruction}\n"
        "You also have access to web search (DuckDuckGo), academic paper search (Semantic Scholar), "
        "real-time stock market data (Alpha Vantage), and specialized arithmetic or domain tools.\n"
        "Guidelines:\n"
        "- Choose the most appropriate tool for the user's inquiry.\n"
        "- If you can answer simple greetings or general knowledge directly, do so without unnecessary tool calls.\n"
        "- When tools return data, synthesize the information clearly and cite the sources where applicable."
    )

    return SystemMessage(content=content)
