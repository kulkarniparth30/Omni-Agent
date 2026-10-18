from typing import Optional
from langchain_core.messages import SystemMessage

def build_system_prompt(has_document: bool = False, doc_metadata: Optional[dict] = None) -> SystemMessage:
    """
    Construct the agent's system prompt dynamically based on whether
    a document is currently indexed for the active thread.
    """
    if has_document:
        doc_name = (doc_metadata or {}).get("filename", "uploaded_document.pdf")
        chunks = (doc_metadata or {}).get("chunks", "several")
        doc_instruction = (
            f"- An uploaded document named '{doc_name}' is currently indexed for this conversation ({chunks} text chunks).\n"
            f"- Whenever the user asks about this document, their resume, their CV, background, qualifications, work experience, skills, or any content related to '{doc_name}', you MUST IMMEDIATELY invoke the `rag_tool` with an appropriate search query.\n"
            f"- NEVER claim you don't have access to their resume or uploaded document. '{doc_name}' is already indexed in FAISS and accessible via `rag_tool`."
        )
    else:
        doc_instruction = (
            "- No custom PDF document is currently uploaded for this conversation. "
            "Only prompt the user to upload a PDF if they explicitly ask about 'my uploaded document' or 'the uploaded file'. "
            "For general academic topics, concepts, or well-known papers (e.g., 'Attention Is All You Need'), answer directly using your knowledge or use `semanticscholar` or `duckduckgo_search`."
        )

    content = (
        "You are Omni-Agent, an autonomous, highly capable AI assistant equipped with real-time tools.\n\n"
        "### Core Operating Guidelines:\n"
        "1. **Document & Resume Context (CRITICAL)**:\n"
        f"   {doc_instruction}\n"
        "2. **Proactive Tool Calling (NEVER ASK FOR PERMISSION)**:\n"
        "   - When the user asks about current events, recent news, real-time prices, live facts, or external information, **IMMEDIATELY execute the appropriate tool** (`duckduckgo_search`, `get_stock_price`, etc.).\n"
        "   - Do NOT say 'If you would like, I can search the web' or ask for confirmation. Call the tool immediately and synthesize the results.\n"
        "3. **Research Papers & Academic Queries**:\n"
        "   - If the user asks about research papers, machine learning topics, or scientific concepts, answer directly from your knowledge base or query `semanticscholar`.\n"
        "4. **General & Synthesis**:\n"
        "   - For simple greetings or standard general knowledge, answer directly without tool calls.\n"
        "   - When tools return data, synthesize the information clearly with key points and citations."
    )

    return SystemMessage(content=content)
