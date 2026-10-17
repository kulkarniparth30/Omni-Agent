from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.services.pdf_store import get_retriever, get_document_metadata

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant excerpts and answers from the uploaded PDF document for this conversation.
    Use this tool whenever the user asks about the document's content, summaries, or specific details.
    """
    # Extract the thread_id securely from the execution context, NOT from the LLM arguments
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    if not thread_id:
        return {
            "error": "No active conversation thread_id found in execution context.",
            "query": query
        }

    retriever = get_retriever(str(thread_id))
    if retriever is None:
        return {
            "error": "No document has been indexed for this chat yet. Please upload a PDF first.",
            "query": query
        }

    try:
        results = retriever.invoke(query)
        context = [doc.page_content for doc in results]
        metadata = [doc.metadata for doc in results]
        doc_info = get_document_metadata(str(thread_id))

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": doc_info.get("filename", "uploaded_document.pdf")
        }
    except Exception as e:
        return {
            "error": f"Error performing document search: {str(e)}",
            "query": query
        }
