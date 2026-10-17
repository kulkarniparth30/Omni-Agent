from typing import Dict, Any, Optional
import tempfile
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from app.config import get_settings

# Thread-level in-memory cache of active retrievers and document metadata
_THREAD_RETRIEVERS: Dict[str, VectorStoreRetriever] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def get_retriever(thread_id: Optional[str]) -> Optional[VectorStoreRetriever]:
    """Fetch the active retriever for a thread if one exists."""
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None

def get_document_metadata(thread_id: Optional[str]) -> dict:
    """Fetch stored document metadata (filename, page count, chunks) for a thread."""
    if thread_id and str(thread_id) in _THREAD_METADATA:
        return _THREAD_METADATA[str(thread_id)]
    return {}

def has_document(thread_id: Optional[str]) -> bool:
    """Check whether a thread has an indexed document."""
    return get_retriever(thread_id) is not None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None, embeddings = None) -> dict:
    """
    Ingest a PDF from bytes, split into text chunks, index with FAISS,
    and register the retriever for the given thread_id.
    """
    if not file_bytes:
        raise ValueError("No bytes received for document ingestion.")

    if embeddings is None:
        from app.agent.llm import get_embeddings
        embeddings = get_embeddings()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        t_key = str(thread_id)
        _THREAD_RETRIEVERS[t_key] = retriever
        _THREAD_METADATA[t_key] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[t_key]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
