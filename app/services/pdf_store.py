import os
import json
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import get_settings

logger = logging.getLogger("app.services.pdf_store")

# In-memory LRU/lookup cache for active retrievers and metadata
_THREAD_RETRIEVERS: Dict[str, VectorStoreRetriever] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_thread_index_dir(thread_id: str) -> Path:
    """Return the filesystem directory path for a thread's persisted FAISS index."""
    settings = get_settings()
    base_dir = Path(settings.INDEXES_DIR)
    return base_dir / str(thread_id)

def get_retriever(thread_id: Optional[str], embeddings=None) -> Optional[VectorStoreRetriever]:
    """
    Fetch the active retriever for a thread.
    Uses Lazy Loading: checks in-memory cache first; if not found,
    checks if a persisted FAISS index exists on disk and reloads it.
    """
    if not thread_id:
        return None

    t_key = str(thread_id)

    # 1. Check in-memory cache
    if t_key in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[t_key]

    # 2. Check disk persistence
    index_dir = _get_thread_index_dir(t_key)
    faiss_file = index_dir / "index.faiss"

    if index_dir.exists() and faiss_file.exists():
        try:
            logger.info("Lazy-loading persisted FAISS index from disk for thread '%s'...", t_key)
            if embeddings is None:
                from app.agent.llm import get_embeddings
                embeddings = get_embeddings()

            vector_store = FAISS.load_local(
                folder_path=str(index_dir),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            # Populate in-memory cache
            _THREAD_RETRIEVERS[t_key] = retriever

            # Load metadata JSON if present
            meta_file = index_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    _THREAD_METADATA[t_key] = json.load(f)

            return retriever
        except Exception as exc:
            logger.error("Failed to load persisted index for thread '%s': %s", t_key, exc, exc_info=True)
            return None

    return None

def get_document_metadata(thread_id: Optional[str]) -> dict:
    """Fetch stored document metadata (filename, page count, chunks) for a thread."""
    if not thread_id:
        return {}

    t_key = str(thread_id)
    if t_key in _THREAD_METADATA:
        return _THREAD_METADATA[t_key]

    # Attempt to read from disk if not yet in memory
    index_dir = _get_thread_index_dir(t_key)
    meta_file = index_dir / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                _THREAD_METADATA[t_key] = data
                return data
        except Exception as exc:
            logger.warning("Could not parse metadata.json for thread '%s': %s", t_key, exc)

    return {}

def has_document(thread_id: Optional[str]) -> bool:
    """Check whether a thread has an indexed document (in memory or on disk)."""
    if not thread_id:
        return False
    t_key = str(thread_id)
    if t_key in _THREAD_RETRIEVERS:
        return True
    index_dir = _get_thread_index_dir(t_key)
    return (index_dir / "index.faiss").exists()

def list_persisted_threads() -> List[str]:
    """Scan the disk indexes directory and return all thread IDs with stored indexes."""
    settings = get_settings()
    base_dir = Path(settings.INDEXES_DIR)
    if not base_dir.exists():
        return []
    threads = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item / "index.faiss").exists():
            threads.append(item.name)
    return sorted(threads)

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None, embeddings = None) -> dict:
    """
    Ingest a PDF from bytes, split into text chunks, index with FAISS,
    save the index and metadata to disk, and register the retriever for thread_id.
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
        if not chunks:
            raise ValueError("The uploaded PDF contains no extractable text.")

        # 1. Build in-memory FAISS vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        t_key = str(thread_id)

        # 2. Persist FAISS index to disk
        index_dir = _get_thread_index_dir(t_key)
        index_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_dir))

        # 3. Persist metadata.json to disk
        metadata_payload = {
            "thread_id": t_key,
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        meta_file = index_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        # 4. Cache in memory
        _THREAD_RETRIEVERS[t_key] = retriever
        _THREAD_METADATA[t_key] = metadata_payload

        logger.info(
            "Persisted FAISS index for thread '%s' to '%s' (%d chunks from %d pages).",
            t_key, index_dir, len(chunks), len(docs)
        )
        return metadata_payload

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def delete_thread_index(thread_id: str) -> bool:
    """Delete the persisted index and clear in-memory cache for a thread."""
    t_key = str(thread_id)
    _THREAD_RETRIEVERS.pop(t_key, None)
    _THREAD_METADATA.pop(t_key, None)

    index_dir = _get_thread_index_dir(t_key)
    if index_dir.exists() and index_dir.is_dir():
        shutil.rmtree(index_dir)
        logger.info("Deleted persisted index for thread '%s'.", t_key)
        return True
    return False
