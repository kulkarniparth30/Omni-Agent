from pydantic import BaseModel, Field

class DocumentUploadResponse(BaseModel):
    """Response returned upon successfully indexing a PDF."""
    thread_id: str = Field(..., description="Thread associated with this document")
    filename: str = Field(..., description="Original name of the uploaded PDF")
    documents: int = Field(..., description="Number of extracted pages")
    chunks: int = Field(..., description="Total text chunks indexed into FAISS")
    status: str = Field(default="indexed", description="Status of the ingestion")
