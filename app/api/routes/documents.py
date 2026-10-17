from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.documents import DocumentUploadResponse
from app.services.pdf_store import ingest_pdf
from app.core.exceptions import DocumentProcessingError

router = APIRouter(prefix="/threads", tags=["Documents"])

@router.post("/{thread_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(
    thread_id: str,
    file: UploadFile = File(..., description="PDF file to index for this conversation thread")
):
    """
    Upload and index a PDF document for a specific conversation thread.
    Splits text into chunks, indexes them into FAISS, and binds the retriever to thread_id.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        summary = ingest_pdf(
            file_bytes=content,
            thread_id=thread_id,
            filename=file.filename
        )

        return DocumentUploadResponse(
            thread_id=thread_id,
            filename=summary["filename"],
            documents=summary["documents"],
            chunks=summary["chunks"],
            status="indexed"
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise DocumentProcessingError(message=str(exc))
