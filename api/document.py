from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Any, List
from services.container import Services
from services.vector_store_service import VectorStoreService
from pathlib import Path
import tempfile
import os

router = APIRouter(prefix="/api/documents")
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

class DocumentMetadata(BaseModel):
    filename: str

class UploadDocumentResponse(BaseModel):
    message: str
    document: Any  # Or DocumentMetadata if you know the structure
    thread_id: str

class ListDocumentsResponse(BaseModel):
    thread_id: str
    documents: List[Any]  # Or List[DocumentMetadata]
    count: int

class ClearDocumentsResponse(BaseModel):
    thread_id: str
    message: str
    documents_removed: int

def _get_doc_service(request: Request) -> VectorStoreService:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=500, detail="Services not initialized")
    return services.documents

def _require_thread_id(thread_id: str) -> str:
    if thread_id == "":
        raise HTTPException(status_code=400, detail="thread_id is required")
    return thread_id

@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    """
    Upload a document and add it to the thread's RAG system.
    
    Args:
        file: Uploaded file from client (multipart/form-data)
        thread_id: Thread to associate document with
        
    Returns:
        JSON with document metadata (filename, chunks, size, etc.)
    """
    _require_thread_id(thread_id)
    doc_service = _get_doc_service(request)

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        doc_metadata = doc_service.add_document_to_vector_store(
            thread_id=thread_id,
            file_path=temp_path,
            filename=file.filename,
        )
        return UploadDocumentResponse(
            thread_id=thread_id,
            message="Document uploaded successfully",
            document=doc_metadata,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/list/{thread_id}", response_model=ListDocumentsResponse)
async def list_documents(thread_id: str, request: Request):
    """
    Get list of all documents uploaded for a thread.
    
    This is useful for:
    - Displaying uploaded files in the UI
    - Showing users what documents are available for RAG
    - Tracking document metadata (size, chunks, upload time)
    
    Args:
        thread_id: Thread to get documents for
        
    Returns:
        JSON with list of document metadata
        
    Learning Note:
        - This is a GET request (idempotent, cacheable)
        - Thread_id comes from URL path parameter
        - Returns empty list if no documents uploaded
    """
    doc_service = _get_doc_service(request)
    documents = doc_service.get_thread_documents(thread_id)
    return ListDocumentsResponse(
        thread_id=thread_id,
        documents=documents,
        count=len(documents),
    )

@router.delete("/clear/{thread_id}", response_model=ClearDocumentsResponse)
async def clear_documents(thread_id: str, request: Request):
    """
    Clear all documents for a thread.
    
    This endpoint:
    1. Removes all document chunks from vector store
    2. Deletes document metadata
    3. Frees up memory
    
    Use cases:
    - User clicks "Clear documents" button
    - User closes chatbot (automatic cleanup)
    - Session timeout (background cleanup)
    
    Args:
        thread_id: Thread to clear documents for
        
    Returns:
        JSON with confirmation message
        
    Learning Note:
        - DELETE is the correct HTTP method for resource removal
        - This is crucial for preventing memory leaks
        - In production, consider async background cleanup
    """
    doc_service = _get_doc_service(request)
    docs_before = len(doc_service.get_thread_documents(thread_id))
    doc_service.clear_thread_documents(thread_id)
    return ClearDocumentsResponse(
        thread_id=thread_id,
        message="Documents cleared successfully",
        documents_removed=docs_before,
    )