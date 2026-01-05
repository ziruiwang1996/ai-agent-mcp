import json
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from services.container import Services
from pathlib import Path
import tempfile
import os
import logging

router = APIRouter(prefix="/api/chat")

# Use Uvicorn's logger so INFO logs reliably appear in the uvicorn console.
logger = logging.getLogger("uvicorn.error")

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

def _get_services(request: Request) -> Services:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=500, detail="Services not initialized")
    return services

def _require_thread_id(thread_id: str) -> str:
    if thread_id == "":
        raise HTTPException(status_code=400, detail="thread_id is required")
    return thread_id

def _ensure_thread_config(services: Services, thread_id: str) -> dict:
    thread_configs = services.thread_configs
    if thread_id in thread_configs:
        return thread_configs.get(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    thread_configs.set(thread_id, config)
    return config

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class InitializeRequest(BaseModel):
    thread_id: str

class InitializeResponse(BaseModel):
    thread_id: str
    status: str
    chat_initialized: bool

class ResetRequest(BaseModel):
    thread_id: str

class ResetResponse(BaseModel):
    thread_id: str
    message: str
    documents_cleared: bool
    cache_stats: dict   

@router.post("/initialize", response_model=InitializeResponse)
async def initialize_chat(init_request: InitializeRequest, request: Request):
    services = _get_services(request)

    thread_id = init_request.thread_id or ""
    if thread_id == "":
        thread_id = str(uuid.uuid4())

    logger.info("POST /api/chat/initialize thread_id=%s", thread_id)

    await services.chat.initialize()
    _ensure_thread_config(services, thread_id)

    return InitializeResponse(
        thread_id=thread_id,
        status="ok",
        chat_initialized=services.chat.chat_agent is not None,
    )

@router.post("/batch", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):
    services = _get_services(request)
    chat_service = services.chat
    if hasattr(chat_service, "is_initialized") and not chat_service.is_initialized():
        raise HTTPException(
            status_code=409,
            detail="Chat is not initialized. Call POST /api/chat/initialize first.",
        )

    thread_id = _require_thread_id(chat_request.thread_id)
    logger.info(
        "POST /api/chat/batch thread_id=%s message_len=%s",
        thread_id,
        len(chat_request.message) if chat_request.message is not None else None,
    )
    config = _ensure_thread_config(services, thread_id)
    config["recursion_limit"] = 25

    output = await chat_service.chat(user_input=chat_request.message, config=config)
    return ChatResponse(response=output, thread_id=thread_id)

@router.post("/stream")
async def chat_stream(chat_request: ChatRequest, request: Request):
    """Stream chat responses for real-time interaction."""
    async def generate_response():
        try:
            services = _get_services(request)
            chat_service = services.chat
            if hasattr(chat_service, "is_initialized") and not chat_service.is_initialized():
                yield f"data: {json.dumps({'content': 'Chat is not initialized. Call POST /chat/initialize first.', 'type': 'error'})}\n\n"
                return

            thread_id = _require_thread_id(chat_request.thread_id)
            logger.info(
                "POST /api/chat/stream thread_id=%s message_len=%s",
                thread_id,
                len(chat_request.message) if chat_request.message is not None else None,
            )
            config = _ensure_thread_config(services, thread_id)
            
            # Send thread_id first
            yield f"data: {json.dumps({'thread_id': thread_id, 'type': 'thread_id'})}\n\n"
            
            # Add recursion limit to prevent infinite tool-calling loops
            config["recursion_limit"] = 25

            async for chunk in chat_service.astream_chat(user_input=chat_request.message, config=config):
                if chunk:
                    yield f"data: {json.dumps({'content': chunk, 'type': 'content'})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'content': f'Error: {str(e)}', 'type': 'error'})}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/event-stream")

@router.post("/reset")
async def reset_chat(reset_request: ResetRequest, request: Request):
    """
    Reset a chat history and clear uploaded documents (RAG cleanup)
    Keep the same thread id for continuity.
    """
    services = _get_services(request)
    thread_id = _require_thread_id(reset_request.thread_id)

    logger.info("POST /api/chat/reset thread_id=%s", thread_id)


    _ensure_thread_config(services, thread_id)
    services.chat.clear_thread_documents(thread_id)

    # Clear thread conversation history (LangGraph checkpointer).
    services.chat.clear_thread_history(thread_id)

    return ResetResponse(
        thread_id=thread_id,
        message="Chat thread reset",
        documents_cleared=True,
        cache_stats=services.thread_configs.get_stats(),
    )

@router.post("/documents/upload")
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
    logger.info(
        "POST /api/chat/documents/upload thread_id=%s filename=%s",
        thread_id,
        getattr(file, "filename", None),
    )
    services = _get_services(request)
    chat_service = services.chat
    if hasattr(chat_service, "is_initialized") and not chat_service.is_initialized():
        raise HTTPException(status_code=409, detail="Chat is not initialized. Call POST /api/chat/initialize first.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        doc_metadata = chat_service.add_document_to_thread(
            thread_id=thread_id,
            file_path=temp_path,
            filename=file.filename,
        )
        return {
            "message": "Document uploaded successfully",
            "document": doc_metadata,
            "thread_id": thread_id,
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/documents/list/{thread_id}")
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
    services = _get_services(request)
    documents = services.chat.get_thread_documents(thread_id)
    return {"thread_id": thread_id, "documents": documents, "count": len(documents)}


@router.delete("/documents/clear/{thread_id}")
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
    services = _get_services(request)
    docs_before = len(services.chat.get_thread_documents(thread_id))
    services.chat.clear_thread_documents(thread_id)
    return {
        "thread_id": thread_id,
        "message": "Documents cleared successfully",
        "documents_removed": docs_before,
    }