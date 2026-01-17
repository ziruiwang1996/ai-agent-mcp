import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.container import Services

router = APIRouter(prefix="/api/chat")

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

    thread_id = _require_thread_id(init_request.thread_id or "")

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

    _ensure_thread_config(services, thread_id)
    services.documents.clear_thread_documents(thread_id)

    # Clear thread conversation history (LangGraph checkpointer).
    services.chat.clear_chat_history(thread_id)

    return ResetResponse(
        thread_id=thread_id,
        message="Chat thread reset",
        documents_cleared=True,
        cache_stats=services.thread_configs.get_stats(),
    )

