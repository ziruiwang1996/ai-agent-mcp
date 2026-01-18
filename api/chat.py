import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.container import Services

router = APIRouter(prefix="/api/chat")

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    thread_id: str
    response: str

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

def _get_services(request: Request) -> Services:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=500, detail="Services not initialized")
    return services

def _require_thread_id(thread_id: str) -> str:
    if thread_id == "":
        raise HTTPException(status_code=400, detail="thread_id is required")
    return thread_id

def _update_or_set_thread_config(services: Services, thread_id: str) -> dict:
    thread_configs = services.thread_configs
    if thread_id in thread_configs:
        return thread_configs.get(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    thread_configs.set(thread_id, config)
    return config

@router.post("/initialize", response_model=InitializeResponse)
async def initialize_chat(init_request: InitializeRequest, request: Request):
    services = _get_services(request)
    thread_id = _require_thread_id(init_request.thread_id or "")

    if not services.chat.is_initialized():
        await services.chat.initialize()
    _update_or_set_thread_config(services, thread_id)

    return InitializeResponse(
        thread_id=thread_id,
        status="ok",
        chat_initialized=services.chat.is_initialized(),
    )

@router.post("/batch", response_model=ChatResponse)
async def chat_batch(chat_request: ChatRequest, request: Request):
    services = _get_services(request)
    chat_service = services.chat
    if not chat_service.is_initialized():
        raise HTTPException(
            status_code=409,
            detail="Chat is not initialized. Call POST /api/chat/initialize first.",
        )

    thread_id = _require_thread_id(chat_request.thread_id)
    config = _update_or_set_thread_config(services, thread_id)
    config["recursion_limit"] = 25 # never used?

    output = await chat_service.chat(user_input=chat_request.message, config=config)
    return ChatResponse(response=output, thread_id=thread_id)

@router.post("/stream")
async def chat_stream(chat_request: ChatRequest, request: Request):
    """Stream chat responses for real-time interaction."""
    services = _get_services(request)
    chat_service = services.chat
    if not chat_service.is_initialized():
        raise HTTPException(
            status_code=409,
            detail="Chat is not initialized. Call POST /api/chat/initialize first.",
        )

    thread_id = _require_thread_id(chat_request.thread_id)
    config = _update_or_set_thread_config(services, thread_id)
    config["recursion_limit"] = 25

    async def event_stream():
        try:
            async for chunk in chat_service.astream_chat(chat_request.message, config):
                if not chunk:
                    continue
                payload = json.dumps({"token": chunk})
                yield f"data: {payload}\n\n"
            yield "event: done\ndata: {\"done\": true}\n\n"
        except Exception as e:
            payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {payload}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

@router.post("/reset", response_model=ResetResponse)
async def reset_chat(reset_request: ResetRequest, request: Request):
    """
    Reset a chat history and clear uploaded documents (RAG cleanup)
    Keep the same thread id for continuity.
    """
    services = _get_services(request)
    thread_id = _require_thread_id(reset_request.thread_id)

    _update_or_set_thread_config(services, thread_id)
    services.documents.clear_thread_documents(thread_id)
    services.chat.clear_chat_history(thread_id)

    return ResetResponse(
        thread_id=thread_id,
        message="Chat thread reset",
        documents_cleared=True,
        cache_stats=services.thread_configs.get_stats(),
    )
