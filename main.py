from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from services.container import Services, build_services
from api import chat, interpret, tools, evidence
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = build_services(max_threads=50)
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)
app.include_router(interpret.router)
app.include_router(tools.router)
app.include_router(evidence.router)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your client domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello, this is the Med Helper server."}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    services_obj: Services | None = getattr(app.state, "services", None)
    if services_obj is None:
        return {"status": "unhealthy", "detail": "services not initialized"}
    return {
        "status": "healthy",
        "chat_agent_initialized": services_obj.chat.chat_agent is not None,
        "thread_cache": services_obj.thread_configs.get_stats(),
    }

@app.get("/api/threads/stats")
async def get_thread_stats():
    services_obj: Services | None = getattr(app.state, "services", None)
    if services_obj is None:
        return {"cache_info": None, "message": "services not initialized"}
    return {"cache_info": services_obj.thread_configs.get_stats(), "message": "Thread cache statistics"}

class APIKeyRequest(BaseModel):
    key: str
    provider: str

@app.post("/api/setkey")
def set_api_key(payload: APIKeyRequest):
    pass