from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from client.langchain_client import GeminiMCPChatbot
from host_app.bounded_thread_cache import *
from host_app.routers import chat, document

# Create chatbot instance - use production config if in Docker environment
config_file = "server_config_production.json" if os.path.exists("/.dockerenv") else "server_config.json"
chatbot = GeminiMCPChatbot(config_file=config_file, timeout=90.0)

# Create bounded thread cache with cleanup callback
# Adjust max_threads:
# - 50 threads ≈ minimal memory footprint (good for small servers)
# - 100 threads ≈ balanced (good for most use cases)
# - 500 threads ≈ high capacity (good for busy servers with lots of RAM)
thread_configs = BoundedThreadCache(
    max_threads=50,
    cleanup_callback=lambda thread_id: cleanup_thread_resources(thread_id, chatbot)
)

# Set up dependencies for routers
chat.set_dependencies(chatbot, thread_configs)
document.set_dependencies(chatbot)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the chatbot when the server starts."""
    await chatbot.initialize()
    yield
    """Clean up when the server shuts down."""
    # Add any cleanup if needed
    pass

app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)
app.include_router(document.router)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Streamlit app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello, this is the Life Science Research Agent server with MCP tools."}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "tools_available": len(chatbot.tools),
        "model": chatbot.model_name,
        "thread_cache": thread_configs.get_stats()
    }

@app.get("/threads/stats")
async def get_thread_stats():
    """
    Get thread cache statistics.
    
    Returns information about:
    - Current number of active threads
    - Maximum capacity
    - Cache utilization percentage
    
    Useful for monitoring and capacity planning.
    """
    return {
        "cache_info": thread_configs.get_stats(),
        "message": "Thread cache statistics"
    }

@app.get("/tools")
async def get_available_tools():
    """Get list of available MCP tools."""
    tools_info = []
    for tool in chatbot.tools:
        tool_info = {
            "name": getattr(tool, "name", "Unknown"),
            "description": getattr(tool, "description", "No description available")
        }
        tools_info.append(tool_info)
    
    return {
        "tools_count": len(chatbot.tools),
        "tools": tools_info
    }