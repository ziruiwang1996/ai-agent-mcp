from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import asyncio

# Import your new LangChain chatbot
from client.langchain_gemini_client import GeminiMCPChatbot

# Create chatbot instance
chatbot = GeminiMCPChatbot()

# Request models
class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
# Global dictionary to store thread configurations
thread_configs: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the chatbot when the server starts."""
    await chatbot.initialize()
    yield
    """Clean up when the server shuts down."""
    # Add any cleanup if needed
    pass

app = FastAPI(lifespan=lifespan)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Streamlit app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.post("/connect")
# async def connect(api_key: str):
#     try:
#         await chatbot.connect(api_key)
#         return 200
#     except Exception as e:
#         print(f"Error connecting to Gemini API: {str(e)}")
#         return 400

@app.get("/")
async def root():
    return {"message": "Hello, this is the Life Science Research Agent server with MCP tools."}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "tools_available": len(chatbot.tools),
        "model": chatbot.model_name
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """Process chat queries with thread management."""
    try:
        # Get or create thread configuration
        thread_id = chat_request.thread_id
        if thread_id and thread_id in thread_configs:
            config = thread_configs[thread_id]
        else:
            config = chatbot.new_thread_config()
            thread_id = config["configurable"]["thread_id"]
            thread_configs[thread_id] = config
        
        # Process the query
        from langchain_core.messages import HumanMessage
        output = await chatbot.app.ainvoke(
            {"messages": [HumanMessage(content=chat_request.query)]}, 
            config
        )
        
        # Extract response
        last_message = output.get("messages", [])[-1] if output.get("messages") else None
        if last_message:
            response_text = getattr(last_message, "content", "No response generated")
        else:
            response_text = "No response generated"
        
        return ChatResponse(response=response_text, thread_id=thread_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """Stream chat responses for real-time interaction."""
    async def generate_response():
        try:
            # Get or create thread configuration
            thread_id = chat_request.thread_id
            if thread_id and thread_id in thread_configs:
                config = thread_configs[thread_id]
            else:
                config = chatbot.new_thread_config()
                thread_id = config["configurable"]["thread_id"]
                thread_configs[thread_id] = config
            
            # Send thread_id first
            yield f"data: {json.dumps({'thread_id': thread_id, 'type': 'thread_id'})}\n\n"
            
            # Process the query
            from langchain_core.messages import HumanMessage
            output = await chatbot.app.ainvoke(
                {"messages": [HumanMessage(content=chat_request.query)]}, 
                config
            )
            
            # Extract and stream response
            last_message = output.get("messages", [])[-1] if output.get("messages") else None
            if last_message:
                response_text = getattr(last_message, "content", "No response generated")
                
                # Stream the response word by word
                words = response_text.split()
                for word in words:
                    yield f"data: {json.dumps({'content': word + ' ', 'type': 'content'})}\n\n"
                    await asyncio.sleep(0.05)  # Small delay for streaming effect
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            else:
                yield f"data: {json.dumps({'content': 'No response generated', 'type': 'error'})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'content': f'Error: {str(e)}', 'type': 'error'})}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/plain")

@app.post("/chat/reset")
async def reset_chat(request: Request):
    """Reset a chat thread or create a new one."""
    data = await request.json()
    thread_id = data.get("thread_id")
    
    if thread_id and thread_id in thread_configs:
        del thread_configs[thread_id]
    
    # Create new thread
    config = chatbot.new_thread_config()
    new_thread_id = config["configurable"]["thread_id"]
    thread_configs[new_thread_id] = config
    
    return {"thread_id": new_thread_id, "message": "Chat thread reset"}

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