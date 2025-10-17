from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import asyncio
import os
import tempfile
from pathlib import Path
from langchain_core.messages import HumanMessage

# Import your new LangChain chatbot
from client.langchain_client import GeminiMCPChatbot

# Create chatbot instance - use production config if in Docker environment
import os
config_file = "server_config_production.json" if os.path.exists("/.dockerenv") else "server_config.json"
chatbot = GeminiMCPChatbot(config_file=config_file, timeout=90.0)

# Request models
class ChatRequest(BaseModel):
    query: Optional[str] = None  # Support both 'query' and 'message'
    message: Optional[str] = None
    thread_id: Optional[str] = None
    
    @property
    def user_message(self) -> str:
        """Get the user message from either query or message field."""
        return self.message or self.query or ""

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
        print(f"📨 Received chat request - thread_id: {thread_id}")
        
        if thread_id and thread_id in thread_configs:
            config = thread_configs[thread_id]
            print(f"✓ Using existing thread config for: {thread_id}")
        else:
            config = chatbot.new_thread_config()
            thread_id = config["configurable"]["thread_id"]
            thread_configs[thread_id] = config
            print(f"✓ Created new thread: {thread_id}")
        
        # Check if thread has documents
        if thread_id in chatbot.thread_vector_stores:
            num_docs = len(chatbot.get_thread_documents(thread_id))
            print(f"📚 Thread has {num_docs} document(s) uploaded")
        else:
            print(f"ℹ️  Thread has no documents uploaded yet")
        
        # Process the query
        user_message = chat_request.user_message
        print(f"💬 Query: '{user_message[:100]}...'")
        print(f"🔧 Config being passed: {config}")
        print(f"🔧 Config type: {type(config)}")
        
        output = await chatbot.app.ainvoke(
            {"messages": [HumanMessage(content=user_message)]}, 
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
            print(f"📨 Received stream request - thread_id: {thread_id}")
            
            if thread_id and thread_id in thread_configs:
                config = thread_configs[thread_id]
                print(f"✓ Using existing thread config for: {thread_id}")
            else:
                config = chatbot.new_thread_config()
                thread_id = config["configurable"]["thread_id"]
                thread_configs[thread_id] = config
                print(f"✓ Created new thread: {thread_id}")
            
            # Send thread_id first
            yield f"data: {json.dumps({'thread_id': thread_id, 'type': 'thread_id'})}\n\n"
            
            # Process the query

            user_message = chat_request.user_message
            print(f"💬 Streaming query: '{user_message[:100]}...'")
            print(f"🔧 Config being passed: {config}")
            print(f"🔧 Config type: {type(config)}")
            
            output = await chatbot.app.ainvoke(
                {"messages": [HumanMessage(content=user_message)]}, 
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
    """
    Reset a chat thread or create a new one.
    
    This endpoint:
    1. Deletes the old thread configuration
    2. Clears all uploaded documents for that thread (RAG cleanup)
    3. Creates a new thread with fresh state
    
    Learning Note:
        - This ensures each new conversation starts fresh
        - Document cleanup prevents memory leaks
        - User's previous documents are completely removed
    """
    data = await request.json()
    thread_id = data.get("thread_id")
    
    # Clean up old thread
    if thread_id and thread_id in thread_configs:
        del thread_configs[thread_id]
        
        # ═══════════════════════════════════════════════════════════════
        # IMPORTANT: Clear uploaded documents for this thread
        # ═══════════════════════════════════════════════════════════════
        chatbot.clear_thread_documents(thread_id)
    
    # Create new thread
    config = chatbot.new_thread_config()
    new_thread_id = config["configurable"]["thread_id"]
    thread_configs[new_thread_id] = config
    
    return {
        "thread_id": new_thread_id, 
        "message": "Chat thread reset",
        "documents_cleared": thread_id is not None
    }

# ═══════════════════════════════════════════════════════════════
# DOCUMENT MANAGEMENT ENDPOINTS (RAG)
# ═══════════════════════════════════════════════════════════════

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    """
    Upload a document and add it to the thread's RAG system.
    
    Process:
    1. Validate file type (PDF, TXT, etc.)
    2. Save uploaded file temporarily
    3. Process file and add to vector store
    4. Return document metadata
    5. Clean up temporary file
    
    Args:
        file: Uploaded file from client (multipart/form-data)
        thread_id: Thread to associate document with
        
    Returns:
        JSON with document metadata (filename, chunks, size, etc.)
        
    Learning Notes:
        - UploadFile is FastAPI's wrapper for multipart uploads
        - We use temporary files to avoid disk clutter
        - File validation prevents malicious uploads
        - Thread_id is required to isolate documents per user session
    """
    try:
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Validate file type
        # ═══════════════════════════════════════════════════════════════
        allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Save file to temporary location
        # ═══════════════════════════════════════════════════════════════
        # tempfile.NamedTemporaryFile creates a unique temp file
        # delete=False means we manage deletion ourselves
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Read uploaded file content
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        print(f"📄 Uploaded file saved to: {temp_path}")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Process document and add to vector store
        # ═══════════════════════════════════════════════════════════════
        try:
            # This does the heavy lifting:
            # - Loads PDF/TXT
            # - Splits into chunks
            # - Creates embeddings
            # - Stores in thread's vector store
            doc_metadata = chatbot.add_document_to_thread(
                thread_id=thread_id,
                file_path=temp_path,
                filename=file.filename
            )
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Document uploaded successfully",
                    "document": doc_metadata,
                    "thread_id": thread_id
                }
            )
        
        finally:
            # ═══════════════════════════════════════════════════════════════
            # STEP 4: Clean up temporary file
            # ═══════════════════════════════════════════════════════════════
            # Always delete temp file, even if processing fails
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"🗑️  Temporary file deleted: {temp_path}")
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        # Catch all other errors and return 500
        print(f"❌ Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@app.get("/documents/list/{thread_id}")
async def list_documents(thread_id: str):
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
    try:
        documents = chatbot.get_thread_documents(thread_id)
        
        return {
            "thread_id": thread_id,
            "documents": documents,
            "count": len(documents)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.delete("/documents/clear/{thread_id}")
async def clear_documents(thread_id: str):
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
    try:
        # Get count before deletion for response
        docs_before = len(chatbot.get_thread_documents(thread_id))
        
        # Clear all documents for this thread
        chatbot.clear_thread_documents(thread_id)
        
        return {
            "thread_id": thread_id,
            "message": "Documents cleared successfully",
            "documents_removed": docs_before
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

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