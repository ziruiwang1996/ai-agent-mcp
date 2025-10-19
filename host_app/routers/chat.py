import json
import asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage

# Module-level variables that will be injected
chatbot = None
thread_configs = None
def set_dependencies(chatbot_instance, thread_configs_instance):
    """Set the chatbot and thread_configs instances."""
    global chatbot, thread_configs
    chatbot = chatbot_instance
    thread_configs = thread_configs_instance

router = APIRouter()

class ChatRequest(BaseModel):
    query: Optional[str] = None  
    message: Optional[str] = None
    thread_id: Optional[str] = None
    
    @property
    def user_message(self) -> str:
        """Get the user message from either query or message field."""
        return self.message or self.query or ""

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """Process chat queries with thread management."""
    try:
        # Get or create thread configuration
        thread_id = chat_request.thread_id
        print(f"Received chat request - thread_id: {thread_id}")
        
        if thread_id and thread_id in thread_configs:
            config = thread_configs.get(thread_id)
            print(f"Using existing thread config for: {thread_id}")
        else:
            config = chatbot.new_thread_config()
            thread_id = config["configurable"]["thread_id"]
            thread_configs.set(thread_id, config)
            print(f"Created new thread: {thread_id}")
            print(f"Cache stats: {thread_configs.get_stats()}")
        
        # Process the query
        user_message = chat_request.user_message
        print(f"Query: '{user_message[:100]}...'")
        print(f"Config being passed: {config}")
        print(f"Config type: {type(config)}")
        
        # Add recursion limit to prevent infinite tool-calling loops
        config["recursion_limit"] = 25
        
        output = await asyncio.wait_for(
                chatbot.app.ainvoke(
                    {"messages": [HumanMessage(content=user_message)]}, 
                    config
                ), 
                timeout=60.0
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

@router.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """Stream chat responses for real-time interaction."""
    async def generate_response():
        try:
            # Get or create thread configuration
            thread_id = chat_request.thread_id
            print(f"Received stream request - thread_id: {thread_id}")
            
            if thread_id and thread_id in thread_configs:
                config = thread_configs.get(thread_id)
                print(f"Using existing thread config for: {thread_id}")
            else:
                config = chatbot.new_thread_config()
                thread_id = config["configurable"]["thread_id"]
                thread_configs.set(thread_id, config)
                print(f"Created new thread: {thread_id}")
                print(f"Cache stats: {thread_configs.get_stats()}")
            
            # Send thread_id first
            yield f"data: {json.dumps({'thread_id': thread_id, 'type': 'thread_id'})}\n\n"
            
            # Process the query
            user_message = chat_request.user_message
            print(f"Streaming query: '{user_message[:100]}...'")
            print(f"Config being passed: {config}")
            print(f"Config type: {type(config)}")
            
            # Add recursion limit to prevent infinite tool-calling loops
            config["recursion_limit"] = 25
            
            output = await asyncio.wait_for(
                chatbot.app.ainvoke(
                    {"messages": [HumanMessage(content=user_message)]}, 
                    config
                ), 
                timeout=60.0
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

@router.post("/chat/reset")
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
        thread_configs.remove(thread_id)
        # Clear uploaded documents for this thread
        chatbot.clear_thread_documents(thread_id)
    
    # Create new thread
    config = chatbot.new_thread_config()
    new_thread_id = config["configurable"]["thread_id"]
    thread_configs.set(new_thread_id, config)
    
    return {
        "thread_id": new_thread_id, 
        "message": "Chat thread reset",
        "documents_cleared": thread_id is not None,
        "cache_stats": thread_configs.get_stats()
    }