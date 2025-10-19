from fastapi import APIRouter, UploadFile, HTTPException, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import os

chatbot = None
def set_dependencies(chatbot_instance):
    """Set the chatbot instance."""
    global chatbot
    chatbot = chatbot_instance

router = APIRouter()

@router.post("/documents/upload")
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
        allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )
        
        # Save file to temporary location
        # tempfile.NamedTemporaryFile creates a unique temp file
        # delete=False means we manage deletion ourselves
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Read uploaded file content
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        print(f"Uploaded file saved to: {temp_path}")

        # Process document and add to vector store
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
            # Clean up temporary file
            # Always delete temp file, even if processing fails
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Temporary file deleted: {temp_path}")
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        # Catch all other errors and return 500
        print(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@router.get("/documents/list/{thread_id}")
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


@router.delete("/documents/clear/{thread_id}")
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