from typing import Any
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from langchain_core.documents import Document
import os

class VectorStoreService:
    def __init__(self, embedding: str = "models/gemini-embedding-001"):
        self.thread_vector_stores: dict[str, InMemoryVectorStore] = {}
        self.thread_documents: dict[str, list[dict[str, Any]]] = {}
        self.embedding_model: Any = GoogleGenerativeAIEmbeddings(model=embedding)

    def is_thread_has_vector_store(self, thread_id: str) -> bool:
        return thread_id in self.thread_vector_stores
    
    def is_vector_store_empty(self, thread_id: str) -> bool:
        if thread_id not in self.thread_vector_stores:
            return True
        vector_store = self.thread_vector_stores[thread_id]
        if hasattr(vector_store, 'store') and len(vector_store.store) == 0:
            return True
        return False

    def get_or_create_vector_store(self, thread_id: str) -> InMemoryVectorStore:
        """
            - InMemoryVectorStore is RAM-based (fast but not persistent)
            - Each store uses the same embedding model for consistency
            - For production, consider ChromaDB or Pinecone for persistence
        """
        if thread_id not in self.thread_vector_stores:
            print(f"Creating new vector store for thread: {thread_id}")
            self.thread_vector_stores[thread_id] = InMemoryVectorStore(self.embedding_model)
        return self.thread_vector_stores[thread_id]

    def add_document_to_vector_store(self, thread_id: str, file_path: str, filename: str) -> dict[str, Any]:
        """
        Load a document and add it to the thread's vector store.
        
        Process:
        1. Load PDF/TXT/MD file (using appropriate loader)
        2. Split into chunks (for efficient retrieval)
        3. Create embeddings for each chunk
        4. Store in thread-specific vector store
        5. Track metadata for user visibility
        
        Args:
            thread_id: Unique identifier for the chat thread
            file_path: Path to the document file on disk
            filename: Original filename (for display purposes)
            
        Returns:
            Dictionary with document metadata (filename, chunks, size, etc.)
        """
        try:     
            file_extension = Path(file_path).suffix.lower()
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            docs = loader.load()
            
            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,           # Maximum characters per chunk
                chunk_overlap=200,         # Overlap to maintain context between chunks
                add_start_index=True       # Track where each chunk came from
            )
            all_splits = text_splitter.split_documents(docs)
            
            # Get or create vector store for this thread
            vector_store = self.get_or_create_vector_store(thread_id)
            
            # Add document chunks to vector store
            vector_store.add_documents(documents=all_splits)
            
            # Track document metadata
            doc_metadata = {
                "filename": filename,
                "num_chunks": len(all_splits),
                "upload_time": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "file_type": file_extension[1:].upper()
            }
            
            # Initialize documents list for this thread if needed
            if thread_id not in self.thread_documents:
                self.thread_documents[thread_id] = []
            self.thread_documents[thread_id].append(doc_metadata)
            
            print(f"Added document '{filename}' to thread {thread_id}: {len(all_splits)} chunks")
            return doc_metadata
            
        except Exception as e:
            print(f"Error adding document: {e}")
            raise
    
    def retrieve_context_for_query(
            self, 
            thread_id: str, 
            query: str, 
            k: int = 4
        ) -> list[tuple[Document, float]]:
        if thread_id not in self.thread_vector_stores:
            return []
        
        vector_store = self.thread_vector_stores[thread_id]
        if hasattr(vector_store, 'similarity_search_with_score'):
            retrieved_docs = vector_store.similarity_search_with_score(query, k=k)
            print(f"Retrieved {len(retrieved_docs)} relevant chunks (with scores) for query")
            return retrieved_docs
            
        # fall back: no scores then all 1.0
        retrieved_docs = vector_store.similarity_search(query, k=k)
        print(f"Retrieved {len(retrieved_docs)} relevant chunks for query")
        return [(d, 1.0) for d in retrieved_docs]
    
    def clear_thread_documents(self, thread_id: str) -> None:
        """
        This is called when:
        - User resets the chat
        - User closes the chatbot session
        - Session timeout occurs
        """
        if thread_id in self.thread_vector_stores:
            del self.thread_vector_stores[thread_id]
            print(f"Cleared vector store for thread: {thread_id}")
        
        if thread_id in self.thread_documents:
            del self.thread_documents[thread_id]
            print(f"Cleared document metadata for thread: {thread_id}")
    
    def get_thread_documents(self, thread_id: str) -> list[dict[str, Any]]:
        return self.thread_documents.get(thread_id, [])