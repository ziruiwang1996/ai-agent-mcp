import os
from pathlib import Path
import asyncio
import uuid
from typing import Optional, List, Dict, Any, TypedDict, Annotated, Literal
import json
from client.utils import expand_env_in_text
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Set GOOGLE_API_KEY from GEMINI_API_KEY if not already set
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class GeminiMCPChatbot:
    """
    A chatbot that integrates Gemini AI with Model Context Protocol (MCP) servers
    for enhanced tool capabilities.
    
    This chatbot implements:
    1. MCP tool integration for external capabilities (arXiv, PDB, etc.)
    2. Thread-scoped RAG (Retrieval Augmented Generation) for document Q&A
    3. Streaming responses for real-time interaction
    4. Session-based vector stores that cleanup automatically
    """
    
    def __init__(self, 
                 model_name: str = "gemini-2.5-flash",
                 model_provider: str = "google_genai",
                 config_file: str = "server_config_production.json",
                 embedding: str = "models/gemini-embedding-001",
                 timeout: float = 60.0):
        """
        Initialize the chatbot with configuration parameters.
        
        Args:
            model_name: Name of the Gemini model to use (e.g., gemini-2.5-flash)
            model_provider: Provider for the model (google_genai for Gemini)
            config_file: MCP server configuration file path
            embedding: Embedding model to use for vector similarity search
            timeout: Timeout for MCP server connections in seconds
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.config_file = config_file
        self.embedding = embedding
        self.timeout = timeout
        
        # Initialize MCP and model components
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[Any] = []
        self.model = None
        self.model_with_tools = None
        self.trimmer = None
        self.app = None
        
        # ═══════════════════════════════════════════════════════════════
        # RAG (Retrieval Augmented Generation) Components
        # ═══════════════════════════════════════════════════════════════
        
        # Thread-scoped vector stores: Each chat thread gets its own vector store
        # This ensures user documents are isolated and can be cleaned up per session
        # Key: thread_id (UUID string) -> Value: InMemoryVectorStore instance
        self.thread_vector_stores: Dict[str, InMemoryVectorStore] = {}
        
        # Document metadata tracking: Store info about uploaded documents per thread
        # Key: thread_id -> Value: List of document metadata dicts
        # Each metadata dict contains: {filename, num_chunks, upload_time, file_size}
        self.thread_documents: Dict[str, List[Dict[str, Any]]] = {}
        
        # Embedding model instance for creating vector embeddings
        # This is used when creating new vector stores for each thread
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=self.embedding)
        
        # Create default prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.rag_prompt_template = PromptTemplate.from_template(
            """You are a helpful assistant that answers questions using retrieved context. Use the following pieces of 
            context to answer the question at the end. If you don’t know the answer, just say you don’t know — don’t try 
            to make up an answer.\nContext:\n{context}\n\nQuestion:\n{question}\nHelpful Answer:"""
        )
    
    async def initialize_mcp_client(self) -> None:
        """Initialize MCP client and retrieve tools."""
        print("Starting MCP initialization...")
        
        try:
            config_path = os.path.join(os.path.dirname(__file__), self.config_file)
            with open(config_path, "r") as file:
                raw = file.read()
                expanded = expand_env_in_text(raw)
                data = json.loads(expanded)
                servers = data.get("mcpServers", {})
                
                print(f"Found {len(servers)} MCP servers in config")
                
                if not servers:
                    print("No valid MCP servers found in configuration")
                    self.tools = []
                    self.client = None
                else:
                    print(f"Creating MCP client with {len(servers)} servers...")
                    
                    # Debug: Show server configurations
                    for name, config in servers.items():
                        print(f"  {name}: command={config.get('command')}, args={config.get('args')}")
                        
                        # Verify MCP server files exist
                        if config.get('args'):
                            script_path = config['args'][0]
                            if os.path.exists(script_path):
                                print(f"Script exists: {script_path}")
                            else:
                                print(f"Script missing: {script_path}")
                    
                    try:
                        self.client = MultiServerMCPClient(servers)
                        
                        print("Getting tools (this may take a moment)...")
                        # Add a timeout to prevent hanging
                        self.tools = await asyncio.wait_for(self.client.get_tools(), timeout=self.timeout)
                        print(f"Successfully retrieved {len(self.tools)} tools from MCP servers")
                    except asyncio.TimeoutError:
                        print(f"Timeout ({self.timeout}s) while getting tools from MCP servers")
                        print("This usually indicates MCP server startup issues")
                        print("Falling back to basic model without MCP tools")
                        self.tools = []
                        self.client = None
                    except Exception as e:
                        print(f"Error getting tools from MCP servers: {e}")
                        print(f"Error type: {type(e).__name__}")
                        print("Falling back to basic model without MCP tools")
                        self.tools = []
                        self.client = None
                        
        except Exception as e:
            print(f"Error initializing MCP servers: {e}")
            print("Continuing with basic model without MCP tools")
            self.tools = []
            self.client = None
    
    def initialize_model(self) -> None:
        """Initialize the Gemini model and related components."""
        print("Initializing Gemini model...")
        self.model = init_chat_model(self.model_name, model_provider=self.model_provider)
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        print("Setting up message trimmer...")
        self.trimmer = trim_messages(
            max_tokens=2000,
            strategy="last",
            token_counter=self.model_with_tools,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
    
    def create_workflow(self) -> None:
        """
        Create the LangGraph workflow with tool execution loop and RAG integration.
        
        Workflow Architecture:
        ┌─────────┐
        │  START  │
        └────┬────┘
             │
             ▼
        ┌─────────────────┐
        │  MODEL NODE     │  ◄─── RAG enhancement happens here
        │  (call_model)   │       (retrieves docs if available)
        └────┬────────────┘
             │
             ▼
        ┌─────────────────┐
        │  Has tool_calls?│
        └────┬────────┬───┘
             │        │
        YES  │        │ NO
             │        │
             ▼        ▼
        ┌────────┐  ┌────┐
        │ TOOLS  │  │END │
        └───┬────┘  └────┘
            │
            └──────► (loop back to MODEL)
        
        Notes:
        - MessagesState maintains conversation history
        - MemorySaver persists state across turns (per thread)
        - Tool execution is automatic if model requests it
        - RAG context injection is transparent to the workflow
        """
        workflow = StateGraph(state_schema=MessagesState)

        # ═══════════════════════════════════════════════════════════════
        # NODE DEFINITIONS
        # ═══════════════════════════════════════════════════════════════
        
        # Core model node - handles RAG + tool binding
        # Note: We can't pass config directly here, it comes from ainvoke()
        workflow.add_node("model", self.call_model)

        # Tools node - executes tool calls requested by the model
        # ToolNode automatically handles tool execution and error handling
        tools_node = ToolNode(self.tools) if self.tools else None
        if tools_node is not None:
            workflow.add_node("tools", tools_node)

        # ═══════════════════════════════════════════════════════════════
        # ROUTING LOGIC
        # ═══════════════════════════════════════════════════════════════
        
        def should_continue(state: MessagesState):
            """
            Decide whether to execute tools or finish.
            
            Returns:
                - "tools": If model requested tool execution
                - "end": If model generated final response
                
            Note:
                - LangChain models return tool_calls attribute when they want to use tools
                - This enables agentic behavior: model decides when to use tools
                - Without tools, model just generates text responses
            """
            messages = state.get("messages", [])
            if not messages:
                return "end"
            
            last = messages[-1]
            
            # Check if last message has tool calls
            # tool_calls = [{"name": "search_arxiv", "args": {...}}, ...]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools" if tools_node is not None else "end"
            
            return "end"

        # ═══════════════════════════════════════════════════════════════
        # GRAPH CONSTRUCTION
        # ═══════════════════════════════════════════════════════════════
        
        # Start edge: conversation begins at model node
        workflow.add_edge(START, "model")

        # If tools exist, create tool execution loop
        if tools_node is not None:
            workflow.add_conditional_edges("model", should_continue, {
                "tools": "tools",  # Execute tools if requested
                "end": END,        # Finish if no tools needed
            })
            # After tool execution, return to model with results
            workflow.add_edge("tools", "model")
        else:
            # No tools available, model -> end
            workflow.add_edge("model", END)

        # ═══════════════════════════════════════════════════════════════
        # COMPILE WORKFLOW
        # ═══════════════════════════════════════════════════════════════
        
        # MemorySaver: Persists conversation state across turns
        # Each thread_id gets its own conversation history
        self.app = workflow.compile(checkpointer=MemorySaver())
        
        print("✓ Workflow compiled with RAG + MCP tool support")

    # ═══════════════════════════════════════════════════════════════
    # DOCUMENT MANAGEMENT METHODS (RAG)
    # ═══════════════════════════════════════════════════════════════
    
    def get_or_create_vector_store(self, thread_id: str) -> InMemoryVectorStore:
        """
        Get existing vector store for a thread or create a new one.
        
        This implements lazy initialization - vector stores are only created
        when a user uploads their first document for a thread.
        
        Args:
            thread_id: Unique identifier for the chat thread
            
        Returns:
            InMemoryVectorStore instance for this thread
            
        Learning Note:
            - InMemoryVectorStore is RAM-based (fast but not persistent)
            - Each store uses the same embedding model for consistency
            - For production, consider ChromaDB or Pinecone for persistence
        """
        if thread_id not in self.thread_vector_stores:
            print(f"Creating new vector store for thread: {thread_id}")
            self.thread_vector_stores[thread_id] = InMemoryVectorStore(self.embedding_model)
        return self.thread_vector_stores[thread_id]
    
    def add_document_to_thread(self, thread_id: str, file_path: str, filename: str) -> Dict[str, Any]:
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
            
        Learning Note:
            - Chunk size (1000 chars) balances context vs. precision
            - Overlap (200 chars) prevents information loss at boundaries
            - Smaller chunks = more precise but less context
            - Larger chunks = more context but less precise matches
            - Different loaders handle different file formats
        """
        try:
            # Step 1: Load the document with appropriate loader based on file type
            # We need different loaders for different file formats            
            file_extension = Path(file_path).suffix.lower()
            
            # Choose the correct loader based on file extension
            if file_extension == '.pdf':
                # PyPDFLoader: Extracts text from PDF files page by page
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                # TextLoader: Handles plain text and markdown files
                # encoding='utf-8' ensures proper handling of special characters
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.docx':
                # Docx2txtLoader: Extracts text from Microsoft Word documents
                # Uses docx2txt library to parse .docx files
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load the document (returns list of Document objects)
            docs = loader.load()
            
            # Step 2: Split document into chunks
            # RecursiveCharacterTextSplitter tries to split on natural boundaries
            # (paragraphs, then sentences, then words) for better semantic coherence
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,           # Maximum characters per chunk
                chunk_overlap=200,         # Overlap to maintain context between chunks
                add_start_index=True       # Track where each chunk came from
            )
            all_splits = text_splitter.split_documents(docs)
            
            # Step 3: Get or create vector store for this thread
            vector_store = self.get_or_create_vector_store(thread_id)
            
            # Step 4: Add document chunks to vector store
            # This creates embeddings (vector representations) for each chunk
            # and stores them for similarity search
            vector_store.add_documents(documents=all_splits)
            
            # Step 5: Track document metadata
            from datetime import datetime
            
            # Different metadata for different file types
            # PDFs have pages, text files don't
            doc_metadata = {
                "filename": filename,
                "num_chunks": len(all_splits),
                "upload_time": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "file_type": file_extension[1:].upper()  # e.g., "PDF", "TXT", "MD"
            }
            
            # Add page count only for PDFs (text files are single-document)
            if file_extension == '.pdf':
                doc_metadata["num_pages"] = len(docs)
            else:
                doc_metadata["num_pages"] = 1  # Text files treated as single page
            
            # Initialize documents list for this thread if needed
            if thread_id not in self.thread_documents:
                self.thread_documents[thread_id] = []
            
            self.thread_documents[thread_id].append(doc_metadata)
            
            print(f"✓ Added document '{filename}' to thread {thread_id}: {len(all_splits)} chunks")
            return doc_metadata
            
        except Exception as e:
            print(f"✗ Error adding document: {e}")
            raise
    
    def retrieve_context_for_query(self, thread_id: str, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant document chunks for a query using similarity search.
        
        How it works:
        1. Convert query to embedding vector
        2. Compare with all document chunk embeddings (cosine similarity)
        3. Return top-k most similar chunks
        
        Args:
            thread_id: Thread to search documents in
            query: User's question or prompt
            k: Number of top results to return (default: 4)
            
        Returns:
            List of Document objects with relevant content
            
        Note:
            - Similarity search uses cosine similarity in embedding space
            - k=4 is a good default (not too much/little context)
            - Adjust k based on your chunk size and model context window
        """
        if thread_id not in self.thread_vector_stores:
            # No documents uploaded for this thread
            return []
        
        vector_store = self.thread_vector_stores[thread_id]
        
        # Perform similarity search
        # This finds chunks whose embeddings are closest to the query embedding
        retrieved_docs = vector_store.similarity_search(query, k=k)
        
        print(f"Retrieved {len(retrieved_docs)} relevant chunks for query")
        return retrieved_docs
    
    def clear_thread_documents(self, thread_id: str) -> None:
        """
        Clear all documents for a specific thread.
        
        This is called when:
        - User resets the chat
        - User closes the chatbot session
        - Session timeout occurs
        
        Args:
            thread_id: Thread to clear documents for
            
        Learning Note:
            - Important for memory management in production
            - Prevents memory leaks from abandoned sessions
            - Consider adding automatic cleanup for old threads (e.g., >30min idle)
        """
        if thread_id in self.thread_vector_stores:
            del self.thread_vector_stores[thread_id]
            print(f"✓ Cleared vector store for thread: {thread_id}")
        
        if thread_id in self.thread_documents:
            del self.thread_documents[thread_id]
            print(f"✓ Cleared document metadata for thread: {thread_id}")
    
    def should_use_rag(self, query: str, thread_id: str) -> bool:
        """
        Intelligently decide whether to perform RAG retrieval for a query.
        
        This optimization prevents unnecessary retrieval operations for queries that:
        - Are greetings or small talk
        - Request tool usage (let tools handle it)
        - Are very short (likely not document-related)
        
        Strategies used:
        1. **No documents check**: Skip if no documents uploaded
        2. **Query length check**: Very short queries unlikely to need documents
        3. **Keyword detection**: Look for document-related keywords
        4. **Tool request detection**: Skip if user wants to use external tools
        
        Args:
            query: User's query text
            thread_id: Thread to check for documents
            
        Returns:
            True if RAG should be used, False otherwise
            
        Learning Note:
            This saves ~100-300ms per query by avoiding unnecessary embeddings
            and similarity searches when documents aren't relevant.
        """
        # ═══════════════════════════════════════════════════════════════
        # STRATEGY 1: No documents available → Skip RAG
        # ═══════════════════════════════════════════════════════════════
        if thread_id not in self.thread_vector_stores:
            return False
        
        # ═══════════════════════════════════════════════════════════════
        # STRATEGY 2: Very short queries → Likely not document-related
        # ═══════════════════════════════════════════════════════════════
        # Examples: "hi", "ok", "thanks"
        if len(query.strip()) < 10:
            return False
        
        # ═══════════════════════════════════════════════════════════════
        # STRATEGY 3: Greetings and small talk → Skip RAG
        # ═══════════════════════════════════════════════════════════════
        greeting_patterns = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "how are you", "what's up", "sup"
        ]
        query_lower = query.lower().strip()
        
        # If query is ONLY a greeting (not a real question)
        if any(query_lower == pattern or query_lower.startswith(pattern + " ") for pattern in greeting_patterns):
            if "?" not in query and len(query.split()) < 5:
                return False
        
        # ═══════════════════════════════════════════════════════════════
        # STRATEGY 4: Tool/MCP requests → Let tools handle it
        # ═══════════════════════════════════════════════════════════════
        # If user explicitly asks for external data, documents aren't needed
        tool_keywords = [
            "search arxiv", "search pdb", "find protein", "clinical trial",
            "search pubmed", "look up", "find on", "search for papers",
            "what tools", "available tools", "can you search"
        ]
        
        if any(keyword in query_lower for keyword in tool_keywords):
            return False
        
        # ═══════════════════════════════════════════════════════════════
        # STRATEGY 5: Document-related keywords → USE RAG
        # ═══════════════════════════════════════════════════════════════
        # Strong signals that user wants to query their documents
        document_keywords = [
            "document", "paper", "article", "file", "uploaded", "in the",
            "according to", "based on", "from the", "mentioned", "states",
            "summarize", "summary", "explain", "what does", "section",
            "page", "chapter", "quote", "reference"
        ]
        
        if any(keyword in query_lower for keyword in document_keywords):
            return True
        
        # ═══════════════════════════════════════════════════════════════
        # DEFAULT: Use RAG for medium/long queries
        # ═══════════════════════════════════════════════════════════════
        # If query is substantial (5+ words) and not excluded above,
        # it's likely a real question that might benefit from documents
        word_count = len(query.split())
        return word_count >= 5
    
    def get_thread_documents(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Get list of uploaded documents for a thread.
        
        Args:
            thread_id: Thread to get documents for
            
        Returns:
            List of document metadata dictionaries
        """
        return self.thread_documents.get(thread_id, [])

        
    async def call_model(self, state: MessagesState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Process messages through the model with optional RAG enhancement.
        
        This is the core processing function that:
        1. Checks if user has uploaded documents (RAG available)
        2. If RAG: Retrieves relevant context and augments prompt
        3. If no RAG: Uses standard prompt
        4. Trims message history to fit context window
        5. Invokes model and returns response
        
        Args:
            state: Current conversation state with messages
            config: RunnableConfig containing thread_id and other settings
            
        Returns:
            Dictionary with updated messages list
            
        Note:
            - RAG augmentation happens BEFORE model invocation
            - Retrieved context is injected into the system prompt
            - This allows model to "see" document content without fine-tuning
            - Falls back gracefully if no documents are available
        """
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Extract thread_id from config
        # ═══════════════════════════════════════════════════════════════
        thread_id = None
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")
        
        print(f"🔧 call_model - config type: {type(config)}, has configurable: {'configurable' in config if config else False}")
        if config:
            print(f"🔧 call_model - config keys: {config.keys() if hasattr(config, 'keys') else 'N/A'}")
            print(f"🔧 call_model - thread_id extracted: {thread_id}")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Trim message history to fit context window
        # ═══════════════════════════════════════════════════════════════
        # This prevents "context length exceeded" errors
        # Keeps recent messages and system prompt, discards old ones
        trimmed_messages = self.trimmer.invoke(state["messages"])
        
        # Get the latest user message (the current query)
        last_message = trimmed_messages[-1] if trimmed_messages else None
        user_query = last_message.content if last_message and hasattr(last_message, "content") else ""
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Smart RAG Decision (NEW - Performance Optimization!)
        # ═══════════════════════════════════════════════════════════════
        # Instead of ALWAYS retrieving when docs exist, we intelligently
        # decide if this specific query would benefit from RAG
        should_retrieve = thread_id and self.should_use_rag(user_query, thread_id)
        print(f"🔍 Should use RAG: {should_retrieve} (thread_id={thread_id})")
        
        if thread_id and thread_id in self.thread_vector_stores:
            print(f"📚 Thread {thread_id[:8]} has documents in vector store")
            print(f"🔍 Query: '{user_query[:100]}...' (should_use_rag={should_retrieve})")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: RAG Enhancement (only if deemed necessary)
        # ═══════════════════════════════════════════════════════════════
        if should_retrieve:
            # Retrieve relevant document chunks
            # Increased k=6 for better coverage of resume content
            retrieved_docs = self.retrieve_context_for_query(thread_id, user_query, k=6)
            
            if retrieved_docs:
                print(f"✓ RAG: Retrieved {len(retrieved_docs)} relevant document chunks")
                # Format retrieved context for injection into prompt
                context_text = "\n\n".join([
                    f"[Document Excerpt {i+1}]:\n{doc.page_content}"
                    for i, doc in enumerate(retrieved_docs)
                ])
                
                # Create RAG-enhanced prompt template
                # This is the KEY to RAG: we inject retrieved content into the prompt
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are a helpful assistant. You have access to document excerpts that may be relevant to the user's question.\n\n"
                     "RELEVANT DOCUMENT CONTEXT:\n"
                     "═══════════════════════════════════════\n"
                     f"{context_text}\n"
                     "═══════════════════════════════════════\n\n"
                     "Instructions:\n"
                     "1. Use the document context above to answer questions when relevant\n"
                     "2. If the context doesn't contain the answer, say so - don't make things up\n"
                     "3. Cite which document excerpt you're using (e.g., 'According to Document Excerpt 1...')\n"
                     "4. You can also use your general knowledge and available tools when appropriate\n\n"
                     "Answer all questions to the best of your ability."),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                
                prompt = rag_prompt.invoke({"messages": trimmed_messages})
                print(f"✓ RAG: Augmented prompt with {len(retrieved_docs)} document excerpts")
            else:
                # No relevant documents found, use standard prompt
                print("⚠️ RAG: No relevant documents retrieved despite should_retrieve=True")
                prompt = self.prompt_template.invoke({"messages": trimmed_messages})
        else:
            # ═══════════════════════════════════════════════════════════════
            # STEP 5: Standard prompt (no RAG)
            # ═══════════════════════════════════════════════════════════════
            if thread_id and thread_id in self.thread_vector_stores:
                print(f"ℹ️ RAG: Skipped retrieval (query not deemed document-related)")
            prompt = self.prompt_template.invoke({"messages": trimmed_messages})
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Invoke model with (possibly augmented) prompt
        # ═══════════════════════════════════════════════════════════════
        response = await self.model_with_tools.ainvoke(prompt)
        return {"messages": response}
    
    async def initialize(self) -> None:
        """Initialize all components of the chatbot."""
        print("Starting initialization...")
        await self.initialize_mcp_client()
        self.initialize_model()
        self.create_workflow()
        print("Initialization complete!")
    
    def new_thread_config(self) -> Dict[str, Any]:
        """Generate a new thread configuration with unique ID."""
        return {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    def display_session_info(self, config: Dict[str, Any]) -> None:
        """Display session and tool information."""
        print(f"Session started. thread_id={config['configurable']['thread_id']}")
        
        # Show available tools
        if self.tools:
            print(f"Available tools: {len(self.tools)} MCP tools loaded")
            tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in self.tools[:5]]
            print(f"Sample tools: {', '.join(tool_names)}")
            if len(self.tools) > 5:
                print(f"... and {len(self.tools) - 5} more")
        else:
            print("No MCP tools available - using basic chat mode")
        
        print("Interactive Gemini chat (type /exit to quit, /reset to start a new thread)\n")
    
    async def process_user_input(self, user_input: str, config: Dict[str, Any]) -> Optional[str]:
        """Process user input and return response."""
        try:
            output = await self.app.ainvoke({"messages": [HumanMessage(user_input)]}, config)
            
            # Extract the assistant's last message
            last = output.get("messages", [])[-1] if output.get("messages") else None
            if last is not None:
                try:
                    last.pretty_print()
                    return None  # pretty_print handles the output
                except Exception:
                    # Fallback to raw content
                    content = getattr(last, "content", None)
                    return f"Assistant: {content}\n"
            return None
        except Exception as e:
            return f"Error: {e}"
    
    async def run_interactive_chat(self) -> None:
        """Run the interactive chat loop."""
        await self.initialize()
        # Generate a unique thread_id so MemorySaver keeps history for this session
        config = self.new_thread_config()
        self.display_session_info(config)
        
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user:
                continue
            if user.lower() in {"/exit", "/quit"}:
                print("Bye.")
                break
            if user.lower() == "/reset":
                # Start a new thread to clear prior context with a new unique id
                config = self.new_thread_config()
                print(f"Context reset. New thread_id={config['configurable']['thread_id']}\n")
                continue

            response = await self.process_user_input(user, config)
            if response:
                print(response)


def create_chatbot(config_file: str = "server_config_python_only.json", 
                   model_name: str = "gemini-2.5-flash",
                   timeout: float = 30.0) -> GeminiMCPChatbot:
    """
    Factory function to create a configured chatbot instance.
    
    Args:
        config_file: MCP server configuration file
        model_name: Gemini model to use
        timeout: Timeout for MCP connections
    
    Returns:
        Configured GeminiMCPChatbot instance
    """
    return GeminiMCPChatbot(
        model_name=model_name,
        config_file=config_file,
        timeout=timeout
    )


async def main():
    """Main function to run the chatbot."""
    chatbot = create_chatbot()
    await chatbot.run_interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())