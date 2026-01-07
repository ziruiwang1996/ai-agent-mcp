from agent.mcp_agent import MCPAgent
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datetime import datetime
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages
from typing import Optional, List, Dict, Any
from langgraph.graph.state import CompiledStateGraph
import os
import asyncio
import json

class ChatAgent(MCPAgent):
    def __init__(
        self,
        chat_model: BaseChatModel,
        mcp_config_key: str,
        system_message: str,
        embedding: str
    ):
        super().__init__(chat_model, mcp_config_key, system_message)
        self.timeout: int = 60 
        self.trimmer: list[BaseMessage] = None
        self.embedding_model: Any = GoogleGenerativeAIEmbeddings(model=embedding)
        self.checkpointer: Any = None
        self.thread_vector_stores: Dict[str, InMemoryVectorStore] = {}
        self.thread_documents: Dict[str, List[Dict[str, Any]]] = {}
        self.app: Optional[CompiledStateGraph] = None

    async def initialize(self) -> None:
        await super().initialize()
        self.set_context_window()
        self.create_workflow()

    def _count_tokens_fallback(self, messages: list[BaseMessage]) -> int:
        """Approximate token counter used when the model can't count tokens.
        This avoids runtime errors in trim_messages() for models that don't
        implement get_num_tokens_from_messages().
        """
        total_chars = 0
        for msg in messages or []:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                total_chars += len(json.dumps(content, ensure_ascii=False))
            else:
                total_chars += len(str(content))
        return max(1, total_chars // 4)

    def set_context_window(self, window_size: int = 2000) -> None:
        token_counter = (
            self.chat_model
            if hasattr(self.chat_model, "get_num_tokens_from_messages")
            else self._count_tokens_fallback
        )
        self.trimmer = trim_messages(
            max_tokens=window_size,
            strategy="last",
            token_counter=token_counter,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

    def create_workflow(self) -> None:
        """Create the LangGraph workflow with tool execution loop and RAG integration.
        Notes:
        - MessagesState maintains conversation history
        - MemorySaver persists state across turns (per thread)
        - Tool execution is automatic if model requests it
        - RAG context injection is transparent to the workflow
        """
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", self.call_model)
        tools_node = ToolNode(self.tools) if self.tools else None
        if tools_node is not None:
            workflow.add_node("tools", tools_node)
        
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
            # Check if last message has tool calls
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools" if tools_node is not None else "end"         
            return "end"
        
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

        self.checkpointer = MemorySaver()
        self.app = workflow.compile(checkpointer=self.checkpointer)

    def clear_thread_history(self, thread_id: str) -> None:
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None:
            return
        delete_fn = getattr(checkpointer, "delete_thread", None)
        if callable(delete_fn):
            delete_fn(thread_id)

    async def call_model(self, state: MessagesState, config: RunnableConfig) -> Dict[str, Any]:
        # Extract thread_id from config
        thread_id = None
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")
        print(f"call_model - config type: {type(config)}, has configurable: {'configurable' in config if config else False}")
        if config:
            print(f"call_model - config keys: {config.keys() if hasattr(config, 'keys') else 'N/A'}")
            print(f"call_model - thread_id extracted: {thread_id}")

        # Trim message history to fit context window
        if not self.trimmer:
            self.set_context_window()
        trimmed_messages = self.trimmer.invoke(state["messages"])
        last_message = trimmed_messages[-1] if trimmed_messages else None
        user_query = last_message.content if last_message and hasattr(last_message, "content") else ""

        # RAG incorporation 
        should_retrieve = thread_id and self.should_use_rag(user_query, thread_id)
        print(f"Should use RAG: {should_retrieve} (thread_id={thread_id})")
        if thread_id and thread_id in self.thread_vector_stores:
            print(f"Thread {thread_id[:8]} has documents in vector store")
            print(f"Query: '{user_query[:100]}...' (should_use_rag={should_retrieve})")
        if should_retrieve:
            # Retrieve relevant document chunks
            retrieved_docs = self.retrieve_context_for_query(thread_id, user_query)
            
            if retrieved_docs:
                print(f"RAG: Retrieved {len(retrieved_docs)} relevant document chunks")
                context_text = "\n\n".join([
                    f"[Document Excerpt {i+1}]:\n{doc.page_content}"
                    for i, doc in enumerate(retrieved_docs)
                ])
                
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     f"{self.system_message}\n\n"
                     "You have access to document excerpts that may be relevant to the user's question.\n\n"
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
                
                messages_no_system = [m for m in trimmed_messages if not isinstance(m, SystemMessage)]
                prompt = rag_prompt.invoke({"messages": messages_no_system})
                print(f"RAG: Augmented prompt with {len(retrieved_docs)} document excerpts")
            else:
                print("RAG: No relevant documents retrieved despite should_retrieve=True")
                base_prompt = ChatPromptTemplate.from_messages([
                    ("system", self.system_message),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                messages_no_system = [m for m in trimmed_messages if not isinstance(m, SystemMessage)]
                prompt = base_prompt.invoke({"messages": messages_no_system})
        else:
            # Standard prompt (no RAG)
            print(f"RAG: Skipped retrieval (query not deemed document-related)")
            base_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message),
                MessagesPlaceholder(variable_name="messages"),
            ])
            messages_no_system = [m for m in trimmed_messages if not isinstance(m, SystemMessage)]
            prompt = base_prompt.invoke({"messages": messages_no_system})
        
        # Invoke model with (possibly augmented) prompt
        if not self.agent:
            return {"messages": [AIMessage(content="Agent failed to initialize; check model setup logs.")]}

        try:
            agent_out = await asyncio.wait_for(
                self.agent.ainvoke({"messages": prompt.to_messages()}),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return {
                "messages": [
                    AIMessage(content=f"Request timed out after {self.timeout}s. Please try again.")
                ]
            }
        return {"messages": agent_out.get("messages", [])}
    
    def should_use_rag(self, query: str, thread_id: str) -> bool:
        # STRATEGY 1: No vector store available → Skip RAG
        if thread_id not in self.thread_vector_stores:
            return False
        
        # STRATEGY 2: Vector store exists but is empty → Skip RAG
        # Check multiple ways to determine if vector store has content
        vector_store = self.thread_vector_stores[thread_id]
        # Check the internal store attribute (InMemoryVectorStore specific)
        if hasattr(vector_store, 'store') and len(vector_store.store) == 0:
            print(f"Vector store exists for thread {thread_id[:8]} but store is empty")
            return False
        # Check document metadata tracking
        if thread_id not in self.thread_documents or len(self.thread_documents[thread_id]) == 0:
            print(f"Vector store exists for thread {thread_id[:8]} but no documents tracked")
            return False
        
        # STRATEGY 3: Very short queries → Likely not document-related
        if len(query.strip()) < 10:
            return False
        
        # STRATEGY 4: Greetings and small talk → Skip RAG
        greeting_patterns = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "how are you", "what's up", "sup"
        ]
        query_lower = query.lower().strip()
        # If query is ONLY a greeting (not a real question)
        if any(query_lower == pattern or query_lower.startswith(pattern + " ") for pattern in greeting_patterns):
            if "?" not in query and len(query.split()) < 5:
                return False
        
        # STRATEGY 5: Document-related keywords → USE RAG
        # Strong signals that user wants to query their documents
        document_keywords = [
            "document", "paper", "article", "file", "uploaded", "in the",
            "according to", "based on", "from the", "mentioned", "states",
            "summarize", "summary", "explain", "what does", "section",
            "page", "chapter", "quote", "reference"
        ]
        if any(keyword in query_lower for keyword in document_keywords):
            return True
        
        # DEFAULT: Use RAG for medium/long queries
        # If query is substantial (10+ words) and not excluded above,
        # it's likely a real question that might benefit from documents
        word_count = len(query.split())
        return word_count >= 10

    def get_or_create_vector_store(self, thread_id: str) -> InMemoryVectorStore:
        """
        Note:
            - InMemoryVectorStore is RAM-based (fast but not persistent)
            - Each store uses the same embedding model for consistency
            - For production, consider ChromaDB or Pinecone for persistence
        """
        if thread_id not in self.thread_vector_stores:
            print(f"Creating new vector store for thread: {thread_id}")
            self.thread_vector_stores[thread_id] = InMemoryVectorStore(self.embedding_model)
        return self.thread_vector_stores[thread_id]

    def add_document_to_vector_store(self, thread_id: str, file_path: str, filename: str) -> Dict[str, Any]:
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
        """
        if thread_id not in self.thread_vector_stores:
            return []
        
        vector_store = self.thread_vector_stores[thread_id]
        # Perform similarity search
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
        """
        if thread_id in self.thread_vector_stores:
            del self.thread_vector_stores[thread_id]
            print(f"Cleared vector store for thread: {thread_id}")
        
        if thread_id in self.thread_documents:
            del self.thread_documents[thread_id]
            print(f"Cleared document metadata for thread: {thread_id}")
    
    def get_thread_documents(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Get list of uploaded documents for a thread.
        """
        return self.thread_documents.get(thread_id, [])
    
    async def process_input(self, user_input: str, config: Dict[str, Any]) -> Optional[str]:
        try:
            output = await asyncio.wait_for(
                self.app.ainvoke({"messages": [HumanMessage(user_input)]}, config),
                timeout=self.timeout,
            )
            # Extract the assistant's last message
            last_message = output.get("messages", [])[-1] if output.get("messages") else None
            if last_message:
                response_text = getattr(last_message, "content", "No response generated")
            else:
                response_text = "No response generated"
            return response_text
        except asyncio.TimeoutError:
            return f"Error: Request timed out after {self.timeout}s"
        except Exception as e:
            return f"Error: {e}"