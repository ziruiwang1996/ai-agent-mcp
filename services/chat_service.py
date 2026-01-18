from typing import Any, Optional
import json
from agent.agent_registry import AgentRegistry
from agent.mcp_agent import MCPAgent
from services.vector_store_service import VectorStoreService
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
import asyncio

class ChatService:
    def __init__(self):
        self._vs_service = VectorStoreService()
        self._agent_registry = AgentRegistry.get_instance()
        self._chat_agent: Optional[MCPAgent] = None
        self._trimmer: list[BaseMessage] = None

        self._workflow = StateGraph(state_schema=MessagesState)
        self._register_nodes()
        self._checkpointer = MemorySaver()
        self._app = self._workflow.compile(checkpointer=self._checkpointer)
    
    async def initialize(self) -> None:
        if self._chat_agent is not None:
            return
        agent = await self._agent_registry.resolve("chat_agent")
        self._chat_agent = agent
        self.set_context_window()

    def is_initialized(self) -> bool:
        return self._chat_agent is not None
    
    def clear_chat_history(self, thread_id: str) -> None:
        checkpointer = self._checkpointer
        if checkpointer is None:
            return
        delete_fn = getattr(checkpointer, "delete_thread", None)
        if callable(delete_fn):
            delete_fn(thread_id)
    
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
            self._chat_agent.chat_model
            if hasattr(self._chat_agent.chat_model, "get_num_tokens_from_messages")
            else self._count_tokens_fallback
        )
        self._trimmer = trim_messages(
            max_tokens=window_size,
            strategy="last",
            token_counter=token_counter,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

    def _register_nodes(self) -> None:
        self._workflow.add_node("model", self._call_model)
        tools_node = ToolNode(self._chat_agent.tools) if self._chat_agent.tools else None
        if tools_node is not None:
            self._workflow.add_node("tools", tools_node)
        
        self._workflow.add_edge(START, "model")
        # If tools exist, create tool execution loop
        if tools_node is not None:
            self._workflow.add_conditional_edges("model", self._should_continue, {
                "tools": "tools",  # Execute tools if requested
                "end": END,        # Finish if no tools needed
            })
            # After tool execution, return to model with results
            self._workflow.add_edge("tools", "model")
        else:
            # No tools available, model -> end
            self._workflow.add_edge("model", END)
    
    def _should_continue(state: MessagesState) -> str:
            """
            Decide whether to execute tools or finish.
                
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
                return "tools"    
            return "end"

    async def _call_model(self, state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        # Extract thread_id from config
        thread_id = config["configurable"].get("thread_id", None)
        if not thread_id:
            raise ValueError("thread_id is required in RunnableConfig for chat service")

        # Trim message history to fit context window
        if not self._trimmer:
            self.set_context_window()
        trimmed_messages = self._trimmer.invoke(state["messages"])
        last_message = trimmed_messages[-1] if trimmed_messages else None
        user_query = last_message.content if last_message and hasattr(last_message, "content") else ""

        # RAG incorporation 
        should_retrieve = self._should_use_rag(user_query, thread_id)
        print(f"Should use RAG: {should_retrieve} (thread_id={thread_id})")

        injected_messages = trimmed_messages
        if should_retrieve:
            retrieved_docs = self._vs_service.retrieve_context_for_query(thread_id, user_query)
            
            if retrieved_docs:
                ## only include high-confidence docs (score >= 0.75)
                context_text = "\n\n".join([
                    f"[Document Excerpt {i+1}]:\n{doc.page_content}"
                    for i, doc in enumerate(retrieved_docs) if doc[1] >= 0.75
                ])

                messages_no_system = [m for m in trimmed_messages if not isinstance(m, SystemMessage)]
                injected_messages = [
                    SystemMessage(
                        content=(
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
                            "Answer all questions to the best of your ability."
                        )
                    ),
                    *messages_no_system
                ]
        
        # Invoke model with (possibly augmented) prompt
        if not self._chat_agent:
            return {"messages": [AIMessage(content="Agent failed to initialize; check model setup logs.")]}

        try:
            agent_out = await asyncio.wait_for(
                self._chat_agent.chat_model.ainvoke({"messages": injected_messages}),
                timeout=60.0,
            )
            return {"messages": agent_out.get("messages", [])}
        except asyncio.TimeoutError:
            return {"messages": [AIMessage(content=f"Request timed out after 60s. Please try again.")]}
    
    def _should_use_rag(self, query: str, thread_id: str) -> bool:
        # STRATEGY 1: No vector store available → Skip RAG
        if not self._vs_service.is_thread_has_vector_store(thread_id):
            return False
        
        # STRATEGY 2: Vector store exists but is empty → Skip RAG
        # Check multiple ways to determine if vector store has content
        if self._vs_service.is_vector_store_empty(thread_id):
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
    
    async def chat(self, user_input: str, config: dict[str, Any]) -> Optional[str]:
        try:
            output = await asyncio.wait_for(
                self._app.ainvoke({"messages": [HumanMessage(user_input)]}, config),
                timeout=60.0,
            )
            # Extract the assistant's last message
            last_message = output.get("messages", [])[-1] if output.get("messages") else None
            if last_message:
                response_text = getattr(last_message, "content", "No response generated")
            else:
                response_text = "No response generated"
            return response_text
        except asyncio.TimeoutError:
            return f"Error: Request timed out after 60.0s"
        except Exception as e:
            return f"Error: {str(e)}"