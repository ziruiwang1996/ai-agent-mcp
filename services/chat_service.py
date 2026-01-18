from typing import Any, Optional, AsyncIterator
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
        self._agent_registry = AgentRegistry()
        self._chat_agent: Optional[MCPAgent] = None
        self.trimmer: list[BaseMessage] = None

        self._workflow = StateGraph(state_schema=MessagesState)
        self.checkpointer = MemorySaver()
        self.app = None  # Will be set in initialize()
    
    async def initialize(self) -> None:
        if self._chat_agent is not None:
            return
        agent = await self._agent_registry.resolve("chat_agent")
        self._chat_agent = agent
        self.set_context_window()
        self._register_nodes()
        self.app = self._workflow.compile(checkpointer=self.checkpointer)

    def is_initialized(self) -> bool:
        return self._chat_agent is not None
    
    def clear_chat_history(self, thread_id: str) -> None:
        checkpointer = getattr(self, "checkpointer", None)
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

    def _iter_text_chunks(self, text: str, chunk_size: int = 48) -> list[str]:
        if not text:
            return []
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _extract_stream_text(self, event: Any) -> Optional[str]:
        if not isinstance(event, dict):
            return None
        event_type = event.get("event") or ""
        if "stream" not in event_type:
            return None
        data = event.get("data") or {}
        chunk = data.get("chunk") or data.get("delta") or data.get("output")
        if isinstance(chunk, str):
            return chunk
        if hasattr(chunk, "content"):
            return getattr(chunk, "content", None)
        if isinstance(chunk, dict):
            content = chunk.get("content") or chunk.get("text")
            if isinstance(content, str):
                return content
        return None
    
    def set_context_window(self, window_size: int = 2000) -> None:
        token_counter = (
            self._chat_agent.chat_model
            if hasattr(self._chat_agent.chat_model, "get_num_tokens_from_messages")
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
        if not self.trimmer:
            self.set_context_window()
        trimmed_messages = self.trimmer.invoke(state["messages"])
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
                    f"[Document Excerpt {i+1}]:\n{doc[0].page_content}"
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
            # Pass only the list of messages to the model, not a dict
            agent_out = await asyncio.wait_for(
                self._chat_agent.chat_model.ainvoke(injected_messages),
                timeout=60.0,
            )
            return {"messages": [agent_out]}
        except asyncio.TimeoutError:
            return {"messages": [AIMessage(content=f"Request timed out after 60s. Please try again.")]}
    
    def _should_use_rag(self, query: str, thread_id: str) -> bool:
        # STRATEGY 1: No vector store available → Skip RAG
        has_vector_store = self._vs_service.is_thread_has_vector_store(thread_id)
        print(f"[RAG DEBUG] has_vector_store={has_vector_store} for thread_id={thread_id}")
        if not has_vector_store:
            print("[RAG DEBUG] Skipping RAG: No vector store for thread.")
            return False

        # STRATEGY 2: Vector store exists but is empty → Skip RAG
        is_empty = self._vs_service.is_vector_store_empty(thread_id)
        print(f"[RAG DEBUG] is_vector_store_empty={is_empty} for thread_id={thread_id}")
        if is_empty:
            print("[RAG DEBUG] Skipping RAG: Vector store is empty.")
            return False

        # STRATEGY 3: Very short queries → Likely not document-related
        query_len = len(query.strip())
        print(f"[RAG DEBUG] query_len={query_len} for query='{query}'")
        if query_len < 10:
            print("[RAG DEBUG] Skipping RAG: Query too short.")
            return False

        # STRATEGY 4: Greetings and small talk → Skip RAG
        greeting_patterns = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "how are you", "what's up", "sup"
        ]
        query_lower = query.lower().strip()
        if any(query_lower == pattern or query_lower.startswith(pattern + " ") for pattern in greeting_patterns):
            if "?" not in query and len(query.split()) < 5:
                print("[RAG DEBUG] Skipping RAG: Query is a greeting.")
                return False

        # STRATEGY 5: Document-related keywords → USE RAG
        document_keywords = [
            "document", "paper", "article", "file", "uploaded", "in the",
            "according to", "based on", "from the", "mentioned", "states",
            "summarize", "summary", "explain", "what does", "section",
            "page", "chapter", "quote", "reference"
        ]
        keyword_match = any(keyword in query_lower for keyword in document_keywords)
        print(f"[RAG DEBUG] keyword_match={keyword_match} for query='{query_lower}'")
        if keyword_match:
            print("[RAG DEBUG] Using RAG: Document-related keyword found.")
            return True

        # DEFAULT: Use RAG for medium/long queries
        word_count = len(query.split())
        print(f"[RAG DEBUG] word_count={word_count} for query='{query}'")
        use_rag = word_count >= 10
        if use_rag:
            print("[RAG DEBUG] Using RAG: Query is long enough.")
        else:
            print("[RAG DEBUG] Skipping RAG: Query not long enough.")
        return use_rag
    
    async def chat(self, user_input: str, config: dict[str, Any]) -> Optional[str]:
        try:
            output = await asyncio.wait_for(
                self.app.ainvoke({"messages": [HumanMessage(user_input)]}, config),
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
        
    ## To Do: Implement streaming chat method
    async def astream_chat(self, user_input: str, config: dict[str, Any]) -> AsyncIterator[str]:
        if not self.app:
            yield "Error: Chat service not initialized."
            return
        payload = {"messages": [HumanMessage(user_input)]}

        try:
            if hasattr(self.app, "astream_events"):
                try:
                    async for event in self.app.astream_events(payload, config):
                        chunk = self._extract_stream_text(event)
                        if chunk:
                            yield chunk
                    return
                except TypeError:
                    async for event in self.app.astream_events(payload, config, version="v1"):
                        chunk = self._extract_stream_text(event)
                        if chunk:
                            yield chunk
                    return

            if hasattr(self.app, "astream"):
                last_content = ""
                async for state in self.app.astream(payload, config):
                    if not isinstance(state, dict):
                        continue
                    messages = state.get("messages", [])
                    if not messages:
                        continue
                    last = messages[-1]
                    content = getattr(last, "content", None)
                    if not isinstance(content, str):
                        continue
                    if content.startswith(last_content):
                        delta = content[len(last_content):]
                    else:
                        delta = content
                    last_content = content
                    if delta:
                        yield delta
                return

            output = await asyncio.wait_for(self.app.ainvoke(payload, config), timeout=60.0)
            last_message = output.get("messages", [])[-1] if output.get("messages") else None
            response_text = getattr(last_message, "content", "No response generated") if last_message else "No response generated"
            for chunk in self._iter_text_chunks(response_text):
                yield chunk
        except asyncio.TimeoutError:
            yield "Error: Request timed out after 60.0s"
        except Exception as e:
            yield f"Error: {str(e)}"
