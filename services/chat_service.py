from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from agent.chat_agent import ChatAgent
from agent.model_registry import ModelRegistry

class ChatService:
    def __init__(self,
        *,
        mcp_server_name: str = "chat_agent",
        system_message: str = "You are a helpful assistant. Answer all questions to the best of your ability.",
        embedding: str = "models/gemini-embedding-001",
    ):
        self._model_registry = ModelRegistry()
        self._mcp_server_name = mcp_server_name
        self._system_message = system_message
        self._embedding = embedding
        self._chat_agent: Optional[ChatAgent] = None

    @property
    def chat_agent(self) -> Optional[ChatAgent]:
        return self._chat_agent

    async def initialize(self) -> None:
        if self._chat_agent is not None:
            return
        chat_model = self._model_registry.resolve("gemini")
        agent = ChatAgent(
            chat_model,
            self._mcp_server_name,
            self._system_message,
            self._embedding,
        )
        await agent.initialize()
        self._chat_agent = agent

    def is_initialized(self) -> bool:
        return self._chat_agent is not None

    async def chat(self, user_input: str, config: Dict[str, Any]) -> str:
        if self._chat_agent is None:
            raise RuntimeError("Chat is not initialized. Call POST /chat/initialize first.")
        return await self._chat_agent.process_input(user_input, config)

    async def astream_chat(self, user_input: str, config: Dict[str, Any]) -> AsyncIterator[str]:
        agent = self._chat_agent
        if agent is None:
            raise RuntimeError("Chat is not initialized. Call POST /chat/initialize first.")
        if agent is None or not hasattr(agent, "app"):
            yield await self.chat(user_input=user_input, config=config)
            return

        # IMPORTANT:
        # Prefer streaming through the LangGraph app (agent.app) because it uses the
        # checkpointer (MemorySaver) and therefore preserves chat history by thread_id.
        # The lc_agent streaming path below is effectively stateless because it only
        # receives the current turn's messages.
        app = getattr(agent, "app", None)
        input_payload = {"messages": [HumanMessage(content=user_input)]}

        if app is not None and hasattr(app, "astream_events"):
            async for event in app.astream_events(input_payload, config, version="v1"):
                event_name = event.get("event")
                if event_name not in {"on_chat_model_stream", "on_llm_stream"}:
                    continue
                data = event.get("data") or {}
                chunk = data.get("chunk")
                content = getattr(chunk, "content", None)
                if content:
                    yield content
            return

        if app is not None and hasattr(app, "astream"):
            sent = ""
            async for state in app.astream(input_payload, config):
                if not isinstance(state, dict):
                    continue
                messages = state.get("messages")
                if not messages:
                    continue
                last_message = messages[-1]
                full = getattr(last_message, "content", None)
                if not isinstance(full, str) or not full:
                    continue
                if full.startswith(sent):
                    delta = full[len(sent):]
                else:
                    delta = full
                if delta:
                    yield delta
                    sent = full
            return

        thread_id = None
        if isinstance(config, dict):
            thread_id = (config.get("configurable") or {}).get("thread_id")

        system_message = getattr(agent, "system_message", self._system_message)
        if thread_id and hasattr(agent, "should_use_rag") and hasattr(agent, "retrieve_context_for_query"):
            try:
                if agent.should_use_rag(user_input, thread_id):
                    retrieved_docs = agent.retrieve_context_for_query(thread_id, user_input)
                    if retrieved_docs:
                        context_text = "\n\n".join(
                            f"[Document Excerpt {i+1}]:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)
                        )
                        system_message = (
                            f"{system_message}\n\n"
                            "RELEVANT DOCUMENT CONTEXT:\n"
                            "═══════════════════════════════════════\n"
                            f"{context_text}\n"
                            "═══════════════════════════════════════\n\n"
                            "Use the document context when relevant. If it doesn't contain the answer, say so."
                        )
            except Exception:
                pass

        # Fallback: stream directly from the underlying agent if needed.
        # NOTE: This does not automatically include saved chat history.
        lc_agent = getattr(agent, "agent", None)
        agent_input = {"messages": [SystemMessage(content=system_message), HumanMessage(content=user_input)]}
        if lc_agent is not None and hasattr(lc_agent, "astream_events"):
            async for event in lc_agent.astream_events(agent_input, config, version="v1"):
                event_name = event.get("event")
                if event_name not in {"on_chat_model_stream", "on_llm_stream"}:
                    continue
                data = event.get("data") or {}
                chunk = data.get("chunk")
                content = getattr(chunk, "content", None)
                if content:
                    yield content
            return
        yield await self.chat(user_input=user_input, config=config)

    def clear_thread_documents(self, thread_id: str) -> None:
        # Thread eviction can happen before chat is initialized.
        if self._chat_agent is None:
            return
        self._chat_agent.clear_thread_documents(thread_id)

    def clear_thread_history(self, thread_id: str) -> None:
        if self._chat_agent is None:
            return
        self._chat_agent.clear_thread_history(thread_id)

    def add_document_to_thread(self, thread_id: str, file_path: str, filename: str) -> Dict[str, Any]:
        if self._chat_agent is None:
            raise RuntimeError("Chat is not initialized. Call POST /chat/initialize first.")
        return self._chat_agent.add_document_to_vector_store(
            thread_id=thread_id,
            file_path=file_path,
            filename=filename,
        )

    def get_thread_documents(self, thread_id: str) -> List[Dict[str, Any]]:
        if self._chat_agent is None:
            return []
        return self._chat_agent.get_thread_documents(thread_id)