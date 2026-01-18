from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from services.chat_service import ChatService


class _DummyVectorStore:
    def __init__(self, *, has_store: bool = False, empty: bool = True, docs=None):
        self._has_store = has_store
        self._empty = empty
        self._docs = docs or []

    def is_thread_has_vector_store(self, thread_id: str) -> bool:
        return self._has_store

    def is_vector_store_empty(self, thread_id: str) -> bool:
        return self._empty

    def retrieve_context_for_query(self, thread_id: str, query: str):
        return self._docs


class _DummyChatModel:
    def __init__(self):
        self.last_messages = None

    async def ainvoke(self, payload):
        self.last_messages = payload.get("messages")
        return {"messages": [AIMessage(content="ok")]}


def test_chat_service_initialization():
    with patch("services.chat_service.VectorStoreService", return_value=_DummyVectorStore()):
        with patch("services.chat_service.AgentRegistry") as MockRegistry:
            mock_agent = AsyncMock()
            mock_agent.chat_model = object()
            mock_registry = MockRegistry.get_instance.return_value
            mock_registry.resolve = AsyncMock(return_value=mock_agent)
            chat_service = ChatService()
            import asyncio

            asyncio.run(chat_service.initialize())
            assert chat_service.is_initialized()


def test_count_tokens_fallback_handles_non_string():
    chat_service = ChatService.__new__(ChatService)
    messages = [HumanMessage(content=["a", "b"])]
    assert chat_service._count_tokens_fallback(messages) >= 1


def test_should_use_rag_respects_empty_store():
    chat_service = ChatService.__new__(ChatService)
    chat_service._vs_service = _DummyVectorStore()

    assert chat_service._should_use_rag("Longer query about documents", "thread-1") is False


def test_clear_chat_history_uses_checkpointer():
    chat_service = ChatService.__new__(ChatService)
    delete_thread = Mock()
    chat_service._checkpointer = SimpleNamespace(delete_thread=delete_thread)

    chat_service.clear_chat_history("thread-123")
    delete_thread.assert_called_once_with("thread-123")


def test_should_use_rag_rules():
    chat_service = ChatService.__new__(ChatService)
    chat_service._vs_service = _DummyVectorStore(has_store=False, empty=True)
    assert chat_service._should_use_rag("Tell me about aspirin dosage", "thread-1") is False

    chat_service._vs_service = _DummyVectorStore(has_store=True, empty=True)
    assert chat_service._should_use_rag("Tell me about aspirin dosage", "thread-1") is False

    chat_service._vs_service = _DummyVectorStore(has_store=True, empty=False)
    assert chat_service._should_use_rag("Hi", "thread-1") is False
    assert chat_service._should_use_rag("hello there", "thread-1") is False
    assert chat_service._should_use_rag("Summarize the document", "thread-1") is True
    assert chat_service._should_use_rag("Tell me what this says in the file", "thread-1") is True


def test_call_model_requires_thread_id():
    chat_service = ChatService.__new__(ChatService)
    chat_service._vs_service = _DummyVectorStore()
    chat_service._chat_agent = SimpleNamespace(chat_model=_DummyChatModel())
    chat_service._trimmer = SimpleNamespace(invoke=lambda messages: messages)

    import asyncio

    with pytest.raises(ValueError, match="thread_id is required"):
        asyncio.run(chat_service._call_model({"messages": [HumanMessage(content="hi")]}, {"configurable": {}}))


def test_call_model_injects_rag_context():
    chat_service = ChatService.__new__(ChatService)
    docs = [
        (Document(page_content="High confidence"), 0.8),
        (Document(page_content="Low confidence"), 0.4),
    ]
    chat_service._vs_service = _DummyVectorStore(has_store=True, empty=False, docs=docs)
    chat_service._chat_agent = SimpleNamespace(chat_model=_DummyChatModel())
    chat_service._trimmer = SimpleNamespace(invoke=lambda messages: messages)

    import asyncio

    result = asyncio.run(
        chat_service._call_model(
            {"messages": [HumanMessage(content="Summarize the document contents please")]},
            {"configurable": {"thread_id": "thread-1"}},
        )
    )
    assert result["messages"][-1].content == "ok"
    injected = chat_service._chat_agent.chat_model.last_messages
    assert isinstance(injected[0], SystemMessage)
    assert "High confidence" in injected[0].content
    assert "Low confidence" not in injected[0].content


def test_call_model_timeout_returns_error(monkeypatch: pytest.MonkeyPatch):
    chat_service = ChatService.__new__(ChatService)
    chat_service._vs_service = _DummyVectorStore()
    chat_service._chat_agent = SimpleNamespace(chat_model=_DummyChatModel())
    chat_service._trimmer = SimpleNamespace(invoke=lambda messages: messages)

    import asyncio

    async def _raise_timeout(*args, **kwargs):
        raise asyncio.TimeoutError()

    monkeypatch.setattr("services.chat_service.asyncio.wait_for", _raise_timeout)
    result = asyncio.run(
        chat_service._call_model(
            {"messages": [HumanMessage(content="hi")]},
            {"configurable": {"thread_id": "thread-1"}},
        )
    )
    assert "timed out" in result["messages"][0].content.lower()
