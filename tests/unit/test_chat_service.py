from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage

from services.chat_service import ChatService


class _DummyVectorStore:
    def is_thread_has_vector_store(self, thread_id: str) -> bool:
        return False

    def is_vector_store_empty(self, thread_id: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_chat_service_initialization():
    with patch("services.chat_service.VectorStoreService", return_value=_DummyVectorStore()):
        with patch("services.chat_service.AgentRegistry") as MockRegistry:
            mock_agent = AsyncMock()
            mock_agent.chat_model = object()
            mock_registry = MockRegistry.get_instance.return_value
            mock_registry.resolve = AsyncMock(return_value=mock_agent)
            chat_service = ChatService()
            await chat_service.initialize()
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
    chat_service.checkpointer = SimpleNamespace(delete_thread=delete_thread)

    chat_service.clear_chat_history("thread-123")
    delete_thread.assert_called_once_with("thread-123")
