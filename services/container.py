from __future__ import annotations
from dataclasses import dataclass
from services.chat_service import ChatService
from services.agent_orchestrator import AgentOrchestrator
from services.thread_service import ThreadService, cleanup_thread_resources

@dataclass(slots=True)
class Services:
    """Application service container.

    Store long-lived, shareable services here and hang this object off
    FastAPI's `app.state.services`.

    This makes dependencies explicit and easy to override in tests.
    """
    chat: ChatService
    orchestrator: AgentOrchestrator
    thread_configs: ThreadService


def build_services(*, max_threads: int = 50) -> Services:
    chat = ChatService()
    orchestrator = AgentOrchestrator()
    thread_configs = ThreadService(
        max_threads=max_threads,
        cleanup_callback=lambda thread_id: cleanup_thread_resources(thread_id, chat),
    )

    return Services(
        chat=chat,
        orchestrator=orchestrator,
        thread_configs=thread_configs,
    )