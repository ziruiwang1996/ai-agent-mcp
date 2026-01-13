from __future__ import annotations
from dataclasses import dataclass
from services.chat_service import ChatService
from services.evidence_service import EvidenceService
from services.label_service import LabelService
from services.thread_service import ThreadService, cleanup_thread_resources

@dataclass(slots=True)
class Services:
    """Application service container.

    Store long-lived, shareable services here and hang this object off
    FastAPI's `app.state.services`.

    This makes dependencies explicit and easy to override in tests.
    """
    chat: ChatService
    label: LabelService
    evidence: EvidenceService
    thread_configs: ThreadService


def build_services(*, max_threads: int = 50) -> Services:
    chat = ChatService()
    label = LabelService()
    evidence = EvidenceService()
    thread_configs = ThreadService(
        max_threads=max_threads,
        cleanup_callback=lambda thread_id: cleanup_thread_resources(thread_id, chat),
    )

    return Services(
        chat=chat,
        label=label,
        evidence=evidence,
        thread_configs=thread_configs,
    )