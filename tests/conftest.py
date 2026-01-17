from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.container import Services
from services.thread_service import ThreadService


@dataclass
class DummyChat:
    chat_agent: object | None = None
    initialized: bool = False
    cleared_threads: list[str] = field(default_factory=list)

    async def initialize(self) -> None:
        self.initialized = True
        self.chat_agent = object()

    def is_initialized(self) -> bool:
        return self.initialized

    async def chat(self, user_input: str, config: dict[str, Any]) -> str:
        return f"echo: {user_input}"

    async def astream_chat(self, user_input: str, config: dict[str, Any]) -> AsyncIterator[str]:
        yield "echo: "
        yield user_input

    def clear_chat_history(self, thread_id: str) -> None:
        self.cleared_threads.append(thread_id)


@dataclass
class DummyLabel:
    async def execute_workflow(self, input_data: dict[str, Any]) -> str:
        return "interpretation"


@dataclass
class DummyEvidence:
    async def execute_workflow(self, input_data: dict[str, Any]) -> dict[str, str]:
        return {
            "faers_explanation": "faers",
            "rwe_explanation": "rwe",
            "clinical_trials_explanation": "clinical",
            "summary": "summary",
        }


@dataclass
class DummyDocuments:
    documents: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add_document_to_vector_store(self, thread_id: str, file_path: str, filename: str) -> dict[str, Any]:
        size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        metadata = {
            "filename": filename,
            "num_chunks": 1,
            "upload_time": "now",
            "file_size": size,
            "file_type": filename.split(".")[-1].upper(),
        }
        self.documents.setdefault(thread_id, []).append(metadata)
        return metadata

    def get_thread_documents(self, thread_id: str) -> list[dict[str, Any]]:
        return self.documents.get(thread_id, [])

    def clear_thread_documents(self, thread_id: str) -> None:
        self.documents.pop(thread_id, None)


@pytest.fixture
def mock_services() -> Services:
    thread_configs = ThreadService(max_threads=10)
    return Services(
        chat=DummyChat(),
        label=DummyLabel(),
        evidence=DummyEvidence(),
        documents=DummyDocuments(),
        thread_configs=thread_configs,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, mock_services: Services) -> TestClient:
    import main

    monkeypatch.setattr(main, "build_services", lambda max_threads=50: mock_services)
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def documents_app(mock_services: Services):
    from fastapi import FastAPI
    from api import document

    app = FastAPI()
    app.include_router(document.router)
    app.state.services = mock_services
    return app


@pytest.fixture
def documents_client(documents_app) -> TestClient:
    with TestClient(documents_app) as test_client:
        yield test_client


@pytest.fixture
def external_services():
    if os.getenv("RUN_EXTERNAL") != "1":
        pytest.skip("Set RUN_EXTERNAL=1 to enable external-call tests.")
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is required for external embedding calls.")

    from services.vector_store_service import VectorStoreService

    return SimpleNamespace(documents=VectorStoreService())
