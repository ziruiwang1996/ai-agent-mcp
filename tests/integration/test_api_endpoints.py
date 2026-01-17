from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_available_tools(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    class DummyAgent:
        def get_available_tools(self):
            return [{"name": "tool", "description": "desc"}]

    class DummyRegistry:
        def initialized_agents(self):
            return ["chat_agent"]

        def get_initialized_agent(self, key):
            return DummyAgent()

        def is_agent_initialized(self, key):
            return True

    monkeypatch.setattr(
        "api.tools.AgentRegistry.get_instance",
        lambda: DummyRegistry(),
    )

    response = client.get("/api/tools")
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    assert data["agents"]["chat_agent"]["tools_count"] == 1


def test_initialize_chat(client: TestClient):
    response = client.post("/api/chat/initialize", json={"thread_id": "thread-1"})
    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"] == "thread-1"
    assert data["chat_initialized"] is True


@pytest.mark.parametrize(
    "message",
    ["Hello! What can you help me with?", "Can you summarize uploaded documents?"],
)
def test_simple_chat(message: str, client: TestClient):
    client.post("/api/chat/initialize", json={"thread_id": "thread-2"})
    chat_data = {"message": message, "thread_id": "thread-2"}
    response = client.post("/api/chat/batch", json=chat_data)
    assert response.status_code == 200
    data = response.json()
    assert data["response"].startswith("echo:")


def test_chat_requires_initialization(client: TestClient):
    response = client.post("/api/chat/batch", json={"message": "hi", "thread_id": "thread-3"})
    assert response.status_code == 409


def test_reset_chat_clears_documents(client: TestClient):
    client.post("/api/chat/initialize", json={"thread_id": "thread-4"})
    response = client.post("/api/chat/reset", json={"thread_id": "thread-4"})
    assert response.status_code == 200
    data = response.json()
    assert data["documents_cleared"] is True


def test_thread_stats(client: TestClient):
    response = client.get("/api/threads/stats")
    assert response.status_code == 200
    data = response.json()
    assert "cache_info" in data


@pytest.mark.external
def test_external_chat_initialize():
    if os.getenv("RUN_EXTERNAL") != "1":
        pytest.skip("Set RUN_EXTERNAL=1 to enable external-call tests.")
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is required for external chat tests.")

    import main

    with TestClient(main.app) as test_client:
        response = test_client.post("/api/chat/initialize", json={"thread_id": "ext-thread"})
        assert response.status_code == 200
