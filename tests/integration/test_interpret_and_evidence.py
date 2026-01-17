from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


def test_interpret_requires_thread_id(client: TestClient):
    response = client.post(
        "/api/interpret",
        json={
            "thread_id": "",
            "drug_name": "Aspirin",
            "section_name": "usage",
            "section_content": "Pain relief",
        },
    )
    assert response.status_code == 400


def test_interpret_returns_payload(client: TestClient):
    response = client.post(
        "/api/interpret",
        json={
            "thread_id": "thread-1",
            "drug_name": "Aspirin",
            "section_name": "usage",
            "section_content": "Pain relief",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["interpretation"] == "interpretation"


def test_evidence_requires_thread_id(client: TestClient):
    response = client.post(
        "/api/evidence",
        json={
            "thread_id": "",
            "drug_set_id": "set-1",
            "drug_name": "Drug",
            "age": "30",
            "sex": "f",
            "weight": "70",
            "is_pregnant": False,
            "is_breastfeeding": False,
        },
    )
    assert response.status_code == 400


def test_evidence_returns_report(client: TestClient):
    response = client.post(
        "/api/evidence",
        json={
            "thread_id": "thread-2",
            "drug_set_id": "set-1",
            "drug_name": "Drug",
            "age": "30",
            "sex": "f",
            "weight": "70",
            "is_pregnant": False,
            "is_breastfeeding": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == "summary"


@pytest.mark.external
def test_external_interpret():
    if os.getenv("RUN_EXTERNAL") != "1":
        pytest.skip("Set RUN_EXTERNAL=1 to enable external-call tests.")
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is required for external interpret tests.")

    import main
    from fastapi.testclient import TestClient

    with TestClient(main.app) as test_client:
        response = test_client.post(
            "/api/interpret",
            json={
                "thread_id": "ext-thread",
                "drug_name": "Aspirin",
                "section_name": "usage",
                "section_content": "Pain relief",
            },
        )
        assert response.status_code == 200
