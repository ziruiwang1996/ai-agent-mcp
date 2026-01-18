from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


def test_interpret_returns_payload(client: TestClient):
    response = client.post(
        "/api/interpret",
        json={
            "drug_name": "Aspirin",
            "section_name": "usage",
            "section_content": "Pain relief",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["interpretation"] == "interpretation"


def test_interpret_validation_error(client: TestClient):
    response = client.post(
        "/api/interpret",
        json={
            "drug_name": "Aspirin",
            "section_name": "usage",
        },
    )
    assert response.status_code == 422


def test_evidence_returns_report(client: TestClient):
    response = client.post(
        "/api/evidence",
        json={
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


def test_evidence_validation_error(client: TestClient):
    response = client.post(
        "/api/evidence",
        json={
            "drug_set_id": "set-1",
            "drug_name": "Drug",
            "age": "30",
        },
    )
    assert response.status_code == 422


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
                "drug_name": "Aspirin",
                "section_name": "usage",
                "section_content": "Pain relief",
            },
        )
        assert response.status_code == 200
