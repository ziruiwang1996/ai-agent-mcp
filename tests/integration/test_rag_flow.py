from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import document


def test_document_upload_list_and_clear(documents_client: TestClient, tmp_path: Path):
    file_path = tmp_path / "test_document.txt"
    file_path.write_text("hello world", encoding="utf-8")

    with file_path.open("rb") as handle:
        response = documents_client.post(
            "/api/documents/upload",
            files={"file": ("test_document.txt", handle)},
            data={"thread_id": "thread-1"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["thread_id"] == "thread-1"

    response = documents_client.get("/api/documents/list/thread-1")
    assert response.status_code == 200
    docs = response.json()
    assert docs["count"] == 1

    response = documents_client.delete("/api/documents/clear/thread-1")
    assert response.status_code == 200
    cleared = response.json()
    assert cleared["documents_removed"] == 1


def test_document_upload_rejects_extension(documents_client: TestClient, tmp_path: Path):
    file_path = tmp_path / "test_document.exe"
    file_path.write_text("nope", encoding="utf-8")

    with file_path.open("rb") as handle:
        response = documents_client.post(
            "/api/documents/upload",
            files={"file": ("test_document.exe", handle)},
            data={"thread_id": "thread-2"},
        )
    assert response.status_code == 400


@pytest.mark.external
def test_external_document_upload(tmp_path: Path, external_services):
    file_path = tmp_path / "external_doc.txt"
    file_path.write_text("external call", encoding="utf-8")

    app = FastAPI()
    app.include_router(document.router)
    app.state.services = external_services

    with TestClient(app) as test_client:
        with file_path.open("rb") as handle:
            response = test_client.post(
                "/api/documents/upload",
                files={"file": ("external_doc.txt", handle)},
                data={"thread_id": "ext-thread"},
            )
        assert response.status_code == 200
