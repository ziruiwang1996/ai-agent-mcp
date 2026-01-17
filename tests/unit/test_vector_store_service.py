from __future__ import annotations

import pytest
from langchain_core.documents import Document

from services.vector_store_service import VectorStoreService


class _DummyEmbeddings:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


class _DummyVectorStore:
    def __init__(self):
        self.added: list[Document] = []

    def add_documents(self, documents):
        self.added.extend(documents)

    def similarity_search_with_score(self, query, k=4):
        return [(Document(page_content="doc"), 0.9)]

    def similarity_search(self, query, k=4):
        return [Document(page_content="doc")]


class _DummyVectorStoreNoScore:
    def similarity_search(self, query, k=4):
        return [Document(page_content="doc")]


def test_vector_store_creation_and_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "services.vector_store_service.GoogleGenerativeAIEmbeddings",
        lambda model: _DummyEmbeddings(),
    )
    vs = VectorStoreService()
    thread_id = "test-thread"
    assert not vs.is_thread_has_vector_store(thread_id)

    vs.get_or_create_vector_store(thread_id)
    assert vs.is_thread_has_vector_store(thread_id)
    assert vs.is_vector_store_empty(thread_id)

    vs.clear_thread_documents(thread_id)
    assert not vs.is_thread_has_vector_store(thread_id)


def test_add_document_to_vector_store_records_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(
        "services.vector_store_service.GoogleGenerativeAIEmbeddings",
        lambda model: _DummyEmbeddings(),
    )

    class DummyLoader:
        def __init__(self, path, encoding=None):
            self.path = path

        def load(self):
            return [Document(page_content="hello")]

    class DummySplitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def split_documents(self, docs):
            return docs

    vs = VectorStoreService()
    dummy_store = _DummyVectorStore()
    monkeypatch.setattr(vs, "get_or_create_vector_store", lambda _thread_id: dummy_store)
    monkeypatch.setattr("services.vector_store_service.TextLoader", DummyLoader)
    monkeypatch.setattr("services.vector_store_service.RecursiveCharacterTextSplitter", DummySplitter)

    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello", encoding="utf-8")
    metadata = vs.add_document_to_vector_store("thread-1", str(file_path), "doc.txt")

    assert metadata["filename"] == "doc.txt"
    assert metadata["num_chunks"] == 1
    assert metadata["file_type"] == "TXT"
    assert dummy_store.added


def test_add_document_to_vector_store_rejects_extension(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(
        "services.vector_store_service.GoogleGenerativeAIEmbeddings",
        lambda model: _DummyEmbeddings(),
    )
    vs = VectorStoreService()
    file_path = tmp_path / "doc.exe"
    file_path.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        vs.add_document_to_vector_store("thread-1", str(file_path), "doc.exe")


def test_retrieve_context_for_query_with_scores(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "services.vector_store_service.GoogleGenerativeAIEmbeddings",
        lambda model: _DummyEmbeddings(),
    )
    vs = VectorStoreService()
    thread_id = "thread-1"
    vs.thread_vector_stores[thread_id] = _DummyVectorStore()

    results = vs.retrieve_context_for_query(thread_id, "query", k=1)
    assert len(results) == 1
    assert results[0][1] == 0.9


def test_retrieve_context_for_query_without_scores(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "services.vector_store_service.GoogleGenerativeAIEmbeddings",
        lambda model: _DummyEmbeddings(),
    )
    vs = VectorStoreService()
    thread_id = "thread-2"
    vs.thread_vector_stores[thread_id] = _DummyVectorStoreNoScore()

    results = vs.retrieve_context_for_query(thread_id, "query", k=1)
    assert len(results) == 1
    assert results[0][1] == 1.0
