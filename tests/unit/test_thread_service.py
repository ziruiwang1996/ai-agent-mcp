from __future__ import annotations

from services.thread_service import ThreadService


def test_thread_service_eviction_and_cleanup():
    cleaned: list[str] = []

    def cleanup(thread_id: str) -> None:
        cleaned.append(thread_id)

    cache = ThreadService(max_threads=2, cleanup_callback=cleanup)
    cache.set("thread-1", {"config": 1})
    cache.set("thread-2", {"config": 2})
    cache.set("thread-3", {"config": 3})

    assert "thread-1" not in cache
    assert "thread-2" in cache
    assert "thread-3" in cache
    assert cleaned == ["thread-1"]


def test_thread_service_stats_and_get():
    cache = ThreadService(max_threads=5)
    cache.set("thread-1", {"config": 1})

    assert cache.get("thread-1") == {"config": 1}
    stats = cache.get_stats()
    assert stats["current_threads"] == 1
    assert stats["max_threads"] == 5
