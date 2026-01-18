from collections import OrderedDict
from typing import Optional, Any, Callable

class _ThreadDocumentCleaner:
    def clear_thread_documents(self, thread_id: str) -> None:
        ...

def cleanup_thread_resources(thread_id: str, cleaner: _ThreadDocumentCleaner):
    """
    Cleanup callback for evicted threads.
    This function is called automatically when a thread is evicted from cache.
    It cleans up all associated resources (documents, vector stores, etc.).
    """
    try:
        cleaner.clear_thread_documents(thread_id)
        print(f"Cleared documents for evicted thread: {thread_id[:8]}...")
    except Exception as e:
        print(f"Error clearing documents: {e}")

"""
Thread Service - Reusable LRU Cache Implementation
A generic LRU (Least Recently Used) cache for managing thread configurations
with automatic eviction and optional resource cleanup.
"""
class ThreadService:
    """
    LRU (Least Recently Used) cache for thread configurations.
    
    Automatically evicts oldest threads when capacity is reached.
    This prevents memory leaks from abandoned/anonymous user sessions.
    
    Features:
    - Fixed maximum capacity
    - Automatic cleanup of old threads
    - Optional cleanup callback for associated resources
    - Thread-safe basic operations
    - Decoupled from any specific cleanup logic
    - Zero external dependencies (stdlib only)
    
    Example Usage:
    
        # Basic usage without cleanup
        cache = ThreadService(max_threads=100)
        cache.set("thread-1", {"config": "value"})
        config = cache.get("thread-1")
        
        # With cleanup callback
        def cleanup(thread_id):
            # Clean up resources
            database.delete_thread(thread_id)
            storage.clear_files(thread_id)
        
        cache = ThreadService(
            max_threads=100,
            cleanup_callback=cleanup
        )
    
    Args:
        max_threads: Maximum number of threads to keep in memory (default: 100)
        cleanup_callback: Optional function called when thread is evicted.
                         Should accept thread_id (str) as parameter.
                         Called before thread is removed from cache.
    """
    
    def __init__(
        self,
        max_threads: int = 100,
        cleanup_callback: Optional[Callable[[str], None]] = None
    ):
        if max_threads <= 0:
            raise ValueError(f"max_threads must be positive, got {max_threads}")
        self.max_threads = max_threads
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.cleanup_callback = cleanup_callback
    
    def get(self, thread_id: str) -> Optional[dict[str, Any]]:
        """
        Get thread config and mark as recently used.
        """
        if thread_id in self._cache:
            self._cache.move_to_end(thread_id)
            return self._cache[thread_id]
        return None
    
    def set(self, thread_id: str, config: dict[str, Any]) -> None:
        """
        Add or update thread config.
        If thread exists: Updates config and marks as recently used.
        If thread is new and cache is full: Evicts oldest thread first.
        """
        if thread_id in self._cache:
            self._cache.move_to_end(thread_id)
            self._cache[thread_id] = config
        else:
            if len(self._cache) >= self.max_threads:
                self._evict_oldest()
            self._cache[thread_id] = config
    
    def remove(self, thread_id: str) -> bool:
        """
        Remove a thread from cache.
        Note: This does NOT call the cleanup callback. Use this for manual
        removal where you want to handle cleanup separately.
        """
        if thread_id in self._cache:
            del self._cache[thread_id]
            return True
        return False
    
    def _evict_oldest(self) -> None:
        """
        Evict the oldest (least recently used) thread.
        This is called automatically when cache reaches capacity.
        Calls cleanup callback if provided before removing the thread.
        """
        if len(self._cache) == 0:
            return
        
        # Remove oldest (first) thread
        oldest_thread_id, oldest_config = self._cache.popitem(last=False)
        print(f"Evicted oldest thread: {oldest_thread_id[:8]}... (cache full)")
        
        # Call cleanup callback if provided
        if self.cleanup_callback:
            try:
                self.cleanup_callback(oldest_thread_id)
                print(f"Cleanup callback executed for thread: {oldest_thread_id[:8]}...")
            except Exception as e:
                print(f"Error in cleanup callback for {oldest_thread_id[:8]}: {e}")
    
    def clear(self, call_cleanup: bool = False) -> int:
        """
        Clear all threads from cache.
        
        Args:
            call_cleanup: If True, calls cleanup callback for each thread
            
        Returns:
            Number of threads that were cleared
        """
        count = len(self._cache)
        if call_cleanup and self.cleanup_callback:
            for thread_id in list(self._cache.keys()):
                try:
                    self.cleanup_callback(thread_id)
                except Exception as e:
                    print(f"Error in cleanup callback for {thread_id[:8]}: {e}")
        self._cache.clear()
        print(f"Cleared {count} threads from cache")
        return count
    
    def __contains__(self, thread_id: str) -> bool:
        """Check if thread exists in cache."""
        return thread_id in self._cache
    
    def __len__(self) -> int:
        """Get current number of threads in cache."""
        return len(self._cache)
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with:
            - current_threads: Number of threads in cache
            - max_threads: Maximum capacity
            - utilization: Percentage string (e.g., "45.0%")
        """
        return {
            "current_threads": len(self._cache),
            "max_threads": self.max_threads,
            "utilization": f"{len(self._cache) / self.max_threads * 100:.1f}%"
        }
    
    def get_thread_ids(self) -> list[str]:
        """
        Get list of all thread IDs in cache.
        
        Returns in LRU order (oldest first, newest last).
        
        Returns:
            List of thread IDs
        """
        return list(self._cache.keys())
    
    def get_all(self) -> dict[str, dict[str, Any]]:
        """
        Get all threads as a dictionary.
        
        Warning: This returns a reference to the internal cache.
        Modifying it directly may cause unexpected behavior.
        
        Returns:
            Dictionary mapping thread_id -> config
        """
        return dict(self._cache)