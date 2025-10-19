from collections import OrderedDict
from typing import Optional, Dict, Any, Callable
from client.langchain_client import GeminiMCPChatbot

def cleanup_thread_resources(thread_id: str, chatbot: GeminiMCPChatbot):
    """
    Cleanup callback for evicted threads.
    
    This function is called automatically when a thread is evicted from cache.
    It cleans up all associated resources (documents, vector stores, etc.).
    
    Args:
        thread_id: ID of the thread being evicted
    """
    try:
        chatbot.clear_thread_documents(thread_id)
        print(f"Cleared documents for evicted thread: {thread_id[:8]}...")
    except Exception as e:
        print(f"Error clearing documents: {e}")

"""
Bounded Thread Cache - Reusable LRU Cache Implementation

A generic LRU (Least Recently Used) cache for managing thread configurations
with automatic eviction and optional resource cleanup.
"""
class BoundedThreadCache:
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
        cache = BoundedThreadCache(max_threads=100)
        cache.set("thread-1", {"config": "value"})
        config = cache.get("thread-1")
        
        # With cleanup callback
        def cleanup(thread_id):
            # Clean up resources
            database.delete_thread(thread_id)
            storage.clear_files(thread_id)
        
        cache = BoundedThreadCache(
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
        """
        Initialize the bounded cache.
        
        Args:
            max_threads: Maximum number of threads (must be > 0)
            cleanup_callback: Optional cleanup function
        
        Raises:
            ValueError: If max_threads <= 0
        """
        if max_threads <= 0:
            raise ValueError(f"max_threads must be positive, got {max_threads}")
        self.max_threads = max_threads
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.cleanup_callback = cleanup_callback
    
    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get thread config and mark as recently used.
        
        This implements the "recently used" part of LRU - accessing an item
        moves it to the end of the queue, making it less likely to be evicted.
        
        Args:
            thread_id: Thread ID to retrieve
            
        Returns:
            Thread config dict if found, None otherwise
        """
        if thread_id in self.cache:
            # Move to end (mark as recently used)
            self.cache.move_to_end(thread_id)
            return self.cache[thread_id]
        return None
    
    def set(self, thread_id: str, config: Dict[str, Any]) -> None:
        """
        Add or update thread config.
        
        If thread exists: Updates config and marks as recently used.
        If thread is new and cache is full: Evicts oldest thread first.
        
        Args:
            thread_id: Thread ID
            config: Configuration dictionary
        """
        if thread_id in self.cache:
            # Update existing - move to end (mark as recently used)
            self.cache.move_to_end(thread_id)
            self.cache[thread_id] = config
        else:
            # Check if we need to evict oldest thread
            if len(self.cache) >= self.max_threads:
                self._evict_oldest()
            
            # Add new thread
            self.cache[thread_id] = config
    
    def remove(self, thread_id: str) -> bool:
        """
        Remove a thread from cache.
        
        Note: This does NOT call the cleanup callback. Use this for manual
        removal where you want to handle cleanup separately.
        
        Args:
            thread_id: Thread ID to remove
            
        Returns:
            True if thread was removed, False if not found
        """
        if thread_id in self.cache:
            del self.cache[thread_id]
            return True
        return False
    
    def _evict_oldest(self) -> None:
        """
        Evict the oldest (least recently used) thread.
        
        This is called automatically when cache reaches capacity.
        Calls cleanup callback if provided before removing the thread.
        """
        if len(self.cache) == 0:
            return
        
        # Remove oldest (first) thread
        oldest_thread_id, oldest_config = self.cache.popitem(last=False)
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
        count = len(self.cache)
        if call_cleanup and self.cleanup_callback:
            for thread_id in list(self.cache.keys()):
                try:
                    self.cleanup_callback(thread_id)
                except Exception as e:
                    print(f"Error in cleanup callback for {thread_id[:8]}: {e}")
        self.cache.clear()
        print(f"Cleared {count} threads from cache")
        return count
    
    def __contains__(self, thread_id: str) -> bool:
        """
        Check if thread exists in cache.
        
        Supports: if thread_id in cache: ...
        
        Args:
            thread_id: Thread ID to check
            
        Returns:
            True if thread exists, False otherwise
        """
        return thread_id in self.cache
    
    def __len__(self) -> int:
        """
        Get current number of threads in cache.
        
        Supports: len(cache)
        
        Returns:
            Number of threads currently in cache
        """
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with:
            - current_threads: Number of threads in cache
            - max_threads: Maximum capacity
            - utilization: Percentage string (e.g., "45.0%")
        """
        return {
            "current_threads": len(self.cache),
            "max_threads": self.max_threads,
            "utilization": f"{len(self.cache) / self.max_threads * 100:.1f}%"
        }
    
    def get_thread_ids(self) -> list[str]:
        """
        Get list of all thread IDs in cache.
        
        Returns in LRU order (oldest first, newest last).
        
        Returns:
            List of thread IDs
        """
        return list(self.cache.keys())
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all threads as a dictionary.
        
        Warning: This returns a reference to the internal cache.
        Modifying it directly may cause unexpected behavior.
        
        Returns:
            Dictionary mapping thread_id -> config
        """
        return dict(self.cache)



if __name__ == "__main__":
    print("BoundedThreadCache - Example Usage\n" + "="*50)
    
    # Example 1: Basic usage
    print("\n1. Basic usage (no cleanup)")
    cache = BoundedThreadCache(max_threads=3)
    
    cache.set("thread-1", {"user": "Alice", "messages": 5})
    cache.set("thread-2", {"user": "Bob", "messages": 3})
    cache.set("thread-3", {"user": "Charlie", "messages": 7})
    
    print(f"Stats: {cache.get_stats()}")
    print(f"Get thread-2: {cache.get('thread-2')}")
    
    # This will evict thread-1 (oldest)
    cache.set("thread-4", {"user": "David", "messages": 2})
    print(f"\nAfter adding thread-4:")
    print(f"Thread IDs: {cache.get_thread_ids()}")
    print(f"thread-1 exists: {cache.get('thread-1') is not None}")
    
    # Example 2: With cleanup callback
    print("\n2. With cleanup callback")
    
    cleanup_log = []
    
    def my_cleanup(thread_id: str):
        cleanup_log.append(thread_id)
        print(f"   🧹 Cleaning up resources for {thread_id}")
    
    cache2 = BoundedThreadCache(max_threads=2, cleanup_callback=my_cleanup)
    cache2.set("session-1", {"data": "A"})
    cache2.set("session-2", {"data": "B"})
    cache2.set("session-3", {"data": "C"})  # Will evict session-1
    
    print(f"Cleanup was called for: {cleanup_log}")
    
    # Example 3: Clear all with cleanup
    print("\n3. Clear all threads")
    cache2.clear(call_cleanup=True)
    print(f"Total cleanups called: {len(cleanup_log)}")
    
    print("\n" + "="*50)
    print("Examples completed!")
