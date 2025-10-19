"""
Test Script for RAG Integration

This script demonstrates and tests the RAG workflow.
Run this after starting the server to verify everything works.
"""

import requests
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if server is running."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    if response.ok:
        data = response.json()
        print(f"✓ Server is healthy")
        print(f"  - Status: {data['status']}")
        print(f"  - Model: {data['model']}")
        print(f"  - Tools available: {data['tools_available']}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False


def test_create_thread():
    """Create a new chat thread."""
    print("\n" + "="*60)
    print("TEST 2: Create Thread")
    print("="*60)
    
    response = requests.post(f"{API_BASE_URL}/chat/reset", json={})
    if response.ok:
        data = response.json()
        thread_id = data['thread_id']
        print(f"✓ Thread created: {thread_id}")
        return thread_id
    else:
        print(f"✗ Thread creation failed: {response.status_code}")
        return None


def test_upload_document(thread_id, file_path):
    """Upload a test document."""
    print("\n" + "="*60)
    print("TEST 3: Upload Document")
    print("="*60)
    
    if not Path(file_path).exists():
        print(f"✗ Test file not found: {file_path}")
        print(f"  Create a test PDF file or update the path")
        return False
    
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/pdf')}
        data = {'thread_id': thread_id}
        
        response = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            data=data,
            timeout=30
        )
    
    if response.ok:
        result = response.json()
        doc = result['document']
        print(f"✓ Document uploaded successfully")
        print(f"  - Filename: {doc['filename']}")
        print(f"  - Chunks: {doc['num_chunks']}")
        print(f"  - Pages: {doc.get('num_pages', 'N/A')}")
        print(f"  - Size: {doc['file_size'] / 1024:.1f} KB")
        return True
    else:
        print(f"✗ Upload failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return False


def test_list_documents(thread_id):
    """List uploaded documents."""
    print("\n" + "="*60)
    print("TEST 4: List Documents")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/documents/list/{thread_id}")
    if response.ok:
        data = response.json()
        print(f"✓ Found {data['count']} document(s)")
        for doc in data['documents']:
            print(f"  - {doc['filename']}: {doc['num_chunks']} chunks")
        return True
    else:
        print(f"✗ List failed: {response.status_code}")
        return False


def test_chat_with_rag(thread_id, query):
    """Test chat with RAG context."""
    print("\n" + "="*60)
    print("TEST 5: Chat with RAG")
    print("="*60)
    print(f"Query: {query}")
    print("-" * 60)
    
    payload = {
        "query": query,
        "thread_id": thread_id
    }
    
    response = requests.post(
        f"{API_BASE_URL}/chat/stream",
        json=payload,
        stream=True
    )
    
    if response.ok:
        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    if data['type'] == 'content':
                        print(data['content'], end='', flush=True)
                        full_response += data['content']
                    elif data['type'] == 'done':
                        break
                except json.JSONDecodeError:
                    continue
        
        print("\n" + "-" * 60)
        print(f"✓ RAG chat completed ({len(full_response)} chars)")
        return True
    else:
        print(f"✗ Chat failed: {response.status_code}")
        return False


def test_chat_without_rag(thread_id, query):
    """Test chat without documents (baseline)."""
    print("\n" + "="*60)
    print("TEST 6: Chat without RAG (Baseline)")
    print("="*60)
    print(f"Query: {query}")
    print("-" * 60)
    
    payload = {
        "query": query,
        "thread_id": thread_id
    }
    
    response = requests.post(
        f"{API_BASE_URL}/chat/stream",
        json=payload,
        stream=True
    )
    
    if response.ok:
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    if data['type'] == 'content':
                        print(data['content'], end='', flush=True)
                    elif data['type'] == 'done':
                        break
                except json.JSONDecodeError:
                    continue
        print("\n" + "-" * 60)
        print("✓ Baseline chat completed")
        return True
    else:
        print(f"✗ Chat failed: {response.status_code}")
        return False


def test_clear_documents(thread_id):
    """Clear all documents."""
    print("\n" + "="*60)
    print("TEST 7: Clear Documents")
    print("="*60)
    
    response = requests.delete(f"{API_BASE_URL}/documents/clear/{thread_id}")
    if response.ok:
        data = response.json()
        print(f"✓ Cleared {data['documents_removed']} document(s)")
        return True
    else:
        print(f"✗ Clear failed: {response.status_code}")
        return False


def run_full_test_suite():
    """Run complete test suite."""
    print("\n" + "="*60)
    print("🚀 RAG INTEGRATION TEST SUITE")
    print("="*60)
    print("This will test the complete RAG workflow:")
    print("1. Server health")
    print("2. Thread creation")
    print("3. Document upload")
    print("4. Document listing")
    print("5. RAG-enhanced chat")
    print("6. Baseline chat")
    print("7. Document cleanup")
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Server not responding. Is it running?")
        return
    
    # Test 2: Create thread
    thread_id = test_create_thread()
    if not thread_id:
        print("\n❌ Cannot proceed without thread")
        return
    
    # Test 3 & 4: Upload and list (with RAG)
    # You need to provide a test PDF file
    test_file = "test_document.pdf"  # UPDATE THIS PATH
    
    print(f"\n📄 To test document upload, create a test PDF at: {test_file}")
    print("   Or update the 'test_file' variable in this script")
    
    if Path(test_file).exists():
        if test_upload_document(thread_id, test_file):
            test_list_documents(thread_id)
            
            # Test 5: Chat with RAG
            test_chat_with_rag(
                thread_id,
                "What is this document about? Summarize the main points."
            )
            
            # Test 7: Clear documents
            test_clear_documents(thread_id)
    else:
        print(f"\n⚠️  Skipping upload test (no test file)")
    
    # Test 6: Chat without RAG (baseline)
    test_chat_without_rag(
        thread_id,
        "What is machine learning?"
    )
    
    print("\n" + "="*60)
    print("✅ TEST SUITE COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Create a test PDF and re-run for full RAG test")
    print("2. Try uploading different file types (TXT, MD)")
    print("3. Test with multiple documents")
    print("4. Test concurrent users (multiple threads)")


if __name__ == "__main__":
    run_full_test_suite()
