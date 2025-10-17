#!/usr/bin/env python3
"""
Debug script for RAG system testing.
This helps diagnose why the RAG system might not be working properly.
"""

import requests
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_document_upload_and_query():
    """Test the full RAG flow with debugging."""
    
    print("=" * 70)
    print("RAG DEBUG TEST")
    print("=" * 70)
    
    # Step 1: Create a new thread
    print("\n1️⃣  Creating new thread...")
    response = requests.post(f"{API_BASE_URL}/threads/new")
    if response.status_code != 200:
        print(f"❌ Failed to create thread: {response.status_code}")
        return
    
    thread_id = response.json()["thread_id"]
    print(f"✅ Thread created: {thread_id}")
    
    # Step 2: Upload your resume
    print("\n2️⃣  Upload your resume...")
    resume_path = input("Enter the path to your resume (PDF/TXT/DOCX): ").strip()
    
    if not Path(resume_path).exists():
        print(f"❌ File not found: {resume_path}")
        return
    
    with open(resume_path, 'rb') as f:
        files = {'file': (Path(resume_path).name, f)}
        params = {'thread_id': thread_id}
        
        response = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            params=params
        )
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"Error: {response.text}")
        return
    
    doc_info = response.json()
    print(f"✅ Document uploaded successfully!")
    print(f"   Filename: {doc_info['document']['filename']}")
    print(f"   Chunks: {doc_info['document']['num_chunks']}")
    print(f"   Pages: {doc_info['document'].get('num_pages', 'N/A')}")
    
    # Step 3: List documents to verify
    print("\n3️⃣  Verifying document in system...")
    response = requests.get(f"{API_BASE_URL}/documents/list/{thread_id}")
    if response.status_code == 200:
        docs = response.json()
        print(f"✅ Found {docs['count']} document(s) in thread")
    
    # Step 4: Test queries
    print("\n4️⃣  Testing RAG queries...")
    print("=" * 70)
    
    test_queries = [
        "What work experience do I have at Regeneron?",
        "What are my responsibilities at Regeneron?",
        "Can you summarize my work experience?",
        "Tell me about my education background",
    ]
    
    print("\nTesting multiple queries. Watch the server logs for RAG debug info!")
    print("(Look for messages like: '📚 Thread has documents', '✓ RAG: Retrieved X chunks')\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 70)
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "query": query,  # Use 'query' field (also supports 'message')
                "thread_id": thread_id
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "No response")
            print(f"🤖 Answer: {answer[:500]}...")
            
            # Check if the answer seems document-aware
            if any(indicator in answer.lower() for indicator in 
                   ["according to", "based on", "your resume", "regeneron", "document excerpt"]):
                print("✅ Response appears to use document context!")
            else:
                print("⚠️  Response may not be using document context")
        else:
            print(f"❌ Query failed: {response.status_code}")
        
        print()
    
    print("=" * 70)
    print("DEBUG TEST COMPLETE")
    print("=" * 70)
    print("\n💡 Tips:")
    print("1. Check the server console for RAG debug messages")
    print("2. Look for: '✓ RAG: Retrieved X relevant document chunks'")
    print("3. If you see '⚠️ RAG: Skipped retrieval', the query keywords may need updating")
    print("4. If you see '⚠️ RAG: No relevant documents retrieved', check chunk size/overlap")
    print(f"\n🔗 Thread ID for further testing: {thread_id}")

if __name__ == "__main__":
    try:
        test_document_upload_and_query()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
