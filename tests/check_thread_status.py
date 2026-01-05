#!/usr/bin/env python3
"""
Quick script to check thread status and documents.
Run this to see what threads exist and if they have documents.
"""

import requests
import sys

API_BASE_URL = "http://localhost:8000"

def check_thread_documents(thread_id):
    """Check if a thread has documents."""
    print(f"\n{'='*70}")
    print(f"Checking Thread: {thread_id}")
    print(f"{'='*70}")
    
    # Check documents
    response = requests.get(f"{API_BASE_URL}/documents/list/{thread_id}")
    
    if response.status_code == 200:
        data = response.json()
        count = data.get('count', 0)
        docs = data.get('documents', [])
        
        print(f"\nDocuments in thread: {count}")
        
        if count > 0:
            for i, doc in enumerate(docs, 1):
                print(f"\n  Document {i}:")
                print(f"    Filename: {doc.get('filename')}")
                print(f"    Chunks: {doc.get('num_chunks')}")
                print(f"    Pages: {doc.get('num_pages')}")
                print(f"    Type: {doc.get('file_type')}")
                print(f"    Uploaded: {doc.get('upload_time')}")
        else:
            print(f"\nNo documents found in this thread!")
            print(f"You need to upload a document first using:")
            print(f"POST {API_BASE_URL}/documents/upload?thread_id={thread_id}")
    else:
        print(f"\nError checking documents: {response.status_code}")
        print(f"     {response.text}")
    
    print(f"\n{'='*70}\n")

def test_query_with_thread(thread_id, query):
    """Test a query with a specific thread_id."""
    print(f"\n{'='*70}")
    print(f"Testing Query with Thread")
    print(f"{'='*70}")
    print(f"Thread ID: {thread_id}")
    print(f"Query: {query}")
    print(f"\nSending request...")
    
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "query": query,
            "thread_id": thread_id
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse received:")
        print(f"Thread ID: {result.get('thread_id')}")
        print(f"\nAnswer:")
        print(f"   {result.get('response')[:500]}...")
    else:
        print(f"\nError: {response.status_code}")
        print(f"   {response.text}")
    
    print(f"\n{'='*70}\n")

def main():
    print(f"\n{'='*70}")
    print("THREAD STATUS CHECKER")
    print(f"{'='*70}\n")
    
    if len(sys.argv) > 1:
        # Thread ID provided as argument
        thread_id = sys.argv[1]
        check_thread_documents(thread_id)
        
        # Ask if user wants to test a query
        print("Would you like to test a query with this thread? (y/n): ", end='')
        if input().strip().lower() == 'y':
            print("Enter your query: ", end='')
            query = input().strip()
            if query:
                test_query_with_thread(thread_id, query)
    else:
        print("Usage:")
        print(f"  python {sys.argv[0]} <thread_id>")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} 12345678-1234-1234-1234-123456789abc")
        print()
        print("Tip: The thread_id is returned when you:")
        print("   1. Upload a document")
        print("   2. Send your first chat message")
        print()
        print("To find your thread_id:")
        print("   - Check the response from document upload")
        print("   - Check the response from your chat messages")
        print("   - Look in the server logs for 'Created new thread: ...'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
