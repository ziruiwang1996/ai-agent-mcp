#!/usr/bin/env python3
"""
Test client for the Gemini MCP Chatbot API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_available_tools():
    """Test the tools endpoint."""
    print("🛠️  Testing available tools...")
    response = requests.get(f"{BASE_URL}/tools")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Tools available: {data['tools_count']}")
    for tool in data['tools']:
        print(f"  - {tool['name']}: {tool['description']}")
    print()

def test_simple_chat():
    """Test a simple chat interaction."""
    print("💬 Testing simple chat...")
    
    chat_data = {
        "query": "Hello! What can you help me with?"
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Thread ID: {data['thread_id']}")
    print(f"Response: {data['response']}")
    print()
    
    return data['thread_id']

def test_chat_with_tools(thread_id=None):
    """Test chat with MCP tools."""
    print("🔬 Testing chat with MCP tools...")
    
    chat_data = {
        "query": "Can you search for recent papers about machine learning in drug discovery?",
        "thread_id": thread_id
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Thread ID: {data['thread_id']}")
    print(f"Response: {data['response'][:500]}...")  # Truncate long responses
    print()
    
    return data['thread_id']

def test_streaming_chat():
    """Test streaming chat."""
    print("📡 Testing streaming chat...")
    
    chat_data = {
        "query": "Explain how CRISPR gene editing works"
    }
    
    response = requests.post(f"{BASE_URL}/chat/stream", json=chat_data, stream=True)
    print(f"Status: {response.status_code}")
    
    print("Streaming response:")
    for line in response.iter_lines():
        if line:
            line_text = line.decode('utf-8')
            if line_text.startswith('data: '):
                try:
                    data = json.loads(line_text[6:])  # Remove 'data: ' prefix
                    if data['type'] == 'thread_id':
                        print(f"Thread ID: {data['thread_id']}")
                    elif data['type'] == 'content':
                        print(data['content'], end='', flush=True)
                    elif data['type'] == 'done':
                        print("\n✅ Stream completed")
                        break
                    elif data['type'] == 'error':
                        print(f"\n❌ Error: {data['content']}")
                        break
                except json.JSONDecodeError:
                    continue
    print()

def test_reset_chat(thread_id):
    """Test resetting a chat thread."""
    print("🔄 Testing chat reset...")
    
    reset_data = {"thread_id": thread_id}
    response = requests.post(f"{BASE_URL}/chat/reset", json=reset_data)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"New Thread ID: {data['thread_id']}")
    print(f"Message: {data['message']}")
    print()

def main():
    """Run all tests."""
    print("🚀 Starting API tests for Gemini MCP Chatbot\n")
    
    try:
        # Test basic endpoints
        test_health_check()
        test_available_tools()
        
        # Test chat functionality
        thread_id = test_simple_chat()
        thread_id = test_chat_with_tools(thread_id)
        
        # Test streaming
        test_streaming_chat()
        
        # Test reset
        test_reset_chat(thread_id)
        
        print("✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    main()