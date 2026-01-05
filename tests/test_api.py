#!/usr/bin/env python3
"""
Test client for the Gemini MCP Chatbot API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_label_interpretation():
    data = {
        "drug_name": "NAPROXEN",
        "section": "indications_and_usage",
        "content": """Naproxen tablets and naproxen sodium tablets are indicated for: the relief of the signs and symptoms of: 
        • rheumatoid arthritis • osteoarthritis • ankylosing spondylitis • Polyarticular Juvenile Idiopathic Arthritis Naproxen 
        tablets and naproxen sodium tablets are also indicated for: the relief of signs and symptoms of: • tendonitis • bursitis 
        • acute gout the management of: • pain • primary dysmenorrhea Naproxen tablets and naproxen sodium tablets are non-steroidal 
        anti-inflammatory drugs indicated for: the relief of the signs and symptoms of: • rheumatoid arthritis • osteoarthritis • 
        ankylosing spondylitis • polyarticular juvenile idiopathic arthritis Naproxen tablets and naproxen sodium tablets are also 
        indicated for: the relief of signs and symptoms of: • tendonitis • bursitis • acute gout the management of: • pain • primary dysmenorrhea"""
    }
    print("🧪 Testing label interpretation endpoint...")
    response = requests.post(f"{BASE_URL}/interpretation", json=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Drug Name: {result['drug_name']}")
        print(f"Section: {result['section']}")
        print(f"Interpretation: {result['interpretation'][:500]}...")  # Truncate long interpretations
    else:
        print(f"Error: {response.text}")


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
    agents = data.get("agents", {})
    for agent_name, info in agents.items():
        print(f"Agent: {agent_name} (initialized={info.get('initialized')})")
        for tool in info.get("tools", []):
            print(f"  - {tool.get('name')}: {tool.get('description')}")
    print()


def initialize_chat():
    print("⚙️  Initializing chat agent...")
    response = requests.post(f"{BASE_URL}/chat/initialize")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
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

        # Chat must be initialized on-demand
        initialize_chat()
        
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
    # main()
    test_label_interpretation()