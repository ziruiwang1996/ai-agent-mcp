# Gemini MCP Chatbot API Documentation

## Overview

Your chatbot is now accessible via a REST API! The FastAPI server provides multiple endpoints to interact with your LangChain-based Gemini chatbot that integrates with MCP (Model Context Protocol) servers.

## 🚀 Starting the Server

```bash
cd /Users/ziruiwang/Projects/chatbot-project/agent-server
source server-env/bin/activate
python3 -m uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at: `http://localhost:8000`

## 📚 API Endpoints

### 1. Health Check
**GET** `/health`

Returns server status and tool information.

**Response:**
```json
{
  "status": "healthy",
  "tools_available": 7,
  "model": "gemini-2.5-flash"
}
```

### 2. Available Tools
**GET** `/tools`

Lists all available MCP tools.

**Response:**
```json
{
  "tools_count": 7,
  "tools": [
    {
      "name": "search_papers",
      "description": "Search for academic papers"
    },
    {
      "name": "search_drug",
      "description": "Search drug information"
    }
  ]
}
```

### 3. Simple Chat
**POST** `/chat`

Send a message and get a complete response.

**Request:**
```json
{
  "query": "Hello! What can you help me with?",
  "thread_id": "optional-existing-thread-id"
}
```

**Response:**
```json
{
  "response": "Hello! I'm a research assistant...",
  "thread_id": "ae885e93-8278-4e90-8876-fc6c8423ba9c"
}
```

### 4. Streaming Chat
**POST** `/chat/stream`

Get real-time streaming responses (Server-Sent Events).

**Request:**
```json
{
  "query": "Explain CRISPR gene editing",
  "thread_id": "optional-existing-thread-id"
}
```

**Response:** (Stream of events)
```
data: {"thread_id": "abc123", "type": "thread_id"}
data: {"content": "CRISPR ", "type": "content"}
data: {"content": "is ", "type": "content"}
data: {"content": "a ", "type": "content"}
...
data: {"type": "done"}
```

### 5. Reset Chat Thread
**POST** `/chat/reset`

Reset conversation history and create a new thread.

**Request:**
```json
{
  "thread_id": "existing-thread-id"
}
```

**Response:**
```json
{
  "thread_id": "new-thread-id",
  "message": "Chat thread reset"
}
```

## 💻 Usage Examples

### Python Example

```python
import requests

# Simple chat
response = requests.post("http://localhost:8000/chat", json={
    "query": "Search for papers about machine learning in drug discovery"
})
data = response.json()
print(f"Response: {data['response']}")
print(f"Thread ID: {data['thread_id']}")

# Continue conversation
response = requests.post("http://localhost:8000/chat", json={
    "query": "Can you summarize the first paper?",
    "thread_id": data['thread_id']  # Use same thread
})
```

### JavaScript/Node.js Example

```javascript
// Simple chat
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        query: 'What are the latest developments in gene therapy?'
    })
});
const data = await response.json();
console.log('Response:', data.response);
console.log('Thread ID:', data.thread_id);
```

### cURL Example

```bash
# Simple chat
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello! What can you help me with?"}'

# Get available tools
curl "http://localhost:8000/tools"

# Health check
curl "http://localhost:8000/health"
```

## 🔧 Features

### ✅ **What's Implemented:**

1. **Thread Management**: Conversations maintain context across multiple messages
2. **MCP Tool Integration**: Access to 7 research tools (arXiv, FDA, clinical trials, PDB, etc.)
3. **Streaming Responses**: Real-time response streaming for better UX
4. **Error Handling**: Robust error handling with meaningful messages
5. **Health Monitoring**: Health check and tool status endpoints
6. **CORS Support**: Cross-origin requests enabled for web apps

### 🛠️ **Available MCP Tools:**

- **search_papers**: Search academic papers on arXiv
- **extract_info**: Extract drug information from FDA database
- **search_drug**: Search drug details
- **search_clinical_trials**: Find clinical trials
- **search_pdb_ids**: Search protein structure database
- And more...

## 🌐 Integration Ideas

### Frontend Applications:
- **React/Vue/Angular**: Use streaming endpoints for real-time chat
- **Mobile Apps**: REST API works with any HTTP client
- **Streamlit**: Easy integration with requests library
- **Jupyter Notebooks**: Direct API calls for research workflows

### Example Frontend Integration:

```javascript
// React component example
const [messages, setMessages] = useState([]);
const [currentThread, setCurrentThread] = useState(null);

const sendMessage = async (query) => {
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            query, 
            thread_id: currentThread 
        })
    });
    
    const data = await response.json();
    setCurrentThread(data.thread_id);
    setMessages(prev => [...prev, 
        { role: 'user', content: query },
        { role: 'assistant', content: data.response }
    ]);
};
```

## 🔒 Security Considerations

For production deployment:
1. Add authentication/authorization
2. Rate limiting
3. Input validation
4. HTTPS encryption
5. Environment variable management
6. Restrict CORS origins

## 📈 Next Steps

1. **Test the API**: Use the test client (`python3 test_api.py`)
2. **Build a Frontend**: Create a web or mobile interface
3. **Add Authentication**: Implement user management
4. **Deploy**: Host on cloud platforms (AWS, GCP, Azure)
5. **Monitor**: Add logging and analytics

Your chatbot is now ready to be accessed programmatically! 🎉