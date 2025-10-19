# Life Science AI Agent

A production-ready AI agent framework powered by Google Gemini and LangChain, integrated with Model Context Protocol (MCP) servers for enhanced research capabilities. Features include thread-scoped RAG (Retrieval Augmented Generation), document processing, and bounded thread caching for memory management.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Thread & Memory Management](#thread--memory-management)
- [RAG System](#rag-system)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

---

## Features

### 🤖 **AI Capabilities**
- **Gemini 2.5 Flash** integration via LangChain
- **Agentic tool use** - Model autonomously decides when to use tools
- **Streaming responses** for real-time interaction
- **Thread-based conversation** history with context preservation

### 🔧 **MCP Tool Integration**
- **arXiv Server**: Search and retrieve academic papers
- **OpenFDA Server**: Query drug and device data
- **ClinicalTrials Server**: Access clinical trial information
- **PDB Server**: Search protein structure database
- **Extensible**: Easy to add more MCP servers

### 📚 **RAG (Retrieval Augmented Generation)**
- **Thread-scoped document storage** - Each conversation has isolated documents
- **Smart retrieval decision** - Only retrieves when query is document-related
- **Multi-format support** - PDF, TXT, DOCX, Markdown
- **Automatic cleanup** - Documents removed when threads expire

### 🧠 **Memory Management**
- **Bounded Thread Cache** - LRU eviction prevents memory leaks
- **Configurable capacity** - Adjust based on server resources
- **Automatic cleanup callbacks** - Documents cleaned up on thread eviction
- **Production-ready** for anonymous/unauthenticated environments

### 🌐 **FastAPI Backend**
- **RESTful API** with auto-generated docs
- **CORS support** for web frontends
- **Streaming endpoints** via Server-Sent Events
- **Health monitoring** and statistics endpoints

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                       │
│                    (main.py, routers/)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │  Chat   │  │Document │  │ Health  │
    │ Router  │  │ Router  │  │ Router  │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
         ┌────────────────────────┐
         │  GeminiMCPChatbot      │
         │  (langchain_client.py) │
         └────┬───────────┬───────┘
              │           │
      ┌───────▼───┐   ┌───▼────────┐
      │ LangGraph │   │ MCP Client │
      │ Workflow  │   │  (Tools)   │
      └───┬───────┘   └───┬────────┘
          │               │
      ┌───▼───────┐   ┌───▼─────────┐
      │  Gemini   │   │ MCP Servers │
      │   Model   │   │ (4 servers) │
      └───────────┘   └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          ┌──────┐   ┌──────┐   ┌──────┐
          │arXiv │   │ FDA  │   │ PDB  │
          └──────┘   └──────┘   └──────┘
```

### Project Structure

```
agent-server/
├── main.py                          # FastAPI app entry point
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker configuration
├── docker-entrypoint.sh            # Container startup script
│
├── client/
│   ├── langchain_client.py         # GeminiMCPChatbot (core logic)
│   ├── server_config.json          # MCP server configuration (dev)
│   ├── server_config_production.json
│   └── utils.py                    # Helper functions
│
├── host_app/
│   ├── bounded_thread_cache.py     # LRU cache for threads
│   └── routers/
│       ├── chat.py                 # Chat endpoints
│       └── document.py             # Document upload endpoints
│
├── mcp-server/
│   ├── arxiv_server.py             # Academic paper search
│   ├── openfda_server.py           # FDA drug data
│   ├── clinicaltrials_server.py   # Clinical trial data
│   └── pdb_server.py               # Protein structure data
│
└── arxiv_papers/                   # Paper cache directory
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Google Gemini API Key
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd agent-server
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv server-env
   source server-env/bin/activate  # On Windows: server-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file:
   ```env
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

5. **Run the server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Verify it's working**
   
   Open http://localhost:8000/docs for interactive API documentation

### Docker Deployment

```bash
# Build the image
docker build -t ai-agent:latest .

# Run the container
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key ai-agent:latest
```

---

## API Documentation

### Base URL
- **Local**: `http://localhost:8000`
- **Docker**: `http://localhost:8000`

### Core Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "tools_available": 7,
  "model": "gemini-2.5-flash",
  "thread_cache": {
    "current_threads": 45,
    "max_threads": 100,
    "utilization": "45.0%"
  }
}
```

#### 2. Available Tools
```http
GET /tools
```

**Response:**
```json
{
  "tools_count": 7,
  "tools": [
    {
      "name": "search_papers",
      "description": "Search for papers on arXiv"
    },
    {
      "name": "search_drug",
      "description": "Search FDA drug database"
    }
  ]
}
```

#### 3. Simple Chat
```http
POST /chat
Content-Type: application/json

{
  "query": "Find papers about CRISPR gene editing",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "response": "I found several papers about CRISPR...",
  "thread_id": "abc123-def456-..."
}
```

#### 4. Streaming Chat
```http
POST /chat/stream
Content-Type: application/json

{
  "query": "Explain machine learning in drug discovery",
  "thread_id": "optional-thread-id"
}
```

**Response:** (Server-Sent Events)
```
data: {"thread_id": "abc123", "type": "thread_id"}
data: {"content": "Machine ", "type": "content"}
data: {"content": "learning ", "type": "content"}
...
data: {"type": "done"}
```

#### 5. Upload Document
```http
POST /document/upload
Content-Type: multipart/form-data

file: <file>
thread_id: abc123-def456-...
```

**Response:**
```json
{
  "filename": "research_paper.pdf",
  "num_chunks": 42,
  "num_pages": 10,
  "file_size": 245760,
  "file_type": "PDF",
  "thread_id": "abc123-def456-..."
}
```

#### 6. List Documents
```http
GET /document/list?thread_id=abc123-def456-...
```

#### 7. Clear Documents
```http
POST /document/clear
Content-Type: application/json

{
  "thread_id": "abc123-def456-..."
}
```

#### 8. Reset Thread
```http
POST /chat/reset
Content-Type: application/json

{
  "thread_id": "abc123-def456-..."
}
```

#### 9. Thread Statistics
```http
GET /threads/stats
```

**Response:**
```json
{
  "cache_info": {
    "current_threads": 45,
    "max_threads": 100,
    "utilization": "45.0%"
  }
}
```

### Usage Examples

#### Python Client
```python
import requests

# Start a conversation
response = requests.post("http://localhost:8000/chat", json={
    "query": "Search for papers about quantum computing"
})
data = response.json()
thread_id = data['thread_id']

# Upload a document
with open("paper.pdf", "rb") as f:
    requests.post(
        f"http://localhost:8000/document/upload?thread_id={thread_id}",
        files={"file": f}
    )

# Ask about the document
response = requests.post("http://localhost:8000/chat", json={
    "query": "Summarize the uploaded paper",
    "thread_id": thread_id
})
```

#### JavaScript/Node.js
```javascript
// Simple chat
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        query: 'What can you help me with?'
    })
});
const data = await response.json();
console.log(data.response);
```

#### cURL
```bash
# Health check
curl http://localhost:8000/health

# Simple chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!"}'

# Upload document
curl -X POST "http://localhost:8000/document/upload?thread_id=abc123" \
  -F "file=@document.pdf"
```

---

## Thread & Memory Management

### Bounded Thread Cache

The system uses an LRU (Least Recently Used) cache to prevent memory leaks in anonymous environments.

#### How It Works

```python
# In main.py
from host_app.bounded_thread_cache import BoundedThreadCache

def cleanup_thread_resources(thread_id: str):
    """Called when thread is evicted."""
    chatbot.clear_thread_documents(thread_id)

thread_configs = BoundedThreadCache(
    max_threads=100,
    cleanup_callback=cleanup_thread_resources
)
```

#### Key Features

1. **Fixed Capacity** - Maximum number of active threads
2. **LRU Eviction** - Oldest/least-used threads removed first
3. **Automatic Cleanup** - Documents deleted when thread evicted
4. **Statistics** - Monitor cache usage via `/threads/stats`

#### Capacity Guidelines

| Max Threads | Use Case | Memory Impact |
|-------------|----------|---------------|
| 50 | Small server, low traffic | ~10-50 MB |
| 100 | Default, balanced | ~20-100 MB |
| 200 | Medium traffic | ~40-200 MB |
| 500 | High traffic, lots of RAM | ~100-500 MB |

#### Configuration

```python
# Adjust in main.py based on your server:
thread_configs = BoundedThreadCache(
    max_threads=100,  # Change this value
    cleanup_callback=cleanup_thread_resources
)
```

#### Monitoring

```bash
# Check cache statistics
curl http://localhost:8000/threads/stats

# Watch in real-time
watch -n 5 'curl -s http://localhost:8000/threads/stats | jq'
```

### When Threads Are Evicted

```
🗑️  Evicted oldest thread: abc12345... (cache full)
✓ Cleared documents for evicted thread: abc12345...
```

**Important**: Users should save important conversation history client-side, as threads can be evicted when the cache is full.

---

## RAG System

### Overview

The RAG (Retrieval Augmented Generation) system allows the chatbot to answer questions about uploaded documents.

### Supported Formats

- **PDF** - Parsed page-by-page
- **TXT** - Plain text files
- **DOCX** - Microsoft Word documents
- **MD** - Markdown files

### How It Works

1. **Document Upload** → Split into chunks → Create embeddings → Store in vector DB
2. **Query** → Smart decision: "Is this about documents?" → Retrieve relevant chunks
3. **Response** → Inject document context into prompt → Model answers with context

### Smart Retrieval Decision

The system intelligently decides when to use RAG:

```python
# Triggers RAG:
"Tell me about my work at Company X"  # ✅ Document keywords
"Summarize the uploaded paper"        # ✅ Document reference
"What does my resume say about..."    # ✅ Personal document

# Skips RAG:
"Hello!"                              # ❌ Greeting
"Search arXiv for papers"             # ❌ Tool request
"What's the weather?"                 # ❌ No document keywords
```

### Configuration

```python
# In client/langchain_client.py
def add_document_to_thread(self, ...):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Characters per chunk
        chunk_overlap=200,    # Overlap for context
        add_start_index=True
    )
```

**Tuning Guidelines:**
- **Smaller chunks (800)** = More precise, less context
- **Larger chunks (1500)** = More context, less precise
- **More overlap (300)** = Better continuity, more storage

### Retrieval Count

```python
# In client/langchain_client.py
retrieved_docs = self.retrieve_context_for_query(thread_id, query, k=4)
```

- **k=4** - Default, balanced
- **k=6-8** - For complex documents (resumes, research papers)
- **k=2-3** - For simple documents (short memos)

### Document Keywords

The system recognizes these keywords for RAG triggering:

```python
document_keywords = [
    "document", "paper", "article", "file", "uploaded",
    "resume", "cv", "work experience", "responsibilities",
    "according to", "based on", "mentioned", "states",
    "summarize", "summary", "explain", "section", "page"
]
```

### Debugging RAG

Look for these logs:

**✅ RAG Working:**
```
📚 Thread abc12345 has documents in vector store
🔍 Query: 'tell me about my resume...' (should_use_rag=True)
✓ RAG: Retrieved 6 relevant document chunks
✓ RAG: Augmented prompt with 6 document excerpts
```

**⚠️ RAG Issues:**
```
⚠️ Vector store exists for thread abc12345 but store is empty
⚠️ RAG: No relevant documents retrieved despite should_retrieve=True
ℹ️ RAG: Skipped retrieval (query not deemed document-related)
```

### Empty Vector Store Check

The system detects when a vector store exists but has no documents:

```python
# Two-level checking:
# 1. Check internal store attribute
if hasattr(vector_store, 'store') and len(vector_store.store) == 0:
    return False

# 2. Check document metadata
if thread_id not in self.thread_documents or len(self.thread_documents[thread_id]) == 0:
    return False
```

### Common Issues

#### Issue 1: "I cannot access documents"
**Solution**: Make sure you're using the same `thread_id` for upload and chat.

```python
# ✅ Correct
thread_id = "abc123"
upload_document(thread_id, file)
chat(thread_id, "Tell me about the document")

# ❌ Wrong
upload_document("thread-1", file)
chat("thread-2", "Tell me about the document")  # Different thread!
```

#### Issue 2: No relevant documents retrieved
**Solution**: Use more specific keywords from your document.

```python
# Instead of:
"Tell me about the company"  # Too vague

# Try:
"What does my resume say about my work at Company X?"  # Specific
```

#### Issue 3: RAG not triggering
**Solution**: Use document-related keywords.

```python
# Instead of:
"What did I do?"  # No document keywords

# Try:
"Based on my uploaded resume, what did I do?"  # Has keywords
```

---

## Deployment

### Local Development

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

1. **Build the image:**
   ```bash
   docker build -t ai-agent:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 \
     -e GEMINI_API_KEY=your_key \
     ai-agent:latest
   ```

### Docker Hub + Render

1. **Build for multiple platforms:**
   ```bash
   docker buildx create --name mybuilder --use
   docker buildx build \
     --platform linux/amd64,linux/arm64 \
     -t yourusername/ai-agent:latest \
     --push .
   ```

2. **Deploy on Render:**
   - Create new Web Service
   - Select "Deploy existing image from registry"
   - Image: `yourusername/ai-agent:latest`
   - Add environment variable: `GEMINI_API_KEY`
   - Deploy!

3. **Update deployment:**
   ```bash
   docker build -t yourusername/ai-agent:latest .
   docker push yourusername/ai-agent:latest
   # Render will auto-deploy or manually trigger
   ```

### Production Considerations

- **Environment Variables**: Use secrets management
- **CORS**: Restrict origins in production
- **Rate Limiting**: Add rate limiting middleware
- **Authentication**: Implement user auth
- **Monitoring**: Add logging and metrics
- **HTTPS**: Use TLS/SSL certificates
- **Scaling**: Consider horizontal scaling

---

## Troubleshooting

### Circular Import Error

**Symptom:**
```
ImportError: cannot import name 'chatbot' from partially initialized module 'main'
```

**Solution:** Use dependency injection pattern:

```python
# In routers/chat.py
chatbot = None
thread_configs = None

def set_dependencies(chatbot_instance, thread_configs_instance):
    global chatbot, thread_configs
    chatbot = chatbot_instance
    thread_configs = thread_configs_instance

# In main.py (after creating objects)
from host_app.routers import chat, document

chat.set_dependencies(chatbot, thread_configs)
document.set_dependencies(chatbot)

app.include_router(chat.router)
app.include_router(document.router)
```

### Infinite Tool Calling Loop

**Symptom:** Server hangs, logs show repeated `CallToolRequest`.

**Solution:** Add recursion limit:

```python
# In routers/chat.py
config["recursion_limit"] = 25

output = await asyncio.wait_for(
    chatbot.app.ainvoke({"messages": [...]}, config),
    timeout=120.0
)
```

### Thread ID Issues

**Symptom:** `thread_id=None` in logs, documents not found.

**Solution:** Always use the same thread_id for upload and chat:

```python
# 1. Create or get thread_id
response = requests.post("http://localhost:8000/chat", json={"query": "hi"})
thread_id = response.json()['thread_id']

# 2. Upload with same thread_id
requests.post(
    f"http://localhost:8000/document/upload?thread_id={thread_id}",
    files={"file": open("doc.pdf", "rb")}
)

# 3. Chat with same thread_id
requests.post("http://localhost:8000/chat", json={
    "query": "Tell me about the document",
    "thread_id": thread_id  # SAME ID!
})
```

### MCP Servers Not Starting

**Symptom:** `tools_available: 0` in health check.

**Solution:**

1. Check server configuration:
   ```python
   # client/server_config.json
   {
     "mcpServers": {
       "arxiv_server": {
         "command": "python3",
         "args": ["mcp-server/arxiv_server.py"]  # Correct path?
       }
     }
   }
   ```

2. Verify MCP server files exist:
   ```bash
   ls -la mcp-server/*.py
   ```

3. Test individual server:
   ```bash
   python3 mcp-server/arxiv_server.py
   ```

### Docker Build Issues

**Symptom:** Build fails or image doesn't run.

**Solutions:**

```bash
# Clear cache and rebuild
docker build --no-cache -t ai-agent:latest .

# For multi-platform (Apple Silicon → Cloud)
docker buildx build \
  --platform linux/amd64 \
  -t ai-agent:latest \
  --load .

# Check logs
docker logs <container-id>
```

### Memory Issues

**Symptom:** Server runs out of memory, crashes.

**Solutions:**

1. Reduce thread cache size:
   ```python
   thread_configs = BoundedThreadCache(max_threads=50)  # Lower
   ```

2. Monitor cache utilization:
   ```bash
   watch -n 5 'curl -s http://localhost:8000/threads/stats'
   ```

3. Implement aggressive cleanup:
   ```python
   # Clear threads older than 30 minutes
   # (Custom implementation needed)
   ```

---

## Development Notes

### Debugging Configuration Passing

If `thread_id` is not reaching `call_model`:

```python
# Add debug logging
print(f"Config being passed: {config}")
print(f"Config type: {type(config)}")
print(f"Config keys: {config.keys() if hasattr(config, 'keys') else 'N/A'}")
```

Look for:
- Config type (dict vs RunnableConfig)
- Presence of "configurable" key
- thread_id value

### Architecture Decisions

#### Why Bounded Cache?
- Anonymous users create threads indefinitely
- Prevents memory leaks
- LRU eviction is fair and predictable

#### Why Thread-Scoped RAG?
- User privacy (documents isolated per thread)
- Automatic cleanup (when thread evicted)
- No cross-user document leakage

#### Why Dependency Injection for Routers?
- Breaks circular import
- Testable in isolation
- Follows SOLID principles

#### Why Smart RAG Triggering?
- Saves ~100-300ms per query
- Prevents unnecessary embeddings
- Better UX (faster non-RAG responses)

### Testing

```bash
# Health check
curl http://localhost:8000/health

# Simple chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!"}'

# Upload document
curl -X POST "http://localhost:8000/document/upload?thread_id=test-123" \
  -F "file=@test.pdf"

# Test RAG
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the uploaded document",
    "thread_id": "test-123"
  }'
```

### Configuration Files

- **`server_config.json`** - Development (local paths)
- **`server_config_production.json`** - Production (Docker paths)
- **`.env`** - API keys and secrets (never commit!)

### Performance Tuning

**Thread Cache:**
- More threads = More memory, better UX
- Fewer threads = Less memory, frequent evictions

**RAG:**
- Larger chunks = More context, less precise
- More chunks (k) = Better recall, slower
- More overlap = Better continuity, more storage

**Model:**
- `gemini-2.5-flash` - Fast, cheap, good quality
- `gemini-2.0-pro` - Slower, more expensive, better quality

---

## License

[Your License Here]

## Contributors

[Your Name/Team]

## Acknowledgments

- **LangChain** - AI framework
- **LangGraph** - Workflow orchestration
- **FastAPI** - Web framework
- **Google Gemini** - LLM
- **MCP (Model Context Protocol)** - Tool integration standard

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [Your repo URL]
- Documentation: [Your docs URL]
- Email: [Your email]

---

**Built with ❤️ for life science research**
