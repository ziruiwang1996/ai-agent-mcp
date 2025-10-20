# Life Science AI Agent

A production-ready AI agent powered by Google Gemini and LangChain, featuring Model Context Protocol (MCP) servers for research capabilities, thread-scoped RAG (Retrieval Augmented Generation), and intelligent memory management.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [RAG System](#rag-system)
- [Memory Management](#memory-management)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Gemini 2.5 Flash** with agentic tool use and streaming responses
- **MCP Servers**: 4 specialized servers with 7 tools total
  - arXiv: Search papers, get available folders, retrieve topic papers
  - OpenFDA: Search drug and device information
  - ClinicalTrials: Search clinical trial data
  - PDB: Search protein structures, extract data, find similar sequences
- **Thread-scoped RAG**: Upload and query documents per conversation
- **Smart memory management**: LRU cache with configurable capacity (default: 50 threads)
- **Multi-format support**: PDF, TXT, DOCX, Markdown
- **RESTful API**: FastAPI with auto-generated docs and CORS support

---

## Project Structure

```
agent-server/
├── main.py                          # FastAPI app entry point & server configuration
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker container configuration
├── docker-entrypoint.sh            # Container startup script
├── render.yaml                     # Render deployment configuration
├── .env                            # Environment variables (create this)
│
├── client/                         # LangChain chatbot core
│   ├── langchain_client.py         # GeminiMCPChatbot class (main logic)
│   ├── server_config.json          # MCP server config (development)
│   ├── server_config_production.json # MCP server config (production/Docker)
│   └── utils.py                    # Helper functions
│
├── host_app/                       # FastAPI application
│   ├── bounded_thread_cache.py     # LRU cache for thread management
│   └── routers/                    # API endpoints
│       ├── chat.py                 # Chat endpoints (/chat, /chat/stream, /chat/reset)
│       └── document.py             # Document endpoints (/documents/*)
│
├── mcp-server/                     # Model Context Protocol servers
│   ├── arxiv_server.py             # Academic paper search (3 tools)
│   ├── openfda_server.py           # FDA drug database (1 tool)
│   ├── clinicaltrials_server.py   # Clinical trials data (1 tool)
│   └── pdb_server.py               # Protein structure database (3 tools)
│
├── arxiv_papers/                   # Cached arXiv papers (created at runtime)
│
├── server-env/                     # Virtual environment (create with venv)
│
└── test/                           # Test files
    ├── test_api.py
    ├── test_rag.py
    ├── test_rag_debug.py
    └── check_thread_status.py
```

### Key Files

- **`main.py`**: Initializes FastAPI app, chatbot, thread cache, and routers
- **`client/langchain_client.py`**: Core chatbot with MCP integration, RAG, and LangGraph workflow
- **`host_app/bounded_thread_cache.py`**: LRU cache preventing memory leaks
- **`host_app/routers/chat.py`**: Chat endpoints with streaming support
- **`host_app/routers/document.py`**: Document upload/management for RAG
- **`mcp-server/*.py`**: Independent MCP servers providing specialized tools

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

## API Reference

**Base URL:** `http://localhost:8000`

### Core Endpoints

#### 0. Root
```http
GET /
```

**Response:**
```json
{
  "message": "Hello, this is the Life Science Research Agent server with MCP tools."
}
```

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

**Note:** You can use either `"query"` or `"message"` field for the user input.

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
POST /documents/upload
Content-Type: multipart/form-data

file: <file>
thread_id: abc123-def456-...
```

**Response:**
```json
{
  "message": "Document uploaded successfully",
  "document": {
    "filename": "research_paper.pdf",
    "num_chunks": 42,
    "num_pages": 10,
    "file_size": 245760,
    "file_type": "PDF"
  },
  "thread_id": "abc123-def456-..."
}
```

#### 6. List Documents
```http
GET /documents/list/{thread_id}
```

**Response:**
```json
{
  "thread_id": "abc123-def456-...",
  "documents": [...],
  "count": 2
}
```

#### 7. Clear Documents
```http
DELETE /documents/clear/{thread_id}
```

**Response:**
```json
{
  "thread_id": "abc123-def456-...",
  "message": "Documents cleared successfully",
  "documents_removed": 2
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

**Response:**
```json
{
  "thread_id": "new-abc123-...",
  "message": "Chat thread reset",
  "documents_cleared": true,
  "cache_stats": {...}
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
    "current_threads": 23,
    "max_threads": 50,
    "utilization": "46.0%"
  },
  "message": "Thread cache statistics"
}
```

### Usage Examples

**Python:**
```python
import requests

response = requests.post("http://localhost:8000/chat", 
    json={"query": "Search for papers about quantum computing"})
thread_id = response.json()['thread_id']

# Upload a document (note: /documents/ not /document/)
with open("paper.pdf", "rb") as f:
    files = {"file": f}
    data = {"thread_id": thread_id}
    requests.post("http://localhost:8000/documents/upload", files=files, data=data)

# Ask about the document
requests.post("http://localhost:8000/chat", 
    json={"query": "Summarize the paper", "thread_id": thread_id})
```

**cURL:**
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query": "Hello!"}'
curl -X POST http://localhost:8000/documents/upload -F "file=@document.pdf" -F "thread_id=abc123"
```

---

## MCP Tools

The agent has access to 7 specialized tools across 4 MCP servers:

### arXiv Server (3 tools)
- **search_papers**: Search for academic papers by topic (returns up to 5 papers)
- **get_available_folders**: List all cached paper topics
- **get_topic_papers**: Retrieve papers for a specific cached topic

### OpenFDA Server (1 tool)
- **search_drug**: Search FDA database for drug and device information

### ClinicalTrials Server (1 tool)
- **search_clinical_trials**: Query clinical trials by condition, intervention, or other criteria

### PDB Server (3 tools)
- **search_pdb_ids**: Search for protein structures using text queries
- **extract_pdb_data**: Get detailed data for a specific PDB ID
- **search_similar_sequence**: Find protein structures with similar sequences (supports protein, DNA, RNA)

The model **autonomously decides** when to use these tools based on user queries.

---

## Memory Management

The system uses an LRU (Least Recently Used) cache to prevent memory leaks by limiting active threads.

### Configuration

Edit `main.py` to adjust capacity:

```python
thread_configs = BoundedThreadCache(
    max_threads=50,  # Current default, adjust based on server resources
    cleanup_callback=cleanup_thread_resources
)
```

**Capacity Guidelines:**

| Max Threads | Use Case | Memory |
|-------------|----------|--------|
| 50 | Default, small-medium server | ~10-50 MB |
| 100 | Medium traffic | ~20-100 MB |
| 200 | High traffic | ~40-200 MB |

### Monitoring

```bash
curl http://localhost:8000/threads/stats
watch -n 5 'curl -s http://localhost:8000/threads/stats | jq'
```

**Note:** Threads are evicted when cache is full. Save important conversations client-side.

---

## RAG System

Upload documents (PDF, TXT, DOCX, MD) and ask questions about them. Each thread has isolated document storage.

### How It Works

1. Upload → Split into chunks → Create embeddings → Store in vector DB
2. Query → Smart decision: "Is this document-related?" → Retrieve relevant chunks
3. Generate → Inject context into prompt → Model answers with document context

### Smart Retrieval

The system automatically detects document-related queries:

**Triggers RAG:**
- "Summarize the uploaded paper"
- "What does my resume say about..."
- "According to the document..."

**Skips RAG:**
- "Hello!"
- "Search arXiv for papers" (uses MCP tools instead)

### Configuration

Edit `client/langchain_client.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap for context
)

retrieved_docs = self.retrieve_context_for_query(thread_id, query, k=4)
```

**Tuning:**
- Chunk size: 800 (precise) to 1500 (more context)
- Retrieval count (k): 2-3 (simple docs) to 6-8 (complex docs)

### Common Issues

**Documents not found:**
- Ensure you use the same `thread_id` for upload and queries

**RAG not triggering:**
- Use document keywords: "summarize", "according to", "uploaded", "document"

---

## Deployment

### Local Development

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t ai-agent:latest .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key ai-agent:latest
```

### Docker Hub + Render

```bash
# Build multi-platform image
docker buildx create --name mybuilder --use
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/ai-agent:latest --push .
```

On Render:
1. Create new Web Service
2. Select "Deploy existing image from registry"
3. Image: `yourusername/ai-agent:latest`
4. Add environment variable: `GEMINI_API_KEY`

### Production Checklist

- Use secrets management for API keys
- Restrict CORS origins
- Add rate limiting
- Implement authentication
- Enable HTTPS
- Configure logging and monitoring

---

## Troubleshooting

### MCP Servers Not Starting

Check health endpoint shows `tools_available: 0`:

```bash
ls -la mcp-server/*.py  # Verify files exist
python3 mcp-server/arxiv_server.py  # Test individual server
```

Verify `client/server_config.json` paths are correct.

### Documents Not Found

Always use the same `thread_id` for upload and queries:

```python
response = requests.post("http://localhost:8000/chat", json={"query": "hi"})
thread_id = response.json()['thread_id']

# Use this thread_id for both upload and chat (note: /documents/ not /document/)
files = {"file": open("doc.pdf", "rb")}
data = {"thread_id": thread_id}
requests.post("http://localhost:8000/documents/upload", files=files, data=data)
requests.post("http://localhost:8000/chat", json={"thread_id": thread_id, "query": "..."})
```

### Server Hangs / Infinite Loop

The code already has protection with recursion limit (25) and timeout (60s) in `routers/chat.py`:

```python
config["recursion_limit"] = 25
output = await asyncio.wait_for(chatbot.app.ainvoke(...), timeout=60.0)
```

If issues persist, reduce recursion_limit to 15 or increase timeout to 90.0.

### Memory Issues

Reduce thread cache size in `main.py`:

```python
thread_configs = BoundedThreadCache(max_threads=50)
```

Monitor usage:
```bash
watch -n 5 'curl -s http://localhost:8000/threads/stats'
```

### Docker Build Fails

```bash
docker build --no-cache -t ai-agent:latest .
docker buildx build --platform linux/amd64 -t ai-agent:latest --load .  # For Apple Silicon
```

---

## License

MIT

---