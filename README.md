# Med Helper Agent Server

A FastAPI service that wraps a LangGraph-powered chat agent and a set of Model Context Protocol (MCP) servers for medication safety research. The server orchestrates Google Gemini models with purpose-built analysis prompts, thread-scoped retrieval-augmented generation (RAG), and domain-specific tooling for drug labels, adverse event reports, clinical trials, and real-world evidence.

## Table of Contents
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the API](#running-the-api)
- [API Overview](#api-overview)
  - [Root and Health](#root-and-health)
  - [Chat Workflow](#chat-workflow)
  - [Document Management](#document-management)
  - [Tool Discovery](#tool-discovery)
  - [Label Interpretation](#label-interpretation)
  - [Evidence Reports](#evidence-reports)
- [MCP Servers and Tools](#mcp-servers-and-tools)
- [Development and Testing](#development-and-testing)
- [Troubleshooting](#troubleshooting)

## Features
- Google Gemini 2.5 Flash chat model managed through LangChain's `init_chat_model` and cached by `ModelRegistry`.
- LangGraph state machine with checkpointing (`MemorySaver`) that enables agentic tool use and streaming responses per conversation thread.
- Thread-scoped RAG: PDF, TXT, Markdown, and DOCX uploads are chunked, embedded with Google Generative AI embeddings, and stored in in-memory vector stores keyed by `thread_id`.
- Multi-agent orchestrator that can assemble evidence reports by coordinating label, FAERS, clinical trial, and PubMed analysis agents plus a summarizer.
- Rich MCP server suite exposing FDA label data, FAERS queries, clinical trial searches, PubMed real-world evidence retrieval, shared utilities, and carefully crafted analysis prompts.
- Service container pattern (`services/container.py`) that centralizes long-lived services (chat, orchestrator, thread cache) for easy dependency injection and testing.

## Architecture Overview
1. **FastAPI application** (`main.py`)
   - Configures CORS, registers routers, and bootstraps shared services on startup.
2. **Services layer** (`services/`)
   - `ChatService` owns the LangGraph chat agent and document RAG helpers.
   - `AgentOrchestrator` lazily initializes specialized MCP agents for evidence synthesis.
   - `ThreadService` tracks per-thread LangGraph configuration and eviction cleanup.
3. **API layer** (`api/`)
   - `/api/chat` for initialization, batch responses, SSE streaming, resets, and document upload/list/clear.
   - `/api/tools` for introspecting tool availability by agent.
   - `/api/interpret` for patient-friendly FDA label section summaries.
   - `/api/evidence` placeholder routes where orchestrated evidence reports will be exposed.
4. **Agent layer** (`agent/`)
   - `ChatAgent` extends `MCPAgent`, wiring LangGraph workflow, RAG integration, and tool execution loop.
   - `model_registry.py` resolves Google Gemini or optional HuggingFace models via environment-driven configuration.
5. **MCP servers** (`mcp_servers/`)
   - Standalone FastMCP entry points that expose HTTP integrations, local references, and analysis prompts for each research domain.

## Project Structure
```
agent-server/
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-entrypoint.sh
├── render.yaml
├── agent/
│   ├── __init__.py
│   ├── chat_agent.py
│   ├── mcp_agent.py
│   └── model_registry.py
├── api/
│   ├── __init__.py
│   ├── chat.py
│   ├── evidence.py
│   ├── interpret.py
│   └── tools.py
├── assets/
│   ├── drug_adverse_event_fields.yaml
│   └── drug_labeling_fields.yaml
├── mcp_servers/
│   ├── adverse_event_server.py
│   ├── clinical_trial_server.py
│   ├── drug_label_server.py
│   ├── pubmed_server.py
│   ├── shared_tool_server.py
│   └── mcp_server_config.json
├── services/
│   ├── __init__.py
│   ├── agent_orchestrator.py
│   ├── chat_service.py
│   ├── container.py
│   └── thread_service.py
├── tests/
│   ├── check_thread_status.py
│   ├── test_api.py
│   ├── test_rag.py
│   └── test_rag_debug.py
├── requirements.txt
└── server-env/
    └── ... (local virtual environment, optional)
```

## Getting Started
### Prerequisites
- Python 3.12 or newer
- Google Gemini API key (`GOOGLE_API_KEY`)
- (Optional) Merriam-Webster Medical Dictionary key (`MERRIAM_WEBSTER_API_KEY`) for drug label definitions
- Recommended: virtual environment such as `python -m venv`

### Installation
```bash
git clone <repo-url>
cd agent-server
python3 -m venv server-env
source server-env/bin/activate  # Windows: server-env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your-gemini-api-key
MERRIAM_WEBSTER_API_KEY=optional-if-using-dictionary-tool
```
`ModelRegistry` loads `.env` on demand, so missing `GOOGLE_API_KEY` prevents model initialization.

## Running the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger UI: http://localhost:8000/docs
- Root JSON: `GET /` returns the welcome banner
- Health check: `GET /health` confirms chat service and thread cache status

## API Overview
### Root and Health
- `GET /` → `{"message": "Hello, this is the Med Helper server."}`
- `GET /health` → reports service status, whether the chat agent is initialized, and thread cache stats
- `GET /threads/stats` → dumps the thread LRU cache size, utilization, and limits

### Chat Workflow
1. Initialize (creates `thread_id` and bootstraps the LangGraph agent if needed):
   ```bash
   curl -X POST http://localhost:8000/api/chat/initialize \
     -H "Content-Type: application/json" \
     -d '{"thread_id": ""}'
   ```
2. Send requests with the returned `thread_id`:
   ```bash
   curl -X POST http://localhost:8000/api/chat/batch \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "<thread_id>", "message": "Explain current label guidance"}'
   ```
3. Stream responses via Server-Sent Events:
   ```bash
   curl -N -X POST http://localhost:8000/api/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "<thread_id>", "message": "Summarize the uploaded PDF"}'
   ```
4. Reset the thread (clears RAG vector store and LangGraph checkpoint) without creating a new ID:
   ```bash
   curl -X POST http://localhost:8000/api/chat/reset \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "<thread_id>"}'
   ```

### Document Management
Supported extensions: `.pdf`, `.txt`, `.md`, `.docx`.

- Upload: `POST /api/chat/documents/upload` (multipart form with `file` + `thread_id`)
- List: `GET /api/chat/documents/list/{thread_id}`
- Clear: `DELETE /api/chat/documents/clear/{thread_id}`

Uploaded files are stored on disk temporarily, chunked with `RecursiveCharacterTextSplitter`, embedded with Google embeddings, and indexed in a per-thread `InMemoryVectorStore`.

### Tool Discovery
- `GET /api/tools` lists every known agent and the tools it exposes. Agents that have not been initialized return `initialized: false` with an empty tool list.
- `GET /api/tools/{agent_name}` narrows the view to a specific agent (e.g., `chat`, `label`, `faers`).

### Label Interpretation
`POST /api/interpret` expects `{ "drug_name": "...", "section": "...", "content": "..." }` and returns a plain-language rephrasing using the `label_agent`. The orchestrator handles agent initialization, so no manual setup is required besides providing the section text.

### Evidence Reports
`api/evidence.py` defines request models and response schema for multi-source evidence summaries. The orchestrator already implements `evidence_report`, but the routes are currently marked TODO and respond with HTTP 500 until the wiring is completed.

## MCP Servers and Tools
Each server is a FastMCP executable launched via `mcp_servers/mcp_server_config.json`.

### drug_label
- `get_drug_information(name)` – FDA label lookup via `https://api.fda.gov/drug/label.json`
- `get_health_topic_summary(term)` – MedlinePlus health topic summaries
- `get_medical_term_definition(term)` – Merriam-Webster Medical Dictionary (requires API key)
- Prompts:
  - `generate_full_label_summarization_prompt(drug_name)` – structured full-label report
  - `generate_label_section_interpretation_prompt(drug_name, section_name, section_content)` – patient-friendly section explanation

### adverse_event (FAERS)
- `get_adverse_event_reports(...)` – pulls structured case reports filtered by SPL set ID and demographics
- `get_adverse_reaction_count(...)` – aggregates MedDRA preferred term counts
- Resource: `file:///openfda/faers_searchable_fields` (YAML reference)
- Prompt: `generate_faers_analysis_prompt()` – instructions for adverse event analysis responses

### clinical_trial
- `search_clinical_trials(...)` – ClinicalTrials.gov V2 search with relevance sorting and optional user demographics
- Prompt: `generate_clinical_trial_analysis_prompt()` – guidance for translating trial summaries into user-facing language

### pubmed (Real-world evidence)
- `search_pubmed_real_world_clinical_studies(...)` – PubMed E-utilities query tuned for observational and pragmatic studies
- `fetch_clinical_study_abstract(paper_ids)` – retrieves titles and abstracts for PMIDs
- Resource: `file:///pubmed/query_guidance`
- Prompt: `generate_real_world_evidence_analysis_prompt()` – workflow for synthesizing observational literature

### shared_tool
- Utility MCP server (e.g., shared formatting or helper functions) leveraged by multiple agents. See implementation for currently exposed helpers.

## Development and Testing
- Format / lint: use your preferred tooling (e.g., `ruff`, `black`) – not bundled by default.
- Run tests:
  ```bash
  pytest
  ```
  `tests/test_api.py` covers chat endpoints, while `tests/test_rag.py` exercises document ingestion logic.
- Logging: `api/chat.py` routes log to `uvicorn.error`, making debugging easier during development.

## Troubleshooting
- **Chat not initialized**: `POST /api/chat/initialize` must run once per process. Health endpoint shows `chat_agent_initialized: false` when missing.
- **Missing Google API key**: `ModelRegistry` raises `ValueError` if `GOOGLE_API_KEY` is absent. Confirm `.env` is loaded or export the variable.
- **Vector store empty**: RAG only runs when a thread has uploaded documents. Use `GET /api/chat/documents/list/{thread_id}` to verify.
- **Merriam-Webster errors**: Ensure `MERRIAM_WEBSTER_API_KEY` is set or handle the "not found" message gracefully in downstream prompts.
- **MCP command paths**: `mcp_servers/mcp_server_config.json` expects `${APP_PATH}` and `${PYTHON_PATH}` to be set by the orchestrator. Adjust if running outside the provided tooling.

---
