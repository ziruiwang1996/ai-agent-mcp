# Life Science AI Agent

A modular AI agent framework that connects chatbots (Gemini, HuggingFace) to multiple Model Context Protocol (MCP) servers, including filesystem, arXiv, OpenFDA, ClinicalTrials, and PDB tools. This framework enables tool-augmented reasoning by integrating large language models (LLMs) with external tools and APIs.

---

## Project Structure

```
ai-agent-mcp/
│
├── src/
│   ├── host/
│   │   ├── gemini_chatbot.py        # Gemini chatbot implementation
│   │   └── HuggingFace_LLM.py       # Future integration
│   │
│   ├── client/
│   │   ├── gemini_client.py         # Gemini client for tool management
│   │   ├── huggingface_client.py    # Future integration
│   │   └── server_config.json       # Configuration for MCP servers
│   │
│   ├── mcp-server/
│   │   ├── arxiv_server.py          # MCP server for arXiv paper search
│   │   ├── clinicaltrials_server.py # MCP server for ClinicalTrials data
│   │   ├── openfda_server.py        # MCP server for OpenFDA data
│   │   ├── pdb_server.py            # MCP server for PDB data
│   │   ├── rdkit_server.py          # Future integration
│   │   └── pubmed_server.py         # Future integration
│   │
│   ├── fastapi_app.py               # FastAPI app for chatbot API
│   └── __init__.py
│
├── streamlit_app/
│   └── home_page.py                 # Streamlit app for chatbot UI
│
├── .env                             # Environment variables (e.g., API keys)
├── .gitignore                       # Git ignore file
└── README.md                        # Project documentation
```


---

## Setup

1. **Clone the repository**
    ```sh
    git clone <your-repo-url>
    cd ai-agent-mcp
    ```

2. **Create a virtual environment**
    ```sh
    python3 -m venv .
    source bin/activate
    ```

3. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**
    Create a `.env` file in the project root:
    ```
    GEMINI_API_KEY=your-gemini-api-key
    ```

---

## Configuration

MCP server configuration is in [`mcp-server/server_config.json`](src/client/server_config.json):

```json
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "."
            ]
        },
        
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"]
        },

        "arxiv_server": {
            "command": "python3",
            "args": ["src/mcp-server/arxiv_server.py"]
        },

        "openfda_server": {
            "command": "python3",
            "args": ["src/mcp-server/openfda_server.py"]
        },

        "clinicaltrials_server": {
            "command": "python3",
            "args": ["src/mcp-server/clinicaltrials_server.py"]
        },

        "pdb_server": {
            "command": "python3",
            "args": ["src/mcp-server/pdb_server.py"]
        }
        
    }
}
```

---

## Running the Chatbots

### 1. Start the FastAPI Backend
Run the FastAPI app to expose the chatbot as a RESTful API:
```sh
uvicorn src.fastapi_app:app --reload
```

### 2. Start the Streamlit Frontend
Run the Streamlit app to interact with the chatbot via a web interface:
```sh
streamlit run streamlit_app/home_page.py
```

---

## Running MCP Servers

Each server in `mcp-server/` can be started individually, for example:

```sh
python3 mcp-server/arxiv_server.py
python3 mcp-server/openFDA_server.py
python3 mcp-server/pdb_server.py
```

Or, let the chatbot start them as subprocesses according to `server_config.json`.

---

## Features

### Chatbot Integration
- **Gemini Chatbot**: Powered by Google Gemini for advanced reasoning and tool usage.
- **HuggingFace Chatbot**: (Optional) Integration with HuggingFace models.

### MCP Servers
- **Filesystem Server**: Interact with files and directories.
- **arXiv Server**: Search and summarize academic papers.
- **ClinicalTrials Server**: Retrieve clinical trial data.
- **OpenFDA Server**: Access FDA datasets.
- **PDB Server**: Query protein structure data.

### Frontend
- **Streamlit App**: A user-friendly web interface for interacting with the chatbot.
- **Real-Time Streaming**: Incremental updates for a responsive user experience.

---

## Notes

- Ensure your `.env` file is **not** committed to version control.
- Install any additional dependencies required by individual servers (e.g., `arxiv`, `openfda`, etc.).
- For the filesystem server, you need Node.js and `npx`.

---

## License

[MIT](LICENSE) (or your preferred license)

---

## Acknowledgments

- [Google Gemini](https://ai.google.dev/)
- [HuggingFace](https://huggingface.co/)
- [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)
- [Streamlit](https://streamlit.io)