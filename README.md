# AI Agent MCP

A modular AI agent framework that connects chatbots (Gemini, HuggingFace) to multiple Model Context Protocol (MCP) servers, including filesystem, arXiv, OpenFDA, and PDB tools.

---

## Project Structure

```
ai-agent-mcp/src
│
├── host/
│   └── gemini_chatbot.py
│
├── client/
│   ├── gemini_client.py
│   └── server_config.json
│
├── mcp-server/
│   ├── arxiv_server.py
│   ├── clinicaltrials_server.py
│   ├── openfda_server.py
│   ├── pdb_server.py
│   ├── rdkit_server.py (future integration)
│   └── pubmed_server.py (future integration)
│
├── .env
├── .gitignore
└── ...
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

### Gemini Chatbot

```sh
python3 host/gemini_chatbot.py
```

### HuggingFace Chatbot

```sh
python3 host/huggingface_chatbot.py
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