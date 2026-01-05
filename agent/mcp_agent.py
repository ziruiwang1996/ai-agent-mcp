import re
import os
import json
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional, List, Any
from langgraph.graph.state import CompiledStateGraph

def expand_env_in_text(text: str) -> str:
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")
    def repl(match):
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            raise KeyError(f"Environment variable '{var}' not set for configuration substitution")
        return val
    return pattern.sub(repl, text)

class MCPAgent:
    def __init__(
        self,
        chat_model: BaseChatModel,
        mcp_server_name: str,
        system_message: str
    ):
        self.chat_model: BaseChatModel = chat_model
        self.mcp_server_name: str = mcp_server_name
        self.system_message: str = system_message
        self.tools: List[Any] = []
        self.agent: Optional[CompiledStateGraph] = None

    async def initialize_mcp_client(self) -> None:
        try:
            config_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "mcp_server_config.json")
            )
            with open(config_file_path, 'r') as file:
                raw_config = file.read()
                expanded_config = expand_env_in_text(raw_config)
                config = json.loads(expanded_config)
                servers = config.get(self.mcp_server_name, {})
                client = MultiServerMCPClient(servers)
                self.tools = await client.get_tools()
        except Exception as e:
            print(f"Error initializing MCP client: {e}")

    async def initialize(self) -> None:
        try:
            await self.initialize_mcp_client()
            self.agent = create_agent(
                model=self.chat_model, 
                tools=self.tools
            )
        except Exception as e:
            print(f"Error initializing Label Interpreter agent: {e}")
            import traceback
            traceback.print_exc()
        
    async def process_input(self, user_input: str) -> str:
        if not self.agent:
            return "Agent failed to initialize; check model setup logs."
        messages = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=user_input)
        ]
        response = await self.agent.ainvoke({"messages": messages})
        last = response.get("messages", [])[-1] if response.get("messages") else None
        response = getattr(last, "content", "No response generated") if last else "No response generated"
        return response
    
    def get_available_tools(self) -> list[dict[str, str]]:
        return [
            {
                "name": getattr(tool, "name", "Unknown"),
                "description": getattr(tool, "description", "No description available"),
            }
            for tool in self.tools
        ]