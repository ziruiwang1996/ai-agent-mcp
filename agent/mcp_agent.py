import re
import os
import json
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from typing import Optional, Any
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

def coerce_prompt_to_string(value: Any) -> Optional[str]:
    """Best-effort conversion of MCP prompt payloads into plain strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        return coerce_prompt_to_string(getattr(value, "content", None))
    if isinstance(value, dict):
        return coerce_prompt_to_string(value.get("content"))
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = coerce_prompt_to_string(item)
            if isinstance(text, str) and text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
        return None
    return None

class MCPAgent:
    def __init__(
        self,
        chat_model: BaseChatModel,
        mcp_config_key: str,
        system_prompt: Optional[str] = None 
    ):
        self.chat_model: BaseChatModel = chat_model
        self.mcp_config_key: str = mcp_config_key

        self.system_prompt: Optional[str] = system_prompt
        self.tools: list[Any] = []
        self.resources: list[Any] = []
        self.agent: Optional[CompiledStateGraph] = None

    async def _initialize_mcp_client(self) -> None:
        try:
            config_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "mcp_server_config.json")
            )
            with open(config_file_path, 'r') as file:
                raw_config = file.read()
                expanded_config = expand_env_in_text(raw_config)
                config = json.loads(expanded_config)
                servers = config.get(self.mcp_config_key, {})
                client = MultiServerMCPClient(servers)

                try:
                    self.tools = await client.get_tools()
                except Exception as tool_err:
                    print(f"Warning initializing MCP tools: {tool_err}")
                    self.tools = []

                for server_name in servers.keys():
                    prompts = []
                    try:
                        prompts = await client.get_prompt(server_name, "generate_system_prompt")
                        if prompts:
                            prompt_value = prompts[0]
                            extracted_prompt = coerce_prompt_to_string(prompt_value)
                            if extracted_prompt:
                                self.system_prompt = extracted_prompt
                                break
                            else:
                                print(
                                    f"Warning: unsupported prompt payload from {server_name}: {type(prompt_value).__name__}"
                                )
                    except Exception as prompt_err:
                        print(
                            f"Warning retrieving system prompt from {server_name}: {prompt_err}"
                        )

                    try:
                        resources = await client.get_resources(server_name)
                        self.resources.extend(resources)
                    except Exception as resource_err:
                        print(
                            f"Warning retrieving resources from {server_name}: {resource_err}"
                        )
        except Exception as e:
            print(f"Error initializing MCP client: {e}")

    async def initialize(self) -> None:
        try:
            await self._initialize_mcp_client()
            self.agent = create_agent(
                model=self.chat_model, 
                tools=self.tools,
                system_prompt=self.system_prompt
            )
        except Exception as e:
            print(f"Error initializing Agent: {e}")
            import traceback
            traceback.print_exc()
        
    async def process_input(self, user_input: str) -> str:
        if not self.agent:
            return "Agent failed to initialize; check model setup logs."
        try:
            messages = [HumanMessage(content=user_input)]
            response = await self.agent.ainvoke({"messages": messages})
            print(f"Agent response: {response}")
            last = response.get("messages", [])[-1] if response.get("messages") else None
            response = getattr(last, "content", "No response generated") if last else "No response generated"
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_available_tools(self) -> list[dict[str, str]]:
        return [
            {
                "name": getattr(tool, "name", "Unknown"),
                "description": getattr(tool, "description", "No description available"),
            }
            for tool in self.tools
        ]