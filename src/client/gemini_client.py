import json
from contextlib import AsyncExitStack
from typing import List, Dict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google.genai import types
import os

class GeminiClient:
    def __init__(self):
        self.sessions: List[ClientSession] = [] # each client session establishes a 1-to-1 connection to each server
        self.exit_stack = AsyncExitStack() # context manager that manage the mcp client objects
        self.available_tools: List[types.Tool] = []
        self.tool_session_map: Dict[str, ClientSession] = {} #maps the tool name to the corresponding client session

    async def connect_to_servers(self): 
        """Connect to all configured MCP servers."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "server_config.json")
            with open(config_path, "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)
            # list available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            fc_decl_list = []
            for tool in tools:
                self.tool_session_map[tool.name] = session
                clean_schema = self.clean_schema(tool.inputSchema) # Remove unsupported keys from inputSchema
                fn_decl = types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=clean_schema
                )
                fc_decl_list.append(fn_decl)

            self.available_tools.append(
                types.Tool(function_declarations=fc_decl_list)
            )
            print(f"Finished connecting to {server_name}.", flush=True)

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    def clean_schema(self, obj):
        """Recursively remove 'additionalProperties' and '$schema' from dicts."""
        if isinstance(obj, dict):
            obj = {k: self.clean_schema(v) for k, v in obj.items() if k not in ("additionalProperties", "$schema")}
            return obj
        elif isinstance(obj, list):
            return [self.clean_schema(i) for i in obj]
        else:
            return obj
    
    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()