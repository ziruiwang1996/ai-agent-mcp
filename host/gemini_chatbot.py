from dotenv import load_dotenv
import os
from typing import List, Dict, TypedDict
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import json

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version='v1alpha')
)

class GeminiChatBot:
    def __init__(self):
        self.sessions: List[ClientSession] = [] # each client session establishes a 1-to-1 connection to each server
        self.exit_stack = AsyncExitStack() # context manager that manage the mcp client objects
        self.gemini = client
        self.available_tools: List[types.Tool] = []
        self.tool_session_map: Dict[str, ClientSession] = {} #maps the tool name to the corresponding client session

    async def connect_to_servers(self): 
        """Connect to all configured MCP servers."""
        try:
            config_path = os.path.join(
                os.path.dirname(__file__),
                '..', 'mcp-server', 'server_config.json'
            )
            with open(os.path.abspath(config_path), "r") as file:
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

    async def process_query(self, query:str):
        """
        Send user's query to Gemini, handle any requested tool calls,
        and print the model's final response.
        """
        contents = [types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )]

        config = types.GenerateContentConfig(
            tools=self.available_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=2048,
        )

        response: types.GenerateContentResponse = self.gemini.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=contents,
            config=config
        )

        while True:
            if response.function_calls:
                fc = response.function_calls[0]
                contents.append(response.candidates[0].content)

                fc_name = fc.name
                fc_args = fc.args

                try: 
                    print(f"Calling tool {fc_name} with args {fc_args}")
                    session = self.tool_session_map[fc_name]
                    result = await session.call_tool(fc_name, arguments=fc_args)

                    fc_response = {'result': result.content}
                except Exception as e:
                    # instead of raising the exception, you can let the model handle it
                    fc_response = {'error': str(e)}

                fc_response_part = types.Part.from_function_response(
                    name=fc.name,
                    response=fc_response,
                )
                function_response_content = types.Content(
                    role='tool', parts=[fc_response_part]
                )
                contents.append(function_response_content)

                response = self.gemini.models.generate_content(
                    model="gemini-2.5-flash-preview-04-17",
                    contents=contents,
                    config=config
                )
                    
                if not response.function_calls:
                    print(response.text)
                    break
            else:
            # Pure text response—just print and exit
                print(response.text)
                break
        
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = GeminiChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

import asyncio
if __name__ == "__main__":
    asyncio.run(main())