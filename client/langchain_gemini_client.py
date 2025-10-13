import os
import asyncio
import uuid
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, trim_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
import json

# Load environment variables from .env file
load_dotenv()

# Set GOOGLE_API_KEY from GEMINI_API_KEY if not already set
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

class GeminiMCPChatbot:
    """
    A chatbot that integrates Gemini AI with Model Context Protocol (MCP) servers
    for enhanced tool capabilities.
    """
    
    def __init__(self, 
                 model_name: str = "gemini-2.5-flash",
                 model_provider: str = "google_genai",
                 config_file: str = "server_config.json",
                 timeout: float = 30.0):
        """
        Initialize the chatbot with configuration parameters.
        
        Args:
            model_name: Name of the Gemini model to use
            model_provider: Provider for the model
            config_file: MCP server configuration file
            timeout: Timeout for MCP server connections in seconds
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.config_file = config_file
        self.timeout = timeout
        
        # Initialize components
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[Any] = []
        self.model = None
        self.model_with_tools = None
        self.trimmer = None
        self.app = None
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    async def initialize_mcp_client(self) -> None:
        """Initialize MCP client and retrieve tools."""
        print("Starting MCP initialization...")
        
        try:
            config_path = os.path.join(os.path.dirname(__file__), self.config_file)
            with open(config_path, "r") as file:
                raw = file.read()
                data = json.loads(raw)
                servers = data.get("mcpServers", {})
                
                print(f"Found {len(servers)} MCP servers in config")
                
                if not servers:
                    print("No valid MCP servers found in configuration")
                    self.tools = []
                    self.client = None
                else:
                    print(f"Creating MCP client with {len(servers)} servers...")
                    self.client = MultiServerMCPClient(servers)
                    
                    print("Getting tools (this may take a moment)...")
                    try:
                        # Add a timeout to prevent hanging
                        self.tools = await asyncio.wait_for(self.client.get_tools(), timeout=self.timeout)
                        print(f"Successfully retrieved {len(self.tools)} tools from MCP servers")
                    except asyncio.TimeoutError:
                        print("Timeout while getting tools from MCP servers")
                        print("Falling back to basic model without MCP tools")
                        self.tools = []
                        self.client = None
                    except Exception as e:
                        print(f"Error getting tools from MCP servers: {e}")
                        print("Falling back to basic model without MCP tools")
                        self.tools = []
                        self.client = None
                        
        except Exception as e:
            print(f"Error initializing MCP servers: {e}")
            print("Continuing with basic model without MCP tools")
            self.tools = []
            self.client = None
    
    def initialize_model(self) -> None:
        """Initialize the Gemini model and related components."""
        print("Initializing Gemini model...")
        self.model = init_chat_model(self.model_name, model_provider=self.model_provider)
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        print("Setting up message trimmer...")
        self.trimmer = trim_messages(
            max_tokens=2000,
            strategy="last",
            token_counter=self.model_with_tools,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
    
    def create_workflow(self) -> None:
        """Create the LangGraph workflow."""
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self.call_model)
        self.app = workflow.compile(checkpointer=MemorySaver())
    
    async def call_model(self, state: MessagesState) -> Dict[str, Any]:
        """Process messages through the model."""
        trimmed_messages = self.trimmer.invoke(state["messages"])
        prompt = self.prompt_template.invoke({"messages": trimmed_messages})
        response = await self.model_with_tools.ainvoke(prompt)
        return {"messages": response}
    
    async def initialize(self) -> None:
        """Initialize all components of the chatbot."""
        print("Starting initialization...")
        await self.initialize_mcp_client()
        self.initialize_model()
        self.create_workflow()
        print("Initialization complete!")
    
    def new_thread_config(self) -> Dict[str, Any]:
        """Generate a new thread configuration with unique ID."""
        return {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    def display_session_info(self, config: Dict[str, Any]) -> None:
        """Display session and tool information."""
        print(f"Session started. thread_id={config['configurable']['thread_id']}")
        
        # Show available tools
        if self.tools:
            print(f"Available tools: {len(self.tools)} MCP tools loaded")
            tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in self.tools[:5]]
            print(f"Sample tools: {', '.join(tool_names)}")
            if len(self.tools) > 5:
                print(f"... and {len(self.tools) - 5} more")
        else:
            print("No MCP tools available - using basic chat mode")
        
        print("Interactive Gemini chat (type /exit to quit, /reset to start a new thread)\n")
    
    async def process_user_input(self, user_input: str, config: Dict[str, Any]) -> Optional[str]:
        """Process user input and return response."""
        try:
            output = await self.app.ainvoke({"messages": [HumanMessage(user_input)]}, config)
            
            # Extract the assistant's last message
            last = output.get("messages", [])[-1] if output.get("messages") else None
            if last is not None:
                try:
                    last.pretty_print()
                    return None  # pretty_print handles the output
                except Exception:
                    # Fallback to raw content
                    content = getattr(last, "content", None)
                    return f"Assistant: {content}\n"
            return None
        except Exception as e:
            return f"Error: {e}"
    
    async def run_interactive_chat(self) -> None:
        """Run the interactive chat loop."""
        await self.initialize()
        # Generate a unique thread_id so MemorySaver keeps history for this session
        config = self.new_thread_config()
        self.display_session_info(config)
        
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user:
                continue
            if user.lower() in {"/exit", "/quit"}:
                print("Bye.")
                break
            if user.lower() == "/reset":
                # Start a new thread to clear prior context with a new unique id
                config = self.new_thread_config()
                print(f"Context reset. New thread_id={config['configurable']['thread_id']}\n")
                continue

            response = await self.process_user_input(user, config)
            if response:
                print(response)


def create_chatbot(config_file: str = "server_config_python_only.json", 
                   model_name: str = "gemini-2.5-flash",
                   timeout: float = 30.0) -> GeminiMCPChatbot:
    """
    Factory function to create a configured chatbot instance.
    
    Args:
        config_file: MCP server configuration file
        model_name: Gemini model to use
        timeout: Timeout for MCP connections
    
    Returns:
        Configured GeminiMCPChatbot instance
    """
    return GeminiMCPChatbot(
        model_name=model_name,
        config_file=config_file,
        timeout=timeout
    )


async def main():
    """Main function to run the chatbot."""
    chatbot = create_chatbot()
    await chatbot.run_interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())