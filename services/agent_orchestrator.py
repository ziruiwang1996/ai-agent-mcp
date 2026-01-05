from langchain_core.language_models import BaseChatModel
from typing import Optional
from agent.mcp_agent import MCPAgent
from agent.model_registry import ModelRegistry

class AgentOrchestrator:
    def __init__(self):
        self.label_interpreter_agent: Optional[MCPAgent] = None
        # Placeholders for future services
        self.classifier: Optional[MCPAgent] = None
        self.generator: Optional[MCPAgent] = None
        self.context_relevance_agent: Optional[MCPAgent] = None
        self.FAERS_summarizer_agent: Optional[MCPAgent] = None
        self.critique_guardrail_agent: Optional[MCPAgent] = None
        self._model_registry = ModelRegistry()
        
    async def interpret_label(
        self,
        user_input: str,
        model: str | BaseChatModel = "gemini",
        mcp_server_name: str = "label_interpreter_agent",
        system_message: str = "You are a helpful medical label interpreter agent.",
    ) -> str:
        # Keep signature for compatibility; delegate to the label service.
        if not self.label_interpreter_agent:
            chat_model = self._model_registry.resolve(model)
            self.label_interpreter_agent = MCPAgent(chat_model, mcp_server_name, system_message)
            await self.label_interpreter_agent.initialize()
        return await self.label_interpreter_agent.process_input(user_input)

    def summarize_FAERS(self) -> str:
        pass

    def analyze_patient_context(self) -> str:
        pass