import asyncio
from dataclasses import dataclass
from typing import Union
from agent.mcp_agent import MCPAgent
from agent.model_registry import ModelRegistry

@dataclass(frozen=True, slots=True)
class AgentSpec:
    key: str
    mcp_config_key: str
    chat_model_key: str
    system_message: str

DEFAULT_AGENT_REGISTRY: dict[str, AgentSpec] = {
    "label_agent": AgentSpec(
        key="label_agent",
        mcp_config_key="label_agent",
        chat_model_key="gemini",
        system_message="You are a helpful medical label interpreter agent.",
    ),
    "faers_agent": AgentSpec(
        key="faers_agent",
        mcp_config_key="faers_agent",
        chat_model_key="openai",
        system_message="You are an expert in adverse event report analysis.",
    ),
    "rwe_agent": AgentSpec(
        key="rwe_agent",
        mcp_config_key="rwe_agent",
        chat_model_key="openai",
        system_message="You are an expert in real-world clinical evidence analysis.",
    ),
    "clinical_trials_agent": AgentSpec(
        key="clinical_trials_agent",
        mcp_config_key="clinical_trials_agent",
        chat_model_key="openai",
        system_message="You are an expert in clinical trials data analysis.",
    ),
    "explainer_agent": AgentSpec(
        key="explainer_agent",
        mcp_config_key="explainer_agent",
        chat_model_key="gemini",
        system_message="You are a medical content explainer agent.",
    ),
    "chat_agent": AgentSpec(
        key="chat_agent",
        mcp_config_key="chat_agent",
        chat_model_key="gemini",
        system_message="You are a helpful medication expert. Answer all questions to the best of your ability.",
    ),
}

class AgentRegistry:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            # Initialize attributes only once
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._registry: dict[str, AgentSpec] = DEFAULT_AGENT_REGISTRY
        self._cache: dict[str, MCPAgent] = {}
        self._model_registry = ModelRegistry()
        self._init_locks: dict[str, asyncio.Lock] = {}

    async def _ensure_agent(self, spec: AgentSpec) -> MCPAgent:
        chat_model_instance = self._model_registry.resolve(spec.chat_model_key)
        agent_instance = MCPAgent(
            chat_model_instance,
            spec.mcp_config_key,
            spec.system_message,
        )
        await agent_instance.initialize()
        return agent_instance

    async def resolve(self, ref: Union[str, AgentSpec, MCPAgent]) -> MCPAgent:
        if isinstance(ref, MCPAgent):
            return ref
        
        if isinstance(ref, AgentSpec):
            cache_key = ref.key
            lock = self._init_locks.setdefault(cache_key, asyncio.Lock())
            async with lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]
                agent_instance = await self._ensure_agent(ref)
                self._cache[cache_key] = agent_instance
                return agent_instance
        
        if isinstance(ref, str):
            if ref in self._cache:
                return self._cache[ref]
            if ref not in self._registry:
                raise ValueError(
                    f"Unknown model key '{ref}'. "
                    f"Known keys: {sorted(self._registry.keys())}. "
                    "Pass a BaseChatModel to override directly, or a ModelSpec(name=..., provider=...)."
                )
            spec = self._registry[ref]
            lock = self._init_locks.setdefault(ref, asyncio.Lock())
            async with lock:
                if ref in self._cache:
                    return self._cache[ref]
                agent_instance = await self._ensure_agent(spec)
                self._cache[ref] = agent_instance
                return agent_instance

        raise TypeError(f"Unsupported model reference type: {type(ref).__name__}")

    def spec_for(self, key: str) -> AgentSpec:
        if key not in self._registry:
            raise KeyError(f"Unknown model key '{key}'. Known keys: {sorted(self._registry.keys())}")
        return self._registry[key]

    def keys(self) -> list[str]:
        return sorted(self._registry.keys())
    
    def initialized_agents(self) -> list:
        return list(self._cache.keys())
    
    def is_agent_initialized(self, key: str) -> bool:
        return key in self._cache
    
    def get_initialized_agent(self, key: str) -> MCPAgent | None:
        return self._cache.get(key, None)