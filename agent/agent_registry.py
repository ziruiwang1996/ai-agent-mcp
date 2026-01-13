import asyncio
from dataclasses import dataclass
from typing import Dict, Union
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
}

class AgentRegistry:
    def __init__(self, *, enable_cache: bool = True):
        self._registry: dict[str, AgentSpec] = DEFAULT_AGENT_REGISTRY
        self._enable_cache = enable_cache
        self._cache: Dict[str, MCPAgent] = {}
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
                if self._enable_cache and cache_key in self._cache:
                    return self._cache[cache_key]
                agent_instance = await self._ensure_agent(ref)
                if self._enable_cache:
                    self._cache[cache_key] = agent_instance
                return agent_instance
        
        if isinstance(ref, str):
            if self._enable_cache and ref in self._cache:
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
                if self._enable_cache and ref in self._cache:
                    return self._cache[ref]
                agent_instance = await self._ensure_agent(spec)
                if self._enable_cache:
                    self._cache[ref] = agent_instance
                return agent_instance

        raise TypeError(f"Unsupported model reference type: {type(ref).__name__}")

    def spec_for(self, key: str) -> AgentSpec:
        if key not in self._registry:
            raise KeyError(f"Unknown model key '{key}'. Known keys: {sorted(self._registry.keys())}")
        return self._registry[key]

    def keys(self) -> list[str]:
        return sorted(self._registry.keys())