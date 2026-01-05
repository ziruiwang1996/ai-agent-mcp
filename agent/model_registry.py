from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Union
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
import os
from dotenv import load_dotenv

@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

DEFAULT_MODEL_REGISTRY: Mapping[str, ModelSpec] = {
    "gemini": ModelSpec(name="gemini-2.5-flash", provider="google_genai"),
    "med_gemma": ModelSpec(name="google/medgemma-4b-it", provider="langchain-huggingface"),
    "tx_gemma": ModelSpec(name="google/txgemma-9b-chat", provider="langchain-huggingface"),
}

class ModelRegistry:
    def __init__(
        self,
        registry: Optional[Mapping[str, ModelSpec]] = None,
        *,
        enable_cache: bool = True,
    ):
        self._registry: Mapping[str, ModelSpec] = registry or DEFAULT_MODEL_REGISTRY
        self._enable_cache = enable_cache
        self._cache: Dict[str, BaseChatModel] = {}

    def resolve(self, ref: Union[str, ModelSpec, BaseChatModel]) -> BaseChatModel:
        """Resolve a model reference into a concrete LangChain chat model.

        Accepted refs:
        - BaseChatModel: returned as-is
        - str: treated as a registry key (e.g. "gemini")
        - ModelSpec: created via init_chat_model
        """
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        if isinstance(ref, BaseChatModel):
            return ref

        if isinstance(ref, ModelSpec):
            return init_chat_model(ref.name, model_provider=ref.provider, **(ref.kwargs or {}))

        if isinstance(ref, str):
            key = ref
            if self._enable_cache and key in self._cache:
                return self._cache[key]
            if key not in self._registry:
                raise ValueError(
                    f"Unknown model key '{key}'. "
                    f"Known keys: {sorted(self._registry.keys())}. "
                    "Pass a BaseChatModel to override directly, or a ModelSpec(name=..., provider=...)."
                )
            spec = self._registry[key]
            model = init_chat_model(spec.name, model_provider=spec.provider, **(spec.kwargs or {}))
            if self._enable_cache:
                self._cache[key] = model
            return model

        raise TypeError(f"Unsupported model reference type: {type(ref).__name__}")

    def spec_for(self, key: str) -> ModelSpec:
        if key not in self._registry:
            raise KeyError(f"Unknown model key '{key}'. Known keys: {sorted(self._registry.keys())}")
        return self._registry[key]

    def keys(self) -> list[str]:
        return sorted(self._registry.keys())

    def get_gemini(self) -> BaseChatModel:
        return self.resolve("gemini")

    def get_med_gemma(self) -> BaseChatModel:
        return self.resolve("med_gemma")

    def get_tx_gemma(self) -> BaseChatModel:
        return self.resolve("tx_gemma")