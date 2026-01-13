from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Union
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import certifi

# Align Python SSL clients with certifi so async HTTPS calls succeed in dev/runtime.
_CERT_PATH = certifi.where()
os.environ.setdefault("SSL_CERT_FILE", _CERT_PATH)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _CERT_PATH)

@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

DEFAULT_MODEL_REGISTRY: Mapping[str, ModelSpec] = {
    "gemini": ModelSpec(name="gemini-2.5-flash", provider="google_genai"),
    # "med_gemma": ModelSpec(name="google/medgemma-4b-it", provider=""),
    # "tx_gemma": ModelSpec(name="google/txgemma-9b-chat", provider=""),
    "summarizer": ModelSpec(name="google/bigbird-pegasus-large-arxiv", provider="hf-inference"),
    "openai": ModelSpec(
        name="openai/gpt-oss-20b", 
        provider="huggingface", 
        kwargs={"task": "text-generation", "max_new_tokens": 1024}),
}

class ModelRegistry:
    def __init__(self, *, enable_cache: bool = True):
        self._registry: Mapping[str, ModelSpec] = DEFAULT_MODEL_REGISTRY
        self._enable_cache = enable_cache
        self._cache: Dict[str, BaseChatModel | InferenceClient] = {}

    def resolve(self, ref: Union[str, ModelSpec, BaseChatModel, InferenceClient]) -> BaseChatModel | InferenceClient:
        """Resolve a model reference into a concrete LangChain chat model.

        Accepted refs:
        - BaseChatModel: returned as-is
        - str: treated as a registry key (e.g. "gemini")
        - ModelSpec: created via init_chat_model
        """
        if isinstance(ref, BaseChatModel):
            return ref
        
        if isinstance(ref, InferenceClient):
            return ref

        if isinstance(ref, ModelSpec):
            if ref.provider == "hf-inference":
                model_instance = self._via_inference_client(ref)
            elif ref.provider == "huggingface":
                model_instance = self._via_huggingface_endpoint(ref)
            else:
                model_instance = self._via_init_chat_model(ref)
            if self._enable_cache:
                self._cache[ref.name] = model_instance
            return model_instance

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
            if spec.provider == "hf-inference":
                model_instance = self._via_inference_client(spec)
            elif spec.provider == "huggingface":
                model_instance = self._via_huggingface_endpoint(spec)
            else:
                model_instance = self._via_init_chat_model(spec)
            if self._enable_cache:
                self._cache[ref] = model_instance
            return model_instance

        raise TypeError(f"Unsupported model reference type: {type(ref).__name__}")

    def spec_for(self, key: str) -> ModelSpec:
        if key not in self._registry:
            raise KeyError(f"Unknown model key '{key}'. Known keys: {sorted(self._registry.keys())}")
        return self._registry[key]

    def keys(self) -> list[str]:
        return sorted(self._registry.keys())
    
    def _via_huggingface_endpoint(self, spec: ModelSpec) -> ChatHuggingFace:
        load_dotenv()
        llm = HuggingFaceEndpoint(
            repo_id=spec.name,
            task=spec.kwargs.get("task") if spec.kwargs else None,
            max_new_tokens=spec.kwargs.get("max_new_tokens", 1024) if spec.kwargs else 1024,
            repetition_penalty=1.03,
            provider="auto",
            huggingfacehub_api_token = os.environ["HF_API_TOKEN"]
        )
        chat_model = ChatHuggingFace(llm=llm)
        return chat_model
    
    def _via_init_chat_model(self, spec: ModelSpec) -> BaseChatModel:
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        return init_chat_model(spec.name, model_provider=spec.provider, **(spec.kwargs or {}))

    def _via_inference_client(self, spec: ModelSpec) -> InferenceClient:
        load_dotenv()
        token = os.environ.get("HF_API_TOKEN")
        if not token:
            raise ValueError("HF_API_TOKEN environment variable is not set.")
        # Use Hugging Face Inference Endpoints with hosted models; this stays fully serverless.
        return InferenceClient(model=spec.name, token=token)