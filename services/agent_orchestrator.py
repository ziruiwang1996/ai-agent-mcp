from __future__ import annotations
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from agent.mcp_agent import MCPAgent
from agent.model_registry import ModelRegistry
from huggingface_hub import InferenceClient
from langchain_core.language_models import BaseChatModel

# Provide sane defaults when the orchestrator runs outside container tooling.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("APP_PATH", str(_PROJECT_ROOT))
os.environ.setdefault("PYTHON_PATH", sys.executable)

@dataclass(frozen=True, slots=True)
class _AgentSpec:
    key: str
    mcp_config_key: str
    chat_model_key: str
    system_message: str

@dataclass(frozen=True, slots=True)
class _ModelSpec:
    key: str
    model_key: str

class AgentOrchestrator:
    def __init__(self):
        self._agents_and_models: dict[str, MCPAgent | InferenceClient] = {}
        self._init_locks: dict[str, asyncio.Lock] = {}
        self._model_registry = ModelRegistry()

        self._specs: dict[str, _AgentSpec | _ModelSpec] = {
            "label_agent": _AgentSpec(
                key="label_agent",
                mcp_config_key="label_agent",
                chat_model_key="gemini",
                system_message="You are a helpful medical label interpreter agent.",
            ),
            "faers_agent": _AgentSpec(
                key="faers_agent",
                mcp_config_key="faers_agent",
                chat_model_key="openai",
                system_message="You are an expert in adverse event report analysis.",
            ),
            "rwe_agent": _AgentSpec(
                key="rwe_agent",
                mcp_config_key="rwe_agent",
                chat_model_key="openai",
                system_message="You are an expert in real-world clinical evidence analysis.",
            ),
            "clinical_trials_agent": _AgentSpec(
                key="clinical_trials_agent",
                mcp_config_key="clinical_trials_agent",
                chat_model_key="openai",
                system_message="You are an expert in clinical trials data analysis.",
            ),
            "summarizer": _ModelSpec(
                key="summarizer",
                model_key="bigbird",
            ),
            "critique_guardrail_agent": _AgentSpec(
                key="critique_guardrail_agent",
                mcp_config_key=None,
                chat_model_key="gemini",
                system_message="You are an expert in critiquing medical reports for accuracy and completeness.",
            ),
        }

    async def _get_or_create_agent_or_model(self, spec: _AgentSpec | _ModelSpec) -> MCPAgent | InferenceClient | BaseChatModel:
        if isinstance(spec, _ModelSpec):
            model = self._agents_and_models.get(spec.key)
            if model is not None:
                return model
                
            model = self._model_registry.resolve(spec.model_key)
            self._agents_and_models[spec.key] = model
            return model
        
        agent = self._agents_and_models.get(spec.key)
        if agent is not None:
            return agent

        lock = self._init_locks.setdefault(spec.key, asyncio.Lock())
        async with lock:
            agent = self._agents_and_models.get(spec.key)
            if agent is not None:
                return agent

            chat_model = self._model_registry.resolve(spec.chat_model_key)
            agent = MCPAgent(
                chat_model,
                spec.mcp_config_key,
                spec.system_message,
            )
            await agent.initialize()
            self._agents_and_models[spec.key] = agent
            return agent
        
    def _execute_summarization(self, instance: InferenceClient, input_data: str) -> str:
        try:
            output = instance.summarization(input_data)
            return output.summary_text
        except Exception as e:
            raise RuntimeError(f"Error executing step summarization: {str(e)}") from e

    async def _execute_step(self, step_name: str, step_instance: MCPAgent, input_data: str) -> Dict[str, Any]:
        """Execute a single step with error handling."""
        try:
            output = await step_instance.process_input(input_data)
            return {
                "step": step_name,
                "status": "success",
                "output": output,
                "error": None,
            }
        except Exception as e:
            return {
                "step": step_name,
                "status": "failure",
                "output": None,
                "error": str(e),
            }

    async def interpret_label(self, user_input: str) -> str:
        agent = await self._get_or_create_agent_or_model(
            self._specs["label_agent"]
        )
        out = await self._execute_step("label_interpretation", agent, user_input)
        if out["status"] != "success":
            raise RuntimeError(out.get("error") or "Label interpretation failed")
        return str(out["output"] or "")
    
    async def evidence_report(self, request_data: dict[str, Any]):
        # Lazily initialize agents (thread-safe) and mirror legacy attributes.
        faers_agent, rwe_agent, clinical_trials_agent, summarizer = await asyncio.gather(
            self._get_or_create_agent_or_model(self._specs["faers_agent"]),
            self._get_or_create_agent_or_model(self._specs["rwe_agent"]),
            self._get_or_create_agent_or_model(self._specs["clinical_trials_agent"]),
            self._get_or_create_agent_or_model(self._specs["summarizer"]),
        )

        demographics_prompt = (
            "Patient demographics:\n"
            f"- age: {request_data.get('age')}\n"
            f"- sex: {request_data.get('sex')}\n"
            f"- weight: {request_data.get('weight')}\n"
            f"- is_pregnant: {request_data.get('is_pregnant')}\n"
            f"- is_breastfeeding: {request_data.get('is_breastfeeding')}\n"
            f"- conditions: {request_data.get('conditions')}\n"
            f"- other_medications: {request_data.get('other_medications')}\n"
        )
        drug_set_id = request_data.get("drug_set_id")
        drug_name = request_data.get("drug_name")

        faers_prompt = (
            "Analyze the FDA Adverse Event Report for the medication and provide key findings.\n"
            f"set_id: {drug_set_id}\n"
            f"drug_name: {drug_name}\n\n"
            f"{demographics_prompt}"
        )
        rwe_prompt = (
            "Analyze the real-world clinical evidence (e.g., PubMed) for the medication and provide key findings.\n"
            f"set_id: {drug_set_id}\n"
            f"drug_name: {drug_name}\n\n"
            f"{demographics_prompt}"
        )
        clinical_trials_prompt = (
            "Analyze the clinical trials data for the medication and provide key findings.\n"
            f"set_id: {drug_set_id}\n"
            f"drug_name: {drug_name}\n\n"
            f"{demographics_prompt}"
        )

        #These three are independent; run in parallel to reduce latency.
        faers_out, rwe_out, clinical_trials_out = await asyncio.gather(
            self._execute_step("FAERS_analysis", faers_agent, faers_prompt),
            self._execute_step("RWE_analysis", rwe_agent, rwe_prompt),
            self._execute_step("Clinical_Trials_analysis", clinical_trials_agent, clinical_trials_prompt),
        )

        summary_input = (
            "Summarize the evidence findings from FAERS, PubMed, and Clinical Trials into a concise report.\n"
            f"set_id: {drug_set_id}\n"
            f"drug_name: {drug_name}\n\n"
            f"FAERS findings:\n{faers_out.get('output')}\n\n" if faers_out.get("status") == "success" else ""
            f"RWE findings:\n{rwe_out.get('output')}\n\n" if rwe_out.get("status") == "success" else ""
            f"Clinical Trials findings:\n{clinical_trials_out.get('output')}\n" if clinical_trials_out.get("status") == "success" else ""
            "If no findings are available, do not make up any information."
        )
        print(faers_out, rwe_out, clinical_trials_out)

        if faers_out.get("status") == "failure" and rwe_out.get("status") == "failure" and clinical_trials_out.get("status") == "failure":
            summary_out = "No findings to summarize."
        else:
            summary_out = self._execute_summarization(summarizer, summary_input)

        return {
            "faers_report": faers_out["output"] if faers_out["status"] == "success" else faers_out["error"],
            "rwe_report": rwe_out["output"] if rwe_out["status"] == "success" else rwe_out["error"],
            "clinical_trials_report": clinical_trials_out["output"] if clinical_trials_out["status"] == "success" else clinical_trials_out["error"],
            "summary": summary_out
        }
    
# if __name__ == "__main__":
#     orchestrator = AgentOrchestrator()
#     input = {
#         "age": "62",  
#         "sex": "male",
#         "weight": "85",
#         "is_pregnant": "Unknown",
#         "is_breastfeeding": "Unknown",
#         "conditions": ["hypertension", "diabetes"],
#         "other_medications": ["Lisinopril","Atorvastatin"],
#         "drug_set_id": "a944a167-e3ac-4084-af1c-22b48713471c",
#         "drug_name": "Metformin"     
#     }
#     out = asyncio.run(
#         orchestrator.evidence_report(input)
#     )
#     print(out)