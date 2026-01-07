from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional
from langchain_core.language_models import BaseChatModel
from agent.mcp_agent import MCPAgent
from agent.model_registry import ModelRegistry

@dataclass(frozen=True, slots=True)
class _AgentSpec:
    key: str
    mcp_config_key: str
    chat_model_key: str
    system_message: str

class AgentOrchestrator:
    def __init__(self):
        self._agents: dict[str, MCPAgent] = {}
        self._init_locks: dict[str, asyncio.Lock] = {}
        self._model_registry = ModelRegistry()

        self._specs: dict[str, _AgentSpec] = {
            "label_agent": _AgentSpec(
                key="label_agent",
                mcp_config_key="label_agent",
                chat_model_key="gemini",
                system_message="You are a helpful medical label interpreter agent.",
            ),
            "faers_agent": _AgentSpec(
                key="faers_agent",
                mcp_config_key="faers_agent",
                chat_model_key="gemini",
                system_message="You are an expert in adverse event report analysis.",
            ),
            "rwe_agent": _AgentSpec(
                key="rwe_agent",
                mcp_config_key="rwe_agent",
                chat_model_key="gemini",
                system_message="You are an expert in real-world clinical evidence analysis.",
            ),
            "clinical_trials_agent": _AgentSpec(
                key="clinical_trials_agent",
                mcp_config_key="clinical_trials_agent",
                chat_model_key="gemini",
                system_message="You are an expert in clinical trials data analysis.",
            ),
            "summarizer": _AgentSpec(
                key="summarizer",
                mcp_config_key="evidence_summarizer",
                chat_model_key="gemini",
                system_message="You are an expert in summarizing medical evidence from multiple sources.",
            ),
            "critique_guardrail_agent": _AgentSpec(
                key="critique_guardrail_agent",
                mcp_config_key="critique_guardrail_agent",
                chat_model_key="gemini",
                system_message="You are an expert in critiquing medical reports for accuracy and completeness.",
            ),
        }

    async def _get_or_create_agent(self, spec: _AgentSpec) -> MCPAgent:
        agent = self._agents.get(spec.key)
        if agent is not None:
            return agent

        lock = self._init_locks.setdefault(spec.key, asyncio.Lock())
        async with lock:
            agent = self._agents.get(spec.key)
            if agent is not None:
                return agent

            chat_model = self._model_registry.resolve(spec.chat_model_key)
            agent = MCPAgent(
                chat_model,
                spec.mcp_config_key,
                spec.system_message,
            )
            await agent.initialize()
            self._agents[spec.key] = agent
            return agent

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
        agent = await self._get_or_create_agent(
            self._specs["label_agent"]
        )
        out = await self._execute_step("label_interpretation", agent, user_input)
        if out["status"] != "success":
            raise RuntimeError(out.get("error") or "Label interpretation failed")
        return str(out["output"] or "")

    async def evidence_report(self, request_data: dict[str, Any]):
        # Lazily initialize agents (thread-safe) and mirror legacy attributes.
        faers_agent, pubmed_agent, clinical_trials_agent, summarizer_agent = await asyncio.gather(
            self._get_or_create_agent(self._specs["faers_agent"]),
            self._get_or_create_agent(self._specs["rwe_agent"]),
            self._get_or_create_agent(self._specs["clinical_trials_agent"]),
            self._get_or_create_agent(self._specs["summarizer"]),
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
        pubmed_prompt = (
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

        # These three are independent; run in parallel to reduce latency.
        faers_out, pubmed_out, clinical_trials_out = await asyncio.gather(
            self._execute_step("FAERS_analysis", faers_agent, faers_prompt),
            self._execute_step("PubMed_analysis", pubmed_agent, pubmed_prompt),
            self._execute_step("Clinical_Trials_analysis", clinical_trials_agent, clinical_trials_prompt),
        )

        summary_prompt = (
            "Summarize the evidence findings from FAERS, PubMed, and Clinical Trials into a concise report.\n"
            f"set_id: {drug_set_id}\n"
            f"drug_name: {drug_name}\n\n"
            f"FAERS findings:\n{faers_out.get('output')}\n\n"
            f"PubMed findings:\n{pubmed_out.get('output')}\n\n"
            f"Clinical Trials findings:\n{clinical_trials_out.get('output')}\n"
        )
        summary_out = await self._execute_step("Evidence_Summarization", summarizer_agent, summary_prompt)

        return {
            "faers_report": faers_out["output"] if faers_out["status"] == "success" else "Error generating FAERS report.",
            "pubmed_report": pubmed_out["output"] if pubmed_out["status"] == "success" else "Error generating PubMed report.",
            "clinical_trials_report": clinical_trials_out["output"] if clinical_trials_out["status"] == "success" else "Error generating Clinical Trials report.",
            "summary": summary_out["output"] if summary_out["status"] == "success" else "Error generating summary.",
        }