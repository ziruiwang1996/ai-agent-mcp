from __future__ import annotations
from typing import Any, TypedDict
from agent.agent_registry import AgentRegistry
from agent.mcp_agent import MCPAgent
from agent.model_registry import ModelRegistry
from langgraph.graph import START, END, StateGraph

class WorkflowState(TypedDict, total=False):
    set_id: str
    drug_name: str
    user_profile: dict[str, Any]  # e.g., age/sex/conditions
    faers_evidence: Any
    rwe_evidence: Any
    clinical_trials_evidence: Any
    explanations: dict[str, str]
    summary: str
    join_ready: bool
    explainer_started: bool

class EvidenceService:
    def __init__(self):
        self._model_registry = ModelRegistry.get_instance()
        self._agent_registry = AgentRegistry.get_instance()

        self._workflow = StateGraph(WorkflowState)
        self._register_nodes()
        self.app = self._workflow.compile()

    def _register_nodes(self) -> None:
        # upstream resources collectors 
        self._workflow.add_node("faers_agent", self._run_faers_agent)
        self._workflow.add_node("rwe_agent", self._run_rwe_agent)
        self._workflow.add_node("clinical_trials_agent", self._run_clinical_trials_agent)

        # downstream synthesizers
        self._workflow.add_node("join", self._join_reports)
        self._workflow.add_node("explainer", self._run_explainer)
        self._workflow.add_node("summarizer", self._run_summarizer)

        # Fan out
        self._workflow.add_edge(START, "faers_agent")
        self._workflow.add_edge(START, "rwe_agent")
        self._workflow.add_edge(START, "clinical_trials_agent")

        # Fan in
        self._workflow.add_edge("faers_agent", "join")
        self._workflow.add_edge("rwe_agent", "join")
        self._workflow.add_edge("clinical_trials_agent", "join")

        # Join routes: only proceed when all evidence present
        self._workflow.add_conditional_edges(
            "join",
            self._route_after_join,
            {
                "explain": "explainer",
                "wait": END,  # IMPORTANT: end *this branch*; graph keeps running for other branches
            },
        )
        self._workflow.add_edge("explainer", "summarizer")
        self._workflow.add_edge("summarizer", END)

    async def _execute_step(
            self, 
            step_name: str, 
            step_instance: MCPAgent, 
            input_data: dict[str, Any]
    ) -> dict[str, Any]:
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

    def _generate_input_prompt(self, state: WorkflowState) -> str:
        user_profile = state.get("user_profile", {})
        return (
            f"Drug name: {state.get('drug_name', '')}\n"
            f"Set ID: {state.get('set_id', '')}\n"
            "Patient demographics:\n"
            f"- age: {user_profile.get('age')}\n"
            f"- sex: {user_profile.get('sex')}\n"
            f"- weight: {user_profile.get('weight')}\n"
            f"- is_pregnant: {user_profile.get('is_pregnant')}\n"
            f"- is_breastfeeding: {user_profile.get('is_breastfeeding')}\n"
            f"- conditions: {user_profile.get('conditions')}\n"
            f"- other_medications: {user_profile.get('other_medications')}\n"
        )

    async def _run_faers_agent(self, state: WorkflowState) -> WorkflowState:
        faers_agent = await self._agent_registry.resolve("faers_agent")
        input_prompt = self._generate_input_prompt(state)
        out = await self._execute_step("faers_agent", faers_agent, input_prompt)
        result = out["output"] if out["status"] == "success" else out["error"]
        return {"faers_evidence": result}
    
    async def _run_rwe_agent(self, state: WorkflowState) -> WorkflowState:
        rwe_agent = await self._agent_registry.resolve("rwe_agent")
        input_prompt = self._generate_input_prompt(state)
        out = await self._execute_step("rwe_agent", rwe_agent, input_prompt)
        result = out["output"] if out["status"] == "success" else out["error"]
        return {"rwe_evidence": result}
    
    async def _run_clinical_trials_agent(self, state: WorkflowState) -> WorkflowState:
        clinical_trials_agent = await self._agent_registry.resolve("clinical_trials_agent")
        input_prompt = self._generate_input_prompt(state)
        out = await self._execute_step("clinical_trials_agent", clinical_trials_agent, input_prompt)
        result = out["output"] if out["status"] == "success" else out["error"]
        return {"clinical_trials_evidence": result}
    
    async def _run_explainer(self, state: WorkflowState) -> WorkflowState:
        explainer = await self._agent_registry.resolve("explainer_agent")
        evidence_sources = {
            "faers_adverse_event_reports": state.get("faers_evidence"),
            "real_world_evidence_studies": state.get("rwe_evidence"),
            "clinical_trials_studies": state.get("clinical_trials_evidence"),
        }
        evidence_sources = {k: v for k, v in evidence_sources.items() if v is not None}
        if not evidence_sources:
            return {"explanations": {}}

        explanations: dict[str, str] = {}
        user_profile = state.get("user_profile")
        # Loop across evidence sources so each report receives a dedicated explainer pass.
        for source_name, payload in evidence_sources.items():
            input_prompt = (
                "Generate a patient-facing explanation focused on a single evidence source.\n"
                f"Drug: {state.get('drug_name')}\n"
                f"User profile: {user_profile}\n"
                f"Target evidence source name: {source_name}\n\n"
                f"Target evidence payload (JSON or text): {payload}\n"
            )
            out = await self._execute_step("explainer", explainer, input_prompt)
            result_text = out["output"] if out["status"] == "success" else out["error"]
            if not isinstance(result_text, str):
                result_text = str(result_text)
            explanations[source_name] = result_text

        return {"explanations": explanations}
    
    def _run_summarizer(self, state: WorkflowState) -> WorkflowState:
        try:
            model_instance = self._model_registry.resolve("summarizer")
            explanations = state.get("explanations")
            if isinstance(explanations, dict) and explanations:
                segments = []
                for source_name, text in explanations.items():
                    segments.append(f"{source_name}:\n{text}")
                input_data = "\n\n".join(segments)
            else:
                input_data = state.get("explanations", "") or ""
            if not input_data:
                return {"summary": ""}
            output = model_instance.summarization(input_data)
            return {"summary": output.summary_text}
        except Exception as e:
            raise RuntimeError(f"Error executing step summarization: {str(e)}") from e
        
    def _join_reports(self, state: WorkflowState) -> WorkflowState:
        required = (
            state.get("faers_evidence"),
            state.get("rwe_evidence"),
            state.get("clinical_trials_evidence"),
        )
        if state.get("explainer_started") is True:
            return {}
        # Check if all three collector outputs exist in state (success or failure payload)
        if all(val is not None for val in required):
            return {"explainer_started": True, "join_ready": True}
        # Not ready yet
        return {"join_ready": False}
    
    def _route_after_join(self, state: WorkflowState) -> str:
        if state.get("join_ready") is True and state.get("explainer_started") is True:
            return "explain"
        return "wait"

    async def execute_workflow(self, input_data: dict[str, Any]) -> dict[str, Any]:
        user_profile = {
            "age": input_data.get('age'),
            "sex": input_data.get('sex'),
            "weight": input_data.get('weight'),
            "is_pregnant": input_data.get('is_pregnant'),
            "is_breastfeeding": input_data.get('is_breastfeeding'),
            "conditions": input_data.get('conditions'),
            "other_medications": input_data.get('other_medications')
        }
        output = await self.app.ainvoke(
            {
                "set_id": input_data.get("drug_set_id", ""),
                "drug_name": input_data.get("drug_name", ""),
                "user_profile": user_profile
            }
        )
        explanations = output.get("explanations") or {}
        return {
            "faers_explanation": explanations.get("faers_adverse_event_reports", ""),
            "rwe_explanation": explanations.get("real_world_evidence_studies", ""),
            "clinical_trials_explanation": explanations.get("clinical_trials_studies", ""),
            "summary": output.get("summary", ""),
        }