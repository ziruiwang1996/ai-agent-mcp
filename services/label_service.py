from __future__ import annotations
from typing import Any, TypedDict
from agent.agent_registry import AgentRegistry
from agent.mcp_agent import MCPAgent
from agent.model_registry import ModelRegistry
from langgraph.graph import START, END, StateGraph

class WorkflowState(TypedDict, total=False):
    drug_name: str
    section_name: str
    section_content: str
    interpretation: str
    explanation: str

class LabelService:
    def __init__(self):
        self._model_registry = ModelRegistry.get_instance()
        self._agent_registry = AgentRegistry.get_instance()

        self._workflow = StateGraph(WorkflowState)
        self._register_nodes()
        self._app = self._workflow.compile()
    
    def _register_nodes(self) -> None:
        self._workflow.add_node("interpreter_agent", self._run_label_interpreter)
        self._workflow.add_node("explainer", self._run_explainer)

        self._workflow.add_edge(START, "interpreter_agent")
        self._workflow.add_edge("interpreter_agent", "explainer")
        self._workflow.add_edge("explainer", END)

    async def _run_label_interpreter(self, state: WorkflowState) -> WorkflowState:
        interpreter = await self._agent_registry.resolve("label_agent")
        input_prompt = (
            f"Interpret the following label section for the drug {state['drug_name']}:\n\n"
            f"Section: {state['section_name']}\n\n"
            f"Content: {state['section_content']}\n\n"
        )
        out = await self._execute_step("label_interpretation", interpreter, input_prompt)
        if out["status"] == "success":
            return {"interpretation": out["output"]}
        else:
            return {"interpretation": out["error"]}
        
    async def _run_explainer(self, state: WorkflowState) -> WorkflowState:
        explainer = await self._agent_registry.resolve("explainer_agent")
        input_prompt = (
            "Generate a patient-facing, plain explanation using the label interpretation.\n"
            f"Drug: {state['drug_name']}\n\n"
            f"Interpretation: {state['interpretation']}\n\n"
        )
        out = await self._execute_step("explainer", explainer, input_prompt)
        if out["status"] == "success":
            return {"explanation": out["output"]}
        else:
            return {"explanation": out["error"]}

    async def _execute_step(self, step_name: str, step_instance: MCPAgent, input_data: str) -> dict[str, Any]:
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
        
    async def execute_workflow(self, input_data: dict[str, Any]) -> str:
        output = await self._app.ainvoke(input_data)
        return output['explanation']