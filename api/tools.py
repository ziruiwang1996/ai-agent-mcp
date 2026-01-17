from __future__ import annotations
from fastapi import APIRouter, Request
from agent.agent_registry import AgentRegistry

router = APIRouter(prefix="/api/tools")

@router.get("/")
def get_available_tools(request: Request):
    registry = AgentRegistry.get_instance()

    response = {"agents": {}}
    agent_keys = registry.initialized_agents()
    if not agent_keys:
        return {"agents": {}}
    for key in agent_keys:
        agent = registry.get_initialized_agent(key)
        tools = agent.get_available_tools()
        response["agents"][key] = {
            "tools_count": len(tools),
            "tools": tools,
        }
    return response

@router.get("/{agent_name}")
def get_agent_available_tools(agent_name: str):
    registry = AgentRegistry.get_instance()
    if registry.is_agent_initialized(agent_name):
        agent = registry.get_initialized_agent(agent_name)
        tools = agent.get_available_tools()
        return {
            "agent": agent_name,
            "initialized": True,
            "tools_count": len(tools),
            "tools": tools,
        }

    return {
        "agent": agent_name,
        "initialized": False,
        "tools_count": 0,
        "tools": [],
        "error": f"Agent '{agent_name}' is not initialized.",
    }