from __future__ import annotations
from fastapi import APIRouter, Request
from services.container import Services

router = APIRouter(prefix="/api/tools")

def _serialize_tools(tools: object) -> list[dict[str, str]]:
    items = tools if isinstance(tools, list) else []
    return [
        {
            "name": getattr(tool, "name", "Unknown"),
            "description": getattr(tool, "description", "No description available"),
        }
        for tool in items
    ]


def _get_agent_tools(agent: object | None) -> list[dict[str, str]]:
    """Get tools from an agent.

    Convention in this codebase: agents expose `get_available_tools()`.
    If it's missing, fall back to a `.tools` attribute.
    """
    if agent is None:
        return []

    getter = getattr(agent, "get_available_tools", None)
    if callable(getter):
        tools = getter()
        return tools if isinstance(tools, list) else []

    return _serialize_tools(getattr(agent, "tools", []))


def _discover_agents(services_obj: Services) -> dict[str, object | None]:
    """Discover agents available in the current app state.

    Convention in this codebase:
    - Chat agent lives at `services.chat.chat_agent` (created after /chat/initialize)
    - Specialized agents are stored as attributes on `services.orchestrator` and
      are named like `<capability>_agent`.

    This avoids hardcoding a list of agents in the API layer.
    """
    agents: dict[str, object | None] = {"chat": services_obj.chat.chat_agent}

    orchestrator = getattr(services_obj, "orchestrator", None)
    if orchestrator is None:
        return agents

    for attr_name, value in vars(orchestrator).items():
        if not attr_name.endswith("_agent"):
            continue
        public_name = attr_name.removesuffix("_agent")
        agents[public_name] = value

    return agents


@router.get("/")
def get_available_tools(request: Request):
    """Return available tools grouped by agent.

    Note:
        Agents that have not been initialized yet will return an empty tool list.
        (e.g. chat tools are available only after POST /chat/initialize).
    """
    services_obj: Services | None = getattr(request.app.state, "services", None)
    if services_obj is None:
        return {"agents": {}, "message": "services not initialized"}

    discovered = _discover_agents(services_obj)
    agents = {
        name: {
            "initialized": agent is not None,
            "tools": _get_agent_tools(agent),
        }
        for name, agent in discovered.items()
    }

    # Convenience counts
    for value in agents.values():
        value["tools_count"] = len(value["tools"])

    return {"agents": agents}


@router.get("/{agent_name}")
def get_tools_for_agent(agent_name: str, request: Request):
    """Return available tools for a specific agent.

    Supported agent_name values:
        - chat
        - plus any `<capability>_agent` fields on the orchestrator
    """
    payload = get_available_tools(request)
    agents = payload.get("agents", {}) if isinstance(payload, dict) else {}

    if agent_name not in agents:
        return {
            "agent": agent_name,
            "initialized": False,
            "tools_count": 0,
            "tools": [],
            "error": f"Unknown agent '{agent_name}'. Supported: {sorted(agents.keys())}",
        }

    return {"agent": agent_name, **agents[agent_name]}
