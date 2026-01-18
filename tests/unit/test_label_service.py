from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.label_service import LabelService


def test_label_service_workflow():
    with patch("services.label_service.AgentRegistry") as MockRegistry:
        mock_agent = AsyncMock()
        mock_registry = MockRegistry.get_instance.return_value
        mock_registry.resolve = AsyncMock(return_value=mock_agent)
        label_service = LabelService()
        input_data = {"drug_name": "Aspirin", "section_name": "usage", "section_content": "Pain relief"}
        with patch.object(label_service.app, "ainvoke", AsyncMock(return_value={"explanation": "Test explanation"})):
            import asyncio

            result = asyncio.run(label_service.execute_workflow(input_data))
            assert result == "Test explanation"


def test_label_service_prompts_include_fields():
    label_service = LabelService()

    captured: list[str] = []

    async def fake_execute_step(step_name, step_instance, input_data):
        captured.append(input_data)
        return {"status": "success", "output": "ok", "error": None, "step": step_name}

    label_service._execute_step = fake_execute_step

    import asyncio

    asyncio.run(
        label_service._run_label_interpreter(
            {"drug_name": "Aspirin", "section_name": "usage", "section_content": "Pain relief"}
        )
    )
    asyncio.run(
        label_service._run_explainer({"drug_name": "Aspirin", "interpretation": "interp"})
    )

    assert "Aspirin" in captured[0]
    assert "usage" in captured[0]
    assert "Pain relief" in captured[0]
    assert "interp" in captured[1]


def test_label_service_execute_step_failure():
    label_service = LabelService()

    class BrokenAgent:
        async def process_input(self, input_data):
            raise RuntimeError("boom")

    import asyncio

    result = asyncio.run(label_service._execute_step("label", BrokenAgent(), "input"))
    assert result["status"] == "failure"
    assert "boom" in result["error"]
