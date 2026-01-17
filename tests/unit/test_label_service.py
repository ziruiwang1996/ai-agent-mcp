from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.label_service import LabelService


@pytest.mark.asyncio
async def test_label_service_workflow():
    with patch("services.label_service.AgentRegistry") as MockRegistry:
        mock_agent = AsyncMock()
        mock_registry = MockRegistry.get_instance.return_value
        mock_registry.resolve = AsyncMock(return_value=mock_agent)
        label_service = LabelService()
        input_data = {"drug_name": "Aspirin", "section_name": "usage", "section_content": "Pain relief"}
        with patch.object(label_service.app, "ainvoke", AsyncMock(return_value={"explanation": "Test explanation"})):
            result = await label_service.execute_workflow(input_data)
            assert result == "Test explanation"
