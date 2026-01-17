from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services.evidence_service import EvidenceService


@pytest.mark.asyncio
async def test_execute_workflow_maps_explanations(monkeypatch: pytest.MonkeyPatch):
    service = EvidenceService()
    monkeypatch.setattr(
        service.app,
        "ainvoke",
        AsyncMock(
            return_value={
                "explanations": {"faers_adverse_event_reports": "faers"},
                "summary": "summary",
            }
        ),
    )

    result = await service.execute_workflow(
        {
            "drug_set_id": "set-1",
            "drug_name": "drug",
            "age": "30",
            "sex": "f",
            "weight": "70",
            "is_pregnant": False,
            "is_breastfeeding": False,
        }
    )

    assert result["faers_explanation"] == "faers"
    assert result["summary"] == "summary"


def test_route_after_join():
    service = EvidenceService()
    assert service._route_after_join({"join_ready": True, "explainer_started": True}) == "explain"
    assert service._route_after_join({"join_ready": False}) == "wait"


def test_join_reports_requires_all_inputs():
    service = EvidenceService()
    assert service._join_reports({"faers_evidence": "a"}) == {"join_ready": False}
    assert service._join_reports(
        {
            "faers_evidence": "a",
            "rwe_evidence": "b",
            "clinical_trials_evidence": "c",
        }
    ) == {"explainer_started": True, "join_ready": True}
