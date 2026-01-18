from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services.evidence_service import EvidenceService


def test_execute_workflow_maps_explanations(monkeypatch: pytest.MonkeyPatch):
    service = EvidenceService()
    monkeypatch.setattr(
        service._app,
        "ainvoke",
        AsyncMock(
            return_value={
                "explanations": {"faers_adverse_event_reports": "faers"},
                "summary": "summary",
            }
        ),
    )

    import asyncio

    result = asyncio.run(
        service.execute_workflow(
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


def test_generate_input_prompt_includes_profile_fields():
    service = EvidenceService()
    prompt = service._generate_input_prompt(
        {
            "drug_name": "drug",
            "set_id": "set-1",
            "user_profile": {
                "age": "30",
                "sex": "f",
                "weight": "70",
                "is_pregnant": False,
                "is_breastfeeding": False,
                "conditions": "none",
                "other_medications": "none",
            },
        }
    )
    assert "drug" in prompt
    assert "set-1" in prompt
    assert "age: 30" in prompt
    assert "sex: f" in prompt


def test_run_explainer_fans_out_sources(monkeypatch: pytest.MonkeyPatch):
    service = EvidenceService()

    async def fake_execute_step(step_name, step_instance, input_data):
        return {"status": "success", "output": f"{step_name}-ok", "error": None}

    monkeypatch.setattr(service, "_execute_step", fake_execute_step)

    import asyncio

    result = asyncio.run(
        service._run_explainer(
            {
                "drug_name": "drug",
                "user_profile": {"age": "30"},
                "faers_evidence": "faers",
                "rwe_evidence": "rwe",
                "clinical_trials_evidence": "clinical",
            }
        )
    )
    assert "faers_adverse_event_reports" in result["explanations"]
    assert "real_world_evidence_studies" in result["explanations"]
    assert "clinical_trials_studies" in result["explanations"]


def test_run_summarizer_empty_explanations():
    service = EvidenceService()
    assert service._run_summarizer({"explanations": {}}) == {"summary": ""}


def test_run_summarizer_success(monkeypatch: pytest.MonkeyPatch):
    service = EvidenceService()

    class DummySummary:
        summary_text = "summary"

    class DummyModel:
        def summarization(self, text):
            return DummySummary()

    monkeypatch.setattr(service._model_registry, "resolve", lambda key: DummyModel())
    result = service._run_summarizer({"explanations": {"faers": "text"}})
    assert result["summary"] == "summary"


def test_run_summarizer_error(monkeypatch: pytest.MonkeyPatch):
    service = EvidenceService()
    monkeypatch.setattr(service._model_registry, "resolve", lambda key: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="Error executing step summarization"):
        service._run_summarizer({"explanations": {"faers": "text"}})
