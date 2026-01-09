from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.container import Services

router = APIRouter(prefix="/api/evidence")

class EvidenceRequest(BaseModel):
    drug_set_id: str
    drug_name: str
    age: str
    sex: str
    weight: str
    is_pregnant: bool
    is_breastfeeding: bool
    conditions: Optional[str] = None
    other_medications: Optional[str] = None

class EvidenceResponse(BaseModel):
    drug_set_id: str
    drug_name: str
    faers_report: str
    rwe_report: str
    clinical_trials_report: str
    summary: str

@router.post("", response_model=EvidenceResponse)
async def generate_evidence_report(api_request: EvidenceRequest, request: Request):
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=500, detail="Services not initialized")
    request_data = api_request.model_dump()
    orchestrator = services.orchestrator
    report = await orchestrator.evidence_report(request_data)
    return EvidenceResponse(
        drug_set_id=api_request.drug_set_id,
        drug_name=api_request.drug_name,
        faers_report=report.get("faers_report") or "",
        rwe_report=report.get("rwe_report") or "",
        clinical_trials_report=report.get("clinical_trials_report") or "",
        summary=report.get("summary") or ""
    )
