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
    faers_explanation: str
    rwe_explanation: str
    clinical_trials_explanation: str
    summary: str

@router.post("/", response_model=EvidenceResponse)
async def generate_evidence_report(payload: EvidenceRequest, request: Request):
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=500, detail="Services not initialized")

    request_data = payload.model_dump(exclude={"thread_id"})
    evidence_service = services.evidence
    report = await evidence_service.execute_workflow(request_data)

    return EvidenceResponse(
        drug_set_id=payload.drug_set_id,
        drug_name=payload.drug_name,
        faers_explanation=report.get("faers_explanation") or "",
        rwe_explanation=report.get("rwe_explanation") or "",
        clinical_trials_explanation=report.get("clinical_trials_explanation") or "",
        summary=report.get("summary") or ""
    )