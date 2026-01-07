from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.container import Services

router = APIRouter(prefix="/api/evidence")

class EvidenceRequest(BaseModel):
    drug_set_id: str    
    drug_name: str

class ContextualEvidenceRequest(BaseModel):
    drug_set_id: str
    drug_name: str
    age: str
    sex: str
    weight: str
    is_pregnant: bool
    is_breastfeeding: bool
    conditions: Optional[str]
    other_medications: Optional[str]

class EvidenceResponse(BaseModel):
    drug_set_id: str
    drug_name: str
    faers_report: str
    rwe_report: str
    clinical_trials_report: str
    summary: str

@router.post("/", response_model=EvidenceResponse)
def generate_evidence_report(api_request: EvidenceRequest, request: Request):
    try:
        services: Services | None = getattr(request.app.state, "services", None)
        if services is None:
            raise HTTPException(status_code=500, detail="Services not initialized")
        request_data = api_request.model_dump()
        orchestrator = services.orchestrator
        ## TODO: implement the logic to generate evidence report using orchestrator
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing services: {str(e)}")


@router.post("/personalized", response_model=EvidenceResponse)
def generate_contextual_evidence_report(api_request: ContextualEvidenceRequest, request: Request):
    pass
