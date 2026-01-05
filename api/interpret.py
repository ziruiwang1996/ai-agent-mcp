from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from services.container import Services

router = APIRouter(prefix="/api/interpret")
class InterpretationRequest(BaseModel):
    drug_name: str
    section: str 
    content: str
    
class InterpretationResponse(BaseModel):
    drug_name: str
    section: str
    interpretation: str

@router.post("/", response_model=InterpretationResponse)
async def interpret_drug_section_post(payload: InterpretationRequest, request: Request):
    try:
        services: Services | None = getattr(request.app.state, "services", None)
        if services is None:
            raise HTTPException(status_code=500, detail="Services not initialized")
        orchestrator = services.orchestrator

        request_text = (
            f"Explain {payload.section} section of the FDA label for {payload.drug_name}: \n {payload.content}"
        )
        response = await orchestrator.interpret_label(request_text)
        return InterpretationResponse(
            drug_name=payload.drug_name,
            section=payload.section,
            interpretation=response,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interpreting drug section: {str(e)}")