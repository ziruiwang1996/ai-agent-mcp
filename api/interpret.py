from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from services.container import Services

router = APIRouter(prefix="/api/interpret")
class InterpretationRequest(BaseModel):
    thread_id: str
    drug_name: str
    section_name: str 
    section_content: str
    
class InterpretationResponse(BaseModel):
    drug_name: str
    section_name: str
    interpretation: str

@router.post("/", response_model=InterpretationResponse)
async def interpret_drug_section_post(payload: InterpretationRequest, request: Request):
    try:
        services: Services | None = getattr(request.app.state, "services", None)
        if services is None:
            raise HTTPException(status_code=500, detail="Services not initialized")
        label_service = services.label

        thread_id = (payload.thread_id or "").strip()
        if not thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required")

        input_data = payload.model_dump(exclude={"thread_id"})
        response = await label_service.execute_workflow(input_data, thread_id=thread_id)
        return InterpretationResponse(
            drug_name=payload.drug_name,
            section_name=payload.section_name,
            interpretation=response,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interpreting drug section: {str(e)}")