from pydantic import BaseModel
from typing import Optional, Dict, Any


class ProcessRequest(BaseModel):
    target_url: str
    skip_cleaning: bool = True


class ProcessStartResponse(BaseModel):
    run_id: str
    status: str


class ProcessResult(BaseModel):
    final_email: Optional[str] = None
    profile_data: Optional[Dict[str, Any]] = None
    target_url: Optional[str] = None
    run_id: Optional[str] = None


class ProcessStatusResponse(BaseModel):
    run_id: str
    target_url: str
    status: str
    current_step: str
    steps: Dict[str, str]
    created_at: str
    finished_at: Optional[str]
    result: Optional[ProcessResult]
    error: Optional[str]