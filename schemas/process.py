from typing import Any, Dict, Optional

from pydantic import AnyHttpUrl, BaseModel, field_validator


class ProcessRequest(BaseModel):
    target_url: str
    max_crawl_pages: int = 1
    max_crawl_depth: int = 1
    skip_cleaning: bool = True
    my_service_info: Optional[str] = None
    company_tone: Optional[str] = None

    @field_validator("target_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("target_url debe comenzar con http:// o https://")
        return v


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
    finished_at: Optional[str] = None
    result: Optional[ProcessResult] = None
    error: Optional[str] = None
