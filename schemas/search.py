from typing import Optional
from pydantic import BaseModel


class SearchRequest(BaseModel):
    category: str       # "agencias de marketing"
    city: str           # "Buenos Aires"
    max_results: int = 20


class PlaceResult(BaseModel):
    place_id: str
    name: str
    website: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None


class SearchResponse(BaseModel):
    results: list[PlaceResult]
    total: int
    query: str
