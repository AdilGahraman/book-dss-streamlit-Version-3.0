from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class RatingRequest(BaseModel):
    Age: float
    user_avg_rating: float
    user_consistency: float
    book_avg_rating: float
    book_popularity: float
    book_rating_std: float
    age_book_gap: float
    rating_difference: float

class LikeRequest(BaseModel):
    Age: float
    user_avg_rating: float
    user_consistency: float
    book_avg_rating: float
    book_popularity: float
    age_book_gap: float
    popularity_ratio: float

class NlpRequest(BaseModel):
    text: str

class ForecastRequest(BaseModel):
    months_ahead: int = 6
    future_rating_count: Optional[List[float]] = None  # optional exog input

class ClusterRequest(BaseModel):
    user_avg_rating: float
    user_activity: float
    user_consistency: float
    Age: float

class ApiResponse(BaseModel):
    ok: bool
    data: Dict[str, Any]
