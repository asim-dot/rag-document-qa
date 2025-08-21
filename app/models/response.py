"""
Standardized API response models.
"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.question import QuestionAnswer


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool
    message: str = ""
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QuestionAnswerResponse(APIResponse):
    """Response for question-answer requests."""
    
    data: Optional[QuestionAnswer] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)
    version: str = "1.0.0"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(APIResponse):
    """Error response model."""
    
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None