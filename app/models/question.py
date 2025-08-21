"""
Pydantic models for question-answering operations.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    
    question: str = Field(..., min_length=1, max_length=1000)
    document_id: Optional[str] = None  # Optional: search specific document
    max_chunks: int = Field(default=5, ge=1, le=20)
    include_sources: bool = Field(default=True)


class SourceChunk(BaseModel):
    """Source chunk information for answers."""
    
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    relevance_score: Optional[float] = None
    start_char: int
    end_char: int


class QuestionAnswer(BaseModel):
    """Complete question-answer response."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    confidence_score: Optional[float] = None
    sources: List[SourceChunk] = Field(default_factory=list)
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QASessionStats(BaseModel):
    """Statistics for Q&A session."""
    
    total_questions: int = 0
    total_processing_time: float = 0.0
    average_response_time: float = 0.0
    questions_with_sources: int = 0
    unique_documents_referenced: int = 0