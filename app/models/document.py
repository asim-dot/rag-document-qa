"""
Pydantic models for document-related operations.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            UUID: str
        }


class DocumentMetadata(BaseModel):
    """Document metadata information."""
    
    filename: str
    file_size: int
    file_type: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_timestamp: Optional[datetime] = None
    total_chunks: int = 0
    total_characters: int = 0
    status: str = "uploaded"  # uploaded, processing, processed, failed
    error_message: Optional[str] = None
    
    @validator('file_type', pre=True)
    def validate_file_type(cls, v):
        """Ensure file type is lowercase."""
        return v.lower() if v else v


class Document(BaseModel):
    """Main document model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str
        }


class ProcessingResult(BaseModel):
    """Result of document processing operation."""
    
    success: bool
    document_id: str
    total_chunks: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None