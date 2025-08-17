"""
Custom exception classes for the RAG system.
"""
from typing import Any, Dict, Optional


class RAGException(Exception):
    """Base exception for RAG system."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DocumentProcessingError(RAGException):
    """Raised when document processing fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class OpenAIError(RAGException):
    """Raised when OpenAI API calls fail."""
    pass


class ValidationError(RAGException):
    """Raised when input validation fails."""
    pass


class FileUploadError(RAGException):
    """Raised when file upload operations fail."""
    pass