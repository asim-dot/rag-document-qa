"""
Core document processing functionality.
"""
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import mimetypes

# PDF processing
import PyPDF2

# Document models
from app.models.document import Document, DocumentChunk, DocumentMetadata, ProcessingResult
from app.config import settings
from app.utils.logger import get_logger
from app.utils.exceptions import DocumentProcessingError, ValidationError

logger = get_logger(__name__)


class DocumentProcessor:
    """Handles document text extraction and chunking."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_chunks = settings.max_chunks_per_document
        self.allowed_extensions = settings.allowed_extensions
        
    def process_document(self, file_path: str, filename: str) -> ProcessingResult:
        """
        Process a document file and return structured result.
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            
        Returns:
            ProcessingResult with processing details
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting document processing for: {filename}")
            
            # Validate file
            self._validate_file(file_path, filename)
            
            # Extract text based on file type
            text_content = self._extract_text(file_path, filename)
            
            if not text_content.strip():
                raise DocumentProcessingError(
                    f"No text content extracted from {filename}",
                    error_code="EMPTY_CONTENT"
                )
            
            # Create chunks
            chunks = self._create_chunks(text_content)
            
            if len(chunks) > self.max_chunks:
                logger.warning(f"Document {filename} has {len(chunks)} chunks, truncating to {self.max_chunks}")
                chunks = chunks[:self.max_chunks]
            
            # Create document metadata
            file_stats = os.stat(file_path)
            metadata = DocumentMetadata(
                filename=filename,
                file_size=file_stats.st_size,
                file_type=Path(filename).suffix.lower(),
                total_chunks=len(chunks),
                total_characters=len(text_content),
                status="processed"
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks in {processing_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                document_id=str(metadata.filename),  # We'll use proper UUID later
                total_chunks=len(chunks),
                processing_time=processing_time,
                metadata={
                    "filename": filename,
                    "file_size": file_stats.st_size,
                    "file_type": metadata.file_type,
                    "total_characters": len(text_content),
                    "chunks": [
                        {
                            "chunk_index": i,
                            "text": chunk.text,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char
                        }
                        for i, chunk in enumerate(chunks)
                    ]
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process {filename}: {str(e)}"
            logger.error(error_msg)
            
            return ProcessingResult(
                success=False,
                document_id="",
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def _validate_file(self, file_path: str, filename: str) -> None:
        """Validate file exists and has allowed extension."""
        if not os.path.exists(file_path):
            raise ValidationError(f"File not found: {file_path}")
        
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise ValidationError(
                f"File type {file_ext} not allowed. Supported: {self.allowed_extensions}"
            )
        
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValidationError(
                f"File size {file_size} bytes exceeds maximum {max_size_bytes} bytes"
            )
    
    def _extract_text(self, file_path: str, filename: str) -> str:
        """Extract text content from file based on type."""
        file_ext = Path(filename).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_ext == '.txt':
                return self._extract_txt_text(file_path)
            else:
                raise DocumentProcessingError(
                    f"Unsupported file type: {file_ext}",
                    error_code="UNSUPPORTED_TYPE"
                )
        except Exception as e:
            raise DocumentProcessingError(
                f"Text extraction failed for {filename}: {str(e)}",
                error_code="EXTRACTION_FAILED"
            )
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text_content = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if len(pdf_reader.pages) == 0:
                    raise DocumentProcessingError("PDF has no pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                return "\n\n".join(text_content)
                
        except Exception as e:
            raise DocumentProcessingError(f"PDF processing failed: {str(e)}")
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            # Try UTF-8 first, fallback to other encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            
            raise DocumentProcessingError("Could not decode text file with any supported encoding")
            
        except Exception as e:
            raise DocumentProcessingError(f"Text file processing failed: {str(e)}")
    
    def _create_chunks(self, text: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        chunks = []
        text_length = len(text)
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at word boundaries (avoid cutting words)
            if end < text_length:
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk = DocumentChunk(
                    document_id="",  # Will be set when document is created
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "length": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position (with overlap)
            start = start + self.chunk_size - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks


# Convenience function for easy import
def process_document_file(file_path: str, filename: str) -> ProcessingResult:
    """
    Convenience function to process a document file.
    
    Args:
        file_path: Path to the file
        filename: Original filename
        
    Returns:
        ProcessingResult
    """
    processor = DocumentProcessor()
    return processor.process_document(file_path, filename)