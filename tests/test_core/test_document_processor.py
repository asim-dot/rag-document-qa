"""
Tests for document processing functionality.
"""
import os
import tempfile
from pathlib import Path

import pytest
from PyPDF2 import PdfWriter

from app.core.document_processor import DocumentProcessor, process_document_file
from app.utils.exceptions import DocumentProcessingError, ValidationError


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def setup_method(self):
        """Setup test instance."""
        self.processor = DocumentProcessor()
    
    def test_init(self):
        """Test processor initialization."""
        assert self.processor.chunk_size > 0
        assert self.processor.chunk_overlap >= 0
        assert self.processor.max_chunks > 0
        assert len(self.processor.allowed_extensions) > 0
    
    def test_create_chunks_simple_text(self):
        """Test basic text chunking."""
        text = "This is a simple test. " * 50  # Create text longer than chunk size
        chunks = self.processor._create_chunks(text)
        
        assert len(chunks) > 0
        assert all(len(chunk.text) <= self.processor.chunk_size for chunk in chunks)
        assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))
    
    def test_create_chunks_empty_text(self):
        """Test chunking with empty text."""
        chunks = self.processor._create_chunks("")
        assert len(chunks) == 0
        
        chunks = self.processor._create_chunks("   ")
        assert len(chunks) == 0
    
    def test_create_chunks_overlap(self):
        """Test chunk overlap functionality."""
        # Create text that will result in multiple chunks
        text = "Word " * 200  # Should create multiple chunks
        chunks = self.processor._create_chunks(text)
        
        if len(chunks) > 1:
            # Check that chunks have proper overlap
            chunk1_end = chunks[0].end_char
            chunk2_start = chunks[1].start_char
            assert chunk2_start < chunk1_end  # Should overlap
    
    def test_extract_txt_text(self):
        """Test text file extraction."""
        test_content = "This is a test file.\nWith multiple lines.\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            extracted = self.processor._extract_txt_text(temp_path)
            assert extracted == test_content
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_not_found(self):
        """Test validation with non-existent file."""
        with pytest.raises(ValidationError, match="File not found"):
            self.processor._validate_file("/nonexistent/file.pdf", "file.pdf")
    
    def test_validate_file_wrong_extension(self):
        """Test validation with unsupported extension."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValidationError, match="File type .xyz not allowed"):
                self.processor._validate_file(temp_path, "test.xyz")
        finally:
            os.unlink(temp_path)
    
    def test_process_txt_document(self):
        """Test processing a text document."""
        test_content = "This is a test document. " * 100
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = self.processor.process_document(temp_path, "test.txt")
            
            assert result.success is True
            assert result.total_chunks > 0
            assert result.processing_time > 0
            assert result.error_message is None
            assert result.metadata is not None
            assert "chunks" in result.metadata
        finally:
            os.unlink(temp_path)
    
    def test_process_empty_document(self):
        """Test processing an empty document."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            result = self.processor.process_document(temp_path, "empty.txt")
            
            assert result.success is False
            assert "No text content extracted" in result.error_message
        finally:
            os.unlink(temp_path)
    
    def test_convenience_function(self):
        """Test the convenience function."""
        test_content = "Test content for convenience function."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = process_document_file(temp_path, "test.txt")
            assert result.success is True
        finally:
            os.unlink(temp_path)


# Integration test
def test_full_processing_workflow():
    """Test the complete document processing workflow."""
    # Create test content
    test_content = "This is a comprehensive test. " * 200
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        # Process document
        result = process_document_file(temp_path, "integration_test.txt")
        
        # Verify results
        assert result.success is True
        assert result.total_chunks > 1  # Should create multiple chunks
        assert result.metadata is not None
        
        # Check chunk structure
        chunks_data = result.metadata["chunks"]
        assert len(chunks_data) == result.total_chunks
        
        # Verify chunk properties
        for i, chunk_data in enumerate(chunks_data):
            assert chunk_data["chunk_index"] == i
            assert len(chunk_data["text"]) > 0
            assert chunk_data["start_char"] >= 0
            assert chunk_data["end_char"] > chunk_data["start_char"]
        
    finally:
        os.unlink(temp_path)