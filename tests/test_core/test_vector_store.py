"""
Tests for ChromaDB vector store functionality.
"""
import tempfile
import shutil
from pathlib import Path

import pytest

from app.core.vector_store import ChromaVectorStore, get_vector_store, reset_vector_store
from app.models.document import DocumentChunk
from app.utils.exceptions import VectorStoreError


class TestChromaVectorStore:
    """Test ChromaDB vector store functionality."""
    
    def setup_method(self):
        """Setup test instance with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test vector store with temporary directory
        import app.config
        original_persist_dir = app.config.settings.chroma_persist_directory
        app.config.settings.chroma_persist_directory = self.temp_dir
        
        self.vector_store = ChromaVectorStore()
        
        # Restore original setting
        app.config.settings.chroma_persist_directory = original_persist_dir
    
    def teardown_method(self):
        """Cleanup temporary directory."""
        # Close ChromaDB client first
        if hasattr(self, 'vector_store') and self.vector_store.client:
            try:
                # Force close client connections
                del self.vector_store.client
                del self.vector_store
            except:
                pass
        
        # Windows-specific cleanup with retry
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            import time
            for attempt in range(3):
                try:
                    shutil.rmtree(self.temp_dir)
                    break
                except PermissionError:
                    time.sleep(0.5)  # Wait and retry
                    if attempt == 2:  # Last attempt
                        pass  # Ignore cleanup error
    
    def test_init(self):
        """Test vector store initialization."""
        assert self.vector_store.client is not None
        assert self.vector_store.collection is not None
        assert self.vector_store.collection_name == "documents"
    
    def test_add_chunks(self):
        """Test adding chunks to vector store."""
        # Create test chunks
        chunks = [
            DocumentChunk(
                document_id="test_doc",
                chunk_index=0,
                text="This is the first chunk of text.",
                start_char=0,
                end_char=32
            ),
            DocumentChunk(
                document_id="test_doc",
                chunk_index=1,
                text="This is the second chunk of text.",
                start_char=33,
                end_char=66
            )
        ]
        
        # Create dummy embeddings (1536 dimensions for text-embedding-ada-002)
        embeddings = [
            [0.1] * 1536,  # Dummy embedding 1
            [0.2] * 1536   # Dummy embedding 2
        ]
        
        # Add chunks
        result = self.vector_store.add_chunks(chunks, embeddings, "test_doc")
        assert result is True
        
        # Verify chunks were added
        stats = self.vector_store.get_collection_stats()
        assert stats["total_chunks"] == 2
    
    def test_add_chunks_mismatch(self):
        """Test adding chunks with mismatched embeddings."""
        chunks = [
            DocumentChunk(
                document_id="test_doc",
                chunk_index=0,
                text="Test chunk",
                start_char=0,
                end_char=10
            )
        ]
        
        embeddings = [[0.1] * 1536, [0.2] * 1536]  # 2 embeddings for 1 chunk
        
        with pytest.raises(VectorStoreError, match="Number of chunks and embeddings must match"):
            self.vector_store.add_chunks(chunks, embeddings, "test_doc")
    
    def test_search_similar(self):
        """Test similarity search."""
        # Add test data first
        chunks = [
            DocumentChunk(
                document_id="test_doc",
                chunk_index=0,
                text="Information about artificial intelligence and machine learning.",
                start_char=0,
                end_char=62
            ),
            DocumentChunk(
                document_id="test_doc",
                chunk_index=1,
                text="Details about natural language processing and text analysis.",
                start_char=63,
                end_char=122
            )
        ]
        
        embeddings = [
            [0.1] * 1536,
            [0.9] * 1536  # Different embedding
        ]
        
        self.vector_store.add_chunks(chunks, embeddings, "test_doc")
        
        # Search with query embedding similar to first chunk
        query_embedding = [0.1] * 1536
        results = self.vector_store.search_similar(query_embedding, n_results=2)
        
        assert len(results) == 2
        assert results[0]["text"] == chunks[0].text
        assert results[0]["metadata"]["document_id"] == "test_doc"
        assert results[0]["metadata"]["chunk_index"] == 0
    
    def test_delete_document(self):
        """Test document deletion."""
        # Add test data
        chunks = [
            DocumentChunk(
                document_id="test_doc",
                chunk_index=0,
                text="Test chunk 1",
                start_char=0,
                end_char=12
            ),
            DocumentChunk(
                document_id="test_doc",
                chunk_index=1,
                text="Test chunk 2",
                start_char=13,
                end_char=25
            )
        ]
        
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        self.vector_store.add_chunks(chunks, embeddings, "test_doc")
        
        # Verify chunks exist
        stats = self.vector_store.get_collection_stats()
        assert stats["total_chunks"] == 2
        
        # Delete document
        result = self.vector_store.delete_document("test_doc")
        assert result is True
        
        # Verify chunks are deleted
        stats = self.vector_store.get_collection_stats()
        assert stats["total_chunks"] == 0
    
    def test_get_collection_stats(self):
        """Test collection statistics."""
        # Empty collection
        stats = self.vector_store.get_collection_stats()
        assert stats["total_chunks"] == 0
        assert stats["collection_name"] == "documents"
        
        # Add some data
        chunks = [
            DocumentChunk(
                document_id="doc1",
                chunk_index=0,
                text="Test chunk",
                start_char=0,
                end_char=10
            )
        ]
        embeddings = [[0.1] * 1536]
        self.vector_store.add_chunks(chunks, embeddings, "doc1")
        
        # Check updated stats
        stats = self.vector_store.get_collection_stats()
        assert stats["total_chunks"] == 1
    
    def test_reset_collection(self):
        """Test collection reset."""
        # Add test data
        chunks = [
            DocumentChunk(
                document_id="test_doc",
                chunk_index=0,
                text="Test chunk",
                start_char=0,
                end_char=10
            )
        ]
        embeddings = [[0.1] * 1536]
        self.vector_store.add_chunks(chunks, embeddings, "test_doc")
        
        # Verify data exists
        stats = self.vector_store.get_collection_stats()
        assert stats["total_chunks"] == 1
        
        # Reset collection
        result = self.vector_store.reset_collection()
        assert result is True
        
        # Verify collection is empty
        stats = self.vector_store.get_collection_stats()
        assert stats["total_chunks"] == 0


def test_global_vector_store():
    """Test global vector store instance management."""
    # Reset to ensure clean state
    reset_vector_store()
    
    # Get instance
    store1 = get_vector_store()
    store2 = get_vector_store()
    
    # Should be the same instance
    assert store1 is store2
    
    # Reset and get new instance
    reset_vector_store()
    store3 = get_vector_store()
    
    # Should be different instance
    assert store1 is not store3