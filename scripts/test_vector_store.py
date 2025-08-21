"""
Manual testing script for vector store functionality.
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.vector_store import get_vector_store, reset_vector_store
from app.models.document import DocumentChunk
from app.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def test_vector_store_operations():
    """Test basic vector store operations."""
    logger.info("Starting vector store testing...")
    
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        logger.info("✅ Vector store initialized successfully")
        
        # Get initial stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Initial collection stats: {stats}")
        
        # Create test chunks
        test_chunks = [
            DocumentChunk(
                document_id="test_document_1",
                chunk_index=0,
                text="Artificial intelligence is transforming how we work and live.",
                start_char=0,
                end_char=61
            ),
            DocumentChunk(
                document_id="test_document_1",
                chunk_index=1,
                text="Machine learning algorithms can process vast amounts of data.",
                start_char=62,
                end_char=124
            ),
            DocumentChunk(
                document_id="test_document_1", 
                chunk_index=2,
                text="Natural language processing enables computers to understand text.",
                start_char=125,
                end_char=189
            )
        ]
        
        # Create dummy embeddings (in real usage, these come from OpenAI)
        dummy_embeddings = [
            [0.1 + i * 0.1] * 1536 for i in range(len(test_chunks))
        ]
        
        logger.info(f"Created {len(test_chunks)} test chunks with embeddings")
        
        # Add chunks to vector store
        logger.info("Adding chunks to vector store...")
        result = vector_store.add_chunks(test_chunks, dummy_embeddings, "test_document_1")
        
        if result:
            logger.info("✅ Successfully added chunks to vector store")
        else:
            logger.error("❌ Failed to add chunks to vector store")
            return
        
        # Get updated stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Updated collection stats: {stats}")
        
        # Test similarity search
        logger.info("Testing similarity search...")
        query_embedding = [0.15] * 1536  # Similar to first chunk
        
        search_results = vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=2
        )
        
        logger.info(f"Found {len(search_results)} similar chunks:")
        for i, result in enumerate(search_results):
            logger.info(f"  Result {i+1}:")
            logger.info(f"    Text: {result['text'][:100]}...")
            logger.info(f"    Document ID: {result['metadata']['document_id']}")
            logger.info(f"    Chunk Index: {result['metadata']['chunk_index']}")
            if result['distance'] is not None:
                logger.info(f"    Distance: {result['distance']:.4f}")
        
        # Test document-specific search
        logger.info("Testing document-specific search...")
        doc_results = vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=3,
            document_id="test_document_1"
        )
        
        logger.info(f"Found {len(doc_results)} chunks in specific document")
        
        # Test deletion
        logger.info("Testing document deletion...")
        delete_result = vector_store.delete_document("test_document_1")
        
        if delete_result:
            logger.info("✅ Successfully deleted document")
        else:
            logger.error("❌ Failed to delete document")
        
        # Check final stats
        final_stats = vector_store.get_collection_stats()
        logger.info(f"Final collection stats: {final_stats}")
        
        logger.info("✅ All vector store tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vector_store_operations()