"""
Manual testing script for document processing.
"""
import os
import tempfile
from pathlib import Path

from app.core.document_processor import process_document_file
from app.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def create_test_documents():
    """Create test documents for processing."""
    test_files = []
    
    # Create a test TXT file
    txt_content = """
This is a comprehensive test document for the RAG system.

The document contains multiple paragraphs to test the chunking functionality.
Each paragraph should provide meaningful content for testing purposes.

This system is designed to process documents and answer questions about them.
The chunking algorithm should split this text into manageable pieces while
maintaining context and ensuring proper overlap between chunks.

Additional content helps test the robustness of the processing pipeline.
We want to ensure that the system can handle various types of content
including paragraphs, sentences, and different text structures.

This concludes our test document content.
    """.strip()
    
    # Create temporary TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(txt_content)
        test_files.append((f.name, "test_document.txt"))
    
    return test_files


def test_document_processing():
    """Test document processing with sample files."""
    logger.info("Starting document processing tests...")
    
    # Create test documents
    test_files = create_test_documents()
    
    for file_path, filename in test_files:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing file: {filename}")
        logger.info(f"File path: {file_path}")
        
        try:
            # Process the document
            result = process_document_file(file_path, filename)
            
            if result.success:
                logger.info(f"✅ Processing successful!")
                logger.info(f"   Document ID: {result.document_id}")
                logger.info(f"   Total chunks: {result.total_chunks}")
                logger.info(f"   Processing time: {result.processing_time:.3f} seconds")
                
                if result.metadata:
                    logger.info(f"   File size: {result.metadata['file_size']} bytes")
                    logger.info(f"   File type: {result.metadata['file_type']}")
                    logger.info(f"   Total characters: {result.metadata['total_characters']}")
                    
                    # Show first few chunks
                    chunks = result.metadata.get('chunks', [])
                    logger.info(f"\n   First few chunks:")
                    for i, chunk in enumerate(chunks[:3]):
                        logger.info(f"   Chunk {i}: {chunk['text'][:100]}...")
                        logger.info(f"   Range: {chunk['start_char']}-{chunk['end_char']}")
                
            else:
                logger.error(f"❌ Processing failed!")
                logger.error(f"   Error: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Exception during processing: {e}")
        
        finally:
            # Cleanup
            try:
                os.unlink(file_path)
                logger.info(f"   Cleaned up test file: {file_path}")
            except Exception as e:
                logger.warning(f"   Failed to cleanup {file_path}: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info("Document processing tests completed!")


if __name__ == "__main__":
    test_document_processing()