"""
Manual testing script for Q&A engine functionality.
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.qa_engine import ask_question, get_qa_engine
from app.core.vector_store import get_vector_store
from app.core.embeddings import get_embedding_service
from app.core.document_processor import process_document_file
from app.models.document import DocumentChunk
from app.models.question import QuestionRequest
from app.utils.logger import setup_logging, get_logger
from app.config import settings

# Setup logging
setup_logging()
logger = get_logger(__name__)


def test_full_qa_workflow():
    """Test the complete Q&A workflow with document processing."""
    logger.info("Starting full Q&A workflow test...")
    
    try:
        # Check if OpenAI API key is configured
        if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
            logger.error("❌ OpenAI API key not configured in .env file")
            logger.info("Please add your OpenAI API key to .env file: OPENAI_API_KEY=your_key_here")
            return
        
        # Step 1: Create test document content
        test_content = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
        that can perform tasks that typically require human intelligence. These tasks include learning, 
        reasoning, problem-solving, perception, and language understanding.

        Machine Learning is a subset of AI that enables computers to learn and improve from experience 
        without being explicitly programmed. It uses algorithms and statistical models to analyze and 
        draw insights from data.

        Natural Language Processing (NLP) is another important area of AI that focuses on the interaction 
        between computers and humans through natural language. It enables machines to understand, 
        interpret, and generate human language in a valuable way.

        Deep Learning is a subset of machine learning that uses neural networks with multiple layers 
        to model and understand complex patterns in data. It has been particularly successful in 
        image recognition, speech recognition, and natural language processing tasks.
        """
        
        # Step 2: Create and save test document
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        logger.info("Created test document with AI content")
        
        # Step 3: Process the document
        logger.info("Processing document...")
        processing_result = process_document_file(temp_file_path, "ai_test_document.txt")
        
        if not processing_result.success:
            logger.error(f"❌ Document processing failed: {processing_result.error_message}")
            return
        
        logger.info(f"✅ Document processed: {processing_result.total_chunks} chunks")
        
        # Step 4: Generate embeddings and store in vector database
        logger.info("Generating embeddings and storing in vector database...")
        
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()
        
        # Extract chunk texts
        chunks_data = processing_result.metadata["chunks"]
        chunk_texts = [chunk["text"] for chunk in chunks_data]
        
        # Generate embeddings
        embeddings = embedding_service.generate_embeddings_batch(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Create DocumentChunk objects
        document_chunks = []
        for chunk_data in chunks_data:
            chunk = DocumentChunk(
                document_id="ai_test_document",
                chunk_index=chunk_data["chunk_index"],
                text=chunk_data["text"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"]
            )
            document_chunks.append(chunk)
        
        # Store in vector database
        vector_store.add_chunks(document_chunks, embeddings, "ai_test_document")
        logger.info("✅ Chunks stored in vector database")
        
        # Step 5: Test Q&A functionality
        logger.info("\n" + "="*60)
        logger.info("Testing Q&A functionality...")
        
        test_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What is the difference between AI and machine learning?",
            "What is deep learning used for?",
            "What is natural language processing?",
            "What are neural networks?"  # This might not have direct answer
        ]
        
        qa_engine = get_qa_engine()
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- Question {i} ---")
            logger.info(f"Q: {question}")
            
            try:
                # Ask the question
                response = ask_question(question, max_chunks=3)
                
                logger.info(f"A: {response.answer}")
                logger.info(f"Confidence: {response.confidence_score:.2f}")
                logger.info(f"Processing time: {response.processing_time:.3f}s")
                logger.info(f"Sources used: {len(response.sources)}")
                
                if response.sources:
                    logger.info("Source chunks:")
                    for j, source in enumerate(response.sources[:2]):  # Show first 2 sources
                        logger.info(f"  {j+1}. {source.text[:100]}...")
                        if source.relevance_score:
                            logger.info(f"     Relevance: {source.relevance_score:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Error answering question: {e}")
        
        # Step 6: Show session statistics
        logger.info(f"\n" + "="*60)
        stats = qa_engine.get_session_stats()
        logger.info("Session Statistics:")
        logger.info(f"  Total questions: {stats.total_questions}")
        logger.info(f"  Average response time: {stats.average_response_time:.3f}s")
        logger.info(f"  Questions with sources: {stats.questions_with_sources}")
        
        # Step 7: Test document-specific search
        logger.info(f"\n--- Testing document-specific search ---")
        doc_response = ask_question(
            "What topics are covered in this document?", 
            document_id="ai_test_document",
            max_chunks=5
        )
        logger.info(f"Document-specific answer: {doc_response.answer}")
        
        logger.info("\n✅ Full Q&A workflow test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Q&A workflow test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            os.unlink(temp_file_path)
            logger.info("Cleaned up test file")
        except:
            pass


def test_qa_engine_only():
    """Test Q&A engine without document processing (assumes data exists)."""
    logger.info("Testing Q&A engine with existing data...")
    
    try:
        # Check if OpenAI API key is configured
        if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
            logger.error("❌ OpenAI API key not configured in .env file")
            return
        
        # Get Q&A engine
        qa_engine = get_qa_engine()
        logger.info("✅ Q&A engine initialized")
        
        # Test basic functionality
        test_questions = [
            "Hello, can you help me?",
            "What is machine learning?",
            "How does artificial intelligence work?"
        ]
        
        for question in test_questions:
            logger.info(f"\nTesting question: {question}")
            
            try:
                response = ask_question(question, max_chunks=3)
                logger.info(f"Answer: {response.answer[:200]}...")
                logger.info(f"Confidence: {response.confidence_score:.2f}")
                logger.info(f"Sources: {len(response.sources)}")
                
            except Exception as e:
                logger.error(f"Error with question '{question}': {e}")
        
        logger.info("\n✅ Q&A engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Q&A engine test failed: {e}")


### 5. `app/models/response.py`
# ```python
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