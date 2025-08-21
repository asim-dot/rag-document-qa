"""
Tests for Q&A engine functionality.
"""
import pytest
from unittest.mock import Mock, patch

from app.core.qa_engine import QAEngine, get_qa_engine, reset_qa_engine, ask_question
from app.models.question import QuestionRequest, QuestionAnswer
from app.utils.exceptions import OpenAIError, EmbeddingError, VectorStoreError


class TestQAEngine:
    """Test Q&A engine functionality."""
    
    def setup_method(self):
        """Setup test instance."""
        # Mock the dependencies to avoid actual API calls
        self.mock_vector_store = Mock()
        self.mock_embedding_service = Mock()
        
        # Patch all dependencies before creating QAEngine
        self.patch_vector_store = patch('app.core.qa_engine.get_vector_store', return_value=self.mock_vector_store)
        self.patch_embedding_service = patch('app.core.qa_engine.get_embedding_service', return_value=self.mock_embedding_service)
        self.patch_openai = patch('app.core.qa_engine.openai')
        self.patch_settings = patch('app.core.qa_engine.settings')
        
        # Start all patches
        self.mock_vector_store_patch = self.patch_vector_store.start()
        self.mock_embedding_service_patch = self.patch_embedding_service.start()
        self.mock_openai_patch = self.patch_openai.start()
        self.mock_settings_patch = self.patch_settings.start()
        
        # Configure mock settings
        self.mock_settings_patch.openai_api_key = "test_api_key"
        self.mock_settings_patch.openai_model = "gpt-3.5-turbo"
        
        # Create QA engine with mocked dependencies
        self.qa_engine = QAEngine()
    
    def teardown_method(self):
        """Clean up patches."""
        patch.stopall()
    
    def test_init(self):
        """Test Q&A engine initialization."""
        assert self.qa_engine.model == "gpt-3.5-turbo"
        assert self.qa_engine.vector_store is not None
        assert self.qa_engine.embedding_service is not None
        assert self.qa_engine.session_stats.total_questions == 0
    
    @patch('app.core.qa_engine.openai.chat.completions.create')
    def test_answer_question_with_context(self, mock_openai_chat):
        """Test answering question with relevant context."""
        # Setup mocks
        self.mock_embedding_service.generate_embedding.return_value = [0.1] * 1536
        
        mock_chunks = [
            {
                "id": "doc1_0",
                "text": "Artificial intelligence is a field of computer science.",
                "metadata": {"document_id": "doc1", "chunk_index": 0, "start_char": 0, "end_char": 53},
                "distance": 0.2
            }
        ]
        self.mock_vector_store.search_similar.return_value = mock_chunks
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "According to the source, artificial intelligence is a field of computer science that focuses on creating intelligent machines."
        mock_openai_chat.return_value = mock_response
        
        # Test question
        request = QuestionRequest(question="What is artificial intelligence?")
        response = self.qa_engine.answer_question(request)
        
        # Assertions
        assert isinstance(response, QuestionAnswer)
        assert response.question == "What is artificial intelligence?"
        assert len(response.answer) > 0
        assert len(response.sources) == 1
        assert response.sources[0].document_id == "doc1"
        assert response.confidence_score is not None  # Just check it's not negative
        assert response.processing_time > 0
    
    @patch('app.core.qa_engine.openai.chat.completions.create')
    def test_answer_question_no_context(self, mock_openai_chat):
        """Test answering question without relevant context."""
        # Setup mocks
        self.mock_embedding_service.generate_embedding.return_value = [0.1] * 1536
        self.mock_vector_store.search_similar.return_value = []  # No relevant chunks
        
        # Mock OpenAI response for no-context case
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I don't have specific information about this topic."
        mock_openai_chat.return_value = mock_response
        
        # Test question
        request = QuestionRequest(question="What is quantum computing?")
        response = self.qa_engine.answer_question(request)
        
        # Assertions
        assert isinstance(response, QuestionAnswer)
        assert "No relevant documents found" in response.answer
        assert len(response.sources) == 0
        assert response.confidence_score < 0.5
    
    def test_embedding_error_handling(self):
        """Test handling of embedding errors."""
        # Setup mock to raise error
        self.mock_embedding_service.generate_embedding.side_effect = Exception("Embedding API error")
        
        request = QuestionRequest(question="Test question")
        response = self.qa_engine.answer_question(request)
        
        # Should return error response
        assert "error" in response.answer.lower()
        assert response.confidence_score == 0.0
        assert "error" in response.metadata
    
    def test_vector_store_error_handling(self):
        """Test handling of vector store errors."""
        # Setup mocks
        self.mock_embedding_service.generate_embedding.return_value = [0.1] * 1536
        self.mock_vector_store.search_similar.side_effect = Exception("Vector store error")
        
        request = QuestionRequest(question="Test question")
        response = self.qa_engine.answer_question(request)
        
        # Should return error response
        assert "error" in response.answer.lower()
        assert response.confidence_score == 0.0
    
    def test_build_context(self):
        """Test context building from chunks."""
        chunks = [
            {
                "text": "First chunk of text.",
                "metadata": {"document_id": "doc1", "chunk_index": 0}
            },
            {
                "text": "Second chunk of text.",
                "metadata": {"document_id": "doc1", "chunk_index": 1}
            }
        ]
        
        context = self.qa_engine._build_context(chunks)
        
        assert "First chunk of text" in context
        assert "Second chunk of text" in context
        assert "Source 1" in context
        assert "Source 2" in context
        assert "doc1" in context
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # Test with good chunks
        good_chunks = [
            {"distance": 0.1},
            {"distance": 0.2}
        ]
        confidence1 = self.qa_engine._calculate_confidence(good_chunks, "Clear answer based on sources.")
        
        # Test with poor chunks
        poor_chunks = [
            {"distance": 0.8},
            {"distance": 0.9}
        ]
        confidence2 = self.qa_engine._calculate_confidence(poor_chunks, "I don't know the answer.")
        
        # Good chunks should have higher confidence
        assert confidence1 > confidence2
        assert 0 <= confidence1 <= 1
        assert 0 <= confidence2 <= 1
    
    def test_session_stats(self):
        """Test session statistics tracking."""
        initial_stats = self.qa_engine.get_session_stats()
        assert initial_stats.total_questions == 0
        
        # Simulate processing questions
        self.qa_engine._update_session_stats(1.5, 3)  # 1.5s, 3 sources
        self.qa_engine._update_session_stats(2.0, 0)  # 2.0s, 0 sources
        
        stats = self.qa_engine.get_session_stats()
        assert stats.total_questions == 2
        assert stats.questions_with_sources == 1
        assert stats.total_processing_time == 3.5
        assert stats.average_response_time == 1.75
        
        # Reset stats
        self.qa_engine.reset_session_stats()
        reset_stats = self.qa_engine.get_session_stats()
        assert reset_stats.total_questions == 0


@patch('app.core.qa_engine.get_vector_store')
@patch('app.core.qa_engine.get_embedding_service') 
@patch('app.core.qa_engine.openai')
@patch('app.core.qa_engine.settings')
def test_global_qa_engine(mock_settings, mock_openai, mock_embedding_service, mock_vector_store):
    """Test global Q&A engine instance management."""
    # Configure mocks
    mock_settings.openai_api_key = "test_key"
    mock_settings.openai_model = "gpt-3.5-turbo"
    
    # Reset to ensure clean state
    reset_qa_engine()
    
    # Get instance
    engine1 = get_qa_engine()
    engine2 = get_qa_engine()
    
    # Should be the same instance
    assert engine1 is engine2
    
    # Reset and get new instance
    reset_qa_engine()
    engine3 = get_qa_engine()
    
    # Should be different instance
    assert engine1 is not engine3


@patch('app.core.qa_engine.get_qa_engine')
def test_convenience_function(mock_get_engine):
    """Test the convenience ask_question function."""
    mock_engine = Mock()
    mock_response = QuestionAnswer(
        question="Test question",
        answer="Test answer",
        processing_time=1.0
    )
    mock_engine.answer_question.return_value = mock_response
    mock_get_engine.return_value = mock_engine
    
    response = ask_question("Test question", document_id="doc1", max_chunks=3)
    
    assert response.question == "Test question"
    assert response.answer == "Test answer"
    
    # Verify the request was created correctly
    call_args = mock_engine.answer_question.call_args[0][0]
    assert call_args.question == "Test question"
    assert call_args.document_id == "doc1"
    assert call_args.max_chunks == 3