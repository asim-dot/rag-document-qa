"""
Core Q&A engine for RAG document question-answering.
"""
import time
from typing import List, Optional, Dict, Any, Tuple

import openai

from app.config import settings
from app.core.vector_store import get_vector_store
from app.core.embeddings import get_embedding_service
from app.models.question import QuestionRequest, QuestionAnswer, SourceChunk, QASessionStats
from app.utils.logger import get_logger
from app.utils.exceptions import OpenAIError, VectorStoreError, EmbeddingError

logger = get_logger(__name__)


class QAEngine:
    """Question-Answering engine using RAG approach."""
    
    def __init__(self):
        """Initialize Q&A engine with required services."""
        if not settings.openai_api_key:
            raise OpenAIError("OpenAI API key not configured")
        
        # Set OpenAI API key
        openai.api_key = settings.openai_api_key
        self.model = settings.openai_model
        
        # Get service instances
        self.vector_store = get_vector_store()
        self.embedding_service = get_embedding_service()
        
        # Session statistics
        self.session_stats = QASessionStats()
        
        logger.info(f"Q&A Engine initialized with model: {self.model}")
    
    def answer_question(self, request: QuestionRequest) -> QuestionAnswer:
        """
        Answer a question using RAG approach.
        
        Args:
            request: Question request with parameters
            
        Returns:
            Complete question-answer response
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {request.question[:100]}...")
            
            # Step 1: Generate embedding for the question
            question_embedding = self._generate_question_embedding(request.question)
            
            # Step 2: Search for relevant chunks
            relevant_chunks = self._search_relevant_chunks(
                question_embedding, 
                request.max_chunks,
                request.document_id
            )
            
            # Step 3: Generate answer using retrieved chunks
            answer, confidence = self._generate_answer(request.question, relevant_chunks)
            
            # Step 4: Create source information
            sources = self._create_source_chunks(relevant_chunks)
            
            processing_time = time.time() - start_time
            
            # Update session statistics
            self._update_session_stats(processing_time, len(sources))
            
            # Create response
            response = QuestionAnswer(
                question=request.question,
                answer=answer,
                confidence_score=confidence,
                sources=sources if request.include_sources else [],
                processing_time=processing_time,
                metadata={
                    "chunks_retrieved": len(relevant_chunks),
                    "model_used": self.model,
                    "document_filter": request.document_id
                }
            )
            
            logger.info(f"Question answered in {processing_time:.3f}s with {len(sources)} sources")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to answer question: {str(e)}"
            logger.error(error_msg)
            
            # Return error response
            return QuestionAnswer(
                question=request.question,
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                confidence_score=0.0,
                sources=[],
                processing_time=processing_time,
                metadata={"error": error_msg}
            )
    
    def _generate_question_embedding(self, question: str) -> List[float]:
        """Generate embedding for the question."""
        try:
            return self.embedding_service.generate_embedding(question)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate question embedding: {str(e)}")
    
    def _search_relevant_chunks(
        self, 
        question_embedding: List[float], 
        max_chunks: int,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        try:
            relevant_chunks = self.vector_store.search_similar(
                query_embedding=question_embedding,
                n_results=max_chunks,
                document_id=document_id
            )
            
            logger.debug(f"Found {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search relevant chunks: {str(e)}")
    
    def _generate_answer(
        self, 
        question: str, 
        relevant_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """
        Generate answer using OpenAI with retrieved context.
        
        Args:
            question: User's question
            relevant_chunks: Retrieved relevant chunks
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        try:
            if not relevant_chunks:
                return self._handle_no_context_question(question)
            
            # Build context from relevant chunks
            context = self._build_context(relevant_chunks)
            
            # Create prompt
            prompt = self._create_rag_prompt(question, context)
            
            # Generate answer with OpenAI
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that answers questions based on provided context. Always cite the source when possible and be honest when information is not available in the context."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=500,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(relevant_chunks, answer)
            
            return answer, confidence
            
        except Exception as e:
            raise OpenAIError(f"Failed to generate answer: {str(e)}")
    
    def _build_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Build context string from relevant chunks."""
        context_parts = []
        
        for i, chunk in enumerate(relevant_chunks):
            chunk_text = chunk.get("text", "")
            document_id = chunk.get("metadata", {}).get("document_id", "Unknown")
            chunk_index = chunk.get("metadata", {}).get("chunk_index", i)
            
            context_parts.append(f"[Source {i+1} - Document: {document_id}, Chunk: {chunk_index}]\n{chunk_text}")
        
        return "\n\n".join(context_parts)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create RAG prompt for OpenAI."""
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so clearly.

Context:
{context}

Question: {question}

Please provide a clear, concise answer based on the context above. If you reference specific information, please mention which source it came from (e.g., "According to Source 1...").

Answer:"""
        
        return prompt
    
    def _handle_no_context_question(self, question: str) -> Tuple[str, float]:
        """Handle questions when no relevant context is found."""
        try:
            # Use OpenAI for general knowledge (without RAG)
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Answer the question based on your general knowledge, but mention that this answer is not based on any specific documents."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nNote: No relevant documents were found for this question. Please provide a general answer if possible."
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content.strip()
            return f"No relevant documents found for your question. Based on general knowledge: {answer}", 0.3
            
        except Exception as e:
            return "I apologize, but I couldn't find relevant information in the documents to answer your question.", 0.1
    
    def _calculate_confidence(self, relevant_chunks: List[Dict[str, Any]], answer: str) -> float:
        """
        Calculate confidence score based on various factors.
        
        Args:
            relevant_chunks: Retrieved chunks
            answer: Generated answer
            
        Returns:
            Confidence score between 0 and 1
        """
        if not relevant_chunks:
            return 0.3
        
        # Base confidence from number of sources
        base_confidence = min(0.5 + (len(relevant_chunks) * 0.1), 0.9)
        
        # Adjust based on chunk distances (if available)
        if relevant_chunks and relevant_chunks[0].get("distance") is not None:
            avg_distance = sum(chunk.get("distance", 1.0) for chunk in relevant_chunks) / len(relevant_chunks)
            distance_factor = max(0.5, 1.0 - avg_distance)  # Lower distance = higher confidence
            base_confidence *= distance_factor
        
        # Adjust based on answer characteristics
        if "I don't know" in answer or "not found" in answer.lower():
            base_confidence *= 0.5
        elif "according to" in answer.lower() or "source" in answer.lower():
            base_confidence *= 1.1
        
        return min(base_confidence, 1.0)
    
    def _create_source_chunks(self, relevant_chunks: List[Dict[str, Any]]) -> List[SourceChunk]:
        """Create source chunk objects from search results."""
        sources = []
        
        for chunk in relevant_chunks:
            metadata = chunk.get("metadata", {})
            
            source = SourceChunk(
                chunk_id=chunk.get("id", ""),
                document_id=metadata.get("document_id", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                text=chunk.get("text", ""),
                relevance_score=1.0 - chunk.get("distance", 0.0) if chunk.get("distance") is not None else None,
                start_char=metadata.get("start_char", 0),
                end_char=metadata.get("end_char", 0)
            )
            sources.append(source)
        
        return sources
    
    def _update_session_stats(self, processing_time: float, source_count: int):
        """Update session statistics."""
        self.session_stats.total_questions += 1
        self.session_stats.total_processing_time += processing_time
        self.session_stats.average_response_time = (
            self.session_stats.total_processing_time / self.session_stats.total_questions
        )
        
        if source_count > 0:
            self.session_stats.questions_with_sources += 1
    
    def get_session_stats(self) -> QASessionStats:
        """Get current session statistics."""
        return self.session_stats
    
    def reset_session_stats(self):
        """Reset session statistics."""
        self.session_stats = QASessionStats()


# Global Q&A engine instance
_qa_engine = None


def get_qa_engine() -> QAEngine:
    """Get the global Q&A engine instance."""
    global _qa_engine
    if _qa_engine is None:
        _qa_engine = QAEngine()
    return _qa_engine


def reset_qa_engine():
    """Reset the global Q&A engine instance."""
    global _qa_engine
    _qa_engine = None


def ask_question(question: str, document_id: Optional[str] = None, max_chunks: int = 5) -> QuestionAnswer:
    """
    Convenience function to ask a question.
    
    Args:
        question: Question to ask
        document_id: Optional document ID filter
        max_chunks: Maximum chunks to retrieve
        
    Returns:
        Question answer response
    """
    engine = get_qa_engine()
    request = QuestionRequest(
        question=question,
        document_id=document_id,
        max_chunks=max_chunks
    )
    return engine.answer_question(request)