"""
Embedding generation using OpenAI API.
"""
import time
from typing import List, Optional, Dict, Any

import openai

from app.config import settings
from app.utils.logger import get_logger
from app.utils.exceptions import EmbeddingError

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        if not settings.openai_api_key:
            raise EmbeddingError("OpenAI API key not configured")
        
        # Set OpenAI API key
        openai.api_key = settings.openai_api_key
        self.model = settings.embedding_model
        
        logger.info(f"Embedding service initialized with model: {self.model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if not text.strip():
                raise EmbeddingError("Cannot generate embedding for empty text")
            
            # Clean text (remove excessive whitespace)
            cleaned_text = " ".join(text.split())
            
            # Generate embedding
            response = openai.embeddings.create(
                model=self.model,
                input=cleaned_text
            )
            
            embedding = response.data[0].embedding
            
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to generate embedding: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of each batch
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Clean texts
                cleaned_texts = [" ".join(text.split()) for text in batch_texts]
                
                # Filter out empty texts
                non_empty_texts = [text for text in cleaned_texts if text.strip()]
                
                if not non_empty_texts:
                    # Add empty embeddings for empty texts
                    all_embeddings.extend([[] for _ in batch_texts])
                    continue
                
                # Generate embeddings for batch
                response = openai.embeddings.create(
                    model=self.model,
                    input=non_empty_texts
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Add delay to respect rate limits
                if i + batch_size < len(texts):
                    time.sleep(0.1)  # Small delay between batches
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch_texts)} texts)")
            
            logger.info(f"Generated {len(all_embeddings)} embeddings total")
            return all_embeddings
            
        except Exception as e:
            error_msg = f"Failed to generate batch embeddings: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model."""
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            # Return default dimension for text-embedding-ada-002
            return 1536


# Global embedding service instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def reset_embedding_service():
    """Reset the global embedding service instance."""
    global _embedding_service
    _embedding_service = None