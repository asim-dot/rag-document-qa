"""
ChromaDB vector store operations.
"""
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.models.document import DocumentChunk
from app.utils.logger import get_logger
from app.utils.exceptions import VectorStoreError

logger = get_logger(__name__)


class ChromaVectorStore:
    """ChromaDB vector store for document chunks."""
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.persist_directory = settings.chroma_persist_directory
        self.collection_name = settings.chroma_collection_name
        
        # Ensure persistence directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document chunks"}
            )
            
            logger.info(f"ChromaDB initialized: {self.persist_directory}")
            logger.info(f"Collection: {self.collection_name}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}")
    
    def add_chunks(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: List[List[float]], 
        document_id: str
    ) -> bool:
        """
        Add document chunks with embeddings to vector store.
        
        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            if len(chunks) != len(embeddings):
                raise VectorStoreError("Number of chunks and embeddings must match")
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings_list = []
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = f"{document_id}_{chunk.chunk_index}"
                
                ids.append(chunk_id)
                documents.append(chunk.text)
                embeddings_list.append(embedding)
                
                # Metadata for each chunk
                metadata = {
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "chunk_length": len(chunk.text),
                    **chunk.metadata  # Include any additional metadata
                }
                metadatas.append(metadata)
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to add chunks to vector store: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        n_results: int = 5,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            document_id: Optional document ID filter
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }
            
            # Add document filter if specified
            if document_id:
                query_params["where"] = {"document_id": document_id}
            
            # Search ChromaDB
            results = self.collection.query(**query_params)
            
            # Format results
            similar_chunks = []
            
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    chunk_data = {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results["distances"] else None
                    }
                    similar_chunks.append(chunk_data)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            error_msg = f"Failed to search vector store: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            # Find all chunks for the document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results["ids"]:
                # Delete the chunks
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            else:
                logger.info(f"No chunks found for document {document_id}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete document {document_id}: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            stats = {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
            if count > 0:
                # Get sample of documents to analyze
                sample_results = self.collection.peek(limit=min(10, count))
                if sample_results["metadatas"]:
                    # Count unique documents
                    document_ids = set()
                    for metadata in sample_results["metadatas"]:
                        if metadata and "document_id" in metadata:
                            document_ids.add(metadata["document_id"])
                    
                    stats["sample_document_count"] = len(document_ids)
            
            return stats
            
        except Exception as e:
            error_msg = f"Failed to get collection stats: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def reset_collection(self) -> bool:
        """Reset (clear) the entire collection."""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document chunks"}
            )
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to reset collection: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)


# Global vector store instance
_vector_store = None


def get_vector_store() -> ChromaVectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    return _vector_store


def reset_vector_store():
    """Reset the global vector store instance."""
    global _vector_store
    _vector_store = None