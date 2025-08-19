"""
Application configuration management.
"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    debug: bool = Field(default=False, env="DEBUG")
    
    # OpenAI Settings
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    
    # Database Settings
    database_url: str = Field(default="postgresql://raguser:ragpassword@localhost:5432/ragdb", env="DATABASE_URL")
    postgres_user: str = Field(default="raguser", env="POSTGRES_USER")
    postgres_password: str = Field(default="ragpassword", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="ragdb", env="POSTGRES_DB")
    
    # ChromaDB Settings
    chroma_persist_directory: str = Field(default="./data/vectordb", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="documents", env="CHROMA_COLLECTION_NAME")
    
    # File Upload Settings
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_extensions_str: str = Field(default=".pdf,.txt,.docx", env="ALLOWED_EXTENSIONS")
    upload_directory: str = Field(default="./data/uploads", env="UPLOAD_DIRECTORY")
    
    # Document Processing Settings
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_chunks_per_document: int = Field(default=1000, env="MAX_CHUNKS_PER_DOCUMENT")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
        env="LOG_FORMAT"
    )
    
    @property
    def allowed_extensions(self) -> List[str]:
        """Parse allowed extensions from string."""
        return [ext.strip() for ext in self.allowed_extensions_str.split(',')]
    
    def model_post_init(self, __context) -> None:
        """Create necessary directories after initialization.""" 
        # Ensure directories exist
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.upload_directory).mkdir(parents=True, exist_ok=True)
        Path("./data/processed").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env


# Global settings instance
settings = Settings()


# Utility functions for common paths
def get_upload_path(filename: str) -> Path:
    """Get full path for uploaded file."""
    return Path(settings.upload_directory) / filename


def get_processed_path(filename: str) -> Path:
    """Get full path for processed file."""
    return Path("./data/processed") / filename


def get_vectordb_path() -> Path:
    """Get ChromaDB persistence path."""
    return Path(settings.chroma_persist_directory)