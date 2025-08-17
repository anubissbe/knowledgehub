"""Application settings configuration."""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # App settings
    APP_NAME: str = os.getenv("APP_NAME", "KnowledgeHub")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@postgres:5432/knowledgehub"
    )
    DB_HOST: str = os.getenv("DB_HOST", "postgres")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "knowledgehub")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASS: str = os.getenv("DB_PASS", "postgres")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    
    # Redis settings
    REDIS_URL: str = os.getenv(
        "REDIS_URL",
        "redis://redis:6379/0"
    )
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "3000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # Vector DB settings
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    WEAVIATE_HOST: str = os.getenv("WEAVIATE_HOST", "weaviate")
    WEAVIATE_PORT: int = int(os.getenv("WEAVIATE_PORT", "8080"))
    WEAVIATE_API_KEY: Optional[str] = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_COLLECTION_NAME: str = os.getenv("WEAVIATE_COLLECTION_NAME", "KnowledgeHub")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "knowledgehub")
    
    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    EMBEDDINGS_SERVICE_URL: str = os.getenv("EMBEDDINGS_SERVICE_URL", "http://ai-service:8000")
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"
    
    # Graph DB settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    
    # MinIO settings
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "knowledgehub")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "False").lower() == "true"
    
    # Zep settings
    ZEP_URL: str = os.getenv("ZEP_URL", "http://zep:8000")
    ZEP_API_KEY: Optional[str] = os.getenv("ZEP_API_KEY")
    
    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost:3100",
        "http://localhost:3000", 
        "http://192.168.1.25:3100",
        "http://192.168.1.25:3000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()