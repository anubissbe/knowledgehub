"""Application configuration"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os

# Import Vault integration to load credentials
try:
    from .services.vault import vault_config
except ImportError:
    vault_config = {}


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    APP_NAME: str = "AI Knowledge Hub"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 3000
    API_WORKERS: int = 4
    API_RELOAD: bool = True
    
    # Database settings
    DATABASE_URL: str = "postgresql://khuser:khpassword@localhost:5432/knowledgehub"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50
    
    # Weaviate settings
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: Optional[str] = None
    WEAVIATE_COLLECTION_NAME: str = "Knowledge_chunks"
    
    # S3/MinIO settings
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY_ID: str = "minioadmin"
    S3_SECRET_ACCESS_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "knowledge-hub"
    S3_REGION_NAME: str = "us-east-1"
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDINGS_SERVICE_URL: str = "http://localhost:8100"
    
    # Scraping settings
    MAX_CONCURRENT_SCRAPERS: int = 5
    SCRAPER_TIMEOUT_MS: int = 30000
    SCRAPER_RATE_LIMIT_RPS: float = 1.0
    SCRAPER_USER_AGENT: str = "KnowledgeHubBot/1.0"
    SCRAPER_RETRY_ATTEMPTS: int = 3
    SCRAPER_RETRY_DELAY_MS: int = 1000
    
    # Chunking settings
    MAX_CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100
    
    # Security settings
    SECRET_KEY: str = "change-this-to-a-random-secret-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    API_KEY_HEADER: str = "X-API-Key"
    
    # CORS settings - secure by default
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3102", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3102"
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100
    RATE_LIMIT_BURST_SIZE: int = 10
    
    # MCP Server settings
    MCP_SERVER_HOST: str = "0.0.0.0"
    MCP_SERVER_PORT: int = 3002
    MCP_HEARTBEAT_INTERVAL: int = 30
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)