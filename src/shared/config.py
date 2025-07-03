"""Shared configuration for all services"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for all services"""
    
    def __init__(self):
        # Database
        self.DATABASE_URL = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/knowledgehub"
        )
        
        # Redis
        self.REDIS_URL = os.getenv(
            "REDIS_URL",
            "redis://redis:6379/0"
        )
        
        # Weaviate
        self.WEAVIATE_URL = os.getenv(
            "WEAVIATE_URL",
            "http://weaviate:8080"
        )
        
        # API
        self.API_URL = os.getenv(
            "API_URL",
            "http://api:3000"
        )
        
        # MCP Server
        self.MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "3002"))
        
        # MinIO/S3
        self.S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
        self.S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
        self.S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
        self.S3_BUCKET = os.getenv("S3_BUCKET", "knowledgehub")
        
        # Authentication
        self.SECRET_KEY = os.getenv(
            "SECRET_KEY",
            "your-secret-key-change-in-production"
        )
        self.JWT_ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 30
        
        # Crawling
        self.MAX_CRAWL_DEPTH = int(os.getenv("MAX_CRAWL_DEPTH", "3"))
        self.MAX_PAGES_PER_CRAWL = int(os.getenv("MAX_PAGES_PER_CRAWL", "100"))
        self.CRAWL_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT", "30"))
        
        # Chunking
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Embeddings
        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Search
        self.SEARCH_RESULT_LIMIT = int(os.getenv("SEARCH_RESULT_LIMIT", "10"))
        self.VECTOR_SEARCH_WEIGHT = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7"))
        self.KEYWORD_SEARCH_WEIGHT = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Performance
        self.WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "4"))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
        
        # Monitoring
        self.ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
    
    def get_database_config(self) -> dict:
        """Get database configuration dict"""
        return {
            "url": self.DATABASE_URL,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True
        }
    
    def get_redis_config(self) -> dict:
        """Get Redis configuration dict"""
        return {
            "url": self.REDIS_URL,
            "decode_responses": True,
            "max_connections": 50
        }
    
    def get_s3_config(self) -> dict:
        """Get S3/MinIO configuration dict"""
        return {
            "endpoint_url": self.S3_ENDPOINT,
            "aws_access_key_id": self.S3_ACCESS_KEY,
            "aws_secret_access_key": self.S3_SECRET_KEY,
            "region_name": "us-east-1"
        }
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return not self.is_production