# KnowledgeHub Configuration Guide

This guide covers all configuration options for KnowledgeHub, from basic environment setup to advanced performance tuning.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Service Configuration](#service-configuration)
- [Database Configuration](#database-configuration)
- [Search & AI Configuration](#search--ai-configuration)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Monitoring Configuration](#monitoring-configuration)
- [Advanced Configuration](#advanced-configuration)

## Environment Variables

KnowledgeHub uses environment variables for all configuration. Create a `.env` file or set these in your deployment environment.

### Core Configuration

```bash
# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# PostgreSQL connection string
DATABASE_URL=postgresql://username:password@hostname:port/database
# Example: postgresql://khuser:khpassword@postgres:5432/knowledgehub

# Redis connection string  
REDIS_URL=redis://hostname:port/database
# Example: redis://redis:6379/0

# Weaviate vector database URL
WEAVIATE_URL=http://hostname:port
# Example: http://weaviate-lite:8080

# Weaviate API key (optional)
WEAVIATE_API_KEY=your-weaviate-api-key

# =============================================================================
# OBJECT STORAGE CONFIGURATION
# =============================================================================

# S3/MinIO endpoint URL
S3_ENDPOINT_URL=http://hostname:port
# Example: http://minio:9000 (local MinIO)
# Example: https://s3.amazonaws.com (AWS S3)

# S3/MinIO credentials
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key

# S3/MinIO bucket name
S3_BUCKET=knowledgehub

# S3 region (for AWS S3)
S3_REGION=us-east-1

# =============================================================================
# API CONFIGURATION
# =============================================================================

# API server host and port
API_HOST=0.0.0.0
API_PORT=3000

# Number of worker processes (production)
WORKERS=4

# Enable debug mode (development only)
DEBUG=false

# Secret key for JWT tokens and encryption
SECRET_KEY=your-very-long-secret-key-change-this-in-production

# CORS origins (comma-separated)
CORS_ORIGINS=http://localhost:3101,https://yourdomain.com

# =============================================================================
# EMBEDDINGS & AI CONFIGURATION  
# =============================================================================

# Embeddings service URL
EMBEDDINGS_SERVICE_URL=http://embeddings-service:8100

# Embedding model name
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Model cache directory
MODEL_CACHE_DIR=/app/models

# Enable GPU acceleration
CUDA_VISIBLE_DEVICES=0

# =============================================================================
# FRONTEND CONFIGURATION
# =============================================================================

# API URL for frontend
VITE_API_URL=http://localhost:3000

# WebSocket URL for real-time updates
VITE_WS_URL=ws://localhost:3000

# Frontend environment
NODE_ENV=production
```

### Advanced Configuration

```bash
# =============================================================================
# LOGGING & MONITORING
# =============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Enable structured JSON logging
JSON_LOGGING=true

# Log file path (optional)
LOG_FILE=/var/log/knowledgehub/app.log

# Enable Prometheus metrics
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Enable health check endpoints
HEALTH_CHECK_ENABLED=true

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# API rate limiting (requests per minute)
API_RATE_LIMIT=1000

# Rate limit for search endpoints
SEARCH_RATE_LIMIT=100

# Maximum upload file size
MAX_UPLOAD_SIZE=100MB

# Session timeout (seconds)
SESSION_TIMEOUT=3600

# Enable API key authentication
API_KEY_AUTH_ENABLED=true

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Database connection pool size
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Redis connection pool size  
REDIS_POOL_SIZE=50

# HTTP client timeout (seconds)
HTTP_TIMEOUT=30

# Background job queue size
QUEUE_SIZE=1000

# Worker concurrency
WORKER_CONCURRENCY=4

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

# Search result cache TTL (seconds)
SEARCH_CACHE_TTL=3600

# Source metadata cache TTL (seconds)
SOURCE_CACHE_TTL=900

# Job status cache TTL (seconds)
JOB_CACHE_TTL=60

# System health cache TTL (seconds)
HEALTH_CACHE_TTL=30

# =============================================================================
# SCHEDULER CONFIGURATION
# =============================================================================

# Enable automated scheduling
SCHEDULER_ENABLED=true

# Weekly refresh schedule (cron format)
REFRESH_SCHEDULE=0 2 * * 0

# Batch size for scheduled refreshes
REFRESH_BATCH_SIZE=5

# Delay between refresh jobs (seconds)
REFRESH_DELAY_SECONDS=300
```

## Service Configuration

### API Gateway Configuration

```python
# src/api/config.py
class Settings(BaseSettings):
    # Database settings
    database_url: str
    db_pool_size: int = 20
    db_max_overflow: int = 30
    
    # Redis settings
    redis_url: str
    redis_pool_size: int = 50
    
    # API settings
    debug: bool = False
    workers: int = 1
    api_host: str = "0.0.0.0"
    api_port: int = 3000
    
    # Security settings
    secret_key: str
    api_rate_limit: int = 1000
    cors_origins: List[str] = ["*"]
    
    # Logging settings
    log_level: str = "INFO"
    json_logging: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

### Scraper Configuration

```json
{
  "scraper": {
    "max_depth": 3,
    "max_pages": 1000,
    "crawl_delay": 1.0,
    "timeout": 30,
    "retry_attempts": 3,
    "user_agent": "KnowledgeHub/1.0",
    "follow_patterns": ["**"],
    "exclude_patterns": [
      "**/admin/**",
      "**/private/**",
      "**/*.pdf",
      "**/*.doc*"
    ],
    "content_types": [
      "text/html",
      "text/plain",
      "application/json"
    ],
    "max_file_size": "10MB",
    "javascript_enabled": true,
    "wait_for_load": true,
    "screenshot_errors": false
  }
}
```

### RAG Processor Configuration

```json
{
  "rag_processor": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "min_chunk_size": 100,
    "max_chunk_size": 2000,
    "chunking_strategy": "semantic",
    "preserve_formatting": true,
    "remove_noise": true,
    "embedding_batch_size": 32,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimensions": 384,
    "gpu_enabled": true,
    "device": "auto"
  }
}
```

### Search Configuration

```json
{
  "search": {
    "default_limit": 20,
    "max_limit": 100,
    "similarity_threshold": 0.7,
    "hybrid_search_weight": 0.7,
    "enable_reranking": true,
    "cache_results": true,
    "cache_ttl": 3600,
    "highlight_results": true,
    "max_highlight_length": 200,
    "stemming_enabled": true,
    "fuzzy_search_enabled": true,
    "faceted_search_enabled": true
  }
}
```

## Database Configuration

### PostgreSQL Optimization

```sql
-- postgresql.conf optimizations

-- Memory settings
shared_buffers = 256MB                    # 25% of RAM
effective_cache_size = 1GB                # 75% of RAM
work_mem = 4MB                           # Per connection
maintenance_work_mem = 64MB              # For maintenance operations

-- Connection settings
max_connections = 200                     # Adjust based on load
shared_preload_libraries = 'pg_stat_statements'

-- Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

-- Logging
log_statement = 'mod'                    # Log modifications
log_min_duration_statement = 1000       # Log slow queries (1 second)
log_checkpoints = on
log_connections = on
log_disconnections = on
```

### Database Indexes

```sql
-- Performance indexes for KnowledgeHub

-- Sources table
CREATE INDEX CONCURRENTLY idx_sources_status_updated 
ON knowledge_sources(status, updated_at DESC);

CREATE INDEX CONCURRENTLY idx_sources_type_status 
ON knowledge_sources(source_type, status);

-- Documents table  
CREATE INDEX CONCURRENTLY idx_documents_source_status 
ON documents(source_id, status);

CREATE INDEX CONCURRENTLY idx_documents_content_hash 
ON documents(content_hash) WHERE content_hash IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_documents_url_text 
ON documents USING GIN(to_tsvector('english', url));

-- Chunks table
CREATE INDEX CONCURRENTLY idx_chunks_document_index 
ON document_chunks(document_id, chunk_index);

CREATE INDEX CONCURRENTLY idx_chunks_embedding_id 
ON document_chunks(embedding_id) WHERE embedding_id IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_chunks_content_text 
ON document_chunks USING GIN(to_tsvector('english', content));

-- Jobs table
CREATE INDEX CONCURRENTLY idx_jobs_status_created 
ON scraping_jobs(status, created_at DESC);

CREATE INDEX CONCURRENTLY idx_jobs_source_type_status 
ON scraping_jobs(source_id, job_type, status);

-- Memory table
CREATE INDEX CONCURRENTLY idx_memory_type_priority 
ON memory_items(memory_type, priority);

CREATE INDEX CONCURRENTLY idx_memory_tags 
ON memory_items USING GIN(tags);
```

### Redis Configuration

```redis
# redis.conf optimizations

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1      # Save after 900 sec if at least 1 key changed
save 300 10     # Save after 300 sec if at least 10 keys changed  
save 60 10000   # Save after 60 sec if at least 10000 keys changed

# Performance
tcp-keepalive 300
timeout 0
tcp-backlog 511

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Security
bind 127.0.0.1
protected-mode yes
```

## Search & AI Configuration

### Weaviate Schema Configuration

```json
{
  "class": "Knowledge_chunks",
  "description": "Text chunks from knowledge sources",
  "vectorizer": "text2vec-transformers",
  "moduleConfig": {
    "text2vec-transformers": {
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "options": {
        "waitForModel": true,
        "useGPU": true,
        "useCache": true
      }
    },
    "generative-openai": {
      "model": "gpt-3.5-turbo"
    }
  },
  "properties": [
    {
      "name": "content",
      "dataType": ["text"],
      "description": "The text content of the chunk",
      "moduleConfig": {
        "text2vec-transformers": {
          "skip": false,
          "vectorizePropertyName": false
        }
      }
    },
    {
      "name": "source_id",
      "dataType": ["string"],
      "description": "Reference to knowledge source"
    },
    {
      "name": "document_id", 
      "dataType": ["string"],
      "description": "Reference to source document"
    },
    {
      "name": "chunk_index",
      "dataType": ["int"],
      "description": "Position within document"
    },
    {
      "name": "chunk_type",
      "dataType": ["string"],
      "description": "Type of content (content, heading, code, etc.)"
    },
    {
      "name": "url",
      "dataType": ["string"],
      "description": "Source URL"
    },
    {
      "name": "title",
      "dataType": ["string"], 
      "description": "Document title"
    },
    {
      "name": "metadata",
      "dataType": ["object"],
      "description": "Additional metadata"
    }
  ],
  "vectorIndexConfig": {
    "distance": "cosine",
    "cleanupIntervalSeconds": 300
  },
  "replicationConfig": {
    "factor": 1
  }
}
```

### Embeddings Service Configuration

```yaml
# embeddings-service configuration
model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  cache_dir: "/app/models"
  device: "auto"  # auto, cpu, cuda:0
  max_seq_length: 384
  
server:
  host: "0.0.0.0"
  port: 8100
  workers: 4
  
performance:
  batch_size: 32
  max_batch_wait_time: 100  # milliseconds
  enable_caching: true
  cache_size: 10000
  
monitoring:
  enable_metrics: true
  metrics_port: 9091
```

## Security Configuration

### API Security

```python
# Security middleware configuration
SECURITY_CONFIG = {
    # CORS settings
    "cors": {
        "allow_origins": ["https://yourdomain.com"],
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
        "allow_headers": ["*"],
        "allow_credentials": True,
        "max_age": 3600
    },
    
    # Rate limiting
    "rate_limit": {
        "default": "1000/hour",
        "search": "100/hour", 
        "upload": "10/hour",
        "auth": "30/hour"
    },
    
    # Authentication
    "auth": {
        "api_key_header": "X-API-Key",
        "jwt_secret_key": "your-jwt-secret",
        "jwt_algorithm": "HS256",
        "jwt_expiration": 3600,
        "require_auth": True
    },
    
    # Input validation
    "validation": {
        "max_request_size": "10MB",
        "max_json_size": "1MB", 
        "max_form_size": "5MB",
        "sanitize_input": True
    }
}
```

### Database Security

```sql
-- Create dedicated database user
CREATE USER knowledgehub_app WITH PASSWORD 'secure_password';

-- Grant minimum required permissions
GRANT CONNECT ON DATABASE knowledgehub TO knowledgehub_app;
GRANT USAGE ON SCHEMA public TO knowledgehub_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO knowledgehub_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO knowledgehub_app;

-- Enable row level security (if needed)
ALTER TABLE knowledge_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create security policies
CREATE POLICY source_access_policy ON knowledge_sources
    FOR ALL TO knowledgehub_app
    USING (status = 'active');
```

### Container Security

```dockerfile
# Dockerfile security best practices

# Use non-root user
RUN addgroup --system --gid 1001 appgroup
RUN adduser --system --uid 1001 --gid 1001 appuser

# Set file permissions
COPY --chown=appuser:appgroup . /app
RUN chmod -R 755 /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1
```

## Performance Tuning

### API Performance

```python
# FastAPI performance settings
app = FastAPI(
    title="KnowledgeHub API",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
)

# Connection pooling
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "echo": DEBUG
}

REDIS_CONFIG = {
    "connection_pool_kwargs": {
        "max_connections": 50,
        "retry_on_timeout": True,
        "socket_keepalive": True,
        "socket_keepalive_options": {},
        "health_check_interval": 30
    }
}
```

### Background Worker Optimization

```python
# Worker performance settings
WORKER_CONFIG = {
    "concurrency": 4,                    # Number of concurrent tasks
    "prefetch_multiplier": 2,            # Task prefetch count
    "max_tasks_per_child": 1000,         # Restart worker after N tasks
    "task_time_limit": 3600,             # Max task execution time (seconds)
    "task_soft_time_limit": 3000,        # Soft task time limit
    "worker_max_memory_per_child": 200,  # Max memory per worker (MB)
}

# Queue settings
QUEUE_CONFIG = {
    "default_retry_delay": 60,           # Retry delay (seconds)
    "max_retries": 3,                    # Maximum retry attempts
    "visibility_timeout": 3600,          # Message visibility timeout
    "message_retention_period": 1209600, # 14 days
}
```

### Caching Strategy

```python
# Redis caching configuration
CACHE_CONFIG = {
    "search_results": {
        "ttl": 3600,                     # 1 hour
        "max_size": "100MB",
        "eviction_policy": "lru"
    },
    "source_metadata": {
        "ttl": 900,                      # 15 minutes
        "max_size": "50MB",
        "eviction_policy": "lru"
    },
    "job_status": {
        "ttl": 60,                       # 1 minute
        "max_size": "10MB",
        "eviction_policy": "ttl"
    },
    "embeddings": {
        "ttl": 86400,                    # 24 hours
        "max_size": "500MB",
        "eviction_policy": "lru"
    }
}
```

## Monitoring Configuration

### Prometheus Metrics

```python
# metrics.py - Custom metrics configuration
from prometheus_client import Counter, Histogram, Gauge

# API metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

# Search metrics
search_requests_total = Counter(
    'search_requests_total',
    'Total search requests',
    ['search_type']
)

search_results_total = Histogram(
    'search_results_total',
    'Number of search results returned',
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

# Job metrics
jobs_total = Counter(
    'jobs_total',
    'Total jobs processed',
    ['job_type', 'status']
)

job_duration = Histogram(
    'job_duration_seconds',
    'Job processing duration',
    ['job_type']
)

# System metrics
active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

queue_size = Gauge(
    'queue_size',
    'Current queue size',
    ['queue_name']
)
```

### Logging Configuration

```python
# logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if JSON_LOGGING else "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "knowledgehub": {
            "level": LOG_LEVEL,
            "handlers": ["console", "file"] if LOG_FILE else ["console"],
            "propagate": False
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        }
    },
    "root": {
        "level": LOG_LEVEL,
        "handlers": ["console"]
    }
}
```

## Advanced Configuration

### Multi-tenant Configuration

```python
# Multi-tenant settings
TENANT_CONFIG = {
    "enabled": False,
    "default_tenant": "default",
    "tenant_header": "X-Tenant-ID",
    "tenant_isolation": "schema",  # schema, database, or prefix
    "tenant_limits": {
        "max_sources": 100,
        "max_documents": 10000,
        "max_searches_per_hour": 1000,
        "max_storage_mb": 1024
    }
}
```

### Geographic Distribution

```python
# Multi-region configuration
REGION_CONFIG = {
    "primary_region": "us-east-1",
    "regions": {
        "us-east-1": {
            "database_url": "postgresql://...",
            "redis_url": "redis://...",
            "s3_bucket": "knowledgehub-us-east-1"
        },
        "eu-west-1": {
            "database_url": "postgresql://...", 
            "redis_url": "redis://...",
            "s3_bucket": "knowledgehub-eu-west-1"
        }
    },
    "replication": {
        "enabled": True,
        "sync_interval": 300,  # seconds
        "conflict_resolution": "timestamp"
    }
}
```

### Custom Integrations

```python
# Plugin configuration
PLUGIN_CONFIG = {
    "enabled": True,
    "plugin_directory": "/app/plugins",
    "allowed_plugins": [
        "slack_integration",
        "jira_connector", 
        "confluence_sync"
    ],
    "plugin_settings": {
        "slack_integration": {
            "webhook_url": "https://hooks.slack.com/...",
            "channel": "#knowledge-updates"
        }
    }
}
```

### Configuration Validation

```python
# config_validator.py
from pydantic import BaseSettings, validator
from typing import List, Optional

class KnowledgeHubSettings(BaseSettings):
    database_url: str
    redis_url: str
    weaviate_url: str
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith('postgresql://'):
            raise ValueError('Database URL must be PostgreSQL')
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        if not v.startswith('redis://'):
            raise ValueError('Redis URL must start with redis://')
        return v
    
    api_rate_limit: int = 1000
    
    @validator('api_rate_limit')
    def validate_rate_limit(cls, v):
        if v < 1 or v > 10000:
            raise ValueError('Rate limit must be between 1 and 10000')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

## Configuration Templates

### Development Template

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG

DATABASE_URL=postgresql://dev:dev@localhost:5432/knowledgehub_dev
REDIS_URL=redis://localhost:6379/0
WEAVIATE_URL=http://localhost:8090

S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

API_RATE_LIMIT=10000
CORS_ORIGINS=*

VITE_API_URL=http://localhost:3000
VITE_WS_URL=ws://localhost:3000
```

### Production Template

```bash
# .env.production
DEBUG=false
LOG_LEVEL=INFO
JSON_LOGGING=true

DATABASE_URL=postgresql://khuser:${DB_PASSWORD}@db.example.com:5432/knowledgehub
REDIS_URL=redis://:${REDIS_PASSWORD}@redis.example.com:6379/0
WEAVIATE_URL=http://weaviate.example.com:8080

S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY=${AWS_ACCESS_KEY}
S3_SECRET_KEY=${AWS_SECRET_KEY}
S3_REGION=us-east-1

SECRET_KEY=${SECRET_KEY}
API_RATE_LIMIT=1000
CORS_ORIGINS=https://knowledgehub.example.com

WORKERS=4
METRICS_ENABLED=true

VITE_API_URL=https://api.knowledgehub.example.com
VITE_WS_URL=wss://api.knowledgehub.example.com
```

For more specific configuration scenarios, see:
- [Security Guide](SECURITY.md) for security-focused configuration
- [Deployment Guide](DEPLOYMENT.md) for production deployment configuration  
- [Monitoring Guide](MONITORING.md) for observability configuration
- [Troubleshooting Guide](TROUBLESHOOTING.md) for configuration debugging