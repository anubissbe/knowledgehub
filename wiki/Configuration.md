# Configuration Guide

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

# Enable Prometheus metrics
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

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

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Database connection pool size
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Redis connection pool size  
REDIS_POOL_SIZE=50

# Worker concurrency
WORKER_CONCURRENCY=4

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

# Search result cache TTL (seconds)
SEARCH_CACHE_TTL=3600

# Source metadata cache TTL (seconds)
SOURCE_CACHE_TTL=900

# =============================================================================
# SCHEDULER CONFIGURATION
# =============================================================================

# Enable automated scheduling
SCHEDULER_ENABLED=true

# Weekly refresh schedule (cron format)
REFRESH_SCHEDULE=0 2 * * 0

# Batch size for scheduled refreshes
REFRESH_BATCH_SIZE=5
```

## Service Configuration

### API Gateway Configuration

The API Gateway handles all incoming requests and coordinates between services:

```python
# Key configuration options
{
    "database_pool_size": 20,        # Connection pool size
    "redis_pool_size": 50,           # Redis connections
    "request_timeout": 30,           # Request timeout (seconds)
    "max_request_size": "10MB",      # Maximum request size
    "enable_cors": true,             # CORS support
    "enable_metrics": true           # Prometheus metrics
}
```

### Scraper Configuration

Configure web crawling behavior:

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
    "javascript_enabled": true,
    "wait_for_load": true
  }
}
```

### RAG Processor Configuration

Control content processing and embedding generation:

```json
{
  "rag_processor": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "min_chunk_size": 100,
    "max_chunk_size": 2000,
    "chunking_strategy": "semantic",
    "embedding_batch_size": 32,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "gpu_enabled": true
  }
}
```

### Search Configuration

Fine-tune search behavior:

```json
{
  "search": {
    "default_limit": 20,
    "max_limit": 100,
    "similarity_threshold": 0.7,
    "hybrid_search_weight": 0.7,
    "enable_reranking": true,
    "cache_results": true,
    "cache_ttl": 3600
  }
}
```

## Database Configuration

### PostgreSQL Optimization

For production environments, optimize PostgreSQL settings:

```sql
-- Key performance settings
shared_buffers = 256MB              -- 25% of RAM
effective_cache_size = 1GB          -- 75% of RAM
work_mem = 4MB                      -- Per connection
maintenance_work_mem = 64MB         -- For maintenance
max_connections = 200               -- Connection limit
```

### Database Indexes

Essential indexes for performance:

```sql
-- Sources table
CREATE INDEX idx_sources_status_updated 
ON knowledge_sources(status, updated_at DESC);

-- Documents table  
CREATE INDEX idx_documents_source_status 
ON documents(source_id, status);

-- Chunks table
CREATE INDEX idx_chunks_document_index 
ON document_chunks(document_id, chunk_index);

-- Jobs table
CREATE INDEX idx_jobs_status_created 
ON scraping_jobs(status, created_at DESC);
```

### Redis Configuration

Optimize Redis for caching and queuing:

```redis
# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Performance
tcp-keepalive 300
timeout 0

# Persistence
save 900 1
save 300 10
save 60 10000
```

## Search & AI Configuration

### Weaviate Schema

Configure the vector database schema:

```json
{
  "class": "Knowledge_chunks",
  "vectorizer": "text2vec-transformers",
  "moduleConfig": {
    "text2vec-transformers": {
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "options": {
        "waitForModel": true,
        "useGPU": true,
        "useCache": true
      }
    }
  },
  "vectorIndexConfig": {
    "distance": "cosine"
  }
}
```

### Embeddings Service

Configure the embedding generation service:

```yaml
model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "auto"  # auto, cpu, cuda:0
  max_seq_length: 384
  
performance:
  batch_size: 32
  enable_caching: true
  cache_size: 10000
```

## Security Configuration

### API Security

Configure security middleware:

```python
SECURITY_CONFIG = {
    # CORS settings
    "cors": {
        "allow_origins": ["https://yourdomain.com"],
        "allow_methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_credentials": True
    },
    
    # Rate limiting
    "rate_limit": {
        "default": "1000/hour",
        "search": "100/hour", 
        "upload": "10/hour"
    },
    
    # Authentication
    "auth": {
        "api_key_header": "X-API-Key",
        "require_auth": True
    }
}
```

### Container Security

Security best practices for Docker:

```dockerfile
# Use non-root user
RUN adduser --system --uid 1001 appuser

# Set file permissions
COPY --chown=appuser:appgroup . /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:3000/health || exit 1
```

## Performance Tuning

### API Performance

Optimize API performance:

```python
# Connection pooling
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,
    "pool_recycle": 3600
}

# Redis pooling
REDIS_CONFIG = {
    "max_connections": 50,
    "retry_on_timeout": True,
    "socket_keepalive": True
}
```

### Background Worker Optimization

Configure worker performance:

```python
WORKER_CONFIG = {
    "concurrency": 4,              # Concurrent tasks
    "prefetch_multiplier": 2,      # Task prefetch
    "max_tasks_per_child": 1000,   # Worker restart threshold
    "task_time_limit": 3600        # Max execution time
}
```

### Caching Strategy

Configure caching layers:

```python
CACHE_CONFIG = {
    "search_results": {
        "ttl": 3600,              # 1 hour
        "max_size": "100MB"
    },
    "source_metadata": {
        "ttl": 900,               # 15 minutes
        "max_size": "50MB"
    },
    "embeddings": {
        "ttl": 86400,             # 24 hours
        "max_size": "500MB"
    }
}
```

## Monitoring Configuration

### Prometheus Metrics

Enable monitoring metrics:

```python
# Custom metrics
from prometheus_client import Counter, Histogram, Gauge

api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

search_duration = Histogram(
    'search_duration_seconds',
    'Search request duration',
    ['search_type']
)

active_jobs = Gauge(
    'active_jobs',
    'Number of active jobs',
    ['job_type']
)
```

### Logging Configuration

Configure structured logging:

```python
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json"
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

Enable multi-tenancy:

```python
TENANT_CONFIG = {
    "enabled": true,
    "tenant_header": "X-Tenant-ID",
    "tenant_isolation": "schema",
    "tenant_limits": {
        "max_sources": 100,
        "max_documents": 10000,
        "max_searches_per_hour": 1000
    }
}
```

### Custom Integrations

Configure plugins and integrations:

```python
PLUGIN_CONFIG = {
    "enabled": true,
    "plugin_directory": "/app/plugins",
    "allowed_plugins": [
        "slack_integration",
        "jira_connector"
    ]
}
```

## Configuration Templates

### Development Template

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://dev:dev@localhost:5432/knowledgehub_dev
REDIS_URL=redis://localhost:6379/0
API_RATE_LIMIT=10000
CORS_ORIGINS=*
```

### Production Template

```bash
# .env.production
DEBUG=false
LOG_LEVEL=INFO
JSON_LOGGING=true
DATABASE_URL=postgresql://khuser:${DB_PASSWORD}@db.example.com:5432/knowledgehub
REDIS_URL=redis://:${REDIS_PASSWORD}@redis.example.com:6379/0
SECRET_KEY=${SECRET_KEY}
API_RATE_LIMIT=1000
CORS_ORIGINS=https://knowledgehub.example.com
WORKERS=4
METRICS_ENABLED=true
```

## Next Steps

- [Security Guide](Security) for security-focused configuration
- [Performance](Performance) for optimization strategies
- [Monitoring](Monitoring) for observability setup
- [Troubleshooting](Troubleshooting) for configuration debugging