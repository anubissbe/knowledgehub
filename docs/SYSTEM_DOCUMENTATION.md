# KnowledgeHub System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [API Reference](#api-reference)
5. [Security](#security)
6. [Performance](#performance)
7. [Deployment](#deployment)
8. [Monitoring](#monitoring)

---

## System Overview

KnowledgeHub is an enterprise-grade AI-powered knowledge management platform that combines advanced RAG (Retrieval-Augmented Generation) capabilities with real-time analytics, multi-agent workflows, and comprehensive memory management.

### Key Features
- **Hybrid RAG System**: Vector, sparse, and graph-based search
- **Multi-Agent Orchestration**: LangGraph-powered agent workflows
- **Memory Management**: Zep integration for persistent memory
- **Real-time Analytics**: TimescaleDB for time-series data
- **Knowledge Graph**: Neo4j for relationship mapping
- **Vector Search**: Weaviate and Qdrant for semantic search

### Technology Stack
- **Backend**: Python 3.11, FastAPI
- **Frontend**: React, TypeScript, Vite
- **Databases**: PostgreSQL, TimescaleDB, Neo4j, Redis
- **Vector DBs**: Weaviate, Qdrant
- **Object Storage**: MinIO
- **Container**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana

---

## Architecture

### System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                         Load Balancer                         │
└─────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────▼─────────┐                           ┌────────▼────────┐
│   Web UI        │                           │   API Gateway    │
│  (React/TS)     │◄─────────HTTPS───────────►│   (FastAPI)     │
└─────────────────┘                           └────────┬────────┘
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────┐
                    │                                  │                              │
        ┌───────────▼──────────┐       ┌──────────────▼──────────┐      ┌───────────▼──────────┐
        │   AI Service         │       │   Agent Orchestrator     │      │   Memory Service     │
        │   (Embeddings)       │       │   (LangGraph)           │      │   (Zep)             │
        └──────────┬───────────┘       └──────────┬──────────────┘      └───────────┬──────────┘
                   │                               │                                  │
    ┌──────────────┼───────────────────────────────┼─────────────────────────────────┼──────────┐
    │              │                               │                                  │          │
┌───▼────┐  ┌─────▼─────┐  ┌──────────┐  ┌───────▼──────┐  ┌────────────┐  ┌───────▼───┐  ┌──▼──┐
│Weaviate│  │  Qdrant   │  │PostgreSQL│  │    Neo4j     │  │TimescaleDB │  │   Redis   │  │MinIO│
│(Vector)│  │ (Vector)  │  │  (RDBMS) │  │   (Graph)    │  │(Time-Series)│  │  (Cache)  │  │(S3) │
└────────┘  └───────────┘  └──────────┘  └──────────────┘  └────────────┘  └───────────┘  └─────┘
```

### Microservices Architecture
The system is composed of 15+ microservices, each with specific responsibilities:

1. **API Gateway** - Request routing and authentication
2. **AI Service** - Embedding generation and ML operations
3. **Agent Orchestrator** - Multi-agent workflow coordination
4. **Memory Service** - Context and conversation persistence
5. **RAG Service** - Hybrid retrieval and generation
6. **Graph Service** - Knowledge graph operations
7. **Analytics Service** - Time-series analytics
8. **Search Service** - Multi-modal search capabilities
9. **Cache Service** - Performance optimization
10. **Security Service** - Authentication and authorization

---

## Core Components

### 1. Unified RAG Service
Located at: `/api/services/unified_rag_service.py`

```python
class UnifiedRAGService:
    """Unified RAG service consolidating all implementations"""
    
    async def search(
        self,
        query: str,
        mode: RAGMode = RAGMode.HYBRID,
        filters: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Unified search interface supporting multiple modes:
        - SIMPLE: Basic vector search
        - ADVANCED: Search with filtering
        - PERFORMANCE: Cached optimized search
        - HYBRID: Multi-strategy search
        """
```

### 2. Authentication System
Located at: `/api/security/authentication.py`

```python
class AuthenticationSystem:
    """JWT-based authentication with token management"""
    
    def create_access_token(self, data: dict) -> str:
        """Generate JWT access token"""
        
    def verify_token(self, credentials: HTTPAuthorizationCredentials) -> dict:
        """Verify and decode JWT token"""
```

### 3. Caching System
Located at: `/api/services/caching_system.py`

```python
@cached(ttl=600, prefix="rag")
async def search_documents(query: str):
    """Cached document search with 10-minute TTL"""
    # Expensive search operation
    return results
```

### 4. Database Optimizer
Located at: `/api/services/db_optimizer.py`

Features:
- Connection pooling (20 base + 40 overflow)
- Batch query execution
- Cursor-based pagination
- Concurrent index creation

---

## API Reference

### Base URL
```
https://api.knowledgehub.com
```

### Authentication
All API requests require JWT authentication:
```http
Authorization: Bearer <jwt_token>
```

### Core Endpoints

#### Health Check
```http
GET /health
```
Response:
```json
{
  "status": "healthy",
  "timestamp": 1755428541,
  "services": {
    "api": "operational",
    "database": "connected",
    "redis": "connected",
    "weaviate": "connected"
  }
}
```

#### RAG Search
```http
POST /api/rag/search
Content-Type: application/json

{
  "query": "search text",
  "mode": "hybrid",
  "filters": {
    "category": "documentation"
  },
  "top_k": 10
}
```

#### Memory Operations
```http
POST /api/memory/create
Content-Type: application/json

{
  "user_id": "user123",
  "session_id": "session456",
  "memory_type": "conversation",
  "content": "Memory content",
  "metadata": {}
}
```

#### Agent Workflows
```http
POST /api/agents/execute
Content-Type: application/json

{
  "workflow": "research_assistant",
  "input": {
    "query": "Research topic",
    "depth": "comprehensive"
  }
}
```

### Error Responses
```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {}
}
```

Error Codes:
- `401` - Authentication required
- `403` - Forbidden
- `404` - Resource not found
- `422` - Validation error
- `429` - Rate limit exceeded
- `500` - Internal server error

---

## Security

### Authentication & Authorization
- **JWT-based authentication** with 30-minute token expiration
- **API key management** with Redis storage and encryption
- **Role-based access control** (RBAC) for fine-grained permissions

### Security Headers
```python
Content-Security-Policy: default-src 'self'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
Referrer-Policy: strict-origin-when-cross-origin
```

### Input Validation
- SQL injection protection
- XSS prevention
- Path traversal blocking
- Command injection prevention

### Credentials Management
All sensitive credentials are externalized to environment variables:
```bash
JWT_SECRET_KEY=${JWT_SECRET}
DATABASE_PASSWORD=${DB_PASSWORD}
NEO4J_PASSWORD=${NEO4J_PASSWORD}
REDIS_PASSWORD=${REDIS_PASSWORD}
```

---

## Performance

### Caching Strategy
1. **Multi-layer caching**: Redis + In-memory
2. **Cache TTL**: 5-10 minutes for dynamic content
3. **Cache invalidation**: Pattern-based and event-driven

### Database Optimization
1. **Connection pooling**: 20 base connections, 40 overflow
2. **Query optimization**: Batch execution and cursor pagination
3. **Indexes**: 12+ performance indexes on critical tables

### Async Operations
1. **Concurrency limit**: 100 simultaneous operations
2. **Batch processing**: 50 items per batch
3. **Timeout handling**: 30-second default timeout

### Performance Targets
- API response time: < 100ms (p95)
- Database queries: < 50ms (p95)
- Cache hit rate: > 80%
- Error rate: < 0.1%

---

## Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Node.js 18+
- 16GB RAM minimum
- 50GB disk space

### Quick Start
```bash
# Clone repository
git clone https://github.com/org/knowledgehub.git
cd knowledgehub

# Configure environment
cp .env.example .env.production
# Edit .env.production with your values

# Start services
docker-compose up -d

# Run migrations
docker exec knowledgehub-api-1 python -m alembic upgrade head

# Verify health
curl http://localhost:3000/health
```

### Docker Compose Services
```yaml
services:
  api:
    image: knowledgehub/api:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - postgres
      - redis
      - neo4j
      
  web:
    image: knowledgehub/web:latest
    ports:
      - "3100:3100"
    environment:
      - API_URL=http://api:3000
      
  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
      
  # ... other services
```

### Production Checklist
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database backups configured
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Rate limiting enabled
- [ ] Security headers verified
- [ ] Load testing completed

---

## Monitoring

### Metrics Collection
Prometheus metrics are exposed at `/metrics`:

```python
# Request metrics
http_requests_total{method="GET", endpoint="/api/rag/search", status="200"}
http_request_duration_seconds{method="POST", endpoint="/api/memory/create"}

# System metrics
system_cpu_usage_percent
system_memory_usage_percent

# Database metrics
database_connections_active{database="postgres"}
database_query_duration_seconds{query_type="select"}

# Cache metrics
cache_hits_total{cache_type="redis"}
cache_misses_total{cache_type="redis"}
```

### Alert Rules
```yaml
alerts:
  - name: HighCPUUsage
    condition: cpu_usage > 80
    severity: high
    message: "CPU usage above 80%"
    
  - name: HighMemoryUsage
    condition: memory_usage > 90
    severity: critical
    message: "Memory usage above 90%"
    
  - name: SlowResponseTime
    condition: avg_response_time > 1.0
    severity: medium
    message: "Average response time above 1 second"
```

### Grafana Dashboards
1. **System Overview**: CPU, memory, disk, network
2. **API Performance**: Request rate, response time, error rate
3. **Database Metrics**: Connections, query performance
4. **Cache Performance**: Hit rate, evictions
5. **Business Metrics**: User activity, search queries

### Logging
Structured JSON logging with correlation IDs:
```json
{
  "timestamp": "2025-08-17T11:00:00Z",
  "level": "INFO",
  "correlation_id": "abc123",
  "service": "api",
  "message": "Request processed",
  "metadata": {
    "endpoint": "/api/rag/search",
    "duration_ms": 45,
    "status_code": 200
  }
}
```

---

## Troubleshooting

### Common Issues

#### API Returns 401 Unauthorized
**Solution**: Check JWT token expiration and refresh if needed
```bash
curl -X POST http://api/auth/refresh \
  -H "Authorization: Bearer <refresh_token>"
```

#### High Memory Usage
**Solution**: Check cache size and eviction policies
```bash
docker exec knowledgehub-redis-1 redis-cli INFO memory
```

#### Slow Database Queries
**Solution**: Check missing indexes and query plans
```sql
EXPLAIN ANALYZE SELECT * FROM memories WHERE user_id = '123';
```

#### Service Unavailable
**Solution**: Check service health and restart if needed
```bash
docker-compose ps
docker-compose restart <service_name>
```

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
docker-compose restart api
```

### Support
- GitHub Issues: https://github.com/org/knowledgehub/issues
- Documentation: https://docs.knowledgehub.com
- Email: support@knowledgehub.com

---

*Last Updated: August 17, 2025*  
*Version: 1.0.0*  
*Status: Production Ready*