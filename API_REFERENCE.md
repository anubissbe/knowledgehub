# 🔌 KnowledgeHub API Reference

## Base URL
```
http://localhost:3000/api
```

## Authentication
All endpoints require JWT authentication unless marked as public.

```http
Authorization: Bearer <jwt_token>
```

---

## 🤖 Hybrid RAG Endpoints

### Query Hybrid RAG
```http
POST /api/rag/enhanced/query
```

**Request Body:**
```json
{
  "query": "string",
  "mode": "hybrid|vector|sparse|graph",
  "top_k": 10,
  "rerank": true,
  "include_metadata": true,
  "session_id": "string (optional)",
  "filters": {
    "date_range": ["2024-01-01", "2024-12-31"],
    "source_types": ["documentation", "code"],
    "confidence_threshold": 0.7
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "string",
      "score": 0.95,
      "source": "string",
      "metadata": {},
      "retrieval_method": "vector|sparse|graph"
    }
  ],
  "performance": {
    "total_time_ms": 150,
    "retrieval_time_ms": 100,
    "reranking_time_ms": 50
  },
  "session_id": "string"
}
```

### Ingest Documents
```http
POST /api/rag/enhanced/ingest
```

**Request Body:**
```json
{
  "documents": [
    {
      "content": "string",
      "metadata": {
        "source": "string",
        "title": "string",
        "tags": ["array"]
      }
    }
  ],
  "processing_options": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "extract_entities": true,
    "generate_embeddings": true
  }
}
```

---

## 🎭 Agent Workflow Endpoints

### Execute Workflow
```http
POST /api/agent/workflows/execute
```

**Request Body:**
```json
{
  "workflow_type": "simple_qa|multi_step_research|comparative_analysis|custom",
  "input": {
    "query": "string",
    "context": {},
    "parameters": {}
  },
  "agents": ["researcher", "analyst", "synthesizer"],
  "max_iterations": 5,
  "timeout_seconds": 300
}
```

**Response:**
```json
{
  "workflow_id": "string",
  "status": "running|completed|failed",
  "result": {
    "answer": "string",
    "reasoning": "string",
    "sources": ["array"],
    "confidence": 0.85
  },
  "execution_trace": [
    {
      "agent": "researcher",
      "action": "search",
      "result": {},
      "timestamp": "ISO 8601"
    }
  ]
}
```

### Stream Workflow Execution
```http
POST /api/agent/workflows/stream
```

**Request:** Same as execute workflow

**Response:** Server-Sent Events (SSE) stream
```
event: start
data: {"workflow_id": "123", "timestamp": "2025-08-16T10:00:00Z"}

event: agent_action
data: {"agent": "researcher", "action": "searching", "progress": 0.3}

event: result
data: {"partial_result": "Found relevant information..."}

event: complete
data: {"final_result": {...}, "total_time_ms": 5000}
```

### Get Workflow Status
```http
GET /api/agent/workflows/status/{workflow_id}
```

**Response:**
```json
{
  "workflow_id": "string",
  "status": "running|completed|failed",
  "progress": 0.75,
  "current_agent": "analyst",
  "start_time": "ISO 8601",
  "end_time": "ISO 8601",
  "result": {}
}
```

---

## 🧠 Memory Management Endpoints

### Store Memory
```http
POST /api/memory/store
```

**Request Body:**
```json
{
  "content": "string",
  "type": "episodic|semantic|procedural",
  "metadata": {
    "source": "string",
    "importance": 0.8,
    "tags": ["array"],
    "related_entities": ["array"]
  },
  "session_id": "string",
  "user_id": "string"
}
```

### Recall Memory
```http
GET /api/memory/recall
```

**Query Parameters:**
- `query`: Search query
- `type`: Memory type filter
- `limit`: Number of results (default: 10)
- `threshold`: Relevance threshold (0-1)
- `session_id`: Session context
- `user_id`: User context

**Response:**
```json
{
  "memories": [
    {
      "id": "string",
      "content": "string",
      "type": "episodic|semantic",
      "relevance_score": 0.92,
      "metadata": {},
      "created_at": "ISO 8601"
    }
  ],
  "total_count": 42
}
```

### Search Memory with Zep
```http
POST /api/memory/zep/search
```

**Request Body:**
```json
{
  "query": "string",
  "session_id": "string",
  "search_type": "similarity|mmr",
  "search_scope": "messages|summaries",
  "limit": 10
}
```

---

## 🕷️ Web Ingestion Endpoints

### Start Crawl Job
```http
POST /api/ingestion/crawl
```

**Request Body:**
```json
{
  "url": "https://example.com",
  "mode": "single|sitemap|recursive|selective",
  "options": {
    "max_depth": 3,
    "max_pages": 100,
    "include_patterns": ["*/docs/*"],
    "exclude_patterns": ["*/api/*"],
    "wait_for_selector": ".content",
    "extract_metadata": true
  },
  "processing": {
    "clean_html": true,
    "extract_code_blocks": true,
    "generate_summaries": true
  }
}
```

**Response:**
```json
{
  "job_id": "string",
  "status": "queued",
  "estimated_time_seconds": 120,
  "webhook_url": "string (optional)"
}
```

### Get Ingestion Status
```http
GET /api/ingestion/status/{job_id}
```

**Response:**
```json
{
  "job_id": "string",
  "status": "running|completed|failed",
  "progress": {
    "pages_crawled": 45,
    "total_pages": 100,
    "percentage": 0.45
  },
  "results": {
    "successful": 42,
    "failed": 3,
    "skipped": 0
  },
  "errors": ["array of error messages"],
  "completed_at": "ISO 8601"
}
```

---

## 📊 Analytics & Monitoring Endpoints

### System Health
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "ISO 8601",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "neo4j": "healthy",
    "weaviate": "healthy",
    "zep": "healthy",
    "firecrawl": "degraded"
  },
  "metrics": {
    "uptime_seconds": 86400,
    "total_requests": 1000000,
    "active_sessions": 42
  }
}
```

### RAG Performance Metrics
```http
GET /api/rag/enhanced/performance
```

**Response:**
```json
{
  "retrieval_metrics": {
    "avg_response_time_ms": 150,
    "p95_response_time_ms": 300,
    "p99_response_time_ms": 500,
    "total_queries": 10000,
    "cache_hit_rate": 0.65
  },
  "quality_metrics": {
    "avg_relevance_score": 0.85,
    "reranking_improvement": 0.15,
    "user_satisfaction": 0.92
  },
  "resource_usage": {
    "memory_mb": 2048,
    "cpu_percent": 35,
    "index_size_gb": 10.5
  }
}
```

### Agent Workflow Analytics
```http
GET /api/agent/workflows/analytics
```

**Query Parameters:**
- `start_date`: ISO 8601 date
- `end_date`: ISO 8601 date
- `workflow_type`: Filter by type
- `aggregation`: hour|day|week|month

**Response:**
```json
{
  "summary": {
    "total_executions": 500,
    "success_rate": 0.94,
    "avg_duration_seconds": 12.5,
    "unique_users": 50
  },
  "by_workflow_type": {
    "simple_qa": {
      "count": 300,
      "avg_duration": 5.2,
      "success_rate": 0.98
    },
    "multi_step_research": {
      "count": 150,
      "avg_duration": 25.3,
      "success_rate": 0.89
    }
  },
  "time_series": [
    {
      "timestamp": "ISO 8601",
      "executions": 25,
      "avg_duration": 10.5,
      "errors": 2
    }
  ]
}
```

---

## 🔐 Authentication Endpoints

### Login
```http
POST /api/auth/login
```

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "string",
    "username": "string",
    "roles": ["user", "admin"]
  }
}
```

### Refresh Token
```http
POST /api/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

---

## 🔔 WebSocket Events

### Connection
```javascript
const ws = new WebSocket('ws://localhost:3000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'authenticate',
    token: 'jwt_token'
  }));
};
```

### Event Types

#### Workflow Updates
```json
{
  "type": "workflow_update",
  "data": {
    "workflow_id": "string",
    "status": "running",
    "progress": 0.5,
    "current_agent": "analyst"
  }
}
```

#### Memory Updates
```json
{
  "type": "memory_stored",
  "data": {
    "memory_id": "string",
    "type": "episodic",
    "content": "string"
  }
}
```

#### Ingestion Progress
```json
{
  "type": "ingestion_progress",
  "data": {
    "job_id": "string",
    "pages_crawled": 10,
    "total_pages": 50
  }
}
```

---

## 📋 Error Responses

All endpoints follow a consistent error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "query",
      "issue": "Required field missing"
    }
  },
  "timestamp": "ISO 8601",
  "request_id": "string"
}
```

### Common Error Codes
- `AUTHENTICATION_ERROR` - Invalid or missing authentication
- `AUTHORIZATION_ERROR` - Insufficient permissions
- `VALIDATION_ERROR` - Invalid request parameters
- `NOT_FOUND` - Resource not found
- `RATE_LIMIT_ERROR` - Rate limit exceeded
- `SERVICE_ERROR` - Internal service error
- `TIMEOUT_ERROR` - Request timeout

---

## 🔒 Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **Standard endpoints**: 100 requests/minute
- **Search endpoints**: 50 requests/minute
- **Ingestion endpoints**: 10 requests/minute
- **Workflow endpoints**: 20 requests/minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1629300000
```

---

## 🔄 Versioning

The API uses URL versioning. Current version: `v1`

Future versions will be available at:
```
http://localhost:3000/api/v2/...
```

---

*API Reference Version: 2.0.0*  
*Last Updated: August 2025*