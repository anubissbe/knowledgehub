# KnowledgeHub API Documentation

## API Version: 1.0.0
**Base URL**: `https://api.knowledgehub.com`  
**Protocol**: HTTPS  
**Authentication**: JWT Bearer Token  

---

## Authentication

### Obtain Access Token
```http
POST /auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Refresh Token
```http
POST /auth/refresh
Authorization: Bearer <refresh_token>
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 1800
}
```

---

## RAG Endpoints

### Search Documents
Perform hybrid search across document repositories.

```http
POST /api/rag/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "machine learning algorithms",
  "mode": "hybrid",
  "filters": {
    "category": ["research", "documentation"],
    "date_from": "2024-01-01",
    "date_to": "2025-12-31"
  },
  "top_k": 10,
  "include_metadata": true,
  "rerank": true
}
```

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Search query text |
| mode | enum | No | Search mode: simple, advanced, performance, hybrid (default) |
| filters | object | No | Filter criteria |
| top_k | integer | No | Number of results (default: 10, max: 100) |
| include_metadata | boolean | No | Include document metadata (default: true) |
| rerank | boolean | No | Apply cross-encoder reranking (default: true) |

**Response** (200 OK):
```json
{
  "results": [
    {
      "id": "doc_123",
      "score": 0.95,
      "content": "Document content...",
      "metadata": {
        "title": "Introduction to ML",
        "author": "John Doe",
        "created_at": "2025-01-15T10:00:00Z",
        "category": "research"
      },
      "highlights": [
        "...machine <mark>learning algorithms</mark> are..."
      ]
    }
  ],
  "total": 42,
  "query_time_ms": 125,
  "mode_used": "hybrid"
}
```

### Generate Answer
Generate an answer using RAG pipeline.

```http
POST /api/rag/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "Explain gradient descent optimization",
  "context_size": 5,
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false
}
```

**Response** (200 OK):
```json
{
  "answer": "Gradient descent is an optimization algorithm...",
  "sources": [
    {
      "id": "doc_456",
      "title": "Optimization Techniques",
      "relevance": 0.92
    }
  ],
  "tokens_used": 350,
  "generation_time_ms": 450
}
```

---

## Memory Endpoints

### Create Memory
Store a new memory entry.

```http
POST /api/memory/create
Authorization: Bearer <token>
Content-Type: application/json

{
  "user_id": "user_123",
  "session_id": "session_456",
  "memory_type": "conversation",
  "content": "User asked about gradient descent",
  "metadata": {
    "timestamp": "2025-08-17T10:00:00Z",
    "context": "machine_learning_discussion"
  },
  "relevance_score": 0.85,
  "ttl": 86400
}
```

**Memory Types**:
- `conversation` - Dialog and chat history
- `knowledge` - Facts and information
- `task` - Tasks and actions
- `context` - Environmental context
- `system` - System state and config

**Response** (201 Created):
```json
{
  "memory_id": "mem_789",
  "created_at": "2025-08-17T10:00:00Z",
  "expires_at": "2025-08-18T10:00:00Z"
}
```

### Retrieve Memories
Get memories for a user/session.

```http
GET /api/memory/retrieve?user_id=user_123&session_id=session_456&limit=20
Authorization: Bearer <token>
```

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes | User identifier |
| session_id | string | No | Session identifier |
| memory_type | string | No | Filter by memory type |
| limit | integer | No | Max results (default: 20) |
| offset | integer | No | Pagination offset |

**Response** (200 OK):
```json
{
  "memories": [
    {
      "memory_id": "mem_789",
      "memory_type": "conversation",
      "content": "User asked about gradient descent",
      "relevance_score": 0.85,
      "created_at": "2025-08-17T10:00:00Z",
      "metadata": {}
    }
  ],
  "total": 42,
  "has_more": true
}
```

---

## Agent Endpoints

### List Available Agents
Get all available agent workflows.

```http
GET /api/agents/agents
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "agents": [
    {
      "id": "research_assistant",
      "name": "Research Assistant",
      "description": "Comprehensive research and analysis",
      "capabilities": ["search", "summarize", "analyze"],
      "input_schema": {
        "query": "string",
        "depth": "enum[quick, standard, comprehensive]"
      }
    },
    {
      "id": "code_reviewer",
      "name": "Code Reviewer",
      "description": "Automated code review and suggestions",
      "capabilities": ["review", "suggest", "refactor"]
    }
  ]
}
```

### Execute Agent Workflow
Run an agent workflow with specified input.

```http
POST /api/agents/execute
Authorization: Bearer <token>
Content-Type: application/json

{
  "agent_id": "research_assistant",
  "input": {
    "query": "Latest advances in quantum computing",
    "depth": "comprehensive"
  },
  "session_id": "session_123",
  "async": false
}
```

**Response** (200 OK):
```json
{
  "execution_id": "exec_456",
  "status": "completed",
  "result": {
    "summary": "Quantum computing has seen significant advances...",
    "findings": [
      {
        "topic": "Quantum Supremacy",
        "description": "...",
        "sources": ["paper_1", "paper_2"]
      }
    ],
    "recommendations": ["..."]
  },
  "execution_time_ms": 2500,
  "steps_completed": 5
}
```

---

## Graph Endpoints

### Query Knowledge Graph
Execute Cypher queries on the knowledge graph.

```http
POST /api/graphrag/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "MATCH (d:Document)-[:MENTIONS]->(e:Entity) WHERE e.name = $entity RETURN d LIMIT 10",
  "parameters": {
    "entity": "machine learning"
  }
}
```

**Response** (200 OK):
```json
{
  "results": [
    {
      "d": {
        "id": "doc_123",
        "title": "ML Fundamentals",
        "created_at": "2025-01-15T10:00:00Z"
      }
    }
  ],
  "query_time_ms": 45
}
```

### Get Graph Statistics
Retrieve knowledge graph statistics.

```http
GET /api/graphrag/graph-stats
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "nodes": {
    "total": 15420,
    "by_type": {
      "Document": 5230,
      "Entity": 8450,
      "Concept": 1740
    }
  },
  "relationships": {
    "total": 48350,
    "by_type": {
      "MENTIONS": 22100,
      "RELATES_TO": 15250,
      "CONTAINS": 11000
    }
  },
  "last_updated": "2025-08-17T10:00:00Z"
}
```

---

## WebSocket Endpoints

### Real-time Updates
Connect for real-time updates and streaming responses.

```javascript
const ws = new WebSocket('wss://api.knowledgehub.com/ws');

ws.on('open', () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer <token>'
  }));
  
  // Subscribe to events
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['memory_updates', 'agent_status']
  }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data);
  // Handle different message types
  switch(message.type) {
    case 'memory_update':
      // New memory created
      break;
    case 'agent_complete':
      // Agent execution finished
      break;
  }
});
```

---

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "query",
      "reason": "Query cannot be empty"
    },
    "request_id": "req_abc123",
    "timestamp": "2025-08-17T10:00:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| UNAUTHORIZED | 401 | Missing or invalid authentication |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| VALIDATION_ERROR | 422 | Invalid input parameters |
| RATE_LIMITED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

---

## Rate Limiting

API requests are rate limited per user:
- **Default**: 100 requests per minute
- **Authenticated**: 1000 requests per minute
- **Premium**: 10000 requests per minute

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1755429000
```

---

## Pagination

For endpoints returning lists, use cursor-based pagination:

```http
GET /api/memory/retrieve?cursor=eyJpZCI6MTIzfQ&limit=20
```

Response includes pagination info:
```json
{
  "data": [...],
  "pagination": {
    "next_cursor": "eyJpZCI6MTQzfQ",
    "has_more": true,
    "total": 542
  }
}
```

---

## Versioning

API version is specified in the URL path:
- Current: `/api/v1/...`
- Legacy: `/api/v0/...` (deprecated)

Version sunset dates are announced 6 months in advance.

---

## SDKs and Libraries

### Python
```python
from knowledgehub import Client

client = Client(api_key="your_api_key")
results = client.rag.search("machine learning", top_k=10)
```

### JavaScript/TypeScript
```typescript
import { KnowledgeHub } from '@knowledgehub/sdk';

const client = new KnowledgeHub({ apiKey: 'your_api_key' });
const results = await client.rag.search('machine learning', { topK: 10 });
```

### cURL Examples
```bash
# Search documents
curl -X POST https://api.knowledgehub.com/api/rag/search \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 10}'

# Create memory
curl -X POST https://api.knowledgehub.com/api/memory/create \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123", "content": "Memory content"}'
```

---

## Changelog

### Version 1.0.0 (2025-08-17)
- Initial API release
- RAG endpoints with hybrid search
- Memory management system
- Agent orchestration
- Knowledge graph queries
- WebSocket support

---

*API Documentation Version: 1.0.0*  
*Last Updated: August 17, 2025*  
*OpenAPI Spec: [Download](https://api.knowledgehub.com/openapi.json)*