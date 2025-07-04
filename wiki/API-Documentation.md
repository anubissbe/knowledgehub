# API Documentation

The KnowledgeHub API is a comprehensive REST API built with FastAPI that provides full access to all system functionality. It features automatic OpenAPI documentation, input validation, and real-time capabilities via WebSocket.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
- [Real-time Communication](#real-time-communication)
- [Rate Limiting](#rate-limiting)
- [Pagination](#pagination)
- [SDK Examples](#sdk-examples)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Overview

### Base URL
```
http://localhost:3000
```

### Interactive Documentation
- **Swagger UI**: http://localhost:3000/docs
- **ReDoc**: http://localhost:3000/redoc
- **OpenAPI JSON**: http://localhost:3000/openapi.json

## Authentication

Currently using API key authentication. Include the API key in the header:

```http
X-API-Key: your-api-key-here
```

## Response Format

All API responses follow a consistent format:

```json
{
  "status": "success|error",
  "data": {},
  "message": "Human readable message",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

Error responses include detailed information:

```json
{
  "status": "error",
  "message": "Validation failed",
  "details": {
    "field": "url",
    "error": "Invalid URL format"
  }
}
```

## Endpoints

### System Health

#### Get System Health
```http
GET /health
```

Returns comprehensive system status including all service dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1751572319.7046626,
  "services": {
    "api": "operational",
    "database": "operational", 
    "redis": "operational",
    "weaviate": "operational"
  }
}
```

### Knowledge Sources

#### List Sources
```http
GET /api/v1/sources
```

**Query Parameters:**
- `skip` (int, optional): Number of records to skip (default: 0)
- `limit` (int, optional): Maximum records to return (default: 100)
- `status` (string, optional): Filter by status (active, inactive, error)

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "GitHub Documentation",
    "description": "Official GitHub documentation",
    "base_url": "https://docs.github.com",
    "source_type": "web",
    "status": "active",
    "config": {
      "max_depth": 3,
      "max_pages": 1000,
      "follow_patterns": ["**"],
      "exclude_patterns": ["**/api/**"]
    },
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
    "last_crawl_at": "2024-01-01T00:00:00Z",
    "metadata": {}
  }
]
```

#### Create Source
```http
POST /api/v1/sources
```

**Request Body:**
```json
{
  "name": "Documentation Site",
  "description": "Project documentation",
  "base_url": "https://docs.example.com",
  "source_type": "web",
  "config": {
    "max_depth": 2,
    "max_pages": 500,
    "follow_patterns": ["**"],
    "exclude_patterns": []
  }
}
```

**Response:**
```json
{
  "source": {
    "id": "uuid",
    "name": "Documentation Site",
    // ... full source object
  },
  "job_id": "uuid"
}
```

#### Get Source
```http
GET /api/v1/sources/{source_id}
```

Returns detailed information about a specific source.

#### Update Source
```http
PUT /api/v1/sources/{source_id}
```

**Request Body:** Same as create source

#### Delete Source
```http
DELETE /api/v1/sources/{source_id}
```

Returns a job ID for the deletion process.

#### Trigger Source Refresh
```http
POST /api/v1/sources/{source_id}/refresh
```

**Request Body:**
```json
{
  "force_refresh": false,
  "incremental": true
}
```

### Search

#### Execute Search
```http
POST /api/v1/search
```

**Request Body:**
```json
{
  "query": "How to deploy applications",
  "search_type": "hybrid",
  "source_ids": ["uuid1", "uuid2"],
  "limit": 20,
  "filters": {
    "chunk_type": "content",
    "min_relevance": 0.7
  }
}
```

**Search Types:**
- `hybrid` - Combines vector and keyword search (recommended)
- `vector` - Semantic similarity search
- `keyword` - Traditional text search

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "source_id": "uuid",
      "content": "Text content...",
      "title": "Document Title", 
      "url": "https://example.com/page",
      "relevance_score": 0.95,
      "chunk_index": 0,
      "chunk_type": "content",
      "metadata": {},
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 156,
  "query": "How to deploy applications",
  "search_type": "hybrid",
  "took_ms": 45
}
```

### Jobs

#### List Jobs
```http
GET /api/v1/jobs
```

**Query Parameters:**
- `skip` (int): Pagination offset
- `limit` (int): Results per page
- `status` (string): Filter by status (pending, running, completed, failed, cancelled)
- `job_type` (string): Filter by job type (crawl, delete, refresh)
- `source_id` (string): Filter by source

**Response:**
```json
[
  {
    "id": "uuid",
    "source_id": "uuid", 
    "job_type": "crawl",
    "status": "completed",
    "progress": 100,
    "total_pages": 150,
    "processed_pages": 150,
    "error_message": null,
    "result": {
      "pages_crawled": 150,
      "chunks_created": 1250,
      "elapsed_time": "00:02:15"
    },
    "created_at": "2024-01-01T00:00:00Z",
    "started_at": "2024-01-01T00:00:30Z",
    "completed_at": "2024-01-01T00:02:45Z"
  }
]
```

#### Get Job Details
```http
GET /api/v1/jobs/{job_id}
```

#### Cancel Job
```http
POST /api/v1/jobs/{job_id}/cancel
```

#### Retry Failed Job
```http
POST /api/v1/jobs/{job_id}/retry
```

### Documents

#### List Documents
```http
GET /api/v1/documents
```

**Query Parameters:**
- `source_id` (string): Filter by source
- `status` (string): Filter by status
- `skip` (int): Pagination offset
- `limit` (int): Results per page (max 10,000)

**Response:**
```json
{
  "documents": [
    {
      "id": "uuid",
      "source_id": "uuid",
      "url": "https://example.com/page",
      "title": "Page Title",
      "content_hash": "sha256...",
      "status": "processed",
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z",
      "metadata": {}
    }
  ],
  "total": 1500,
  "skip": 0,
  "limit": 100
}
```

#### Get Document
```http
GET /api/v1/documents/{document_id}
```

#### Get Document Chunks
```http
GET /api/v1/documents/{document_id}/chunks
```

### Memory System

#### Store Memory
```http
POST /api/v1/memories
```

**Request Body:**
```json
{
  "content": "Important information to remember",
  "memory_type": "conversation",
  "priority": "high",
  "tags": ["important", "deployment"],
  "metadata": {
    "context": "deployment discussion"
  }
}
```

#### List Memories
```http
GET /api/v1/memories
```

**Query Parameters:**
- `memory_type` (string): Filter by type
- `priority` (string): Filter by priority
- `tags` (array): Filter by tags
- `skip` (int): Pagination
- `limit` (int): Results per page

#### Search Memories
```http
GET /api/v1/memories/search?q=deployment&memory_type=conversation
```

#### Get Memory Statistics
```http
GET /api/v1/memories/stats
```

**Response:**
```json
{
  "total_memories": 1250,
  "by_type": {
    "conversation": 800,
    "code": 200,
    "decision": 150,
    "error": 100
  },
  "by_priority": {
    "high": 300,
    "medium": 700,
    "low": 250
  },
  "storage_size_mb": 12.5
}
```

## Real-time Communication

### WebSocket Connection
```
ws://localhost:3000/ws
```

**Message Format:**
```json
{
  "type": "job_update",
  "data": {
    "job_id": "uuid",
    "status": "running", 
    "progress": 45,
    "message": "Processing page 45 of 100"
  }
}
```

**Message Types:**
- `job_update` - Job status changes
- `system_alert` - System notifications
- `search_progress` - Search operation progress

## Rate Limiting

API endpoints are rate limited to prevent abuse:

- **Default**: 100 requests per minute per IP
- **Search endpoints**: 20 requests per minute per IP
- **Batch operations**: 5 requests per minute per IP

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Pagination

List endpoints support cursor-based pagination:

```http
GET /api/v1/sources?skip=0&limit=50
```

Responses include pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "skip": 0,
    "limit": 50,
    "total": 1500,
    "has_next": true
  }
}
```

## SDK Examples

### Python
```python
import requests

# Search for content
response = requests.post(
    "http://localhost:3000/api/v1/search",
    json={
        "query": "deployment best practices",
        "search_type": "hybrid",
        "limit": 10
    },
    headers={"X-API-Key": "your-key"}
)

results = response.json()
for result in results["results"]:
    print(f"{result['title']}: {result['content'][:100]}...")
```

### JavaScript
```javascript
// Add a new source
const response = await fetch('http://localhost:3000/api/v1/sources', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-key'
  },
  body: JSON.stringify({
    name: 'Documentation Site',
    base_url: 'https://docs.example.com',
    source_type: 'web'
  })
});

const result = await response.json();
console.log('Source created:', result.source.id);
console.log('Job started:', result.job_id);
```

### cURL
```bash
# Get system health
curl -X GET "http://localhost:3000/health"

# Create a new source
curl -X POST "http://localhost:3000/api/v1/sources" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "name": "Example Docs",
    "base_url": "https://docs.example.com",
    "source_type": "web"
  }'

# Search for content
curl -X POST "http://localhost:3000/api/v1/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "query": "API documentation",
    "search_type": "hybrid",
    "limit": 5
  }'
```

## Performance Tips

1. **Use Hybrid Search**: Combines the best of vector and keyword search
2. **Limit Results**: Use appropriate `limit` values to reduce response times
3. **Filter by Source**: Narrow searches to specific sources when possible
4. **Batch Operations**: Use batch endpoints for bulk operations
5. **WebSocket Updates**: Use WebSocket for real-time job monitoring instead of polling

## Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check API key is included in headers
   - Verify API key is valid and active

2. **422 Validation Error**
   - Check request body matches expected schema
   - Verify all required fields are included

3. **500 Internal Server Error**
   - Check system health endpoint
   - Verify all dependent services are running

4. **Slow Search Response**
   - Reduce result limit
   - Use more specific queries
   - Check Weaviate service status

### Debug Information

Enable debug mode by setting the environment variable:
```bash
DEBUG=true
```

This will include additional debug information in error responses.

## Next Steps

- [Configuration Guide](Configuration) for API configuration options
- [Security Guide](Security) for authentication setup
- [Performance Guide](Performance) for optimization tips
- [Monitoring Guide](Monitoring) for API metrics