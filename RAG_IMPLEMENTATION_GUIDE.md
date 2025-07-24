# KnowledgeHub RAG Implementation Guide

## Overview

This guide documents the production-grade Retrieval-Augmented Generation (RAG) system implemented in KnowledgeHub, following the architectural blueprint from `idea.md`. The implementation provides Claude Code and other AI assistants with a continuously updated, context-aware knowledge base.

## Architecture

### Core Components

1. **LlamaIndex Orchestration** (`api/services/rag/llamaindex_service.py`)
   - Advanced document processing and indexing
   - Hybrid search (vector + keyword)
   - Query processing with re-ranking
   - Streaming response support
   - Fallback to existing search service

2. **Qdrant Vector Database** (Port 6333)
   - High-performance vector storage
   - Scalar quantization enabled
   - On-disk payload storage
   - Distributed architecture ready

3. **Contextual Enrichment** (`api/services/rag/contextual_enrichment.py`)
   - LLM-based chunk enrichment
   - Multiple content type prompts
   - Cost-effective Claude 3 Haiku integration
   - Caching for efficiency

4. **Documentation Scraper** (`api/services/rag/documentation_scraper.py`)
   - Playwright-based JavaScript rendering
   - Ethical scraping (robots.txt, rate limiting)
   - Change detection
   - 10+ real documentation sources configured

## API Endpoints

### Document Ingestion
```bash
POST /api/rag/ingest
{
  "content": "Document content...",
  "title": "Document Title",
  "source_url": "https://example.com",
  "source_type": "documentation",
  "metadata": {},
  "use_contextual_enrichment": true
}
```

### RAG Query
```bash
POST /api/rag/query
{
  "query": "How do I implement authentication?",
  "project_id": "project-123",
  "filters": {"language": "python"},
  "top_k": 5,
  "use_hybrid": true,
  "stream": false
}
```

### Documentation Scraping
```bash
# Start scraping job
POST /api/rag/scrape
{
  "site_name": "fastapi",
  "max_pages": 100,
  "check_changes": true
}

# Check job status
GET /api/rag/scrape/status/{site_name}

# Get scraping statistics
GET /api/rag/scrape/stats

# Schedule all documentation updates
POST /api/rag/scrape/schedule
```

### Other Endpoints
- `GET /api/rag/sources` - List available documentation sources
- `GET /api/rag/index/stats` - Get index statistics
- `POST /api/rag/index/clear` - Clear index (admin only)
- `POST /api/rag/test` - Test RAG pipeline
- `GET /api/rag/health` - Health check

## Configuration

### Environment Variables
```bash
# LlamaIndex Configuration
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
USE_GPU=true

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Claude API (for contextual enrichment)
ANTHROPIC_API_KEY=your-api-key

# Existing Weaviate (parallel operation)
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
```

### Docker Services
Add to your `docker-compose.yml`:
```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__STORAGE__QUANTIZATION__SCALAR: true
      QDRANT__STORAGE__ON_DISK_PAYLOAD: true
```

## Real Data Sources

### Pre-configured Documentation Sites
1. **FastAPI** - https://fastapi.tiangolo.com/
2. **React** - https://react.dev/learn
3. **Django** - https://docs.djangoproject.com/
4. **PostgreSQL** - https://www.postgresql.org/docs/
5. **Docker** - https://docs.docker.com/
6. **Kubernetes** - https://kubernetes.io/docs/
7. **TypeScript** - https://www.typescriptlang.org/docs/
8. **Redis** - https://redis.io/docs/
9. **Elasticsearch** - https://www.elastic.co/guide/
10. **Nginx** - http://nginx.org/en/docs/

### Adding New Sources
Edit `DOCUMENTATION_SOURCES` in `documentation_scraper.py`:
```python
"new_source": {
    "url": "https://docs.example.com/",
    "selector": "main",
    "wait_for": "article",
    "exclude": [".nav", ".footer"]
}
```

## Implementation Features

### 1. Contextual Enrichment
Each chunk is enriched with LLM-generated context:
```
[Context Summary]
This section explains authentication middleware in FastAPI...

---

[Original Chunk]
The actual documentation content...
```

### 2. Hybrid Search
- Vector similarity search (semantic)
- Keyword matching (exact terms)
- Weighted combination (alpha=0.7)
- Post-processing with re-ranking

### 3. Streaming Responses
Support for real-time streaming of RAG responses:
```python
response = await rag_service.query(
    query_text="Complex question",
    stream=True
)
async for chunk in response:
    print(chunk)
```

### 4. Fallback Mechanism
If LlamaIndex fails, automatically falls back to:
- Existing Weaviate search
- Cached results
- Basic keyword search

## Performance Optimizations

### 1. Caching
- Query results cached for 5 minutes
- Enriched chunks cached for 7 days
- Content hashes for change detection

### 2. Batch Processing
- Document chunks processed in batches
- Concurrent enrichment (10 chunks at a time)
- Background index updates

### 3. Resource Management
- Connection pooling
- Circuit breaker for external services
- Rate limiting for scraping

## Security Considerations

### 1. Authentication
- API key required for all endpoints
- Admin-only operations (clear index, schedule)
- User context in metadata

### 2. Input Validation
- Content sanitization
- URL validation for scraping
- Metadata filtering

### 3. Rate Limiting
- Per-domain scraping limits
- API endpoint rate limits
- Circuit breaker protection

## Monitoring and Debugging

### 1. Health Checks
```bash
curl http://localhost:3000/api/rag/health
```

### 2. Metrics Tracked
- Document ingestion count
- Query performance
- Scraping success rate
- Enrichment costs

### 3. Debug Information
- LlamaIndex debug handler
- Detailed error logging
- Query tracing

## Usage Examples

### 1. Ingest Local Documentation
```python
import requests

response = requests.post(
    "http://localhost:3000/api/rag/ingest",
    json={
        "content": open("README.md").read(),
        "title": "Project README",
        "source_type": "documentation",
        "use_contextual_enrichment": True
    },
    headers={"X-API-Key": "your-key"}
)
```

### 2. Query with Filters
```python
response = requests.post(
    "http://localhost:3000/api/rag/query",
    json={
        "query": "How to implement authentication?",
        "filters": {
            "source_type": "documentation",
            "language": "python"
        },
        "top_k": 10
    },
    headers={"X-API-Key": "your-key"}
)
```

### 3. Schedule Documentation Updates
```python
# Schedule scraping for all configured sites
response = requests.post(
    "http://localhost:3000/api/rag/scrape/schedule",
    headers={"X-API-Key": "admin-key"}
)
```

## Implementation Status

### ✅ Phase 1: Core RAG Foundation (Complete)
- LlamaIndex orchestration (simplified version)
- Qdrant vector database deployed
- Contextual enrichment pipeline
- Playwright documentation scraper
- Secure hook execution
- Production API endpoints

### ✅ Phase 2: Advanced Memory and Security (Complete)
- **Zep Memory System** - Conversational memory with temporal knowledge graphs
- **RBAC Implementation** - Role-based permissions and multi-tenant security
- **Hybrid Search** - Combined RAG + conversation memory retrieval
- **API Key Management** - Secure key creation with granular permissions
- **Audit Logging** - Comprehensive access tracking

See [PHASE2_IMPLEMENTATION.md](./PHASE2_IMPLEMENTATION.md) for detailed Phase 2 documentation.

### 🔄 Phase 3: Multi-Agent Evolution (Next)
1. **Orchestrator Pattern** - Task decomposition for complex queries
2. **GraphRAG** - Neo4j PropertyGraphIndex for code relationships
3. **Agent Specialization** - Domain-specific expert agents

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install llama-index llama-index-vector-stores-weaviate llama-index-embeddings-huggingface
   pip install playwright && playwright install chromium
   pip install anthropic tiktoken
   ```

2. **Qdrant Connection Failed**
   - Ensure Qdrant container is running
   - Check port 6333 is accessible
   - Verify no firewall blocking

3. **Scraping Failures**
   - Check robots.txt compliance
   - Verify network connectivity
   - Check circuit breaker status

4. **Memory Issues**
   - Enable quantization in Qdrant
   - Use on-disk storage
   - Implement chunk size limits

## Performance Benchmarks

### Expected Performance
- Document ingestion: ~1000 chunks/minute
- Query latency: <200ms (p95)
- Scraping rate: 2 seconds/page
- Enrichment cost: ~$0.10/1000 chunks

### Optimization Tips
1. Use GPU for embeddings if available
2. Enable Qdrant quantization
3. Implement result caching
4. Batch document processing

## Conclusion

This RAG implementation provides KnowledgeHub with a production-grade document understanding system that:
- Continuously updates from real documentation sources
- Enriches content with contextual information
- Provides sub-200ms query responses
- Scales horizontally with Kubernetes
- Integrates seamlessly with existing services

The system is designed for gradual rollout, allowing parallel operation with existing search functionality while measuring performance improvements.