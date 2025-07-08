# Memory Search Integration

## Overview

The Memory Search Integration bridges the memory system with the existing search infrastructure, providing unified search capabilities across both documents and conversation memories.

## Architecture

### Components

1. **MemorySearchService** (`src/api/memory_system/services/memory_search_service.py`)
   - Extends the base SearchService
   - Provides hybrid vector/keyword search for memories
   - Integrates with existing caching and vector store infrastructure

2. **Unified Search Router** (`src/api/routers/unified_search.py`)
   - Single endpoint for searching both documents and memories
   - Parallel execution of searches for performance
   - Combined results with separate sections

3. **Memory Router Updates** (`src/api/memory_system/api/routers/memory.py`)
   - Enhanced search endpoint using the integrated service
   - Fallback to basic search if vector search fails

## Features

### 1. Hybrid Search Approach
- **Vector Search**: Semantic similarity using embeddings
- **Keyword Search**: PostgreSQL full-text search or ILIKE fallback
- **Filter-based Search**: Search by memory type, importance, user, project

### 2. Unified Search Endpoint
```bash
POST /api/v1/search/unified
{
  "query": "authentication JWT",
  "search_type": "hybrid",
  "limit": 10,
  "include_memories": true,
  "memory_user_id": "user-123",
  "memory_min_importance": 0.5
}
```

### 3. Search Suggestions
```bash
GET /api/v1/search/suggest?q=auth&include_memories=true&user_id=user-123
```

### 4. Caching
- 5-minute TTL cache for search results
- Redis-based caching using deterministic cache keys
- Automatic cache invalidation on memory updates

## API Endpoints

### Memory Search
```bash
POST /api/memory/memories/search
{
  "query": "error handling",
  "user_id": "user-123",
  "memory_types": ["error", "solution"],
  "min_importance": 0.7,
  "use_vector_search": true,
  "limit": 20,
  "offset": 0
}
```

### Unified Search
```bash
POST /api/v1/search/unified
{
  "query": "authentication",
  "search_type": "hybrid",
  "include_memories": true,
  "memory_user_id": "user-123"
}
```

Response:
```json
{
  "query": "authentication",
  "search_type": "hybrid",
  "documents": {
    "results": [...],
    "total": 25,
    "search_time_ms": 150
  },
  "memories": {
    "results": [...],
    "total": 10,
    "search_time_ms": 50
  },
  "total_results": 35,
  "total_search_time_ms": 200
}
```

## Configuration

### Environment Variables
```bash
# Embeddings service (for vector search)
EMBEDDINGS_SERVICE_URL=http://ai-service:8000

# Redis (for caching)
REDIS_URL=redis://redis:6379/0

# Vector store (Weaviate)
WEAVIATE_URL=http://weaviate:8080
```

## Search Ranking

1. **Vector Search Results**
   - Ranked by cosine similarity score
   - Minimum similarity threshold: 0.7

2. **Keyword Search Results**
   - Ranked by importance and recency
   - Full-text search ranking when available

3. **Merged Results**
   - Vector results prioritized
   - Keyword-only results get 0.8x score penalty
   - Final ranking by relevance score and importance

## Error Handling

1. **Embeddings Service Unavailable**
   - Graceful fallback to keyword search
   - Warning logged, no error to user

2. **Vector Store Issues**
   - Fallback to database-only search
   - Cached results used if available

3. **Database Errors**
   - 500 error with appropriate message
   - Transaction rollback

## Performance Considerations

1. **Parallel Execution**
   - Document and memory searches run concurrently
   - Vector and keyword searches run in parallel

2. **Caching Strategy**
   - Results cached for 5 minutes
   - Cache key based on all search parameters

3. **Database Optimization**
   - PostgreSQL full-text search indexes
   - Composite indexes on frequently filtered columns

## Testing

Run the integration test:
```bash
python3 test_integrated_memory_search.py
```

This tests:
- Memory creation and search
- Vector similarity search
- Keyword filtering
- Unified search across documents and memories
- Search suggestions

## Known Limitations

1. **Vector Search Dependency**
   - Requires embeddings service to be running
   - Falls back to keyword search if unavailable

2. **Duplicate Results**
   - May occur with identical memories across sessions
   - Deduplication based on memory ID

3. **Performance at Scale**
   - Vector search performance depends on embedding dimensions
   - Consider pagination for large result sets

## Future Enhancements

1. **Advanced Ranking**
   - Machine learning-based result ranking
   - User preference learning

2. **Real-time Updates**
   - WebSocket support for live search updates
   - Incremental index updates

3. **Search Analytics**
   - Track popular queries
   - Search performance metrics
   - User interaction tracking
