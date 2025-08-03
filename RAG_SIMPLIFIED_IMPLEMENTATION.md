# Simplified RAG Implementation for KnowledgeHub

## Overview

This is a simplified RAG (Retrieval-Augmented Generation) implementation that works with the existing KnowledgeHub infrastructure without requiring complex dependencies that conflict with the current system.

## Key Features

1. **Minimal Dependencies**: Works with existing packages, LlamaIndex is optional
2. **Graceful Fallback**: Falls back to existing search infrastructure if LlamaIndex isn't available
3. **Compatible API**: Same API endpoints as the full RAG implementation
4. **Contextual Enrichment**: Basic chunk enrichment without requiring external LLMs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  /api/rag/ingest    /api/rag/query    /api/rag/health      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Simple RAG Service                         │
│  - Document chunking (simple or LlamaIndex)                 │
│  - Contextual enrichment (basic implementation)             │
│  - Query orchestration                                       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Existing Infrastructure                         │
│  - EmbeddingService (HuggingFace models)                   │
│  - VectorStoreService (Weaviate)                           │
│  - SearchService (hybrid search)                            │
│  - CacheService (Redis)                                     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Option 1: Use with existing dependencies only
No additional installation needed! The simple RAG service works with your current packages.

### Option 2: Add optional LlamaIndex support
If you want to enable LlamaIndex features (better chunking, advanced retrieval):

```bash
# Install optional dependencies in a virtual environment
python -m venv rag_env
source rag_env/bin/activate
pip install -r requirements-rag-optional.txt
```

## Usage

### 1. Document Ingestion

```python
# Example: Ingest a document
POST /api/rag/ingest
{
    "content": "Your document content here...",
    "title": "Document Title",
    "source_url": "https://example.com/doc",
    "source_type": "documentation",
    "metadata": {
        "author": "John Doe",
        "category": "tutorial"
    },
    "use_contextual_enrichment": true
}
```

### 2. Querying

```python
# Example: Query the RAG system
POST /api/rag/query
{
    "query": "How do I implement error handling in Python?",
    "project_id": null,
    "filters": {
        "category": "tutorial"
    },
    "top_k": 5,
    "use_hybrid": true
}
```

### 3. Health Check

```python
# Check system status
GET /api/rag/health

# Response:
{
    "status": "healthy",
    "timestamp": "2024-01-21T10:00:00Z",
    "implementation": "simple_rag",
    "services": {
        "rag_service": "ready",
        "embedding_service": "ready",
        "vector_store": "ready",
        "search_service": "ready"
    },
    "llamaindex_available": false
}
```

## How It Works

### Document Processing

1. **Chunking**: 
   - If LlamaIndex is available: Uses SentenceSplitter for intelligent chunking
   - Fallback: Simple fixed-size chunking with overlap

2. **Contextual Enrichment**:
   - Adds document context to each chunk
   - Includes metadata about the source document
   - Prepends a context summary to improve retrieval

3. **Storage**:
   - Generates embeddings using existing EmbeddingService
   - Stores in Weaviate using VectorStoreService
   - Maintains all metadata for filtering

### Query Processing

1. **Search**:
   - Uses existing hybrid search (vector + keyword)
   - Applies metadata filters if provided
   - Respects project context boundaries

2. **Response Generation**:
   - Formats search results as a coherent response
   - Includes source references
   - Returns metadata for transparency

## Testing

Run the test script to verify the implementation:

```bash
cd /opt/projects/knowledgehub
python test_simple_rag.py
```

## API Compatibility

The simplified implementation maintains full API compatibility with the original RAG endpoints:

- `POST /api/rag/ingest` - Ingest documents
- `POST /api/rag/query` - Query the system
- `GET /api/rag/index/stats` - Get index statistics
- `GET /api/rag/health` - Health check
- `POST /api/rag/test` - Test the pipeline (admin only)

## Performance Considerations

1. **Chunking**: Simple chunking is faster but may split sentences
2. **Embeddings**: Uses cached embeddings when possible
3. **Search**: Hybrid search provides good accuracy with reasonable speed
4. **Caching**: Query results are cached for 5 minutes

## Future Enhancements

When dependency conflicts are resolved, you can enable:

1. **Advanced Chunking**: Semantic splitting, hierarchical parsing
2. **Multi-stage Retrieval**: Re-ranking, diversity optimization
3. **Streaming Responses**: Real-time response generation
4. **Documentation Scraping**: Automatic ingestion from documentation sites
5. **LLM-based Enrichment**: Using Claude for contextual summaries

## Troubleshooting

### Common Issues

1. **Import Errors**: The system gracefully falls back to simple implementation
2. **Vector Store Errors**: Check Weaviate connectivity at `192.168.1.25:8090`
3. **Embedding Errors**: Ensure HuggingFace models are downloaded
4. **Memory Issues**: Reduce chunk size or batch processing

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

This simplified RAG implementation provides core RAG functionality while maintaining compatibility with your existing infrastructure. It's designed to work out-of-the-box without dependency conflicts, while still allowing for future enhancements when needed.