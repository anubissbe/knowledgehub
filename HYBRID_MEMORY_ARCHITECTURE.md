# KnowledgeHub Hybrid Memory Architecture

## Overview
Transform KnowledgeHub into a hybrid memory system that combines Nova Memory's local efficiency with KnowledgeHub's distributed intelligence.

## Architecture Design

### 1. Three-Tier Memory System

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude/AI Interface                       │
├─────────────────────────────────────────────────────────────┤
│                  Memory Router & Optimizer                   │
├──────────────────────┬─────────────────┬───────────────────┤
│   L1: Hot Memory     │  L2: Warm Memory │  L3: Cold Memory │
│   (Nova SQLite)      │  (Redis Cache)   │  (PostgreSQL)    │
│   < 100ms            │  100-500ms       │  > 500ms         │
│   Current context    │  Recent sessions │  Historical data │
│   Active projects    │  Frequent queries│  Archive         │
└──────────────────────┴─────────────────┴───────────────────┘
```

### 2. Key Components

#### A. Local Memory Layer (Nova-inspired)
- **SQLite Database**: Fast local storage
- **Full-text Search**: Instant recall
- **Token Optimizer**: Compress frequent contexts
- **Workflow Tracker**: Project phases
- **Board System**: Task management

#### B. Distributed Intelligence Layer (KnowledgeHub)
- **PostgreSQL**: Persistent storage
- **Weaviate**: Vector search
- **Neo4j**: Knowledge graphs
- **TimescaleDB**: Analytics
- **Document Store**: Scraped content

#### C. Intelligent Sync Layer
- **Smart Caching**: Predictive loading
- **Conflict Resolution**: Version management
- **Compression**: Reduce transfer size
- **Priority Queue**: Important memories first

### 3. Implementation Plan

#### Phase 1: Local Memory Foundation
```python
# New local memory service
class HybridMemoryService:
    def __init__(self):
        self.local_db = SQLiteMemory()  # Nova-style
        self.redis = RedisCache()        # Fast distributed
        self.remote = KnowledgeHubAPI()  # Full features
        
    async def store(self, memory):
        # Store locally first (instant)
        local_id = await self.local_db.save(memory)
        
        # Queue for distributed storage
        await self.redis.queue_sync(local_id)
        
        # Return immediately
        return local_id
        
    async def recall(self, query):
        # Try local first
        if result := await self.local_db.search(query):
            return result
            
        # Then Redis cache
        if result := await self.redis.get(query):
            await self.local_db.cache(result)
            return result
            
        # Finally remote
        result = await self.remote.search(query)
        await self.cache_cascade(result)
        return result
```

#### Phase 2: Token Optimization
```python
class TokenOptimizer:
    def compress_context(self, memories):
        # Remove redundancy
        # Summarize verbose content
        # Keep only relevant fields
        # Use references instead of full text
        
    def estimate_savings(self, context):
        original = count_tokens(context)
        optimized = count_tokens(self.compress(context))
        return (1 - optimized/original) * 100
```

#### Phase 3: Workflow Integration
```python
class WorkflowMemory:
    def track_phase(self, project_id, phase):
        # Nova-style workflow tracking
        
    def get_board_state(self, project_id):
        # Task management board
        
    def map_relationships(self, entities):
        # Knowledge graph builder
```

### 4. Benefits of Hybrid Approach

1. **Speed**: 10-100x faster for common queries
2. **Efficiency**: 50-80% token reduction
3. **Reliability**: Works offline
4. **Scalability**: Distributed when needed
5. **Intelligence**: Full AI features available

### 5. Migration Strategy

1. **Add SQLite layer** alongside existing system
2. **Implement caching logic** gradually
3. **Monitor performance** metrics
4. **Tune cache policies** based on usage
5. **Expand local features** incrementally

### 6. MCP Tool Updates

```yaml
tools:
  # Fast local operations
  - quick_store: Instant memory save
  - quick_recall: Sub-second retrieval
  - context_optimize: Token reduction
  
  # Workflow management  
  - workflow_track: Project phases
  - board_update: Task management
  - relate_entities: Graph building
  
  # Hybrid operations
  - sync_status: Check sync queue
  - cache_stats: Performance metrics
  - priority_set: Important memories
```

### 7. Configuration

```yaml
hybrid_memory:
  local:
    path: ~/.knowledgehub/memory.db
    max_size: 10GB
    index_type: fts5
    compression: zstd
    
  cache:
    redis_url: redis://localhost:6381
    ttl: 3600
    max_memory: 2GB
    
  remote:
    api_url: http://localhost:3000
    sync_interval: 300
    batch_size: 100
    
  optimization:
    token_limit: 4000
    compression_threshold: 0.7
    summary_length: 200
```

### 8. Performance Targets

- Local recall: < 100ms
- Cache hit rate: > 80%
- Token savings: > 60%
- Sync latency: < 5s
- Offline capability: 100%

## Next Steps

1. Implement SQLite memory layer
2. Add token optimization
3. Build sync mechanism
4. Update MCP tools
5. Test performance gains