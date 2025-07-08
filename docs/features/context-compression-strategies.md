# Context Compression Strategies

## Overview

The Context Compression system provides intelligent compression of memory sessions to manage context windows effectively while preserving the most important information. This is essential for maintaining conversation quality while staying within token limits for AI models.

## Architecture

### Core Components

1. **ContextCompressionService** (`src/api/memory_system/core/context_compression.py`)
   - Seven compression strategies with four intensity levels
   - Redis caching for performance optimization
   - Token estimation and compression ratio calculation
   - Graceful fallback and error handling

2. **Context Compression API** (`src/api/memory_system/api/routers/context_compression.py`)
   - `/compression/compress` - Main compression endpoint
   - `/compression/stats/{session_id}` - Compression statistics and recommendations
   - `/compression/compress/preview` - Preview compression results
   - `/compression/cache/{session_id}` - Cache management
   - `/compression/strategies` - Available strategies information

3. **Integration with Context Service** (`src/api/memory_system/services/context_service.py`)
   - Enhanced context retrieval with optional compression
   - Seamless integration with existing context workflows

## Compression Strategies

### 1. Importance-Based Compression

**Strategy**: `importance_based`

**How it works**: Selects memories based on importance scores with configurable thresholds.

**Best for**: Sessions with varying importance levels where critical information must be preserved.

**Characteristics**:
- Preserves high-importance memories (importance >= threshold)
- Summarizes remaining high-importance content
- Configurable importance thresholds per compression level

**Parameters**:
- Light: 70% threshold
- Moderate: 80% threshold  
- Aggressive: 85% threshold
- Extreme: 90% threshold

### 2. Recency-Weighted Compression

**Strategy**: `recency_weighted`

**How it works**: Prioritizes recent memories using exponential decay weighting combined with importance.

**Best for**: Sessions where recent context is most relevant (active development, debugging).

**Characteristics**:
- Exponential decay: `importance * (0.95 ^ hours_ago)`
- Balances recency with importance
- Extracts entities from older memories

**Selection Ratios**:
- Light: 80% of memories
- Moderate: 60% of memories
- Aggressive: 40% of memories
- Extreme: 20% of memories

### 3. Summarization Compression

**Strategy**: `summarization`

**How it works**: Groups memories by type and time windows, then creates summaries.

**Best for**: Sessions with repetitive or groupable content.

**Characteristics**:
- Groups by memory type (DECISION, ERROR, FACT, etc.)
- Preserves critical memory types (decisions, errors)
- Creates group summaries for less critical content
- Maintains structured information organization

### 4. Entity Consolidation Compression

**Strategy**: `entity_consolidation`

**How it works**: Consolidates information about entities to reduce redundancy.

**Best for**: Sessions with many references to the same entities (projects, technologies, people).

**Characteristics**:
- Groups memories by shared entities
- Keeps the most important memory per entity
- Creates entity summaries for consolidated information
- Reduces entity-based redundancy

### 5. Semantic Clustering Compression

**Strategy**: `semantic_clustering`

**How it works**: Groups semantically similar memories based on shared entities and content.

**Best for**: Sessions with thematic content that can be grouped.

**Characteristics**:
- Clusters based on shared entities (2+ common entities)
- Selects cluster representatives
- Creates cluster summaries
- Preserves thematic organization

### 6. Hierarchical Compression

**Strategy**: `hierarchical`

**How it works**: Uses importance tiers for systematic memory selection.

**Best for**: Sessions with clear importance hierarchy.

**Characteristics**:
- Critical tier: importance >= 0.9 (always preserved)
- Important tier: 0.7 <= importance < 0.9 (space permitting)
- Regular tier: importance < 0.7 (summarized)
- Systematic tier-based allocation

### 7. Hybrid Compression (Recommended)

**Strategy**: `hybrid`

**How it works**: Combines multiple strategies for optimal results.

**Best for**: Most sessions - provides balanced compression across different content types.

**Characteristics**:
- Starts with importance-based selection
- Applies recency weighting for remaining space
- Uses entity consolidation for summaries
- Generates comprehensive summaries
- Balanced approach optimizing for various scenarios

## Compression Levels

### Light Compression (10-20% reduction)
- **Use case**: Minimal compression, preserve most content
- **Token reduction**: 10-20%
- **Processing time**: 1-2 seconds
- **Best for**: Large context windows, high-quality preservation needed

### Moderate Compression (30-50% reduction)
- **Use case**: Balanced compression, good for most use cases
- **Token reduction**: 30-50%
- **Processing time**: 1-2 seconds
- **Best for**: Standard context management, balanced quality/efficiency

### Aggressive Compression (60-80% reduction)
- **Use case**: High compression, focus on essentials
- **Token reduction**: 60-80%
- **Processing time**: 2-5 seconds
- **Best for**: Limited context windows, efficiency prioritized

### Extreme Compression (80-90% reduction)
- **Use case**: Maximum compression, only critical information
- **Token reduction**: 80-90%
- **Processing time**: 2-5 seconds
- **Best for**: Very limited context, emergency compression

## API Usage

### Basic Compression

```python
import aiohttp

async def compress_session_context():
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://localhost:3000/api/memory/compression/compress",
            json={
                "session_id": "uuid-here",
                "target_tokens": 4000,
                "strategy": "hybrid",
                "level": "moderate"
            }
        )
        
        if response.status == 200:
            data = await response.json()
            print(f"Compressed from {data['original_memory_count']} to {data['compressed_memory_count']} memories")
            print(f"Compression ratio: {data['compression_ratio']:.2%}")
            print(f"Token estimate: {data['token_estimate']:,}")
            
            compressed_context = data['compressed_context']
            # Use compressed_context['memories'], compressed_context['summary'], etc.
```

### Get Compression Statistics

```python
async def get_compression_stats(session_id):
    async with aiohttp.ClientSession() as session:
        response = await session.get(
            f"http://localhost:3000/api/memory/compression/stats/{session_id}"
        )
        
        if response.status == 200:
            data = await response.json()
            print(f"Original tokens: {data['original_tokens']:,}")
            print("Recommendations:")
            for rec in data['recommendations']:
                print(f"  - {rec}")
            
            # Review compression estimates
            for level, estimates in data['compression_estimates'].items():
                print(f"{level}: {estimates['estimated_tokens']:,} tokens "
                      f"({estimates['compression_ratio']:.0%} reduction)")
```

### Preview Compression

```python
async def preview_compression(session_id):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "http://localhost:3000/api/memory/compression/compress/preview",
            json={
                "session_id": session_id,
                "target_tokens": 3000,
                "strategy": "hybrid",
                "level": "moderate"
            }
        )
        
        if response.status == 200:
            data = await response.json()
            print(f"Preview: {data['compression_ratio']:.2%} reduction")
            print(f"Estimated tokens: {data['token_estimate']:,}")
            print(f"Top entities: {', '.join(data['key_entities'])}")
```

### Integration with Context Service

```python
from sqlalchemy.orm import Session
from memory_system.services.context_service import context_service

async def get_compressed_context(db: Session, session_id: UUID):
    # Option 1: Direct compression service
    compressed = await context_service.get_compressed_context(
        db=db,
        session_id=session_id,
        target_tokens=4000,
        strategy="hybrid",
        level="moderate"
    )
    
    # Option 2: Enhanced context retrieval with compression
    context = await context_service.get_context_for_session(
        db=db,
        session_id=session_id,
        use_compression=True,
        target_tokens=4000
    )
    
    return compressed, context
```

## Performance Characteristics

### Caching Strategy

- **Cache TTL**: 30 minutes (1800 seconds)
- **Cache Key**: MD5 hash of session_id + target_tokens + strategy + level
- **Cache Benefits**: 
  - 90%+ faster retrieval for repeated requests
  - Reduces computational overhead
  - Improves API response times

### Processing Times

| Strategy | Light | Moderate | Aggressive | Extreme |
|----------|-------|----------|------------|---------|
| Importance-based | 0.5-1s | 0.5-1s | 0.8-1.5s | 1-2s |
| Recency-weighted | 0.6-1.2s | 0.8-1.5s | 1-2s | 1.5-3s |
| Summarization | 1-2s | 1.5-2.5s | 2-3s | 2.5-4s |
| Entity consolidation | 0.8-1.5s | 1-2s | 1.5-2.5s | 2-3s |
| Semantic clustering | 1-2s | 1.5-2.5s | 2-3s | 2.5-4s |
| Hierarchical | 0.5-1s | 0.7-1.2s | 1-1.8s | 1.2-2.5s |
| Hybrid | 1.5-2.5s | 2-3s | 2.5-4s | 3-5s |

### Memory Requirements

- **Base memory**: ~50MB for service initialization
- **Per session**: ~1-10MB depending on memory count
- **Peak usage**: ~100MB during large session compression
- **Garbage collection**: Automatic cleanup after compression

## Configuration

### Service Parameters

```python
class ContextCompressionService:
    def __init__(self):
        self.max_context_tokens = 8000      # Conservative limit
        self.token_per_char_ratio = 0.25    # ~4 chars per token
        self.cache_ttl = 1800              # 30 minutes cache
```

### Compression Thresholds

```python
# Importance-based thresholds
IMPORTANCE_THRESHOLDS = {
    CompressionLevel.LIGHT: 0.7,
    CompressionLevel.MODERATE: 0.8,
    CompressionLevel.AGGRESSIVE: 0.85,
    CompressionLevel.EXTREME: 0.9
}

# Recency selection ratios
RECENCY_SELECTION_RATIOS = {
    CompressionLevel.LIGHT: 0.8,
    CompressionLevel.MODERATE: 0.6,
    CompressionLevel.AGGRESSIVE: 0.4,
    CompressionLevel.EXTREME: 0.2
}
```

## Monitoring & Analytics

### Metrics Tracked

- Compression ratios by strategy and level
- Processing times per strategy
- Cache hit rates
- Token estimation accuracy
- User strategy preferences
- Session size distributions

### Logging

```python
# Compression completion
logger.info(
    f"Compressed context for session {session_id}: "
    f"{len(memories)} -> {len(compressed.memories)} memories, "
    f"ratio: {compressed.compression_ratio:.2f}, "
    f"tokens: {compressed.token_estimate}"
)

# Performance monitoring
logger.debug(f"Compression took {processing_time:.2f}s for {strategy} strategy")

# Cache operations
logger.debug(f"Cache {'hit' if cached else 'miss'} for compression key {cache_key}")
```

## Error Handling

### Common Errors

1. **Session Not Found**: Invalid session ID
   - **Response**: 404 with descriptive error
   - **Recovery**: Validate session exists before compression

2. **No Memories Found**: Empty session
   - **Response**: Empty CompressedContext
   - **Recovery**: Graceful degradation, return empty result

3. **Cache Unavailable**: Redis connection issues
   - **Response**: Warning logged, compression proceeds without cache
   - **Recovery**: Automatic fallback to non-cached operation

4. **Compression Failure**: Strategy-specific errors
   - **Response**: 500 with error details
   - **Recovery**: Retry with simpler strategy (importance-based)

### Resilience Features

- **Graceful degradation**: Continues without cache if Redis unavailable
- **Fallback strategies**: Automatic fallback to simpler compression
- **Input validation**: Comprehensive parameter validation
- **Transaction safety**: Database operations are atomic
- **Memory limits**: Built-in protection against excessive memory usage

## Best Practices

### Strategy Selection

1. **Start with hybrid**: Best overall performance for most use cases
2. **Use importance-based**: When critical information must be preserved
3. **Use recency-weighted**: For active development sessions
4. **Use entity consolidation**: For entity-heavy technical discussions
5. **Use summarization**: For sessions with repetitive content

### Level Selection

1. **Light compression**: When context quality is paramount
2. **Moderate compression**: For balanced performance (recommended)
3. **Aggressive compression**: When token limits are strict
4. **Extreme compression**: Only for emergency compression needs

### Performance Optimization

1. **Cache warming**: Pre-compress frequently accessed sessions
2. **Batch operations**: Compress multiple sessions together
3. **Asynchronous processing**: Use background tasks for large compressions
4. **Monitoring**: Track compression metrics for optimization

## Testing

### Comprehensive Test Suite

Run the full test suite:

```bash
python3 test_context_compression.py
```

Tests include:
- All compression strategies and levels
- Performance benchmarks
- Cache management
- Error handling
- Integration testing
- Memory usage analysis

### Test Coverage

- **Strategy testing**: All 7 strategies with 4 levels each
- **Performance testing**: Processing time measurements
- **Cache testing**: Hit/miss ratios and invalidation
- **Integration testing**: Context service integration
- **Error testing**: Graceful failure handling
- **Scale testing**: Large session compression

## Future Enhancements

1. **Machine Learning Integration**: AI-powered strategy selection
2. **Custom Compression Rules**: User-defined compression preferences
3. **Real-time Compression**: Live compression during session
4. **Compression Analytics**: Advanced metrics and insights
5. **Multi-session Compression**: Cross-session context compression
6. **Adaptive Thresholds**: Dynamic threshold adjustment based on usage patterns

## Integration Examples

### Claude-Code Middleware Integration

```python
from memory_system.core.context_compression import context_compression_service

class ContextCompressionMiddleware:
    async def __call__(self, request: Request, call_next):
        # Auto-compress context if session is large
        session_id = request.headers.get("X-Claude-Session-ID")
        if session_id:
            db = next(get_db())
            
            # Check if compression needed
            memory_count = db.query(Memory).filter_by(session_id=session_id).count()
            
            if memory_count > 100:
                # Auto-compress with hybrid strategy
                compressed = await context_compression_service.compress_context(
                    db=db,
                    session_id=UUID(session_id),
                    target_tokens=4000,
                    strategy=CompressionStrategy.HYBRID,
                    level=CompressionLevel.MODERATE
                )
                
                # Inject compressed context into request
                request.state.compressed_context = compressed
        
        return await call_next(request)
```

### Scheduled Compression

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

async def scheduled_compression_job():
    """Compress old sessions to save storage"""
    db = next(get_db())
    
    # Find sessions older than 1 week with high memory count
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    large_sessions = db.query(MemorySession).join(Memory).group_by(
        MemorySession.id
    ).having(
        func.count(Memory.id) > 200
    ).filter(
        MemorySession.started_at < cutoff
    ).all()
    
    for session in large_sessions:
        compressed = await context_compression_service.compress_context(
            db=db,
            session_id=session.id,
            target_tokens=2000,
            strategy=CompressionStrategy.HYBRID,
            level=CompressionLevel.AGGRESSIVE
        )
        
        # Store compressed representation
        # Implementation depends on storage strategy

# Schedule the job
scheduler = AsyncIOScheduler()
scheduler.add_job(
    scheduled_compression_job,
    trigger="cron",
    hour=2,  # Run at 2 AM
    minute=0
)
```