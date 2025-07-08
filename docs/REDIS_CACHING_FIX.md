# Redis Caching Fix for Memory System

## Issue
The session manager was trying to use `redis_client.setex()` directly, but the RedisCache wrapper class doesn't expose this method. It provides a `set()` method instead.

## Solution

### 1. Updated `_cache_session` method in session_manager.py:
```python
async def _cache_session(self, session: MemorySession):
    """Cache session in Redis"""
    try:
        key = f"session:{session.id}"
        # Convert session to dict for caching
        session_data = {
            'id': str(session.id),
            'user_id': session.user_id,
            'project_id': str(session.project_id) if session.project_id else None,
            'parent_session_id': str(session.parent_session_id) if session.parent_session_id else None,
            'started_at': session.started_at.isoformat() if session.started_at else None,
            'ended_at': session.ended_at.isoformat() if session.ended_at else None,
            'session_metadata': session.session_metadata,
            'tags': session.tags,
            'created_at': session.created_at.isoformat() if session.created_at else None,
            'updated_at': session.updated_at.isoformat() if session.updated_at else None
        }
        await redis_client.set(key, session_data, self._cache_ttl)
    except Exception as e:
        logger.warning(f"Failed to cache session: {e}")
```

### 2. Updated `_get_cached_session` method:
Currently returns None to force database queries. This is a safe approach until full cache reconstruction is implemented.

## Implementation Details

The RedisCache wrapper in `/opt/projects/knowledgehub/src/api/services/cache.py`:
- Provides `set(key, value, expiry)` method
- Automatically handles JSON serialization/deserialization
- Uses redis.asyncio for async operations

## Testing Status
- Code changes implemented
- Infrastructure issue (PostgreSQL permission errors) prevents full testing
- Redis caching logic is correct based on the API design

## Next Steps
1. Fix PostgreSQL permission issues
2. Test Redis caching with actual sessions
3. Implement full cache reconstruction in `_get_cached_session`
4. Add cache hit/miss metrics for monitoring