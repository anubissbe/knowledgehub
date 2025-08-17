
import redis
import json
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
from datetime import timedelta
import asyncio

class CachingSystem:
    """Comprehensive caching system with multiple strategies"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
            connection_pool=redis.ConnectionPool(
                max_connections=100,
                connection_class=redis.Connection
            )
        )
        self.local_cache = {}  # In-memory cache for hot data
    
    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback"""
        # Check local cache first
        if key in self.local_cache:
            return self.local_cache[key]
        
        # Check Redis
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set cache with TTL"""
        try:
            # Set in both caches
            self.local_cache[key] = value
            self.redis_client.setex(
                key,
                timedelta(seconds=ttl),
                json.dumps(value)
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def invalidate(self, pattern: str):
        """Invalidate cache by pattern"""
        # Clear local cache
        keys_to_delete = [k for k in self.local_cache if pattern in k]
        for key in keys_to_delete:
            del self.local_cache[key]
        
        # Clear Redis cache
        for key in self.redis_client.scan_iter(f"*{pattern}*"):
            self.redis_client.delete(key)

# Singleton instance
cache_system = CachingSystem()

def cached(ttl: int = 300, prefix: str = "api"):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_system.cache_key(prefix, func.__name__, *args, **kwargs)
            
            # Check cache
            cached_result = await cache_system.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_system.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Usage example:
# @cached(ttl=600, prefix="rag")
# async def search_documents(query: str):
#     # Expensive search operation
#     return results
