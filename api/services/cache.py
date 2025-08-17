"""Redis cache service"""

import redis.asyncio as redis
import json
from typing import Optional, Any
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache service wrapper"""
    
    def __init__(self, url: str):
        self.url = url
        self.client: Optional[redis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.REDIS_MAX_CONNECTIONS
            )
            # Test connection
            if self.client is None:
                raise Exception("Redis client not initialized")
            await self.client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def ping(self) -> bool:
        """Check if Redis is responsive"""
        try:
            if self.client:
                return await self.client.ping()
            return False
        except:
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.client is None:
                return None
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, expiry: int = 3600):
        """Set value in cache with expiry (seconds)"""
        try:
            if self.client is None:
                return
            json_value = json.dumps(value)
            await self.client.setex(key, expiry, json_value)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            if self.client is None:
                return
            await self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    async def keys(self, pattern: str) -> list[str]:
        """Get all keys matching the pattern"""
        try:
            if self.client is None:
                return []
            keys = await self.client.keys(pattern)
            # Redis returns bytes for keys when decode_responses=True is set, 
            # but we've set it, so keys should be strings already
            return keys
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()


# Global Redis client instance
redis_client = RedisCache(settings.REDIS_URL)


async def get_cache_service() -> RedisCache:
    """Get the cache service instance"""
    if not redis_client.client:
        await redis_client.initialize()
    return redis_client

"""Fixed cache service module."""
from typing import Any, Optional
import json

class CacheService:
    """Simple cache service implementation."""
    
    def __init__(self):
        self.cache = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache."""
        self.cache[key] = value
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
    
    async def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()

# Create global instance
cache_service = CacheService()

def get_cache_service():
    """Get cache service instance."""
    return cache_service
