"""
Advanced Cache Manager

Provides multi-tier caching with intelligent cache invalidation,
memory-based caching for hot data, and Redis-based distributed caching.
"""

import asyncio
import hashlib
import json
import time
import weakref
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache strategy options"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on access patterns
    WRITE_THROUGH = "write_through" # Write to cache and storage
    WRITE_BACK = "write_back"      # Write to cache, batch to storage


class CacheTier(Enum):
    """Cache tier levels"""
    MEMORY = "memory"              # In-memory cache (fastest)
    REDIS = "redis"                # Redis cache (fast, distributed)
    DATABASE = "database"          # Database cache (persistent)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    tags: List[str] = None
    size: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """Age of entry in seconds"""
        return time.time() - self.created_at
    
    def touch(self):
        """Update access time and count"""
        self.accessed_at = time.time()
        self.access_count += 1


class MemoryCache:
    """High-performance in-memory cache with multiple strategies"""
    
    def __init__(self, 
                 max_size: int = 10000,
                 max_memory_mb: int = 500,
                 strategy: CacheStrategy = CacheStrategy.LRU,
                 default_ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.default_ttl = default_ttl
        
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.frequency_map: Dict[str, int] = {}  # For LFU
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0,
            'total_requests': 0
        }
        
        # Cleanup task
        self._cleanup_task = None
        
    async def start(self):
        """Start cache maintenance tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop cache maintenance tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.stats['total_requests'] += 1
        
        if key not in self.entries:
            self.stats['misses'] += 1
            return None
        
        entry = self.entries[key]
        
        # Check expiration
        if entry.is_expired:
            await self.delete(key)
            self.stats['misses'] += 1
            return None
        
        # Update access patterns
        entry.touch()
        self._update_access_pattern(key)
        
        self.stats['hits'] += 1
        return entry.value
    
    async def set(self, 
                  key: str, 
                  value: Any, 
                  ttl: Optional[float] = None,
                  tags: List[str] = None) -> bool:
        """Set value in cache"""
        
        # Calculate size
        size = self._calculate_size(value)
        
        # Check if we need to evict
        await self._ensure_capacity(size)
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            ttl=ttl or self.default_ttl,
            tags=tags or [],
            size=size
        )
        
        # Remove old entry if exists
        if key in self.entries:
            old_entry = self.entries[key]
            self.stats['memory_usage'] -= old_entry.size
        
        # Add new entry
        self.entries[key] = entry
        self.stats['memory_usage'] += size
        
        # Update access patterns
        self._update_access_pattern(key)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key not in self.entries:
            return False
        
        entry = self.entries[key]
        self.stats['memory_usage'] -= entry.size
        
        del self.entries[key]
        
        # Clean up access patterns
        if key in self.access_order:
            self.access_order.remove(key)
        if key in self.frequency_map:
            del self.frequency_map[key]
        
        return True
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all entries with any of the specified tags"""
        deleted_count = 0
        keys_to_delete = []
        
        for key, entry in self.entries.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            if await self.delete(key):
                deleted_count += 1
        
        return deleted_count
    
    async def clear(self):
        """Clear all entries"""
        self.entries.clear()
        self.access_order.clear()
        self.frequency_map.clear()
        self.stats['memory_usage'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0
        if self.stats['total_requests'] > 0:
            hit_rate = self.stats['hits'] / self.stats['total_requests']
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'entry_count': len(self.entries),
            'memory_usage_mb': self.stats['memory_usage'] / (1024 * 1024),
            'average_entry_size': self.stats['memory_usage'] / max(len(self.entries), 1)
        }
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, dict)):
                return len(json.dumps(value, default=str))
            else:
                return len(str(value))
        except:
            return 1024  # Default estimate
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for cache strategy"""
        if self.strategy == CacheStrategy.LRU:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        elif self.strategy == CacheStrategy.LFU:
            self.frequency_map[key] = self.frequency_map.get(key, 0) + 1
    
    async def _ensure_capacity(self, new_size: int):
        """Ensure cache has capacity for new entry"""
        # Check memory limit
        while (self.stats['memory_usage'] + new_size > self.max_memory_bytes and 
               len(self.entries) > 0):
            await self._evict_one()
        
        # Check size limit
        while len(self.entries) >= self.max_size and len(self.entries) > 0:
            await self._evict_one()
    
    async def _evict_one(self):
        """Evict one entry based on strategy"""
        if not self.entries:
            return
        
        key_to_evict = None
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            key_to_evict = self.access_order[0] if self.access_order else next(iter(self.entries))
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            if self.frequency_map:
                key_to_evict = min(self.frequency_map, key=self.frequency_map.get)
            else:
                key_to_evict = next(iter(self.entries))
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict oldest entry
            oldest_key = None
            oldest_time = float('inf')
            for key, entry in self.entries.items():
                if entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    oldest_key = key
            key_to_evict = oldest_key
        
        else:  # Adaptive or fallback
            # Use LRU as fallback
            key_to_evict = self.access_order[0] if self.access_order else next(iter(self.entries))
        
        if key_to_evict:
            await self.delete(key_to_evict)
            self.stats['evictions'] += 1
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = []
        current_time = time.time()
        
        for key, entry in self.entries.items():
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)


class AdvancedCacheManager:
    """Multi-tier cache manager with intelligent routing"""
    
    def __init__(self, redis_client=None):
        self.memory_cache = MemoryCache(
            max_size=10000,
            max_memory_mb=500,
            strategy=CacheStrategy.LRU,
            default_ttl=3600
        )
        self.redis_client = redis_client
        
        # Cache patterns for different data types
        self.cache_patterns = {
            'search_results': {'ttl': 1800, 'tier': CacheTier.MEMORY, 'tags': ['search']},
            'embeddings': {'ttl': 86400, 'tier': CacheTier.REDIS, 'tags': ['embeddings']},
            'user_sessions': {'ttl': 3600, 'tier': CacheTier.MEMORY, 'tags': ['sessions']},
            'documents': {'ttl': 7200, 'tier': CacheTier.REDIS, 'tags': ['documents']},
            'analytics': {'ttl': 300, 'tier': CacheTier.MEMORY, 'tags': ['analytics']},
            'system_config': {'ttl': 86400, 'tier': CacheTier.REDIS, 'tags': ['config']}
        }
        
    async def start(self):
        """Start cache manager"""
        await self.memory_cache.start()
        
    async def stop(self):
        """Stop cache manager"""
        await self.memory_cache.stop()
    
    async def get(self, key: str, pattern: str = 'default') -> Optional[Any]:
        """Get value from appropriate cache tier"""
        
        # Try memory cache first (fastest)
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache if available
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value is not None:
                    # Promote to memory cache for hot data
                    config = self.cache_patterns.get(pattern, {})
                    if config.get('tier') == CacheTier.MEMORY:
                        await self.memory_cache.set(
                            key, value, 
                            ttl=config.get('ttl'),
                            tags=config.get('tags', [])
                        )
                    return value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    async def set(self, 
                  key: str, 
                  value: Any, 
                  pattern: str = 'default',
                  ttl: Optional[float] = None,
                  tags: List[str] = None) -> bool:
        """Set value in appropriate cache tier"""
        
        config = self.cache_patterns.get(pattern, {})
        effective_ttl = ttl or config.get('ttl', 3600)
        effective_tags = tags or config.get('tags', [])
        tier = config.get('tier', CacheTier.MEMORY)
        
        success = True
        
        # Always try memory cache for frequently accessed data
        if tier == CacheTier.MEMORY or pattern in ['search_results', 'user_sessions', 'analytics']:
            success &= await self.memory_cache.set(key, value, effective_ttl, effective_tags)
        
        # Redis cache for distributed/persistent data
        if self.redis_client and (tier == CacheTier.REDIS or pattern in ['embeddings', 'documents']):
            try:
                await self.redis_client.set(key, value, effective_ttl)
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
                success = False
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache tiers"""
        success = True
        
        # Delete from memory
        success &= await self.memory_cache.delete(key)
        
        # Delete from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis cache delete error: {e}")
                success = False
        
        return success
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all cache entries for a pattern"""
        config = self.cache_patterns.get(pattern, {})
        tags = config.get('tags', [])
        
        if tags:
            await self.memory_cache.delete_by_tags(tags)
            
            # Redis pattern deletion would need specific implementation
            # based on your Redis setup
    
    async def clear_all(self):
        """Clear all caches"""
        await self.memory_cache.clear()
        
        if self.redis_client:
            try:
                # Be careful with this in production
                pass  # Implement Redis clear if needed
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        
        stats = {
            'memory_cache': memory_stats,
            'redis_cache': {'status': 'available' if self.redis_client else 'unavailable'},
            'patterns': self.cache_patterns
        }
        
        return stats


def cache_result(pattern: str = 'default', 
                ttl: Optional[float] = None,
                key_func: Optional[Callable] = None):
    """Decorator to cache function results"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Simple key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            if hasattr(wrapper, '_cache_manager'):
                cached_result = await wrapper._cache_manager.get(cache_key, pattern)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if hasattr(wrapper, '_cache_manager') and result is not None:
                await wrapper._cache_manager.set(cache_key, result, pattern, ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
cache_manager: Optional[AdvancedCacheManager] = None


def get_cache_manager() -> AdvancedCacheManager:
    """Get or create global cache manager"""
    global cache_manager
    if cache_manager is None:
        # Import here to avoid circular imports
        try:
            from ..services.cache import redis_client
            cache_manager = AdvancedCacheManager(redis_client)
        except ImportError:
            cache_manager = AdvancedCacheManager()
    return cache_manager


async def initialize_cache_manager():
    """Initialize the global cache manager"""
    global cache_manager
    cache_manager = get_cache_manager()
    await cache_manager.start()


async def shutdown_cache_manager():
    """Shutdown the global cache manager"""
    global cache_manager
    if cache_manager:
        await cache_manager.stop()