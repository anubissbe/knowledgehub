"""
Resilient Search Service with Circuit Breaker Protection

Provides a resilient interface to Weaviate vector search with automatic
fallback, caching, and performance optimization.
"""

import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import numpy as np

from .external_service_client import weaviate_client, redis_client
from .circuit_breaker import circuit_breaker, ServiceType, CircuitConfig
from ..models import Memory, Chunk
from ..config import settings

logger = logging.getLogger(__name__)


class ResilientSearchService:
    """
    Resilient search service with circuit breaker protection.
    
    Features:
    - Circuit breaker protection for Weaviate
    - Result caching with Redis fallback
    - Query optimization and batching
    - Graceful degradation to database search
    - Performance monitoring
    """
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes
        self.enable_cache = True
        self.enable_fallback = True
        self.batch_size = 100
        
        # Performance tracking
        self.search_count = 0
        self.cache_hits = 0
        self.fallback_count = 0
    
    async def search_memories(
        self,
        user_id: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search memories with resilient vector search.
        
        Attempts in order:
        1. Cache lookup
        2. Weaviate vector search
        3. Database fallback search
        """
        self.search_count += 1
        
        # Generate cache key
        cache_key = self._generate_cache_key(user_id, query_vector, limit, filters)
        
        # Try cache first
        if self.enable_cache:
            cached_results = await self._get_from_cache(cache_key)
            if cached_results:
                self.cache_hits += 1
                logger.debug(f"Cache hit for search query")
                return cached_results
        
        try:
            # Try Weaviate search
            results = await self._search_weaviate(user_id, query_vector, limit, filters)
            
            # Cache successful results
            if self.enable_cache and results:
                await self._save_to_cache(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            
            # Try fallback search
            if self.enable_fallback:
                self.fallback_count += 1
                return await self._fallback_search(user_id, limit, filters)
            
            # Return empty results if all fails
            return []
    
    @circuit_breaker(
        "weaviate_search",
        service_type=ServiceType.SEARCH,
        config=CircuitConfig(failure_threshold=5, timeout=60.0)
    )
    async def _search_weaviate(
        self,
        user_id: str,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute Weaviate vector search with circuit breaker"""
        
        # Build where filter
        where_filter = {
            "path": ["user_id"],
            "operator": "Equal",
            "valueString": user_id
        }
        
        # Add additional filters if provided
        if filters:
            where_filter = {
                "operator": "And",
                "operands": [where_filter, filters]
            }
        
        # Execute search
        response = await weaviate_client.post(
            "/graphql",
            json_data={
                "query": f"""
                {{
                    Get {{
                        Memory(
                            nearVector: {{
                                vector: {json.dumps(query_vector)}
                            }}
                            limit: {limit}
                            where: {json.dumps(where_filter)}
                        ) {{
                            id
                            content
                            memory_type
                            importance_score
                            created_at
                            metadata
                            _additional {{
                                distance
                                certainty
                            }}
                        }}
                    }}
                }}
                """
            }
        )
        
        # Parse results
        if "data" in response and "Get" in response["data"]:
            memories = response["data"]["Get"].get("Memory", [])
            
            # Format results
            results = []
            for memory in memories:
                result = {
                    **memory,
                    "similarity_score": memory["_additional"]["certainty"],
                    "distance": memory["_additional"]["distance"]
                }
                del result["_additional"]
                results.append(result)
            
            return results
        
        return []
    
    async def _fallback_search(
        self,
        user_id: str,
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback to database search when vector search unavailable"""
        logger.info("Using database fallback search")
        
        try:
            from ..services.resilient_database import resilient_db
            
            # Build query
            query = f"""
                SELECT id, content, memory_type, importance_score, 
                       created_at, metadata
                FROM memories
                WHERE user_id = :user_id
                ORDER BY importance_score DESC, created_at DESC
                LIMIT :limit
            """
            
            params = {"user_id": user_id, "limit": limit}
            
            # Execute query
            results = await resilient_db.execute_complex_query(query, params)
            
            # Add default similarity score for compatibility
            for result in results:
                result["similarity_score"] = 0.5  # Default score
                result["distance"] = 0.5
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    async def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get search results from cache"""
        try:
            cached = await redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        return None
    
    async def _save_to_cache(self, key: str, results: List[Dict[str, Any]]):
        """Save search results to cache"""
        try:
            # Convert datetime objects to strings
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                for field, value in serializable_result.items():
                    if isinstance(value, datetime):
                        serializable_result[field] = value.isoformat()
                serializable_results.append(serializable_result)
            
            await redis_client.set(
                key,
                json.dumps(serializable_results),
                ttl=self.cache_ttl
            )
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")
    
    def _generate_cache_key(
        self,
        user_id: str,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for search query"""
        # Create hash of query parameters
        key_data = {
            "user_id": user_id,
            "vector_hash": hashlib.md5(
                np.array(query_vector).tobytes()
            ).hexdigest()[:8],
            "limit": limit,
            "filters": filters
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"search:{key_hash}"
    
    async def index_memory(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Index memory in Weaviate with circuit breaker protection"""
        try:
            # Prepare object
            memory_object = {
                "id": memory_id,
                "content": content,
                "user_id": metadata.get("user_id"),
                "memory_type": metadata.get("memory_type", "general"),
                "importance_score": metadata.get("importance_score", 0.5),
                "created_at": metadata.get("created_at", datetime.now(timezone.utc).isoformat()),
                "metadata": json.dumps(metadata)
            }
            
            # Index with circuit breaker
            await self._index_in_weaviate(memory_object, embedding)
            return True
            
        except Exception as e:
            logger.error(f"Failed to index memory {memory_id}: {e}")
            return False
    
    @circuit_breaker(
        "weaviate_search",
        service_type=ServiceType.SEARCH,
        config=CircuitConfig(failure_threshold=5, timeout=60.0)
    )
    async def _index_in_weaviate(
        self,
        memory_object: Dict[str, Any],
        embedding: List[float]
    ):
        """Index object in Weaviate with circuit breaker"""
        
        # Create object with vector
        data_object = {
            "class": "Memory",
            "properties": memory_object,
            "vector": embedding
        }
        
        # Index object
        await weaviate_client.post("/objects", json_data=data_object)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from Weaviate with circuit breaker protection"""
        try:
            await self._delete_from_weaviate(memory_id)
            
            # Also clear from cache
            # Note: This is a simplified approach - in production you'd want
            # to invalidate all cache entries that might contain this memory
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    @circuit_breaker(
        "weaviate_search",
        service_type=ServiceType.SEARCH,
        config=CircuitConfig(failure_threshold=5, timeout=60.0)
    )
    async def _delete_from_weaviate(self, memory_id: str):
        """Delete object from Weaviate with circuit breaker"""
        await weaviate_client.delete(f"/objects/Memory/{memory_id}")
    
    async def batch_index_memories(
        self,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Batch index memories with circuit breaker protection"""
        results = {
            "total": len(memories),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        # Process in batches
        for i in range(0, len(memories), self.batch_size):
            batch = memories[i:i + self.batch_size]
            
            try:
                await self._batch_index_weaviate(batch)
                results["successful"] += len(batch)
            except Exception as e:
                logger.error(f"Batch indexing failed: {e}")
                results["failed"] += len(batch)
                results["errors"].append({
                    "batch_start": i,
                    "batch_size": len(batch),
                    "error": str(e)
                })
        
        return results
    
    @circuit_breaker(
        "weaviate_search",
        service_type=ServiceType.SEARCH,
        config=CircuitConfig(failure_threshold=5, timeout=60.0)
    )
    async def _batch_index_weaviate(self, batch: List[Dict[str, Any]]):
        """Batch index objects in Weaviate with circuit breaker"""
        
        # Prepare batch objects
        objects = []
        for item in batch:
            objects.append({
                "class": "Memory",
                "properties": {
                    "id": item["id"],
                    "content": item["content"],
                    "user_id": item["user_id"],
                    "memory_type": item.get("memory_type", "general"),
                    "importance_score": item.get("importance_score", 0.5),
                    "created_at": item.get("created_at", datetime.now(timezone.utc).isoformat()),
                    "metadata": json.dumps(item.get("metadata", {}))
                },
                "vector": item["embedding"]
            })
        
        # Batch index
        await weaviate_client.post("/batch/objects", json_data={"objects": objects})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search service statistics"""
        cache_hit_rate = (
            (self.cache_hits / self.search_count * 100)
            if self.search_count > 0 else 0
        )
        
        fallback_rate = (
            (self.fallback_count / self.search_count * 100)
            if self.search_count > 0 else 0
        )
        
        # Get circuit breaker status
        breaker = circuit_manager.get_circuit_breaker("weaviate_search")
        breaker_status = breaker.get_status() if breaker else None
        
        return {
            "search_count": self.search_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "fallback_count": self.fallback_count,
            "fallback_rate": fallback_rate,
            "cache_enabled": self.enable_cache,
            "fallback_enabled": self.enable_fallback,
            "circuit_breaker": breaker_status
        }


# Global resilient search service
resilient_search = ResilientSearchService()

from ..services.circuit_breaker import circuit_manager