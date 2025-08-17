"""
RAG Performance Monitoring and Cache Management API
Provides endpoints for monitoring and managing the RAG cache optimizer

Author: Adrien Stevens - Python Performance Optimization Expert
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import numpy as np

from ..services.rag_cache_optimizer import get_rag_cache_optimizer, UnifiedMemoryConfig
from ..services.cache import get_cache_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag/performance", tags=["RAG Performance"])


@router.get("/stats")
async def get_performance_stats():
    """Get comprehensive RAG performance statistics"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        stats = await cache_optimizer.get_performance_stats()
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Performance stats error: {str(e)}")


@router.get("/cache/metrics")
async def get_cache_metrics():
    """Get detailed cache performance metrics"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        
        # Get basic metrics
        metrics = cache_optimizer.metrics
        
        # Get L1 cache stats
        l1_stats = {
            'size': len(cache_optimizer.l1_cache.cache),
            'capacity': cache_optimizer.l1_cache.capacity,
            'size_bytes': cache_optimizer.l1_cache.size_bytes,
            'compressed_keys': len(cache_optimizer.l1_cache.compressed_keys)
        }
        
        # Get Redis stats
        redis_client = await get_cache_service()
        redis_info = {}
        try:
            # Basic Redis stats (if available)
            if redis_client.client:
                redis_info = {
                    'connected': await redis_client.ping(),
                    'url': redis_client.url
                }
        except Exception as e:
            redis_info = {'connected': False, 'error': str(e)}
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "cache_metrics": {
                "hit_ratio": metrics.hit_ratio,
                "hit_count": metrics.hit_count,
                "miss_count": metrics.miss_count,
                "eviction_count": metrics.eviction_count,
                "background_writes": metrics.background_writes,
                "memory_usage_mb": metrics.memory_usage_mb
            },
            "l1_cache": l1_stats,
            "l2_cache_redis": redis_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Cache metrics error: {str(e)}")


@router.get("/memory/pools")
async def get_memory_pool_stats():
    """Get memory pool usage statistics"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        
        pool_stats = {}
        for pool_name, pool in cache_optimizer.memory_pools.items():
            pool_stats[pool_name] = pool.get_usage_stats()
        
        # Add system memory info
        import psutil
        memory = psutil.virtual_memory()
        system_memory = {
            'total_mb': memory.total / 1024 / 1024,
            'available_mb': memory.available / 1024 / 1024,
            'used_mb': memory.used / 1024 / 1024,
            'percent': memory.percent
        }
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "memory_pools": pool_stats,
            "system_memory": system_memory
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory pool stats: {e}")
        raise HTTPException(status_code=500, detail=f"Memory pool stats error: {str(e)}")


@router.get("/performance/latency")
async def get_latency_metrics():
    """Get detailed latency metrics for different operations"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        
        latency_stats = {}
        for operation, times in cache_optimizer.operation_times.items():
            if times:
                times_array = np.array(times)
                latency_stats[operation] = {
                    'count': len(times),
                    'avg_ms': float(np.mean(times_array)),
                    'min_ms': float(np.min(times_array)),
                    'max_ms': float(np.max(times_array)),
                    'p50_ms': float(np.percentile(times_array, 50)),
                    'p95_ms': float(np.percentile(times_array, 95)),
                    'p99_ms': float(np.percentile(times_array, 99)),
                    'std_ms': float(np.std(times_array))
                }
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "latency_metrics": latency_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get latency metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Latency metrics error: {str(e)}")


@router.get("/memory/history")
async def get_memory_history():
    """Get memory usage history"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        
        # Convert history to readable format
        history_data = []
        for timestamp, usage_percent in cache_optimizer.memory_usage_history:
            history_data.append({
                'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                'memory_usage_percent': usage_percent
            })
        
        # Sort by timestamp
        history_data.sort(key=lambda x: x['timestamp'])
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "memory_history": history_data,
            "data_points": len(history_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory history: {e}")
        raise HTTPException(status_code=500, detail=f"Memory history error: {str(e)}")


@router.post("/cache/invalidate")
async def invalidate_cache(pattern: Optional[str] = None):
    """Invalidate cache entries, optionally matching a pattern"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        
        await cache_optimizer.invalidate_cache(pattern)
        
        return {
            "status": "success",
            "message": f"Cache invalidated" + (f" (pattern: {pattern})" if pattern else " (all entries)"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache invalidation error: {str(e)}")


@router.post("/memory/optimize")
async def optimize_memory():
    """Trigger manual memory optimization"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        
        # Trigger memory pressure handling
        await cache_optimizer._handle_memory_pressure()
        
        return {
            "status": "success",
            "message": "Memory optimization triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize memory: {e}")
        raise HTTPException(status_code=500, detail=f"Memory optimization error: {str(e)}")


@router.get("/config")
async def get_configuration():
    """Get current RAG cache optimizer configuration"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        config = cache_optimizer.config
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "configuration": {
                "memory_pools": {
                    "cpu_memory_pool_mb": config.cpu_memory_pool_mb,
                    "gpu_memory_pool_mb": config.gpu_memory_pool_mb,
                    "shared_memory_pool_mb": config.shared_memory_pool_mb
                },
                "cache_layers": {
                    "l1_cache_mb": config.l1_cache_mb,
                    "l2_cache_mb": config.l2_cache_mb,
                    "l3_cache_mb": config.l3_cache_mb
                },
                "thresholds": {
                    "memory_pressure_threshold": config.memory_pressure_threshold,
                    "gc_trigger_threshold": config.gc_trigger_threshold,
                    "compression_threshold": config.compression_threshold
                },
                "concurrency": {
                    "max_concurrent_operations": config.max_concurrent_operations,
                    "background_writer_threads": config.background_writer_threads,
                    "prefetch_worker_threads": config.prefetch_worker_threads
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")


@router.get("/cache/keys")
async def get_cache_keys(
    pattern: Optional[str] = Query(None, description="Pattern to match keys"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of keys to return")
):
    """Get cache keys matching optional pattern"""
    try:
        redis_client = await get_cache_service()
        
        if pattern:
            keys = await redis_client.keys(f"*{pattern}*")
        else:
            # Get keys for each cache type
            emb_keys = await redis_client.keys("emb:*")
            qry_keys = await redis_client.keys("qry:*")
            neo4j_keys = await redis_client.keys("neo4j:*")
            keys = emb_keys + qry_keys + neo4j_keys
        
        # Limit results
        keys = keys[:limit]
        
        # Categorize keys
        categorized_keys = {
            'embeddings': [k for k in keys if k.startswith('emb:')],
            'queries': [k for k in keys if k.startswith('qry:')],
            'neo4j': [k for k in keys if k.startswith('neo4j:')],
            'other': [k for k in keys if not any(k.startswith(p) for p in ['emb:', 'qry:', 'neo4j:'])]
        }
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "total_keys": len(keys),
            "pattern": pattern,
            "keys_by_type": categorized_keys
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache keys: {e}")
        raise HTTPException(status_code=500, detail=f"Cache keys error: {str(e)}")


@router.post("/benchmark/cache")
async def benchmark_cache_operations(
    background_tasks: BackgroundTasks,
    num_operations: int = Query(100, ge=10, le=10000, description="Number of operations to benchmark")
):
    """Run cache performance benchmark"""
    try:
        # Start benchmark in background
        background_tasks.add_task(run_cache_benchmark, num_operations)
        
        return {
            "status": "success",
            "message": f"Cache benchmark started with {num_operations} operations",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start cache benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")


async def run_cache_benchmark(num_operations: int):
    """Run cache performance benchmark"""
    logger.info(f"Starting cache benchmark with {num_operations} operations")
    
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        
        # Benchmark embedding caching
        embed_times = []
        retrieve_times = []
        
        for i in range(num_operations):
            # Create test embedding
            test_embedding = np.random.rand(384).astype(np.float32)
            test_key = f"benchmark_emb_{i}"
            
            # Benchmark caching
            start_time = time.time()
            await cache_optimizer.cache_embeddings(test_key, test_embedding)
            cache_time = (time.time() - start_time) * 1000
            embed_times.append(cache_time)
            
            # Benchmark retrieval
            start_time = time.time()
            result = await cache_optimizer.get_cached_embeddings(test_key)
            retrieve_time = (time.time() - start_time) * 1000
            retrieve_times.append(retrieve_time)
            
            if result is None:
                logger.warning(f"Benchmark: Failed to retrieve embedding {test_key}")
        
        # Calculate statistics
        embed_stats = {
            'avg_ms': np.mean(embed_times),
            'p95_ms': np.percentile(embed_times, 95),
            'p99_ms': np.percentile(embed_times, 99),
            'operations_per_sec': 1000 / np.mean(embed_times) if embed_times else 0
        }
        
        retrieve_stats = {
            'avg_ms': np.mean(retrieve_times),
            'p95_ms': np.percentile(retrieve_times, 95),
            'p99_ms': np.percentile(retrieve_times, 99),
            'operations_per_sec': 1000 / np.mean(retrieve_times) if retrieve_times else 0
        }
        
        logger.info(f"Cache benchmark completed: {num_operations} operations")
        logger.info(f"Embed avg: {embed_stats['avg_ms']:.2f}ms, Retrieve avg: {retrieve_stats['avg_ms']:.2f}ms")
        
    except Exception as e:
        logger.error(f"Cache benchmark failed: {e}")


@router.get("/health")
async def health_check():
    """Health check for RAG performance system"""
    try:
        cache_optimizer = await get_rag_cache_optimizer()
        redis_client = await get_cache_service()
        
        # Check components
        checks = {
            'cache_optimizer': 'healthy',
            'redis': 'healthy' if await redis_client.ping() else 'unhealthy',
            'memory_pools': 'healthy',
            'background_workers': 'healthy'
        }
        
        # Check memory pools
        for pool_name, pool in cache_optimizer.memory_pools.items():
            stats = pool.get_usage_stats()
            if stats['utilization'] > 0.95:  # >95% utilization
                checks['memory_pools'] = 'warning'
        
        overall_status = 'healthy'
        if 'unhealthy' in checks.values():
            overall_status = 'unhealthy'
        elif 'warning' in checks.values():
            overall_status = 'warning'
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "uptime": "N/A"  # Could track actual uptime
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
