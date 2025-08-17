"""
Advanced RAG Performance Optimization and Caching System
Implements intelligent multi-layer caching with unified memory management

Author: Adrien Stevens - Python Performance Optimization Expert
Specialization: Python Performance Optimization, Refactoring, Unified Memory
Hardware: 2x Tesla V100-PCIE-16GB GPUs (32GB total VRAM)
"""

import asyncio
import logging
import pickle
import hashlib
import json
import time
import threading
import weakref
import mmap
import zlib
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import psutil
import gc

import redis.asyncio as redis
from sqlalchemy.orm import Session

from ..config import settings
from ..services.cache import get_cache_service
from ..models.document import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class CachePerformanceMetrics:
    """Performance metrics for cache operations"""
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    memory_usage_mb: float = 0.0
    avg_access_time_ms: float = 0.0
    compression_ratio: float = 0.0
    background_writes: int = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


@dataclass 
class UnifiedMemoryConfig:
    """Unified memory management configuration for optimal performance"""
    # Memory pools (optimized for Tesla V100 architecture)
    cpu_memory_pool_mb: int = 8192  # 8GB CPU memory pool
    gpu_memory_pool_mb: int = 16384  # 16GB GPU memory pool (per V100)
    shared_memory_pool_mb: int = 4096  # 4GB shared memory
    
    # Cache layers
    l1_cache_mb: int = 512  # L1: Hot data in memory
    l2_cache_mb: int = 2048  # L2: Warm data compressed  
    l3_cache_mb: int = 4096  # L3: Cold data on disk
    
    # Performance thresholds
    memory_pressure_threshold: float = 0.85
    gc_trigger_threshold: float = 0.90
    eviction_batch_size: int = 100
    compression_threshold: int = 1024  # bytes
    
    # Concurrency
    max_concurrent_operations: int = 16
    background_writer_threads: int = 4
    prefetch_worker_threads: int = 2


class LRUCache:
    """Memory-efficient LRU cache with compression"""
    
    def __init__(self, capacity: int, compress_threshold: int = 1024):
        self.capacity = capacity
        self.compress_threshold = compress_threshold
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.compressed_keys: Set[str] = set()
        self.access_times: Dict[str, float] = {}
        self.size_bytes = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
                
            # Move to end (most recently used)
            value = self.cache[key]
            del self.cache[key]
            
            # Decompress if needed
            if key in self.compressed_keys:
                value = pickle.loads(zlib.decompress(value))
                self.compressed_keys.remove(key)
                
            self.cache[key] = value
            self.access_times[key] = time.time()
            return value
    
    def put(self, key: str, value: Any) -> None:
        with self.lock:
            # Calculate value size
            value_bytes = len(pickle.dumps(value))
            
            # Remove old value if exists
            if key in self.cache:
                del self.cache[key]
                self.compressed_keys.discard(key)
                
            # Compress large values
            if value_bytes > self.compress_threshold:
                compressed = zlib.compress(pickle.dumps(value))
                if len(compressed) < value_bytes * 0.8:  # Only compress if >20% savings
                    value = compressed
                    self.compressed_keys.add(key)
                    value_bytes = len(compressed)
                    
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.size_bytes += value_bytes
            
            # Evict if needed
            while len(self.cache) > self.capacity:
                self._evict_lru()
                
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.cache:
            return
            
        lru_key = next(iter(self.cache))
        del self.cache[lru_key]
        self.compressed_keys.discard(lru_key)
        self.access_times.pop(lru_key, None)


class MemoryPool:
    """Unified memory pool for efficient allocation"""
    
    def __init__(self, size_mb: int):
        self.size_bytes = size_mb * 1024 * 1024
        self.allocated = 0
        self.free_blocks: List[Tuple[int, int]] = [(0, self.size_bytes)]
        self.allocated_blocks: Dict[int, Tuple[int, int]] = {}
        self.lock = threading.Lock()
        
        # Memory-mapped file for disk-based caching
        self.cache_file = Path("/tmp/knowledgehub_memory_pool")
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # Create memory-mapped file
        with open(self.cache_file, "wb") as f:
            f.write(b'\x00' * self.size_bytes)
            
        with open(self.cache_file, "r+b") as f:
            self.mmap = mmap.mmap(f.fileno(), self.size_bytes)
    
    def allocate(self, size: int) -> Optional[int]:
        """Allocate memory block, returns offset or None if failed"""
        with self.lock:
            for i, (start, length) in enumerate(self.free_blocks):
                if length >= size:
                    # Allocate from this block
                    offset = start
                    self.allocated_blocks[offset] = (start, size)
                    
                    # Update free block
                    if length == size:
                        del self.free_blocks[i]
                    else:
                        self.free_blocks[i] = (start + size, length - size)
                        
                    self.allocated += size
                    return offset
            return None
    
    def deallocate(self, offset: int) -> None:
        """Deallocate memory block"""
        with self.lock:
            if offset not in self.allocated_blocks:
                return
                
            start, size = self.allocated_blocks[offset]
            del self.allocated_blocks[offset]
            self.allocated -= size
            
            # Add back to free blocks (could implement coalescing)
            self.free_blocks.append((start, size))
            self.free_blocks.sort()
    
    def write(self, offset: int, data: bytes) -> None:
        """Write data to memory pool"""
        with self.lock:
            if offset in self.allocated_blocks:
                _, size = self.allocated_blocks[offset]
                if len(data) <= size:
                    self.mmap[offset:offset + len(data)] = data
    
    def read(self, offset: int, size: int) -> bytes:
        """Read data from memory pool"""
        with self.lock:
            return bytes(self.mmap[offset:offset + size])
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self.lock:
            return {
                'total_bytes': self.size_bytes,
                'allocated_bytes': self.allocated,
                'free_bytes': self.size_bytes - self.allocated,
                'utilization': self.allocated / self.size_bytes,
                'free_blocks': len(self.free_blocks),
                'allocated_blocks': len(self.allocated_blocks)
            }


class AsyncBackgroundWriter:
    """Asynchronous background writer for cache persistence"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.write_queue = asyncio.Queue(maxsize=max_queue_size)
        self.stats_queue = asyncio.Queue(maxsize=1000)
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        
    async def start_workers(self, num_workers: int = 4):
        """Start background writer workers"""
        self.running = True
        
        # Start write workers
        for i in range(num_workers):
            task = asyncio.create_task(self._write_worker(f"writer-{i}"))
            self.worker_tasks.append(task)
            
        # Start stats worker
        stats_task = asyncio.create_task(self._stats_worker())
        self.worker_tasks.append(stats_task)
        
        logger.info(f"Started {num_workers} background writers + 1 stats worker")
    
    async def queue_write(self, operation: str, key: str, data: Any, ttl: int = 3600):
        """Queue a write operation"""
        if not self.running:
            return
            
        try:
            await self.write_queue.put({
                'operation': operation,
                'key': key,
                'data': data,
                'ttl': ttl,
                'timestamp': time.time()
            }, timeout=1.0)  # Non-blocking with timeout
        except asyncio.TimeoutError:
            logger.warning("Write queue full, dropping write operation")
    
    async def _write_worker(self, worker_id: str):
        """Background write worker"""
        redis_client = await get_cache_service()
        
        while self.running:
            try:
                # Get write operation with timeout
                operation = await asyncio.wait_for(
                    self.write_queue.get(), timeout=1.0
                )
                
                start_time = time.time()
                
                if operation['operation'] == 'set':
                    await redis_client.set(
                        operation['key'], 
                        operation['data'],
                        operation['ttl']
                    )
                elif operation['operation'] == 'delete':
                    await redis_client.delete(operation['key'])
                
                # Queue stats update
                duration_ms = (time.time() - start_time) * 1000
                await self.stats_queue.put({
                    'worker_id': worker_id,
                    'operation': operation['operation'],
                    'duration_ms': duration_ms,
                    'success': True
                })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Background writer {worker_id} error: {e}")
                await self.stats_queue.put({
                    'worker_id': worker_id,
                    'operation': 'error',
                    'error': str(e),
                    'success': False
                })
    
    async def _stats_worker(self):
        """Background stats collection worker"""
        stats = defaultdict(list)
        
        while self.running:
            try:
                stat = await asyncio.wait_for(self.stats_queue.get(), timeout=5.0)
                
                # Collect stats
                if stat['success']:
                    stats[stat['worker_id']].append(stat['duration_ms'])
                    
                # Periodically log stats (every 100 operations)
                total_ops = sum(len(worker_stats) for worker_stats in stats.values())
                if total_ops > 0 and total_ops % 100 == 0:
                    avg_duration = np.mean([
                        duration for worker_stats in stats.values() 
                        for duration in worker_stats
                    ])
                    logger.info(f"Background writers: {total_ops} ops, avg {avg_duration:.1f}ms")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Stats worker error: {e}")
    
    async def stop(self):
        """Stop background workers"""
        self.running = False
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            self.worker_tasks.clear()


class RAGCacheOptimizer:
    """
    Advanced RAG caching system with unified memory management
    
    Features:
    - Multi-layer caching (L1/L2/L3)
    - Intelligent prefetching
    - Compression and memory pooling
    - Background write optimization
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[UnifiedMemoryConfig] = None):
        self.config = config or UnifiedMemoryConfig()
        self.metrics = CachePerformanceMetrics()
        
        # Initialize cache layers
        self.l1_cache = LRUCache(
            capacity=self.config.l1_cache_mb * 1000,  # Rough item estimate
            compress_threshold=self.config.compression_threshold
        )
        
        # Memory pools
        self.memory_pools = {
            'embeddings': MemoryPool(self.config.cpu_memory_pool_mb // 4),
            'queries': MemoryPool(self.config.cpu_memory_pool_mb // 4),
            'results': MemoryPool(self.config.cpu_memory_pool_mb // 4),
            'metadata': MemoryPool(self.config.cpu_memory_pool_mb // 4)
        }
        
        # Background operations
        self.background_writer = AsyncBackgroundWriter()
        self.prefetch_queue = asyncio.Queue(maxsize=1000)
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage_history: List[Tuple[float, float]] = []
        
        # Thread pools for CPU-intensive operations
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_operations
        )
        self.compression_executor = ThreadPoolExecutor(max_workers=4)
        
        # Weak references for automatic cleanup
        self.cache_refs: Dict[str, weakref.ReferenceType] = {}
        
        logger.info("RAG Cache Optimizer initialized with unified memory management")
    
    async def initialize(self):
        """Initialize the cache optimizer"""
        try:
            # Start background workers
            await self.background_writer.start_workers(
                self.config.background_writer_threads
            )
            
            # Start prefetch workers
            for i in range(self.config.prefetch_worker_threads):
                asyncio.create_task(self._prefetch_worker(f"prefetch-{i}"))
            
            # Start memory monitor
            asyncio.create_task(self._memory_monitor())
            
            logger.info("RAG Cache Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Cache Optimizer: {e}")
            raise
    
    async def cache_embeddings(
        self, 
        key: str, 
        embeddings: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: int = 3600
    ) -> bool:
        """
        Cache embeddings with optimized storage
        
        Args:
            key: Cache key
            embeddings: Numpy array of embeddings
            metadata: Optional metadata
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        start_time = time.time()
        
        try:
            # Prepare embedding data for caching
            embedding_data = {
                'embeddings': embeddings.tobytes(),
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype),
                'metadata': metadata or {},
                'cached_at': datetime.utcnow().isoformat()
            }
            
            # Try L1 cache first (hot data)
            cache_key = f"emb:l1:{key}"
            self.l1_cache.put(cache_key, embedding_data)
            
            # Async write to Redis (L2)
            await self.background_writer.queue_write(
                'set', 
                f"emb:l2:{key}", 
                embedding_data, 
                ttl
            )
            
            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            self.operation_times['cache_embeddings'].append(duration_ms)
            self.metrics.background_writes += 1
            
            # Schedule prefetch of related embeddings
            await self._schedule_prefetch('embedding', key, metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache embeddings for {key}: {e}")
            return False
    
    async def get_cached_embeddings(self, key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve cached embeddings with automatic decompression
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (embeddings, metadata) or None
        """
        start_time = time.time()
        
        try:
            # Try L1 cache first
            cache_key = f"emb:l1:{key}"
            cached_data = self.l1_cache.get(cache_key)
            
            if cached_data:
                self.metrics.hit_count += 1
                embeddings = np.frombuffer(
                    cached_data['embeddings'], 
                    dtype=cached_data['dtype']
                ).reshape(cached_data['shape'])
                
                duration_ms = (time.time() - start_time) * 1000
                self.operation_times['get_embeddings_l1'].append(duration_ms)
                
                return embeddings, cached_data['metadata']
            
            # Try L2 cache (Redis)
            redis_client = await get_cache_service()
            redis_key = f"emb:l2:{key}"
            cached_data = await redis_client.get(redis_key)
            
            if cached_data:
                self.metrics.hit_count += 1
                
                # Restore to L1 cache
                self.l1_cache.put(cache_key, cached_data)
                
                embeddings = np.frombuffer(
                    cached_data['embeddings'],
                    dtype=cached_data['dtype']
                ).reshape(cached_data['shape'])
                
                duration_ms = (time.time() - start_time) * 1000
                self.operation_times['get_embeddings_l2'].append(duration_ms)
                
                return embeddings, cached_data['metadata']
            
            # Cache miss
            self.metrics.miss_count += 1
            
            duration_ms = (time.time() - start_time) * 1000
            self.operation_times['get_embeddings_miss'].append(duration_ms)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached embeddings for {key}: {e}")
            self.metrics.miss_count += 1
            return None
    
    async def cache_query_results(
        self,
        query_hash: str,
        results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        ttl: int = 300  # 5 minutes for query results
    ) -> bool:
        """Cache query results with intelligent compression"""
        start_time = time.time()
        
        try:
            query_data = {
                'results': results,
                'metadata': metadata or {},
                'result_count': len(results),
                'cached_at': datetime.utcnow().isoformat()
            }
            
            # L1 cache for immediate reuse
            cache_key = f"qry:l1:{query_hash}"
            self.l1_cache.put(cache_key, query_data)
            
            # Background write to Redis
            await self.background_writer.queue_write(
                'set',
                f"qry:l2:{query_hash}",
                query_data,
                ttl
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.operation_times['cache_query_results'].append(duration_ms)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache query results for {query_hash}: {e}")
            return False
    
    async def get_cached_query_results(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached query results"""
        start_time = time.time()
        
        try:
            # Try L1 cache
            cache_key = f"qry:l1:{query_hash}"
            cached_data = self.l1_cache.get(cache_key)
            
            if cached_data:
                self.metrics.hit_count += 1
                duration_ms = (time.time() - start_time) * 1000
                self.operation_times['get_query_results_l1'].append(duration_ms)
                return cached_data['results']
            
            # Try L2 cache
            redis_client = await get_cache_service()
            redis_key = f"qry:l2:{query_hash}"
            cached_data = await redis_client.get(redis_key)
            
            if cached_data:
                self.metrics.hit_count += 1
                
                # Promote to L1
                self.l1_cache.put(cache_key, cached_data)
                
                duration_ms = (time.time() - start_time) * 1000
                self.operation_times['get_query_results_l2'].append(duration_ms)
                return cached_data['results']
            
            # Cache miss
            self.metrics.miss_count += 1
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached query results for {query_hash}: {e}")
            self.metrics.miss_count += 1
            return None
    
    async def cache_neo4j_results(
        self,
        query_hash: str,
        neo4j_results: List[Dict[str, Any]],
        ttl: int = 600  # 10 minutes for graph queries
    ) -> bool:
        """Cache Neo4j query results with graph-specific optimizations"""
        start_time = time.time()
        
        try:
            # Compress graph structures efficiently
            graph_data = {
                'results': neo4j_results,
                'result_count': len(neo4j_results),
                'cached_at': datetime.utcnow().isoformat(),
                'data_type': 'neo4j_graph'
            }
            
            # Use compression for large graph results
            if len(neo4j_results) > 10:
                compressed_data = await asyncio.get_event_loop().run_in_executor(
                    self.compression_executor,
                    self._compress_graph_data,
                    graph_data
                )
                graph_data = compressed_data
            
            # Cache in L1 and schedule L2 write
            cache_key = f"neo4j:l1:{query_hash}"
            self.l1_cache.put(cache_key, graph_data)
            
            await self.background_writer.queue_write(
                'set',
                f"neo4j:l2:{query_hash}",
                graph_data,
                ttl
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.operation_times['cache_neo4j_results'].append(duration_ms)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache Neo4j results for {query_hash}: {e}")
            return False
    
    def _compress_graph_data(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress graph data using CPU executor"""
        try:
            serialized = pickle.dumps(graph_data['results'])
            compressed = zlib.compress(serialized, level=6)
            
            if len(compressed) < len(serialized) * 0.8:
                return {
                    **graph_data,
                    'results': compressed,
                    'compressed': True,
                    'original_size': len(serialized),
                    'compressed_size': len(compressed)
                }
            else:
                return graph_data
                
        except Exception as e:
            logger.error(f"Graph data compression failed: {e}")
            return graph_data
    
    async def _schedule_prefetch(self, data_type: str, key: str, metadata: Optional[Dict[str, Any]]):
        """Schedule intelligent prefetching based on access patterns"""
        try:
            prefetch_request = {
                'data_type': data_type,
                'key': key,
                'metadata': metadata,
                'scheduled_at': time.time()
            }
            
            await self.prefetch_queue.put(prefetch_request)
            
        except asyncio.QueueFull:
            # Silently drop prefetch requests if queue is full
            pass
    
    async def _prefetch_worker(self, worker_id: str):
        """Background prefetch worker"""
        while True:
            try:
                request = await self.prefetch_queue.get()
                
                # Implement intelligent prefetching logic
                await self._execute_prefetch(request)
                
                # Mark task done
                self.prefetch_queue.task_done()
                
            except Exception as e:
                logger.error(f"Prefetch worker {worker_id} error: {e}")
    
    async def _execute_prefetch(self, request: Dict[str, Any]):
        """Execute prefetch operation"""
        # This could be enhanced with ML-based prediction
        # For now, implement simple related key prefetching
        
        data_type = request['data_type']
        key = request['key']
        
        if data_type == 'embedding':
            # Prefetch related embeddings (could use similarity)
            similar_keys = await self._find_similar_keys(key)
            for similar_key in similar_keys[:3]:  # Limit prefetch
                await self.get_cached_embeddings(similar_key)
    
    async def _find_similar_keys(self, key: str) -> List[str]:
        """Find similar cache keys for prefetching"""
        try:
            redis_client = await get_cache_service()
            pattern = f"emb:l2:{key[:10]}*"  # Simple prefix matching
            similar_keys = await redis_client.keys(pattern)
            return [k.replace('emb:l2:', '') for k in similar_keys[:5]]
        except:
            return []
    
    async def _memory_monitor(self):
        """Background memory monitoring and optimization"""
        while True:
            try:
                # Get system memory info
                memory = psutil.virtual_memory()
                
                # Check memory pressure
                if memory.percent > self.config.memory_pressure_threshold * 100:
                    await self._handle_memory_pressure()
                
                # Record memory usage
                current_time = time.time()
                self.memory_usage_history.append((current_time, memory.percent))
                
                # Keep only recent history (last hour)
                cutoff_time = current_time - 3600
                self.memory_usage_history = [
                    (t, usage) for t, usage in self.memory_usage_history
                    if t > cutoff_time
                ]
                
                # Update metrics
                self.metrics.memory_usage_mb = memory.used / 1024 / 1024
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _handle_memory_pressure(self):
        """Handle memory pressure by evicting cache and triggering GC"""
        logger.warning("Memory pressure detected, optimizing caches")
        
        try:
            # Clear some L1 cache entries
            current_size = len(self.l1_cache.cache)
            target_size = int(current_size * 0.7)  # Remove 30%
            
            items_to_remove = current_size - target_size
            for _ in range(items_to_remove):
                if self.l1_cache.cache:
                    self.l1_cache._evict_lru()
                    self.metrics.eviction_count += 1
            
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Memory optimization: evicted {items_to_remove} items, collected {collected} objects")
            
        except Exception as e:
            logger.error(f"Memory pressure handling failed: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            # Calculate average operation times
            avg_times = {}
            for operation, times in self.operation_times.items():
                if times:
                    avg_times[operation] = {
                        'avg_ms': np.mean(times),
                        'p95_ms': np.percentile(times, 95),
                        'p99_ms': np.percentile(times, 99),
                        'count': len(times)
                    }
            
            # Get memory pool stats
            pool_stats = {}
            for pool_name, pool in self.memory_pools.items():
                pool_stats[pool_name] = pool.get_usage_stats()
            
            # Get system memory
            memory = psutil.virtual_memory()
            
            return {
                'cache_metrics': {
                    'hit_ratio': self.metrics.hit_ratio,
                    'hit_count': self.metrics.hit_count,
                    'miss_count': self.metrics.miss_count,
                    'eviction_count': self.metrics.eviction_count,
                    'background_writes': self.metrics.background_writes
                },
                'performance': {
                    'operation_times': avg_times,
                    'l1_cache_size': len(self.l1_cache.cache),
                    'l1_cache_size_bytes': self.l1_cache.size_bytes
                },
                'memory': {
                    'system_usage_percent': memory.percent,
                    'system_available_mb': memory.available / 1024 / 1024,
                    'pool_stats': pool_stats
                },
                'configuration': {
                    'l1_cache_mb': self.config.l1_cache_mb,
                    'l2_cache_mb': self.config.l2_cache_mb,
                    'max_concurrent': self.config.max_concurrent_operations,
                    'compression_threshold': self.config.compression_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {'error': str(e)}
    
    async def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries matching pattern"""
        try:
            if pattern:
                # Clear matching L1 entries
                keys_to_remove = [
                    key for key in self.l1_cache.cache.keys()
                    if pattern in key
                ]
                
                for key in keys_to_remove:
                    self.l1_cache.cache.pop(key, None)
                    self.l1_cache.compressed_keys.discard(key)
                
                # Clear matching Redis entries
                redis_client = await get_cache_service()
                redis_keys = await redis_client.keys(f"*{pattern}*")
                
                for key in redis_keys:
                    await self.background_writer.queue_write('delete', key, None)
                    
                logger.info(f"Invalidated {len(keys_to_remove)} L1 and {len(redis_keys)} L2 cache entries")
                
            else:
                # Clear all caches
                self.l1_cache.cache.clear()
                self.l1_cache.compressed_keys.clear()
                
                redis_client = await get_cache_service()
                all_keys = await redis_client.keys("emb:*")
                all_keys.extend(await redis_client.keys("qry:*"))
                all_keys.extend(await redis_client.keys("neo4j:*"))
                
                for key in all_keys:
                    await self.background_writer.queue_write('delete', key, None)
                    
                logger.info(f"Invalidated all cache entries ({len(all_keys)} total)")
                
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
    
    def generate_cache_key(self, *components) -> str:
        """Generate consistent cache key from components"""
        key_material = ":".join(str(c) for c in components)
        return hashlib.md5(key_material.encode()).hexdigest()
    
    async def shutdown(self):
        """Clean shutdown of cache optimizer"""
        try:
            logger.info("Shutting down RAG Cache Optimizer")
            
            # Stop background writer
            await self.background_writer.stop()
            
            # Close thread pools
            self.cpu_executor.shutdown(wait=True)
            self.compression_executor.shutdown(wait=True)
            
            # Close memory pools
            for pool in self.memory_pools.values():
                if hasattr(pool, 'mmap'):
                    pool.mmap.close()
            
            logger.info("RAG Cache Optimizer shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during cache optimizer shutdown: {e}")


# Global cache optimizer instance
_cache_optimizer: Optional[RAGCacheOptimizer] = None


async def get_rag_cache_optimizer() -> RAGCacheOptimizer:
    """Get or create the RAG cache optimizer instance"""
    global _cache_optimizer
    
    if _cache_optimizer is None:
        _cache_optimizer = RAGCacheOptimizer()
        await _cache_optimizer.initialize()
    
    return _cache_optimizer
