"""
AsyncIO RAG Performance Optimizer
Advanced asynchronous optimization for RAG operations with unified memory management

Author: Adrien Stevens - Python Performance Optimization Expert
Specialization: AsyncIO optimization, memory management, concurrent operations
"""

import asyncio
import logging
import time
import weakref
from typing import Dict, Any, List, Optional, Callable, Awaitable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import psutil
import gc

import uvloop  # High-performance event loop
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class AsyncOperationMetrics:
    """Metrics for async operations"""
    operation_name: str
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    concurrent_executions: int = 0
    errors: int = 0
    
    def update(self, execution_time_ms: float, error: bool = False):
        """Update metrics with new execution"""
        self.execution_count += 1
        if error:
            self.errors += 1
        else:
            self.total_time_ms += execution_time_ms
            self.avg_time_ms = self.total_time_ms / (self.execution_count - self.errors)
            self.min_time_ms = min(self.min_time_ms, execution_time_ms)
            self.max_time_ms = max(self.max_time_ms, execution_time_ms)


@dataclass
class AsyncRagConfig:
    """Configuration for async RAG optimization"""
    # Connection pooling
    max_concurrent_operations: int = 50
    connection_pool_size: int = 20
    connection_timeout: float = 30.0
    
    # Task batching
    batch_size: int = 10
    batch_timeout_ms: float = 100.0  # Max wait time to form batch
    max_batch_queue_size: int = 1000
    
    # Memory management
    memory_pressure_threshold: float = 0.85
    gc_interval_seconds: float = 60.0
    weak_ref_cleanup_interval: float = 120.0
    
    # Performance optimization
    enable_uvloop: bool = True
    enable_task_batching: bool = True
    enable_connection_pooling: bool = True
    enable_memory_optimization: bool = True


class AsyncConnectionPool:
    """High-performance async connection pool"""
    
    def __init__(self, max_size: int = 20, timeout: float = 30.0):
        self.max_size = max_size
        self.timeout = timeout
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.active_connections = 0
        self.total_connections = 0
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire connection from pool"""
        try:
            # Try to get existing connection
            connection = await asyncio.wait_for(
                self.pool.get(), timeout=0.1
            )
            return connection
        except asyncio.TimeoutError:
            # Create new connection if pool not full
            async with self.lock:
                if self.total_connections < self.max_size:
                    connection = await self._create_connection()
                    self.total_connections += 1
                    self.active_connections += 1
                    return connection
                else:
                    # Wait for available connection
                    connection = await asyncio.wait_for(
                        self.pool.get(), timeout=self.timeout
                    )
                    return connection
    
    async def release(self, connection):
        """Release connection back to pool"""
        if connection and not connection.closed:
            await self.pool.put(connection)
            async with self.lock:
                self.active_connections -= 1
        else:
            async with self.lock:
                self.total_connections -= 1
                self.active_connections -= 1
    
    async def _create_connection(self):
        """Create new connection - to be overridden by subclasses"""
        return object()  # Placeholder
    
    async def close_all(self):
        """Close all connections in pool"""
        connections = []
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                connections.append(conn)
            except asyncio.QueueEmpty:
                break
        
        # Close connections
        for conn in connections:
            if hasattr(conn, 'close'):
                await conn.close()
        
        async with self.lock:
            self.total_connections = 0
            self.active_connections = 0


class AsyncTaskBatcher(Generic[T]):
    """High-performance task batching system"""
    
    def __init__(
        self,
        batch_processor: Callable[[List[T]], Awaitable[List[Any]]],
        batch_size: int = 10,
        batch_timeout_ms: float = 100.0,
        max_queue_size: int = 1000
    ):
        self.batch_processor = batch_processor
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size
        
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.pending_futures: Dict[int, asyncio.Future] = {}
        self.running = False
        self.batch_worker_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.batches_processed = 0
        self.items_processed = 0
        self.avg_batch_size = 0.0
        
    async def start(self):
        """Start the batch processing worker"""
        if self.running:
            return
            
        self.running = True
        self.batch_worker_task = asyncio.create_task(self._batch_worker())
        logger.info("Async task batcher started")
    
    async def stop(self):
        """Stop the batch processing worker"""
        if not self.running:
            return
            
        self.running = False
        
        if self.batch_worker_task:
            self.batch_worker_task.cancel()
            try:
                await self.batch_worker_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining items
        await self._process_remaining_items()
        logger.info("Async task batcher stopped")
    
    async def submit(self, item: T) -> Any:
        """Submit item for batch processing"""
        if not self.running:
            raise RuntimeError("Batcher not running")
            
        future = asyncio.Future()
        item_id = id(item)
        self.pending_futures[item_id] = future
        
        try:
            await self.input_queue.put((item_id, item))
            return await future
        except asyncio.QueueFull:
            self.pending_futures.pop(item_id, None)
            raise RuntimeError("Batch queue is full")
    
    async def _batch_worker(self):
        """Main batch processing worker"""
        batch = []
        batch_item_ids = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Wait for items with timeout
                timeout = (self.batch_timeout_ms / 1000.0)
                
                try:
                    item_id, item = await asyncio.wait_for(
                        self.input_queue.get(), timeout=timeout
                    )
                    batch.append(item)
                    batch_item_ids.append(item_id)
                    
                except asyncio.TimeoutError:
                    # Process partial batch if timeout
                    if batch:
                        await self._process_batch(batch, batch_item_ids)
                        batch.clear()
                        batch_item_ids.clear()
                        last_batch_time = time.time()
                    continue
                
                # Check if batch is ready
                current_time = time.time()
                batch_ready = (
                    len(batch) >= self.batch_size or
                    (current_time - last_batch_time) * 1000 >= self.batch_timeout_ms
                )
                
                if batch_ready:
                    await self._process_batch(batch, batch_item_ids)
                    batch.clear()
                    batch_item_ids.clear()
                    last_batch_time = current_time
                    
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                # Handle error by failing pending futures
                for item_id in batch_item_ids:
                    future = self.pending_futures.pop(item_id, None)
                    if future and not future.done():
                        future.set_exception(e)
                batch.clear()
                batch_item_ids.clear()
    
    async def _process_batch(self, batch: List[T], item_ids: List[int]):
        """Process a batch of items"""
        try:
            # Execute batch processor
            results = await self.batch_processor(batch)
            
            # Distribute results to futures
            for item_id, result in zip(item_ids, results):
                future = self.pending_futures.pop(item_id, None)
                if future and not future.done():
                    future.set_result(result)
            
            # Update metrics
            self.batches_processed += 1
            self.items_processed += len(batch)
            self.avg_batch_size = self.items_processed / self.batches_processed
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Fail all futures in this batch
            for item_id in item_ids:
                future = self.pending_futures.pop(item_id, None)
                if future and not future.done():
                    future.set_exception(e)
    
    async def _process_remaining_items(self):
        """Process any remaining items in queue"""
        batch = []
        batch_item_ids = []
        
        # Collect remaining items
        while not self.input_queue.empty():
            try:
                item_id, item = self.input_queue.get_nowait()
                batch.append(item)
                batch_item_ids.append(item_id)
            except asyncio.QueueEmpty:
                break
        
        # Process final batch
        if batch:
            await self._process_batch(batch, batch_item_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        return {
            'batches_processed': self.batches_processed,
            'items_processed': self.items_processed,
            'avg_batch_size': self.avg_batch_size,
            'queue_size': self.input_queue.qsize(),
            'pending_futures': len(self.pending_futures),
            'running': self.running
        }


class AsyncMemoryOptimizer:
    """Memory optimization for async operations"""
    
    def __init__(self, config: AsyncRagConfig):
        self.config = config
        self.weak_refs: weakref.WeakSet = weakref.WeakSet()
        self.memory_stats = defaultdict(int)
        self.gc_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self):
        """Start memory optimization"""
        if self.running:
            return
            
        self.running = True
        
        # Start garbage collection task
        self.gc_task = asyncio.create_task(self._gc_worker())
        
        # Start weak reference cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("Async memory optimizer started")
    
    async def stop(self):
        """Stop memory optimization"""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel tasks
        if self.gc_task:
            self.gc_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self.gc_task, self.cleanup_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Async memory optimizer stopped")
    
    def track_object(self, obj):
        """Track object for memory optimization"""
        self.weak_refs.add(obj)
        self.memory_stats['tracked_objects'] += 1
    
    async def _gc_worker(self):
        """Background garbage collection worker"""
        while self.running:
            try:
                # Check memory pressure
                memory = psutil.virtual_memory()
                
                if memory.percent > self.config.memory_pressure_threshold * 100:
                    # Force garbage collection
                    collected = gc.collect()
                    logger.info(f"Forced GC: collected {collected} objects")
                    self.memory_stats['gc_collections'] += 1
                    self.memory_stats['gc_collected'] += collected
                
                # Sleep until next check
                await asyncio.sleep(self.config.gc_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"GC worker error: {e}")
    
    async def _cleanup_worker(self):
        """Background cleanup worker"""
        while self.running:
            try:
                # Clean up weak references (this happens automatically)
                # But we can track statistics
                current_refs = len(self.weak_refs)
                self.memory_stats['current_weak_refs'] = current_refs
                
                # Sleep until next cleanup
                await asyncio.sleep(self.config.weak_ref_cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics"""
        memory = psutil.virtual_memory()
        
        return {
            'system_memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent
            },
            'optimizer_stats': dict(self.memory_stats),
            'running': self.running
        }


class AsyncRagOptimizer:
    """
    High-performance AsyncIO optimizer for RAG operations
    
    Features:
    - Connection pooling for database operations
    - Task batching for improved throughput
    - Memory optimization with weak references
    - Performance monitoring and metrics
    - uvloop integration for maximum performance
    """
    
    def __init__(self, config: Optional[AsyncRagConfig] = None):
        self.config = config or AsyncRagConfig()
        
        # Core components
        self.connection_pool: Optional[AsyncConnectionPool] = None
        self.embedding_batcher: Optional[AsyncTaskBatcher] = None
        self.query_batcher: Optional[AsyncTaskBatcher] = None
        self.memory_optimizer: Optional[AsyncMemoryOptimizer] = None
        
        # Performance tracking
        self.operation_metrics: Dict[str, AsyncOperationMetrics] = {}
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)
        
        # State management
        self.running = False
        self.startup_time: Optional[float] = None
        
    async def initialize(self):
        """Initialize the async optimizer"""
        if self.running:
            return
        
        try:
            self.startup_time = time.time()
            
            # Set event loop policy for performance
            if self.config.enable_uvloop:
                try:
                    import uvloop
                    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                    logger.info("uvloop event loop policy set")
                except ImportError:
                    logger.warning("uvloop not available, using default event loop")
            
            # Initialize connection pool
            if self.config.enable_connection_pooling:
                self.connection_pool = AsyncConnectionPool(
                    max_size=self.config.connection_pool_size,
                    timeout=self.config.connection_timeout
                )
            
            # Initialize task batchers
            if self.config.enable_task_batching:
                self.embedding_batcher = AsyncTaskBatcher(
                    batch_processor=self._process_embedding_batch,
                    batch_size=self.config.batch_size,
                    batch_timeout_ms=self.config.batch_timeout_ms,
                    max_queue_size=self.config.max_batch_queue_size
                )
                
                self.query_batcher = AsyncTaskBatcher(
                    batch_processor=self._process_query_batch,
                    batch_size=self.config.batch_size,
                    batch_timeout_ms=self.config.batch_timeout_ms,
                    max_queue_size=self.config.max_batch_queue_size
                )
                
                await self.embedding_batcher.start()
                await self.query_batcher.start()
            
            # Initialize memory optimizer
            if self.config.enable_memory_optimization:
                self.memory_optimizer = AsyncMemoryOptimizer(self.config)
                await self.memory_optimizer.start()
            
            self.running = True
            logger.info("Async RAG optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize async optimizer: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown the async optimizer"""
        if not self.running:
            return
        
        try:
            self.running = False
            
            # Stop task batchers
            if self.embedding_batcher:
                await self.embedding_batcher.stop()
            if self.query_batcher:
                await self.query_batcher.stop()
            
            # Stop memory optimizer
            if self.memory_optimizer:
                await self.memory_optimizer.stop()
            
            # Close connection pool
            if self.connection_pool:
                await self.connection_pool.close_all()
            
            logger.info("Async RAG optimizer shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during async optimizer shutdown: {e}")
    
    async def optimize_operation(
        self,
        operation_name: str,
        operation_func: Callable[[], Awaitable[T]]
    ) -> T:
        """Optimize an async operation with performance tracking"""
        
        start_time = time.time()
        
        # Get or create metrics for this operation
        if operation_name not in self.operation_metrics:
            self.operation_metrics[operation_name] = AsyncOperationMetrics(operation_name)
        
        metrics = self.operation_metrics[operation_name]
        
        try:
            # Acquire semaphore for concurrency control
            async with self.semaphore:
                metrics.concurrent_executions += 1
                
                # Track object for memory optimization
                if self.memory_optimizer:
                    self.memory_optimizer.track_object(operation_func)
                
                # Execute operation
                result = await operation_func()
                
                # Update metrics
                execution_time_ms = (time.time() - start_time) * 1000
                metrics.update(execution_time_ms)
                
                return result
                
        except Exception as e:
            # Update error metrics
            execution_time_ms = (time.time() - start_time) * 1000
            metrics.update(execution_time_ms, error=True)
            raise
        
        finally:
            metrics.concurrent_executions -= 1
    
    async def batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Process embeddings in optimized batches"""
        if not self.embedding_batcher:
            raise RuntimeError("Embedding batcher not initialized")
        
        # Submit texts for batch processing
        tasks = [self.embedding_batcher.submit(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def batch_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process queries in optimized batches"""
        if not self.query_batcher:
            raise RuntimeError("Query batcher not initialized")
        
        # Submit queries for batch processing
        tasks = [self.query_batcher.submit(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _process_embedding_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Process a batch of embedding requests"""
        # This would integrate with actual embedding service
        # For now, return placeholder embeddings
        results = []
        for text in texts:
            # Simulate embedding generation
            embedding = np.random.rand(384).astype(np.float32)
            results.append(embedding)
        
        return results
    
    async def _process_query_batch(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of query requests"""
        # This would integrate with actual query processing
        # For now, return placeholder results
        results = []
        for query in queries:
            result = {
                'query': query.get('query_text', ''),
                'results': [],
                'processed_at': time.time()
            }
            results.append(result)
        
        return results
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        # Collect operation metrics
        operation_stats = {}
        for name, metrics in self.operation_metrics.items():
            operation_stats[name] = {
                'execution_count': metrics.execution_count,
                'avg_time_ms': metrics.avg_time_ms,
                'min_time_ms': metrics.min_time_ms if metrics.min_time_ms \!= float('inf') else 0,
                'max_time_ms': metrics.max_time_ms,
                'concurrent_executions': metrics.concurrent_executions,
                'error_rate': metrics.errors / metrics.execution_count if metrics.execution_count > 0 else 0
            }
        
        # Collect component stats
        component_stats = {}
        
        if self.embedding_batcher:
            component_stats['embedding_batcher'] = self.embedding_batcher.get_stats()
        
        if self.query_batcher:
            component_stats['query_batcher'] = self.query_batcher.get_stats()
        
        if self.memory_optimizer:
            component_stats['memory_optimizer'] = self.memory_optimizer.get_memory_stats()
        
        if self.connection_pool:
            component_stats['connection_pool'] = {
                'total_connections': self.connection_pool.total_connections,
                'active_connections': self.connection_pool.active_connections,
                'pool_size': self.connection_pool.pool.qsize()
            }
        
        # System stats
        system_stats = {
            'uptime_seconds': time.time() - self.startup_time if self.startup_time else 0,
            'running': self.running,
            'concurrent_operations': self.config.max_concurrent_operations - self.semaphore._value
        }
        
        return {
            'system': system_stats,
            'operations': operation_stats,
            'components': component_stats,
            'config': {
                'max_concurrent_operations': self.config.max_concurrent_operations,
                'batch_size': self.config.batch_size,
                'batch_timeout_ms': self.config.batch_timeout_ms,
                'connection_pool_size': self.config.connection_pool_size,
                'uvloop_enabled': self.config.enable_uvloop
            }
        }


# Global async optimizer instance
_async_optimizer: Optional[AsyncRagOptimizer] = None


async def get_async_rag_optimizer() -> AsyncRagOptimizer:
    """Get or create the async RAG optimizer instance"""
    global _async_optimizer
    
    if _async_optimizer is None:
        _async_optimizer = AsyncRagOptimizer()
        await _async_optimizer.initialize()
    
    return _async_optimizer


# Context manager for optimized operations
class OptimizedOperation:
    """Context manager for optimized async operations"""
    
    def __init__(self, operation_name: str, optimizer: Optional[AsyncRagOptimizer] = None):
        self.operation_name = operation_name
        self.optimizer = optimizer
        self.start_time: Optional[float] = None
    
    async def __aenter__(self):
        if not self.optimizer:
            self.optimizer = await get_async_rag_optimizer()
        
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.optimizer and self.start_time:
            execution_time = time.time() - self.start_time
            
            # Update metrics
            if self.operation_name not in self.optimizer.operation_metrics:
                self.optimizer.operation_metrics[self.operation_name] = AsyncOperationMetrics(self.operation_name)
            
            metrics = self.optimizer.operation_metrics[self.operation_name]
            metrics.update(execution_time * 1000, error=exc_type is not None)
