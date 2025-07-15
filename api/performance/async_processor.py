"""
Async Processing Optimizer

Provides optimized async processing with:
- Task queue management
- Concurrent execution optimization
- Resource pooling
- Background job processing
- Memory-efficient streaming
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
from contextlib import asynccontextmanager
import json
import pickle
from queue import Queue
import threading

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Async task with metadata"""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def wait_time(self) -> Optional[float]:
        """Calculate wait time in queue"""
        if self.started_at:
            return self.started_at - self.created_at
        return None


class ResourcePool:
    """Resource pool for managing limited resources"""
    
    def __init__(self, resources: List[Any], max_concurrent: int = None):
        self.resources = resources
        self.max_concurrent = max_concurrent or len(resources)
        self.available = asyncio.Queue()
        self.in_use = set()
        self.stats = {
            'total_requests': 0,
            'current_usage': 0,
            'peak_usage': 0,
            'wait_times': []
        }
        
        # Initialize available resources
        for resource in resources:
            self.available.put_nowait(resource)
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """Acquire a resource from the pool"""
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            if timeout:
                resource = await asyncio.wait_for(self.available.get(), timeout=timeout)
            else:
                resource = await self.available.get()
            
            wait_time = time.time() - start_time
            self.stats['wait_times'].append(wait_time)
            
            self.in_use.add(resource)
            self.stats['current_usage'] = len(self.in_use)
            self.stats['peak_usage'] = max(self.stats['peak_usage'], self.stats['current_usage'])
            
            yield resource
            
        finally:
            if 'resource' in locals():
                self.in_use.discard(resource)
                self.available.put_nowait(resource)
                self.stats['current_usage'] = len(self.in_use)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics"""
        avg_wait_time = sum(self.stats['wait_times']) / max(len(self.stats['wait_times']), 1)
        
        return {
            'total_resources': len(self.resources),
            'available_resources': self.available.qsize(),
            'in_use_resources': len(self.in_use),
            'utilization': len(self.in_use) / len(self.resources),
            'total_requests': self.stats['total_requests'],
            'peak_usage': self.stats['peak_usage'],
            'average_wait_time': avg_wait_time
        }


class AsyncTaskQueue:
    """High-performance async task queue with priority support"""
    
    def __init__(self, max_concurrent: int = 100, max_queue_size: int = 10000):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        # Priority queues for different priority levels
        self.queues = {
            priority: asyncio.Queue(maxsize=max_queue_size) 
            for priority in TaskPriority
        }
        
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_results: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'current_queue_size': 0,
            'peak_queue_size': 0,
            'average_execution_time': 0,
            'average_wait_time': 0
        }
        
        # Worker control
        self.workers: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def start(self, num_workers: int = None):
        """Start the task queue workers"""
        if num_workers is None:
            num_workers = min(self.max_concurrent, 10)
        
        self.workers = []
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {num_workers} async task workers")
    
    async def stop(self, timeout: float = 30.0):
        """Stop the task queue workers"""
        self.shutdown_event.set()
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        # Wait for workers to finish
        if self.workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.workers, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Task queue shutdown timeout - force cancelling workers")
                for worker in self.workers:
                    worker.cancel()
        
        logger.info("Task queue stopped")
    
    async def submit(self, 
                     func: Callable,
                     *args,
                     name: str = None,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     timeout: Optional[float] = None,
                     max_retries: int = 3,
                     **kwargs) -> str:
        """Submit a task to the queue"""
        
        task_id = f"task-{int(time.time() * 1000000)}"
        task = Task(
            id=task_id,
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Check queue capacity
        current_size = sum(q.qsize() for q in self.queues.values())
        if current_size >= self.max_queue_size:
            raise asyncio.QueueFull("Task queue is full")
        
        # Add to appropriate priority queue
        await self.queues[priority].put(task)
        
        self.stats['total_tasks'] += 1
        self.stats['current_queue_size'] = current_size + 1
        self.stats['peak_queue_size'] = max(
            self.stats['peak_queue_size'], 
            self.stats['current_queue_size']
        )
        
        return task_id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result"""
        start_time = time.time()
        
        while True:
            # Check if task is completed
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise Exception(task.error)
                elif task.status == TaskStatus.CANCELLED:
                    raise asyncio.CancelledError("Task was cancelled")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            return True
        
        # Find and remove from queues
        for queue in self.queues.values():
            # Note: asyncio.Queue doesn't have a remove method
            # In a production system, you'd need a more sophisticated approach
            pass
        
        return False
    
    async def _worker(self, worker_name: str):
        """Task queue worker"""
        logger.info(f"Worker {worker_name} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next task from highest priority queue
                task = await self._get_next_task()
                if not task:
                    continue
                
                # Execute task with semaphore for concurrency control
                async with self.semaphore:
                    await self._execute_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _get_next_task(self) -> Optional[Task]:
        """Get next task from priority queues"""
        
        # Check each priority level in order
        for priority in TaskPriority:
            queue = self.queues[priority]
            
            try:
                # Try to get task with short timeout
                task = await asyncio.wait_for(queue.get(), timeout=0.1)
                self.stats['current_queue_size'] -= 1
                return task
            except asyncio.TimeoutError:
                continue
        
        return None
    
    async def _execute_task(self, task: Task):
        """Execute a single task"""
        task.started_at = time.time()
        task.status = TaskStatus.RUNNING
        
        try:
            # Create task future
            if asyncio.iscoroutinefunction(task.func):
                future = asyncio.create_task(task.func(*task.args, **task.kwargs))
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(None, task.func, *task.args, **task.kwargs)
            
            self.running_tasks[task.id] = future
            
            # Execute with timeout
            if task.timeout:
                result = await asyncio.wait_for(future, timeout=task.timeout)
            else:
                result = await future
            
            # Task completed successfully
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            self.stats['completed_tasks'] += 1
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self.stats['cancelled_tasks'] += 1
            
        except Exception as e:
            task.error = str(e)
            task.retry_count += 1
            
            # Retry if possible
            if task.retry_count <= task.max_retries:
                logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                # Re-queue with exponential backoff
                await asyncio.sleep(2 ** task.retry_count)
                await self.queues[task.priority].put(task)
                return
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                self.stats['failed_tasks'] += 1
                logger.error(f"Task {task.id} failed permanently: {e}")
        
        finally:
            # Clean up
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            
            # Store completed task
            self.completed_tasks[task.id] = task
            
            # Update statistics
            if task.execution_time:
                # Update average execution time
                current_avg = self.stats['average_execution_time']
                completed_count = self.stats['completed_tasks']
                self.stats['average_execution_time'] = (
                    (current_avg * (completed_count - 1) + task.execution_time) / completed_count
                )
            
            if task.wait_time:
                # Update average wait time
                current_avg = self.stats['average_wait_time']
                total_count = self.stats['completed_tasks'] + self.stats['failed_tasks']
                self.stats['average_wait_time'] = (
                    (current_avg * (total_count - 1) + task.wait_time) / total_count
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task queue statistics"""
        queue_sizes = {
            priority.name: queue.qsize() 
            for priority, queue in self.queues.items()
        }
        
        return {
            **self.stats,
            'running_tasks': len(self.running_tasks),
            'queue_sizes': queue_sizes,
            'total_queue_size': sum(queue_sizes.values()),
            'worker_count': len(self.workers),
            'max_concurrent': self.max_concurrent
        }


class StreamProcessor:
    """Memory-efficient stream processing"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    async def process_stream(self,
                           data_source: AsyncGenerator[Any, None],
                           processor: Callable[[List[Any]], Awaitable[List[Any]]],
                           output_handler: Optional[Callable[[List[Any]], Awaitable[None]]] = None) -> AsyncGenerator[Any, None]:
        """Process data stream in chunks"""
        
        chunk = []
        
        async for item in data_source:
            chunk.append(item)
            
            if len(chunk) >= self.chunk_size:
                # Process chunk
                processed_chunk = await processor(chunk)
                
                # Handle output
                if output_handler:
                    await output_handler(processed_chunk)
                
                # Yield processed items
                for processed_item in processed_chunk:
                    yield processed_item
                
                # Reset chunk
                chunk = []
        
        # Process remaining items
        if chunk:
            processed_chunk = await processor(chunk)
            
            if output_handler:
                await output_handler(processed_chunk)
            
            for processed_item in processed_chunk:
                yield processed_item
    
    async def batch_process(self,
                           items: List[Any],
                           processor: Callable[[Any], Awaitable[Any]],
                           max_concurrent: int = 10) -> List[Any]:
        """Process items in parallel batches"""
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item(item):
            async with semaphore:
                return await processor(item)
        
        # Create tasks for all items
        tasks = [asyncio.create_task(process_item(item)) for item in items]
        
        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and collect results
        for result in completed_results:
            if not isinstance(result, Exception):
                results.append(result)
            else:
                logger.error(f"Batch processing error: {result}")
        
        return results


class AsyncOptimizer:
    """Main async processing optimizer"""
    
    def __init__(self):
        self.task_queue = AsyncTaskQueue()
        self.stream_processor = StreamProcessor()
        
        # Thread pools for CPU-bound and I/O-bound tasks
        self.cpu_pool = ProcessPoolExecutor(max_workers=4)
        self.io_pool = ThreadPoolExecutor(max_workers=20)
        
        # Resource pools
        self.resource_pools: Dict[str, ResourcePool] = {}
        
    async def start(self):
        """Start the async optimizer"""
        await self.task_queue.start()
        logger.info("Async optimizer started")
    
    async def stop(self):
        """Stop the async optimizer"""
        await self.task_queue.stop()
        
        # Shutdown thread pools
        self.cpu_pool.shutdown(wait=True)
        self.io_pool.shutdown(wait=True)
        
        logger.info("Async optimizer stopped")
    
    def create_resource_pool(self, name: str, resources: List[Any], max_concurrent: int = None) -> ResourcePool:
        """Create a named resource pool"""
        pool = ResourcePool(resources, max_concurrent)
        self.resource_pools[name] = pool
        return pool
    
    def get_resource_pool(self, name: str) -> Optional[ResourcePool]:
        """Get named resource pool"""
        return self.resource_pools.get(name)
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to queue"""
        return await self.task_queue.submit(func, *args, **kwargs)
    
    async def submit_cpu_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit CPU-bound task to process pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.cpu_pool, func, *args, **kwargs)
    
    async def submit_io_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit I/O-bound task to thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, func, *args, **kwargs)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'task_queue': self.task_queue.get_stats(),
            'resource_pools': {
                name: pool.get_stats() 
                for name, pool in self.resource_pools.items()
            },
            'thread_pools': {
                'cpu_pool': {
                    'max_workers': getattr(self.cpu_pool, '_max_workers', 'unknown'),
                    'active_workers': getattr(self.cpu_pool, '_processes', {}).get('qsize', 0) if hasattr(self.cpu_pool, '_processes') else 0,
                    'type': 'ProcessPoolExecutor'
                },
                'io_pool': {
                    'max_workers': getattr(self.io_pool, '_max_workers', 'unknown'),
                    'threads': len(getattr(self.io_pool, '_threads', [])),
                    'type': 'ThreadPoolExecutor'
                }
            }
        }


# Global async optimizer instance
async_optimizer: Optional[AsyncOptimizer] = None


def get_async_optimizer() -> AsyncOptimizer:
    """Get or create global async optimizer"""
    global async_optimizer
    if async_optimizer is None:
        async_optimizer = AsyncOptimizer()
    return async_optimizer


async def initialize_async_optimizer():
    """Initialize the global async optimizer"""
    global async_optimizer
    async_optimizer = get_async_optimizer()
    await async_optimizer.start()


async def shutdown_async_optimizer():
    """Shutdown the global async optimizer"""
    global async_optimizer
    if async_optimizer:
        await async_optimizer.stop()