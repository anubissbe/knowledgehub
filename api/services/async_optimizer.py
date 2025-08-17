
import asyncio
from typing import List, Any, Callable, TypeVar, Optional
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T')

class AsyncOptimizer:
    """Optimizations for async operations"""
    
    def __init__(self):
        self.semaphore = asyncio.Semaphore(100)  # Limit concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=10)  # For CPU-bound tasks
    
    async def gather_with_limit(
        self,
        tasks: List[Callable],
        limit: int = 10
    ) -> List[Any]:
        """Execute tasks with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def limited_task(task):
            async with semaphore:
                return await task()
        
        return await asyncio.gather(
            *[limited_task(task) for task in tasks],
            return_exceptions=True
        )
    
    async def batch_process(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        batch_size: int = 50
    ) -> List[Any]:
        """Process items in batches"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[processor(item) for item in batch]
            )
            results.extend(batch_results)
        
        return results
    
    def timeout_handler(timeout: int = 30):
        """Decorator for handling timeouts"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"{func.__name__} timed out after {timeout}s")
                    return None
            return wrapper
        return decorator
    
    async def run_cpu_bound(self, func: Callable, *args) -> Any:
        """Run CPU-bound task in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

async_optimizer = AsyncOptimizer()

# Usage example:
# @async_optimizer.timeout_handler(timeout=10)
# async def slow_operation():
#     await asyncio.sleep(5)
#     return "completed"
