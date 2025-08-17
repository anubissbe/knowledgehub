
import asyncio
from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor

class AsyncUtils:
    @staticmethod
    async def run_in_executor(func: Callable, *args) -> Any:
        """Run blocking function in executor"""
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        return await loop.run_in_executor(executor, func, *args)
    
    @staticmethod
    async def gather_with_limit(tasks: List, limit: int = 10):
        """Run tasks with concurrency limit"""
        semaphore = asyncio.Semaphore(limit)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        return await asyncio.gather(*[bounded_task(task) for task in tasks])
