"""
Resource Management and Optimization Service
Memory pools, connection pooling, and resource optimization
"""

import asyncio
import gc
import psutil
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import asyncpg
import redis
import aiohttp
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    memory_used_mb: float
    memory_available_mb: float
    cpu_percent: float
    connections_active: int
    connections_idle: int
    cache_size_mb: float
    thread_count: int
    timestamp: float


class MemoryPool:
    """
    Efficient memory pool management for vector operations
    """
    
    def __init__(self, pool_size_mb: int = 2048):
        self.pool_size_mb = pool_size_mb
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.allocated_blocks = {}
        self.free_blocks = deque()
        self.block_size = 1024 * 1024  # 1MB blocks
        self.total_blocks = pool_size_mb
        self.lock = threading.Lock()
        
        # Pre-allocate memory blocks
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Pre-allocate memory blocks"""
        for i in range(self.total_blocks):
            block = np.zeros(self.block_size // 8, dtype=np.float64)  # 8 bytes per float64
            self.free_blocks.append((i, block))
        
        logger.info(f"Memory pool initialized with {self.pool_size_mb}MB ({self.total_blocks} blocks)")
    
    def allocate(self, size_mb: float) -> Optional[List[np.ndarray]]:
        """Allocate memory from pool"""
        blocks_needed = int(np.ceil(size_mb))
        
        with self.lock:
            if len(self.free_blocks) < blocks_needed:
                logger.warning(f"Memory pool exhausted: {blocks_needed} blocks requested, {len(self.free_blocks)} available")
                return None
            
            allocated = []
            allocation_id = id(allocated)
            
            for _ in range(blocks_needed):
                block_id, block = self.free_blocks.popleft()
                allocated.append(block)
                
                if allocation_id not in self.allocated_blocks:
                    self.allocated_blocks[allocation_id] = []
                self.allocated_blocks[allocation_id].append((block_id, block))
            
            logger.debug(f"Allocated {blocks_needed} blocks from memory pool")
            return allocated
    
    def deallocate(self, allocation_id: int):
        """Return memory to pool"""
        with self.lock:
            if allocation_id in self.allocated_blocks:
                blocks = self.allocated_blocks.pop(allocation_id)
                for block_id, block in blocks:
                    # Clear memory
                    block.fill(0)
                    self.free_blocks.append((block_id, block))
                
                logger.debug(f"Deallocated {len(blocks)} blocks to memory pool")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory pool usage statistics"""
        with self.lock:
            return {
                "total_blocks": self.total_blocks,
                "allocated_blocks": sum(len(blocks) for blocks in self.allocated_blocks.values()),
                "free_blocks": len(self.free_blocks),
                "usage_percent": (1 - len(self.free_blocks) / self.total_blocks) * 100,
                "total_size_mb": self.pool_size_mb
            }


class ConnectionPool:
    """
    Advanced connection pooling for databases and services
    """
    
    def __init__(self):
        self.pools = {}
        self.metrics = {}
        self.lock = asyncio.Lock()
        
    async def initialize_pools(self):
        """Initialize all connection pools"""
        
        # PostgreSQL pool
        self.pools['postgresql'] = await asyncpg.create_pool(
            'postgresql://knowledgehub:password@localhost:5433/knowledgehub',
            min_size=10,
            max_size=50,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=10
        )
        
        # Redis connection pool
        self.pools['redis'] = redis.ConnectionPool(
            host='localhost',
            port=6381,
            max_connections=30,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 5   # TCP_KEEPCNT
            }
        )
        
        # HTTP connection pool
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30
        )
        self.pools['http'] = aiohttp.ClientSession(connector=connector)
        
        logger.info("Connection pools initialized")
    
    @asynccontextmanager
    async def get_connection(self, pool_name: str):
        """Get connection from pool with automatic release"""
        async with self.lock:
            if pool_name not in self.metrics:
                self.metrics[pool_name] = {
                    "connections_created": 0,
                    "connections_reused": 0,
                    "total_queries": 0
                }
        
        if pool_name == 'postgresql':
            async with self.pools['postgresql'].acquire() as conn:
                self.metrics[pool_name]["connections_reused"] += 1
                self.metrics[pool_name]["total_queries"] += 1
                yield conn
                
        elif pool_name == 'redis':
            client = redis.Redis(connection_pool=self.pools['redis'])
            self.metrics[pool_name]["connections_reused"] += 1
            yield client
            
        elif pool_name == 'http':
            self.metrics[pool_name]["connections_reused"] += 1
            yield self.pools['http']
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        stats = {}
        
        # PostgreSQL stats
        if 'postgresql' in self.pools:
            pg_pool = self.pools['postgresql']
            stats['postgresql'] = {
                "size": pg_pool.get_size(),
                "free": pg_pool.get_idle_size(),
                "used": pg_pool.get_size() - pg_pool.get_idle_size(),
                "max_size": pg_pool.get_max_size()
            }
        
        # Redis stats
        if 'redis' in self.pools:
            redis_pool = self.pools['redis']
            stats['redis'] = {
                "created_connections": redis_pool.created_connections,
                "available_connections": len(redis_pool._available_connections),
                "in_use_connections": len(redis_pool._in_use_connections),
                "max_connections": redis_pool.max_connections
            }
        
        # Add metrics
        stats['metrics'] = self.metrics
        
        return stats
    
    async def cleanup(self):
        """Clean up all connection pools"""
        if 'postgresql' in self.pools:
            await self.pools['postgresql'].close()
        
        if 'http' in self.pools:
            await self.pools['http'].close()
        
        logger.info("Connection pools cleaned up")


class GarbageCollectionOptimizer:
    """
    Optimized garbage collection for Python memory management
    """
    
    def __init__(self):
        self.gc_stats = []
        self.last_gc_time = time.time()
        self.gc_interval = 60  # seconds
        self.gc_threshold_mb = 500  # Trigger GC if memory growth exceeds this
        self.baseline_memory = None
        
        # Configure GC
        self._configure_gc()
    
    def _configure_gc(self):
        """Configure garbage collection settings"""
        # Set collection thresholds (generation 0, 1, 2)
        gc.set_threshold(700, 10, 10)
        
        # Disable automatic GC for controlled collection
        gc.disable()
        
        # Get baseline memory
        self.baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        logger.info("Garbage collection configured with custom thresholds")
    
    def should_collect(self) -> bool:
        """Determine if garbage collection should run"""
        current_time = time.time()
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Check time interval
        if current_time - self.last_gc_time > self.gc_interval:
            return True
        
        # Check memory growth
        if self.baseline_memory and (current_memory - self.baseline_memory) > self.gc_threshold_mb:
            return True
        
        return False
    
    def collect(self) -> Dict[str, Any]:
        """Perform optimized garbage collection"""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Run garbage collection
        collected = {
            "generation_0": gc.collect(0),
            "generation_1": gc.collect(1),
            "generation_2": gc.collect(2)
        }
        
        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        duration = (time.time() - start_time) * 1000
        
        stats = {
            "timestamp": time.time(),
            "duration_ms": duration,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_freed_mb": memory_before - memory_after,
            "objects_collected": collected,
            "total_objects_collected": sum(collected.values())
        }
        
        self.gc_stats.append(stats)
        self.last_gc_time = time.time()
        self.baseline_memory = memory_after
        
        logger.info(f"GC collected {stats['total_objects_collected']} objects, freed {stats['memory_freed_mb']:.2f}MB")
        
        return stats
    
    async def automatic_collection_loop(self):
        """Run automatic garbage collection loop"""
        while True:
            if self.should_collect():
                self.collect()
            
            await asyncio.sleep(10)  # Check every 10 seconds


class CPUOptimizer:
    """
    CPU optimization with thread pool management and affinity
    """
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.thread_pools = {}
        self.cpu_affinity = None
        
        # Configure CPU optimization
        self._configure_cpu()
    
    def _configure_cpu(self):
        """Configure CPU optimization settings"""
        # Set process priority (nice value)
        try:
            import os
            os.nice(0)  # Normal priority
        except:
            pass
        
        # Configure thread pools
        self.thread_pools['compute'] = ThreadPoolExecutor(
            max_workers=self.cpu_count,
            thread_name_prefix='compute'
        )
        
        self.thread_pools['io'] = ThreadPoolExecutor(
            max_workers=self.cpu_count * 2,
            thread_name_prefix='io'
        )
        
        logger.info(f"CPU optimization configured with {self.cpu_count} cores")
    
    def get_optimal_worker_count(self, task_type: str) -> int:
        """Get optimal worker count for task type"""
        if task_type == 'cpu_bound':
            return self.cpu_count
        elif task_type == 'io_bound':
            return self.cpu_count * 2
        elif task_type == 'memory_bound':
            return max(2, self.cpu_count // 2)
        else:
            return self.cpu_count
    
    async def execute_parallel(self, func, items: List[Any], 
                              task_type: str = 'cpu_bound') -> List[Any]:
        """Execute function in parallel with optimal configuration"""
        worker_count = self.get_optimal_worker_count(task_type)
        pool = self.thread_pools.get('compute' if task_type == 'cpu_bound' else 'io')
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(pool, func, item)
            for item in items
        ]
        
        results = await asyncio.gather(*futures)
        return results
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU usage statistics"""
        return {
            "cpu_count": self.cpu_count,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_percent_per_core": psutil.cpu_percent(interval=1, percpu=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "thread_pools": {
                name: {
                    "max_workers": pool._max_workers,
                    "threads": len(pool._threads)
                }
                for name, pool in self.thread_pools.items()
            }
        }


class ResourceManager:
    """
    Comprehensive resource management orchestrator
    """
    
    def __init__(self):
        self.memory_pool = MemoryPool(pool_size_mb=2048)
        self.connection_pool = ConnectionPool()
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.metrics_history = deque(maxlen=1000)
        self.resource_limits = {
            "max_memory_percent": 80,
            "max_cpu_percent": 85,
            "max_connections": 200
        }
    
    async def initialize(self):
        """Initialize all resource management components"""
        await self.connection_pool.initialize_pools()
        
        # Start automatic GC loop
        asyncio.create_task(self.gc_optimizer.automatic_collection_loop())
        
        # Start resource monitoring
        asyncio.create_task(self.monitor_resources())
        
        logger.info("Resource manager initialized")
    
    async def monitor_resources(self):
        """Continuously monitor resource usage"""
        while True:
            metrics = await self.collect_metrics()
            self.metrics_history.append(metrics)
            
            # Check resource limits
            if metrics.memory_used_mb / (metrics.memory_used_mb + metrics.memory_available_mb) * 100 > self.resource_limits["max_memory_percent"]:
                logger.warning(f"Memory usage high: {metrics.memory_used_mb:.2f}MB")
                self.gc_optimizer.collect()
            
            if metrics.cpu_percent > self.resource_limits["max_cpu_percent"]:
                logger.warning(f"CPU usage high: {metrics.cpu_percent:.1f}%")
            
            await asyncio.sleep(30)  # Monitor every 30 seconds
    
    async def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get connection stats
        conn_stats = await self.connection_pool.get_pool_stats()
        active_connections = sum(
            stats.get('used', 0) + stats.get('in_use_connections', 0)
            for stats in conn_stats.values()
            if isinstance(stats, dict)
        )
        
        idle_connections = sum(
            stats.get('free', 0) + stats.get('available_connections', 0)
            for stats in conn_stats.values()
            if isinstance(stats, dict)
        )
        
        return ResourceMetrics(
            memory_used_mb=memory_info.rss / (1024 * 1024),
            memory_available_mb=psutil.virtual_memory().available / (1024 * 1024),
            cpu_percent=process.cpu_percent(),
            connections_active=active_connections,
            connections_idle=idle_connections,
            cache_size_mb=0,  # Would calculate actual cache size
            thread_count=process.num_threads(),
            timestamp=time.time()
        )
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """Perform resource optimization"""
        optimizations = {
            "timestamp": time.time(),
            "actions": []
        }
        
        # Memory optimization
        memory_stats = self.memory_pool.get_usage_stats()
        if memory_stats["usage_percent"] > 90:
            # Trigger memory cleanup
            gc_stats = self.gc_optimizer.collect()
            optimizations["actions"].append({
                "type": "garbage_collection",
                "freed_mb": gc_stats["memory_freed_mb"]
            })
        
        # Connection optimization
        conn_stats = await self.connection_pool.get_pool_stats()
        for pool_name, stats in conn_stats.items():
            if isinstance(stats, dict) and stats.get("used", 0) > stats.get("max_size", 100) * 0.8:
                optimizations["actions"].append({
                    "type": "connection_pool_expansion",
                    "pool": pool_name
                })
        
        # CPU optimization
        cpu_stats = self.cpu_optimizer.get_cpu_stats()
        if cpu_stats["cpu_percent"] > 80:
            optimizations["actions"].append({
                "type": "cpu_throttling",
                "current_usage": cpu_stats["cpu_percent"]
            })
        
        return optimizations
    
    async def get_resource_report(self) -> Dict[str, Any]:
        """Generate comprehensive resource usage report"""
        current_metrics = await self.collect_metrics()
        
        return {
            "current": {
                "memory_used_mb": current_metrics.memory_used_mb,
                "memory_available_mb": current_metrics.memory_available_mb,
                "cpu_percent": current_metrics.cpu_percent,
                "connections_active": current_metrics.connections_active,
                "connections_idle": current_metrics.connections_idle,
                "thread_count": current_metrics.thread_count
            },
            "memory_pool": self.memory_pool.get_usage_stats(),
            "connection_pools": await self.connection_pool.get_pool_stats(),
            "cpu": self.cpu_optimizer.get_cpu_stats(),
            "gc_stats": self.gc_optimizer.gc_stats[-5:] if self.gc_optimizer.gc_stats else [],
            "recommendations": await self._generate_recommendations()
        }
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate resource optimization recommendations"""
        recommendations = []
        
        current_metrics = await self.collect_metrics()
        
        # Memory recommendations
        memory_usage = current_metrics.memory_used_mb / (current_metrics.memory_used_mb + current_metrics.memory_available_mb) * 100
        if memory_usage > 70:
            recommendations.append(f"Consider increasing memory allocation (current usage: {memory_usage:.1f}%)")
        
        # Connection recommendations
        if current_metrics.connections_active > 150:
            recommendations.append(f"High connection count ({current_metrics.connections_active}), consider connection pooling optimization")
        
        # CPU recommendations
        if current_metrics.cpu_percent > 70:
            recommendations.append(f"High CPU usage ({current_metrics.cpu_percent:.1f}%), consider scaling horizontally")
        
        return recommendations


async def demo_resource_management():
    """Demonstrate resource management capabilities"""
    print("=" * 60)
    print("🔧 Resource Management Demo")
    print("=" * 60)
    
    manager = ResourceManager()
    await manager.initialize()
    
    # Test memory pool
    print("\n📊 Memory Pool Test:")
    allocation = manager.memory_pool.allocate(10)  # 10MB
    if allocation:
        print(f"  ✅ Allocated {len(allocation)} blocks")
        manager.memory_pool.deallocate(id(allocation))
        print(f"  ✅ Deallocated blocks")
    
    stats = manager.memory_pool.get_usage_stats()
    print(f"  Pool usage: {stats['usage_percent']:.1f}%")
    
    # Test connection pool
    print("\n📊 Connection Pool Test:")
    async with manager.connection_pool.get_connection('postgresql') as conn:
        result = await conn.fetchval("SELECT COUNT(*) FROM memories")
        print(f"  ✅ PostgreSQL query result: {result}")
    
    pool_stats = await manager.connection_pool.get_pool_stats()
    print(f"  Connection pools: {list(pool_stats.keys())}")
    
    # Test CPU optimization
    print("\n📊 CPU Optimization Test:")
    
    def cpu_task(n):
        return sum(i**2 for i in range(n))
    
    items = [10000] * 10
    results = await manager.cpu_optimizer.execute_parallel(cpu_task, items, 'cpu_bound')
    print(f"  ✅ Parallel execution completed: {len(results)} tasks")
    
    # Get resource report
    print("\n📊 Resource Report:")
    report = await manager.get_resource_report()
    print(f"  Memory: {report['current']['memory_used_mb']:.2f}MB used")
    print(f"  CPU: {report['current']['cpu_percent']:.1f}%")
    print(f"  Connections: {report['current']['connections_active']} active")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    # Cleanup
    await manager.connection_pool.cleanup()
    
    print("\n✅ Resource management demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_resource_management())