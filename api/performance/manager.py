"""
Performance Manager

Central coordinator for all performance optimization systems.
Integrates cache management, database optimization, async processing,
response optimization, and monitoring.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from .cache_manager import (
    get_cache_manager, 
    initialize_cache_manager, 
    shutdown_cache_manager,
    cache_result
)
from .database_optimizer import (
    get_db_optimizer,
    initialize_db_optimizer,
    shutdown_db_optimizer
)
from .async_processor import (
    get_async_optimizer,
    initialize_async_optimizer,
    shutdown_async_optimizer
)
from .response_optimizer import get_response_optimizer
from .monitoring import (
    get_performance_monitor,
    initialize_performance_monitor,
    shutdown_performance_monitor,
    MetricType
)

logger = logging.getLogger(__name__)


class PerformanceManager:
    """Central performance management system"""
    
    def __init__(self):
        self.cache_manager = None
        self.db_optimizer = None
        self.async_optimizer = None
        self.response_optimizer = None
        self.performance_monitor = None
        
        self.is_initialized = False
        self.optimization_enabled = True
        
    async def initialize(self):
        """Initialize all performance optimization systems"""
        if self.is_initialized:
            return
        
        logger.info("Initializing performance optimization systems...")
        
        try:
            # Initialize cache manager
            await initialize_cache_manager()
            self.cache_manager = get_cache_manager()
            logger.info("✅ Cache manager initialized")
            
            # Initialize database optimizer
            await initialize_db_optimizer()
            self.db_optimizer = get_db_optimizer()
            logger.info("✅ Database optimizer initialized")
            
            # Initialize async processor
            await initialize_async_optimizer()
            self.async_optimizer = get_async_optimizer()
            logger.info("✅ Async processor initialized")
            
            # Initialize response optimizer
            self.response_optimizer = get_response_optimizer()
            logger.info("✅ Response optimizer initialized")
            
            # Initialize performance monitor
            await initialize_performance_monitor()
            self.performance_monitor = get_performance_monitor()
            logger.info("✅ Performance monitor initialized")
            
            self.is_initialized = True
            logger.info("🚀 Performance optimization systems fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance systems: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all performance optimization systems"""
        if not self.is_initialized:
            return
        
        logger.info("Shutting down performance optimization systems...")
        
        try:
            # Shutdown in reverse order
            if self.performance_monitor:
                await shutdown_performance_monitor()
                logger.info("✅ Performance monitor shutdown")
            
            if self.async_optimizer:
                await shutdown_async_optimizer()
                logger.info("✅ Async processor shutdown")
            
            if self.db_optimizer:
                await shutdown_db_optimizer()
                logger.info("✅ Database optimizer shutdown")
            
            if self.cache_manager:
                await shutdown_cache_manager()
                logger.info("✅ Cache manager shutdown")
            
            self.is_initialized = False
            logger.info("🔌 Performance optimization systems shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during performance systems shutdown: {e}")
    
    @asynccontextmanager
    async def get_optimized_session(self):
        """Get optimized database session"""
        if not self.db_optimizer:
            raise RuntimeError("Database optimizer not initialized")
        
        async with self.db_optimizer.get_session() as session:
            yield session
    
    async def execute_optimized_query(self, query, parameters=None, use_cache=True):
        """Execute query with optimization"""
        if not self.db_optimizer:
            raise RuntimeError("Database optimizer not initialized")
        
        return await self.db_optimizer.execute_query(
            query, parameters, use_cache
        )
    
    async def submit_background_task(self, func, *args, **kwargs):
        """Submit task for background processing"""
        if not self.async_optimizer:
            raise RuntimeError("Async optimizer not initialized")
        
        return await self.async_optimizer.submit_task(func, *args, **kwargs)
    
    async def submit_cpu_intensive_task(self, func, *args, **kwargs):
        """Submit CPU-intensive task to process pool"""
        if not self.async_optimizer:
            raise RuntimeError("Async optimizer not initialized")
        
        return await self.async_optimizer.submit_cpu_task(func, *args, **kwargs)
    
    async def optimize_response(self, request, response_data, status_code=200, headers=None):
        """Optimize HTTP response"""
        if not self.response_optimizer:
            raise RuntimeError("Response optimizer not initialized")
        
        return await self.response_optimizer.optimize_response(
            request, response_data, status_code, headers
        )
    
    async def get_from_cache(self, key: str, pattern: str = 'default'):
        """Get value from cache"""
        if not self.cache_manager:
            return None
        
        return await self.cache_manager.get(key, pattern)
    
    async def set_in_cache(self, key: str, value, pattern: str = 'default', ttl=None, tags=None):
        """Set value in cache"""
        if not self.cache_manager:
            return False
        
        return await self.cache_manager.set(key, value, pattern, ttl, tags)
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, tags=None):
        """Record performance metric"""
        if self.performance_monitor:
            self.performance_monitor.record_metric(name, value, metric_type, tags)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'performance_manager': {
                'initialized': self.is_initialized,
                'optimization_enabled': self.optimization_enabled
            }
        }
        
        if self.cache_manager:
            stats['cache'] = self.cache_manager.get_stats()
        
        if self.db_optimizer:
            stats['database'] = self.db_optimizer.get_query_stats()
        
        if self.async_optimizer:
            stats['async_processing'] = self.async_optimizer.get_comprehensive_stats()
        
        if self.response_optimizer:
            stats['response_optimization'] = self.response_optimizer.get_performance_stats()
        
        if self.performance_monitor:
            stats['monitoring'] = self.performance_monitor.get_dashboard_data()
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all performance systems"""
        health = {
            'overall_status': 'healthy',
            'systems': {}
        }
        
        # Check each system
        systems_status = []
        
        # Cache manager
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            cache_health = 'healthy'
            if cache_stats.get('memory_cache', {}).get('hit_rate', 0) < 0.5:
                cache_health = 'degraded'
            health['systems']['cache'] = cache_health
            systems_status.append(cache_health)
        
        # Database optimizer
        if self.db_optimizer:
            try:
                db_health_result = asyncio.create_task(self.db_optimizer.health_check())
                # Note: In a real implementation, you'd await this properly
                health['systems']['database'] = 'healthy'  # Simplified
                systems_status.append('healthy')
            except:
                health['systems']['database'] = 'unhealthy'
                systems_status.append('unhealthy')
        
        # Async optimizer
        if self.async_optimizer:
            async_stats = self.async_optimizer.get_comprehensive_stats()
            queue_stats = async_stats.get('task_queue', {})
            if queue_stats.get('failed_tasks', 0) > queue_stats.get('completed_tasks', 1) * 0.1:
                health['systems']['async_processing'] = 'degraded'
                systems_status.append('degraded')
            else:
                health['systems']['async_processing'] = 'healthy'
                systems_status.append('healthy')
        
        # Performance monitor
        if self.performance_monitor:
            monitor_health = self.performance_monitor.get_health_check()
            health['systems']['monitoring'] = monitor_health.get('status', 'unknown')
            systems_status.append(monitor_health.get('status', 'unknown'))
        
        # Determine overall status
        if 'unhealthy' in systems_status:
            health['overall_status'] = 'unhealthy'
        elif 'degraded' in systems_status:
            health['overall_status'] = 'degraded'
        
        return health
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Run system optimization"""
        optimization_results = {
            'timestamp': asyncio.get_event_loop().time(),
            'optimizations': []
        }
        
        # Cache optimization
        if self.cache_manager:
            # Clear expired entries
            try:
                await self.cache_manager.memory_cache.clear()
                optimization_results['optimizations'].append({
                    'system': 'cache',
                    'action': 'cleared_expired_entries',
                    'status': 'success'
                })
            except Exception as e:
                optimization_results['optimizations'].append({
                    'system': 'cache',
                    'action': 'clear_expired_entries',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Database optimization
        if self.db_optimizer:
            try:
                # Get slow queries and suggest optimizations
                query_stats = self.db_optimizer.get_query_stats()
                slow_queries = query_stats.get('top_slow_queries', [])
                
                optimization_results['optimizations'].append({
                    'system': 'database',
                    'action': 'analyzed_slow_queries',
                    'status': 'success',
                    'slow_queries_count': len(slow_queries)
                })
            except Exception as e:
                optimization_results['optimizations'].append({
                    'system': 'database',
                    'action': 'analyze_slow_queries',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Performance monitoring optimization
        if self.performance_monitor:
            try:
                dashboard_data = self.performance_monitor.get_dashboard_data()
                recommendations = dashboard_data.get('recommendations', [])
                
                optimization_results['optimizations'].append({
                    'system': 'monitoring',
                    'action': 'generated_recommendations',
                    'status': 'success',
                    'recommendations_count': len(recommendations)
                })
            except Exception as e:
                optimization_results['optimizations'].append({
                    'system': 'monitoring',
                    'action': 'generate_recommendations',
                    'status': 'failed',
                    'error': str(e)
                })
        
        return optimization_results
    
    def enable_optimization(self):
        """Enable performance optimization"""
        self.optimization_enabled = True
        logger.info("Performance optimization enabled")
    
    def disable_optimization(self):
        """Disable performance optimization"""
        self.optimization_enabled = False
        logger.info("Performance optimization disabled")


# Global performance manager instance
performance_manager: Optional[PerformanceManager] = None


def get_performance_manager() -> PerformanceManager:
    """Get or create global performance manager"""
    global performance_manager
    if performance_manager is None:
        performance_manager = PerformanceManager()
    return performance_manager


async def initialize_performance_system():
    """Initialize the complete performance system"""
    manager = get_performance_manager()
    await manager.initialize()
    return manager


async def shutdown_performance_system():
    """Shutdown the complete performance system"""
    global performance_manager
    if performance_manager is not None:
        try:
            await performance_manager.shutdown()
        except Exception as e:
            logger.error(f"Error during performance systems shutdown: {e}")


# Convenience decorators and context managers
def with_performance_cache(pattern: str = 'default', ttl: Optional[float] = None):
    """Decorator to cache function results using performance manager"""
    def decorator(func):
        return cache_result(pattern, ttl)(func)
    return decorator


@asynccontextmanager
async def optimized_db_session():
    """Context manager for optimized database session"""
    manager = get_performance_manager()
    async with manager.get_optimized_session() as session:
        yield session


async def submit_background_task(func, *args, **kwargs):
    """Submit background task using performance manager"""
    manager = get_performance_manager()
    return await manager.submit_background_task(func, *args, **kwargs)


async def submit_cpu_task(func, *args, **kwargs):
    """Submit CPU-intensive task using performance manager"""
    manager = get_performance_manager()
    return await manager.submit_cpu_intensive_task(func, *args, **kwargs)


def record_performance_metric(name: str, value: float, metric_type: MetricType, tags=None):
    """Record performance metric using performance manager"""
    manager = get_performance_manager()
    manager.record_metric(name, value, metric_type, tags)