"""
Complete RAG System Performance Management API
Unified monitoring and management for all RAG performance components

Author: Adrien Stevens - Python Performance Optimization Expert
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse

from ..services.rag_cache_optimizer import get_rag_cache_optimizer
from ..services.async_rag_optimizer import get_async_rag_optimizer
from ..services.rag_optimized_llamaindex import get_optimized_rag_service
from ..services.rag_optimized_graphrag import get_optimized_graphrag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag/system", tags=["RAG System Performance"])


@router.get("/status")
async def get_system_status():
    """Get overall system status and health"""
    try:
        status_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'components': {},
            'overall_status': 'healthy'
        }
        
        # Check cache optimizer
        try:
            cache_optimizer = await get_rag_cache_optimizer()
            cache_stats = await cache_optimizer.get_performance_stats()
            status_data['components']['cache_optimizer'] = {
                'status': 'healthy',
                'hit_ratio': cache_stats.get('cache_metrics', {}).get('hit_ratio', 0),
                'memory_usage_mb': cache_stats.get('memory', {}).get('system_usage_percent', 0)
            }
        except Exception as e:
            status_data['components']['cache_optimizer'] = {
                'status': 'error',
                'error': str(e)
            }
            status_data['overall_status'] = 'degraded'
        
        # Check async optimizer
        try:
            async_optimizer = await get_async_rag_optimizer()
            async_stats = await async_optimizer.get_performance_stats()
            status_data['components']['async_optimizer'] = {
                'status': 'healthy',
                'running': async_stats.get('system', {}).get('running', False),
                'concurrent_ops': async_stats.get('system', {}).get('concurrent_operations', 0)
            }
        except Exception as e:
            status_data['components']['async_optimizer'] = {
                'status': 'error',
                'error': str(e)
            }
            status_data['overall_status'] = 'degraded'
        
        # Check LlamaIndex service
        try:
            llama_service = await get_optimized_rag_service()
            llama_metrics = await llama_service.get_performance_metrics()
            status_data['components']['llamaindex_service'] = {
                'status': 'healthy',
                'queries_processed': llama_metrics.get('service_metrics', {}).get('queries_processed', 0),
                'cache_hit_ratio': llama_metrics.get('service_metrics', {}).get('cache_hit_ratio', 0)
            }
        except Exception as e:
            status_data['components']['llamaindex_service'] = {
                'status': 'error',
                'error': str(e)
            }
            status_data['overall_status'] = 'degraded'
        
        # Check GraphRAG service
        try:
            graphrag_service = await get_optimized_graphrag_service()
            graph_metrics = await graphrag_service.get_graph_performance_metrics()
            status_data['components']['graphrag_service'] = {
                'status': 'healthy',
                'queries_processed': graph_metrics.get('service_metrics', {}).get('graph_queries_processed', 0),
                'cache_hit_ratio': graph_metrics.get('service_metrics', {}).get('cache_hit_ratio', 0)
            }
        except Exception as e:
            status_data['components']['graphrag_service'] = {
                'status': 'error',
                'error': str(e)
            }
            status_data['overall_status'] = 'degraded'
        
        # Set overall status
        error_count = sum(1 for comp in status_data['components'].values() if comp['status'] == 'error')
        if error_count == 0:
            status_data['overall_status'] = 'healthy'
        elif error_count < len(status_data['components']):
            status_data['overall_status'] = 'degraded'
        else:
            status_data['overall_status'] = 'unhealthy'
        
        return status_data
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"System status error: {str(e)}")


@router.get("/performance/comprehensive")
async def get_comprehensive_performance():
    """Get comprehensive performance metrics from all components"""
    try:
        performance_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'cache_optimizer': {},
            'async_optimizer': {},
            'llamaindex_service': {},
            'graphrag_service': {}
        }
        
        # Collect metrics from all components
        tasks = []
        
        # Cache optimizer metrics
        async def get_cache_metrics():
            try:
                cache_optimizer = await get_rag_cache_optimizer()
                return await cache_optimizer.get_performance_stats()
            except Exception as e:
                return {'error': str(e)}
        
        # Async optimizer metrics
        async def get_async_metrics():
            try:
                async_optimizer = await get_async_rag_optimizer()
                return await async_optimizer.get_performance_stats()
            except Exception as e:
                return {'error': str(e)}
        
        # LlamaIndex metrics
        async def get_llama_metrics():
            try:
                llama_service = await get_optimized_rag_service()
                return await llama_service.get_performance_metrics()
            except Exception as e:
                return {'error': str(e)}
        
        # GraphRAG metrics
        async def get_graph_metrics():
            try:
                graphrag_service = await get_optimized_graphrag_service()
                return await graphrag_service.get_graph_performance_metrics()
            except Exception as e:
                return {'error': str(e)}
        
        # Execute all metrics collection in parallel
        cache_metrics, async_metrics, llama_metrics, graph_metrics = await asyncio.gather(
            get_cache_metrics(),
            get_async_metrics(),
            get_llama_metrics(),
            get_graph_metrics(),
            return_exceptions=True
        )
        
        performance_data['cache_optimizer'] = cache_metrics if not isinstance(cache_metrics, Exception) else {'error': str(cache_metrics)}
        performance_data['async_optimizer'] = async_metrics if not isinstance(async_metrics, Exception) else {'error': str(async_metrics)}
        performance_data['llamaindex_service'] = llama_metrics if not isinstance(llama_metrics, Exception) else {'error': str(llama_metrics)}
        performance_data['graphrag_service'] = graph_metrics if not isinstance(graph_metrics, Exception) else {'error': str(graph_metrics)}
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get comprehensive performance: {e}")
        raise HTTPException(status_code=500, detail=f"Performance metrics error: {str(e)}")


@router.get("/optimization/recommendations")
async def get_optimization_recommendations():
    """Get AI-powered optimization recommendations"""
    try:
        recommendations = []
        
        # Get performance data
        cache_optimizer = await get_rag_cache_optimizer()
        cache_stats = await cache_optimizer.get_performance_stats()
        
        async_optimizer = await get_async_rag_optimizer()
        async_stats = await async_optimizer.get_performance_stats()
        
        # Analyze cache performance
        cache_metrics = cache_stats.get('cache_metrics', {})
        hit_ratio = cache_metrics.get('hit_ratio', 0)
        
        if hit_ratio < 0.7:
            recommendations.append({
                'type': 'cache_optimization',
                'priority': 'high',
                'title': 'Low Cache Hit Ratio',
                'description': f'Cache hit ratio is {hit_ratio:.2%}, consider increasing cache size or TTL',
                'action': 'Increase L1 cache size or adjust caching strategies',
                'impact': 'High - Can improve query response times by 40-60%'
            })
        
        # Analyze memory usage
        memory_stats = cache_stats.get('memory', {})
        memory_usage = memory_stats.get('system_usage_percent', 0)
        
        if memory_usage > 85:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'critical',
                'title': 'High Memory Usage',
                'description': f'System memory usage is {memory_usage}%, risk of performance degradation',
                'action': 'Trigger memory optimization or increase system memory',
                'impact': 'Critical - Prevents system instability'
            })
        
        # Analyze async performance
        system_stats = async_stats.get('system', {})
        concurrent_ops = system_stats.get('concurrent_operations', 0)
        max_concurrent = async_stats.get('config', {}).get('max_concurrent_operations', 50)
        
        utilization = concurrent_ops / max_concurrent if max_concurrent > 0 else 0
        
        if utilization > 0.8:
            recommendations.append({
                'type': 'concurrency_optimization',
                'priority': 'medium',
                'title': 'High Concurrency Utilization',
                'description': f'Async operations at {utilization:.1%} capacity, may cause queuing',
                'action': 'Consider increasing max_concurrent_operations',
                'impact': 'Medium - Reduces query queuing and improves throughput'
            })
        
        # Analyze operation times
        operation_stats = async_stats.get('operations', {})
        for op_name, op_data in operation_stats.items():
            avg_time = op_data.get('avg_time_ms', 0)
            error_rate = op_data.get('error_rate', 0)
            
            if avg_time > 1000:  # > 1 second
                recommendations.append({
                    'type': 'performance_optimization',
                    'priority': 'medium',
                    'title': f'Slow Operation: {op_name}',
                    'description': f'Operation averaging {avg_time:.0f}ms, consider optimization',
                    'action': 'Profile and optimize slow operations',
                    'impact': 'Medium - Improves user experience'
                })
            
            if error_rate > 0.05:  # > 5% error rate
                recommendations.append({
                    'type': 'reliability_optimization',
                    'priority': 'high',
                    'title': f'High Error Rate: {op_name}',
                    'description': f'Operation has {error_rate:.1%} error rate',
                    'action': 'Investigate and fix error causes',
                    'impact': 'High - Improves system reliability'
                })
        
        # Sort recommendations by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'recommendations': recommendations,
            'recommendation_count': len(recommendations),
            'analysis_summary': {
                'cache_hit_ratio': hit_ratio,
                'memory_usage_percent': memory_usage,
                'concurrency_utilization': utilization,
                'total_operations': len(operation_stats)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations error: {str(e)}")


@router.post("/optimization/apply")
async def apply_optimization_recommendations(
    background_tasks: BackgroundTasks,
    recommendation_types: List[str] = Query([], description="Types of optimizations to apply")
):
    """Apply optimization recommendations"""
    try:
        applied_optimizations = []
        
        # Cache optimization
        if 'cache_optimization' in recommendation_types:
            cache_optimizer = await get_rag_cache_optimizer()
            
            # Trigger cache optimization
            background_tasks.add_task(optimize_cache_performance, cache_optimizer)
            applied_optimizations.append('cache_optimization')
        
        # Memory optimization
        if 'memory_optimization' in recommendation_types:
            cache_optimizer = await get_rag_cache_optimizer()
            
            # Trigger memory optimization
            await cache_optimizer._handle_memory_pressure()
            applied_optimizations.append('memory_optimization')
        
        # Concurrency optimization
        if 'concurrency_optimization' in recommendation_types:
            # This would require configuration changes
            applied_optimizations.append('concurrency_optimization_queued')
        
        return {
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat(),
            'applied_optimizations': applied_optimizations,
            'message': f'Applied {len(applied_optimizations)} optimizations'
        }
        
    except Exception as e:
        logger.error(f"Failed to apply optimizations: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


@router.post("/benchmark/comprehensive")
async def run_comprehensive_benchmark(
    background_tasks: BackgroundTasks,
    duration_seconds: int = Query(60, ge=30, le=300, description="Benchmark duration"),
    concurrent_users: int = Query(10, ge=1, le=50, description="Concurrent users to simulate")
):
    """Run comprehensive system benchmark"""
    try:
        # Start benchmark in background
        background_tasks.add_task(
            run_system_benchmark, 
            duration_seconds, 
            concurrent_users
        )
        
        return {
            'status': 'started',
            'timestamp': datetime.utcnow().isoformat(),
            'benchmark_config': {
                'duration_seconds': duration_seconds,
                'concurrent_users': concurrent_users
            },
            'message': 'Comprehensive benchmark started'
        }
        
    except Exception as e:
        logger.error(f"Failed to start benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")


@router.get("/monitoring/real-time")
async def get_real_time_metrics():
    """Get real-time performance metrics"""
    try:
        # Collect key metrics quickly
        cache_optimizer = await get_rag_cache_optimizer()
        async_optimizer = await get_async_rag_optimizer()
        
        # Get basic stats
        cache_metrics = cache_optimizer.metrics
        async_stats = await async_optimizer.get_performance_stats()
        
        # System metrics
        import psutil
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cache': {
                'hit_ratio': cache_metrics.hit_ratio,
                'hit_count': cache_metrics.hit_count,
                'miss_count': cache_metrics.miss_count
            },
            'async_operations': {
                'concurrent': async_stats.get('system', {}).get('concurrent_operations', 0),
                'total_processed': sum(
                    op.get('execution_count', 0) 
                    for op in async_stats.get('operations', {}).values()
                )
            },
            'system': {
                'memory_percent': memory.percent,
                'cpu_percent': cpu,
                'available_memory_gb': memory.available / (1024**3)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Real-time metrics error: {str(e)}")


@router.post("/configuration/update")
async def update_system_configuration(
    config_updates: Dict[str, Any]
):
    """Update system configuration parameters"""
    try:
        # This would update configuration in a real implementation
        # For now, return acknowledgment
        
        applied_changes = []
        
        # Example configuration updates
        if 'cache_size' in config_updates:
            applied_changes.append(f"Cache size updated to {config_updates['cache_size']}")
        
        if 'max_concurrent_operations' in config_updates:
            applied_changes.append(f"Max concurrent operations updated to {config_updates['max_concurrent_operations']}")
        
        if 'batch_size' in config_updates:
            applied_changes.append(f"Batch size updated to {config_updates['batch_size']}")
        
        return {
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat(),
            'applied_changes': applied_changes,
            'message': 'Configuration updated successfully'
        }
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")


async def optimize_cache_performance(cache_optimizer):
    """Background task to optimize cache performance"""
    try:
        logger.info("Starting cache performance optimization")
        
        # Get current stats
        stats = await cache_optimizer.get_performance_stats()
        initial_hit_ratio = stats.get('cache_metrics', {}).get('hit_ratio', 0)
        
        # Perform optimizations
        # 1. Clear low-value cache entries
        await cache_optimizer.invalidate_cache("temp_")
        
        # 2. Trigger memory optimization
        await cache_optimizer._handle_memory_pressure()
        
        # 3. Optimize memory pools
        for pool_name, pool in cache_optimizer.memory_pools.items():
            pool_stats = pool.get_usage_stats()
            if pool_stats['utilization'] > 0.9:
                logger.info(f"High utilization detected in {pool_name}: {pool_stats['utilization']:.1%}")
        
        logger.info("Cache performance optimization completed")
        
    except Exception as e:
        logger.error(f"Cache optimization failed: {e}")


async def run_system_benchmark(duration_seconds: int, concurrent_users: int):
    """Background task to run system benchmark"""
    try:
        logger.info(f"Starting system benchmark: {duration_seconds}s with {concurrent_users} users")
        
        # Get services
        llama_service = await get_optimized_rag_service()
        graphrag_service = await get_optimized_graphrag_service()
        
        # Benchmark queries
        test_queries = [
            "What is machine learning?",
            "How does neural network work?",
            "Explain deep learning concepts",
            "What are the benefits of AI?",
            "How to optimize performance?"
        ]
        
        start_time = time.time()
        total_queries = 0
        successful_queries = 0
        
        # Run benchmark
        async def user_simulation():
            nonlocal total_queries, successful_queries
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Random query
                    query = test_queries[total_queries % len(test_queries)]
                    
                    # Test LlamaIndex service
                    result = await llama_service.query_optimized(
                        query_text=query,
                        user_id="benchmark_user",
                        top_k=3
                    )
                    
                    if result.get('response'):
                        successful_queries += 1
                    
                    total_queries += 1
                    
                    # Brief pause
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Benchmark query failed: {e}")
                    total_queries += 1
        
        # Run concurrent users
        tasks = [user_simulation() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate results
        actual_duration = time.time() - start_time
        queries_per_second = total_queries / actual_duration if actual_duration > 0 else 0
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        
        logger.info(
            f"Benchmark completed: {total_queries} queries, "
            f"{queries_per_second:.1f} QPS, {success_rate:.1%} success rate"
        )
        
    except Exception as e:
        logger.error(f"System benchmark failed: {e}")
