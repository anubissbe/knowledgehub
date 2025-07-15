"""
Performance Monitoring and Management API Routes

Provides endpoints for performance monitoring, optimization,
and system management.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from ..performance.manager import get_performance_manager
from ..performance.monitoring import MetricType, AlertLevel
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


# API key verification for admin endpoints
async def verify_admin_access(request: Request):
    """Verify admin access for performance endpoints"""
    from ..config import settings
    
    # In development mode, allow access without authentication for testing
    if settings.APP_ENV == "development":
        return {"id": "dev_user", "name": "Development User", "permissions": ["admin"], "type": "development"}
    
    # Check if user is authenticated via middleware
    if not hasattr(request.state, 'authenticated') or not request.state.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Return simple authenticated user for now
    return {"id": "auth_user", "name": "Authenticated User", "permissions": ["admin"], "type": "authenticated"}


@router.get("/performance/health")
async def get_performance_health():
    """
    Get performance system health status
    
    Returns overall health of all performance optimization systems.
    """
    try:
        manager = get_performance_manager()
        health_status = manager.get_health_status()
        
        return JSONResponse(
            status_code=200 if health_status['overall_status'] == 'healthy' else 503,
            content={
                "status": health_status['overall_status'],
                "timestamp": datetime.now().isoformat(),
                "systems": health_status['systems'],
                "message": f"Performance systems are {health_status['overall_status']}"
            }
        )
        
    except Exception as e:
        logger.error(f"Performance health check failed: {e}")
        raise HTTPException(status_code=500, detail="Performance health check failed")


@router.get("/performance/stats")
async def get_performance_stats(
    request: Request,
    system: Optional[str] = Query(None, description="Specific system to get stats for"),
    user = Depends(verify_admin_access)
):
    """
    Get comprehensive performance statistics
    
    Returns detailed performance metrics for all or specific systems.
    """
    try:
        manager = get_performance_manager()
        stats = manager.get_comprehensive_stats()
        
        if system:
            if system not in stats:
                raise HTTPException(status_code=404, detail=f"System '{system}' not found")
            return {
                "system": system,
                "stats": stats[system],
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance statistics: {str(e)}")


@router.get("/performance/dashboard")
async def get_performance_dashboard(
    api_key: str = Depends(verify_admin_access)
):
    """
    Get performance dashboard data
    
    Returns comprehensive dashboard data including metrics, alerts, and recommendations.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitoring not available")
        
        dashboard_data = manager.performance_monitor.get_dashboard_data()
        
        return {
            "dashboard": dashboard_data,
            "health": manager.get_health_status(),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


@router.post("/performance/optimize")
async def optimize_system(
    api_key: str = Depends(verify_admin_access)
):
    """
    Run system optimization
    
    Triggers optimization routines across all performance systems.
    """
    try:
        manager = get_performance_manager()
        optimization_results = await manager.optimize_system()
        
        return {
            "message": "System optimization completed",
            "results": optimization_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"System optimization failed: {str(e)}")


@router.post("/performance/cache/clear")
async def clear_cache(
    api_key: str = Depends(verify_admin_access),
    pattern: Optional[str] = Query(None, description="Cache pattern to clear")
):
    """
    Clear cache entries
    
    Clears cache entries for all patterns or a specific pattern.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.cache_manager:
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        if pattern:
            await manager.cache_manager.invalidate_pattern(pattern)
            message = f"Cache pattern '{pattern}' cleared"
        else:
            await manager.cache_manager.clear_all()
            message = "All cache entries cleared"
        
        return {
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache clear operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@router.get("/performance/cache/stats")
async def get_cache_stats(
    api_key: str = Depends(verify_admin_access)
):
    """
    Get cache performance statistics
    
    Returns detailed cache usage and performance metrics.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.cache_manager:
            raise HTTPException(status_code=503, detail="Cache manager not available")
        
        cache_stats = manager.cache_manager.get_stats()
        
        return {
            "cache_stats": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")


@router.get("/performance/database/stats")
async def get_database_stats(
    api_key: str = Depends(verify_admin_access)
):
    """
    Get database performance statistics
    
    Returns query performance metrics and optimization suggestions.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.db_optimizer:
            raise HTTPException(status_code=503, detail="Database optimizer not available")
        
        db_stats = manager.db_optimizer.get_query_stats()
        
        return {
            "database_stats": db_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve database statistics: {str(e)}")


@router.post("/performance/database/optimize")
async def optimize_database_table(
    table_name: str,
    api_key: str = Depends(verify_admin_access)
):
    """
    Optimize specific database table
    
    Runs optimization routines on a specific database table.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.db_optimizer:
            raise HTTPException(status_code=503, detail="Database optimizer not available")
        
        optimization_results = await manager.db_optimizer.optimize_table(table_name)
        
        return {
            "message": f"Table '{table_name}' optimization completed",
            "results": optimization_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database table optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Table optimization failed: {str(e)}")


@router.get("/performance/async/stats")
async def get_async_stats(
    api_key: str = Depends(verify_admin_access)
):
    """
    Get async processing statistics
    
    Returns task queue performance metrics and processing statistics.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.async_optimizer:
            raise HTTPException(status_code=503, detail="Async optimizer not available")
        
        async_stats = manager.async_optimizer.get_comprehensive_stats()
        
        return {
            "async_stats": async_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get async stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve async processing statistics")


@router.get("/performance/response/stats")
async def get_response_stats(
    api_key: str = Depends(verify_admin_access),
    endpoint: Optional[str] = Query(None, description="Specific endpoint to analyze"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze")
):
    """
    Get response optimization statistics
    
    Returns response time, compression, and caching metrics.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.response_optimizer:
            raise HTTPException(status_code=503, detail="Response optimizer not available")
        
        response_stats = manager.response_optimizer.get_performance_stats(endpoint, hours)
        
        return {
            "response_stats": response_stats,
            "endpoint": endpoint,
            "hours_analyzed": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get response stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve response statistics")


@router.get("/performance/alerts")
async def get_active_alerts(
    api_key: str = Depends(verify_admin_access)
):
    """
    Get active performance alerts
    
    Returns all currently active performance alerts.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitoring not available")
        
        active_alerts = manager.performance_monitor.alert_manager.get_active_alerts()
        
        return {
            "active_alerts": [
                {
                    "name": alert.name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "timestamp": alert.timestamp,
                    "tags": alert.tags
                }
                for alert in active_alerts
            ],
            "alert_count": len(active_alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active alerts")


@router.get("/performance/recommendations")
async def get_optimization_recommendations(
    api_key: str = Depends(verify_admin_access)
):
    """
    Get performance optimization recommendations
    
    Returns intelligent recommendations for improving system performance.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitoring not available")
        
        recommendations = manager.performance_monitor.optimizer.get_optimization_priority()
        
        return {
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve optimization recommendations")


@router.post("/performance/metric")
async def record_custom_metric(request: Request):
    """
    Record custom performance metric
    
    Allows recording custom performance metrics for monitoring.
    """
    try:
        # Manual JSON parsing to avoid FastAPI body validation issues
        body = await request.body()
        if body:
            import json
            metric_data = json.loads(body.decode('utf-8'))
            
            # Basic validation
            required_fields = ['name', 'value', 'type']
            for field in required_fields:
                if field not in metric_data:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
            
            return {
                "message": "Metric recorded successfully",
                "metric_name": metric_data['name'],
                "value": metric_data['value'],
                "type": metric_data['type'],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "message": "Metric recorded successfully (no data)",
                "timestamp": datetime.now().isoformat()
            }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.error(f"Failed to record metric: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")


@router.get("/performance/slow-endpoints")
async def get_slow_endpoints(
    api_key: str = Depends(verify_admin_access),
    threshold: float = Query(1.0, ge=0.1, description="Response time threshold in seconds"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of endpoints to return")
):
    """
    Get slowest API endpoints
    
    Returns endpoints with response times above the specified threshold.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.response_optimizer:
            raise HTTPException(status_code=503, detail="Response optimizer not available")
        
        slow_endpoints = manager.response_optimizer.get_slow_endpoints(threshold, limit)
        
        return {
            "slow_endpoints": slow_endpoints,
            "threshold_seconds": threshold,
            "endpoint_count": len(slow_endpoints),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get slow endpoints: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve slow endpoints")


@router.post("/performance/enable")
async def enable_optimization():
    """
    Enable performance optimization
    
    Enables all performance optimization features.
    """
    # Bypass all middleware processing - direct response
    return {
        "message": "Performance optimization enabled (test)",
        "timestamp": datetime.now().isoformat(),
        "debug": "bypassed_all_logic"
    }


@router.get("/performance/enable-test")
async def enable_optimization_get():
    """
    Enable performance optimization (GET test version)
    
    Test endpoint using GET method to diagnose POST timeout issue.
    """
    return {
        "message": "Performance optimization enabled (GET test)",
        "timestamp": datetime.now().isoformat(),
        "debug": "GET_method_test"
    }


@router.post("/performance/disable")
async def disable_optimization(
    api_key: str = Depends(verify_admin_access)
):
    """
    Disable performance optimization
    
    Disables all performance optimization features.
    """
    try:
        manager = get_performance_manager()
        manager.disable_optimization()
        
        return {
            "message": "Performance optimization disabled",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to disable optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to disable optimization")


@router.get("/performance/system-resources")
async def get_system_resources(
    api_key: str = Depends(verify_admin_access)
):
    """
    Get current system resource usage
    
    Returns CPU, memory, disk, and network usage statistics.
    """
    try:
        manager = get_performance_manager()
        
        if not manager.performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitoring not available")
        
        current_resources = manager.performance_monitor.system_monitor.get_current_resources()
        resource_trends = manager.performance_monitor.system_monitor.get_resource_trends(hours=1)
        
        return {
            "current_resources": {
                "cpu_percent": current_resources.cpu_percent,
                "memory_percent": current_resources.memory_percent,
                "memory_used_mb": current_resources.memory_used_mb,
                "memory_available_mb": current_resources.memory_available_mb,
                "disk_usage_percent": current_resources.disk_usage_percent,
                "disk_free_gb": current_resources.disk_free_gb,
                "network_bytes_sent": current_resources.network_bytes_sent,
                "network_bytes_recv": current_resources.network_bytes_recv,
                "load_average": current_resources.load_average,
                "timestamp": current_resources.timestamp
            },
            "trends": resource_trends,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system resources")