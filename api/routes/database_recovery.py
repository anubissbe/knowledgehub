"""
Database Recovery API Routes

Provides endpoints for monitoring and managing database connection recovery,
health checks, and performance metrics.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.database_recovery import db_recovery, RetryConfig, RetryStrategy, ConnectionState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/database-recovery", tags=["database-recovery"])

# Request/Response Models

class DatabaseHealthResponse(BaseModel):
    """Database health status response"""
    state: str
    last_check: Optional[datetime] = None
    consecutive_failures: int
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_details: Optional[str] = None
    connections: Dict[str, int]
    performance: Dict[str, float]
    circuit_breaker: Dict[str, Any]

class RetryConfigRequest(BaseModel):
    """Retry configuration request"""
    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    strategy: str = Field(default="exponential_backoff")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)

class ConnectionTestRequest(BaseModel):
    """Connection test request"""
    connection_type: str = Field(default="all", pattern="^(all|async|sync|asyncpg|psycopg2)$")
    test_query: str = Field(default="SELECT 1")

# Health and Status Endpoints

@router.get("/health")
async def get_database_health() -> DatabaseHealthResponse:
    """Get current database connection health status"""
    
    try:
        # Perform health check
        is_healthy = await db_recovery.health_check()
        
        # Get connection statistics
        stats = db_recovery.get_connection_stats()
        
        return DatabaseHealthResponse(
            state=stats["health"]["state"],
            last_check=stats["health"]["last_check"],
            consecutive_failures=stats["health"]["consecutive_failures"],
            last_failure=stats["health"]["last_failure"],
            last_success=stats["health"]["last_success"],
            error_details=stats["health"]["error_details"],
            connections=stats["connections"],
            performance=stats["performance"],
            circuit_breaker=stats["circuit_breaker"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get database health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database health: {e}")

@router.get("/status")
async def get_database_status() -> Dict[str, Any]:
    """Get comprehensive database connection status"""
    
    try:
        stats = db_recovery.get_connection_stats()
        
        # Determine overall status
        if stats["health"]["state"] == "connected" and not stats["circuit_breaker"]["open"]:
            overall_status = "healthy"
        elif stats["health"]["state"] == "recovering":
            overall_status = "recovering"
        elif stats["circuit_breaker"]["open"]:
            overall_status = "circuit_breaker_open"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "health": stats["health"],
            "connections": {
                **stats["connections"],
                "pool_utilization": (
                    (stats["connections"]["active"] / 
                     max(stats["connections"]["total"], 1)) * 100
                    if stats["connections"]["total"] > 0 else 0
                )
            },
            "performance": {
                **stats["performance"],
                "health_score": min(stats["performance"]["success_rate"], 100.0)
            },
            "circuit_breaker": stats["circuit_breaker"],
            "retry_config": {
                "max_retries": db_recovery.retry_config.max_retries,
                "initial_delay": db_recovery.retry_config.initial_delay,
                "max_delay": db_recovery.retry_config.max_delay,
                "backoff_multiplier": db_recovery.retry_config.backoff_multiplier,
                "strategy": db_recovery.retry_config.strategy.value
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get database status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database status: {e}")

@router.get("/connections")
async def get_connection_details() -> Dict[str, Any]:
    """Get detailed connection pool information"""
    
    try:
        stats = db_recovery.get_connection_stats()
        
        # Get pool-specific details
        pool_details = {
            "async_engine": {
                "initialized": db_recovery.async_engine is not None,
                "pool_size": getattr(db_recovery.async_engine.pool, 'size', 0) if db_recovery.async_engine else 0,
                "checked_out": getattr(db_recovery.async_engine.pool, 'checked_out', 0) if db_recovery.async_engine else 0
            },
            "asyncpg_pool": {
                "initialized": db_recovery.asyncpg_pool is not None,
                "size": db_recovery.asyncpg_pool._size if db_recovery.asyncpg_pool else 0,
                "free_size": db_recovery.asyncpg_pool._queue.qsize() if db_recovery.asyncpg_pool else 0
            },
            "sync_engine": {
                "initialized": db_recovery.sync_engine is not None,
                "pool_size": getattr(db_recovery.sync_engine.pool, 'size', 0) if db_recovery.sync_engine else 0,
                "checked_out": getattr(db_recovery.sync_engine.pool, 'checked_out', 0) if db_recovery.sync_engine else 0
            },
            "psycopg2_pool": {
                "initialized": db_recovery.psycopg2_pool is not None,
                "minconn": db_recovery.psycopg2_pool.minconn if db_recovery.psycopg2_pool else 0,
                "maxconn": db_recovery.psycopg2_pool.maxconn if db_recovery.psycopg2_pool else 0
            }
        }
        
        return {
            "summary": stats["connections"],
            "pools": pool_details,
            "metrics": {
                "total_connections_created": stats["connections"]["total"],
                "current_active": stats["connections"]["active"],
                "current_idle": stats["connections"]["idle"],
                "avg_wait_time_ms": stats["connections"]["wait_time_ms"]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get connection details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get connection details: {e}")

# Recovery and Management Endpoints

@router.post("/recover")
async def trigger_database_recovery(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger database connection recovery"""
    
    try:
        # Check current state
        stats = db_recovery.get_connection_stats()
        current_state = stats["health"]["state"]
        
        if current_state == "recovering":
            return {
                "message": "Recovery already in progress",
                "state": current_state,
                "initiated": False
            }
        
        # Execute recovery in background
        background_tasks.add_task(db_recovery.recover_connections)
        
        return {
            "message": "Database recovery initiated",
            "previous_state": current_state,
            "state": "recovering",
            "initiated": True,
            "initiated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger recovery: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger recovery: {e}")

@router.post("/test-connection")
async def test_database_connection(request: ConnectionTestRequest) -> Dict[str, Any]:
    """Test database connections"""
    
    results = {}
    
    try:
        # Test async engine
        if request.connection_type in ["all", "async"]:
            try:
                async with db_recovery.get_async_session() as session:
                    result = await session.execute(text(request.test_query))
                    await result.fetchone()
                results["async_engine"] = {"status": "success", "error": None}
            except Exception as e:
                results["async_engine"] = {"status": "failed", "error": str(e)}
        
        # Test asyncpg pool
        if request.connection_type in ["all", "asyncpg"]:
            try:
                async with db_recovery.get_asyncpg_connection() as conn:
                    await conn.fetchval(request.test_query)
                results["asyncpg_pool"] = {"status": "success", "error": None}
            except Exception as e:
                results["asyncpg_pool"] = {"status": "failed", "error": str(e)}
        
        # Test sync engine
        if request.connection_type in ["all", "sync"]:
            try:
                session = db_recovery.get_sync_session()
                try:
                    session.execute(text(request.test_query))
                    results["sync_engine"] = {"status": "success", "error": None}
                finally:
                    session.close()
            except Exception as e:
                results["sync_engine"] = {"status": "failed", "error": str(e)}
        
        # Test psycopg2 pool
        if request.connection_type in ["all", "psycopg2"]:
            try:
                with db_recovery.get_psycopg2_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(request.test_query)
                        cursor.fetchone()
                results["psycopg2_pool"] = {"status": "success", "error": None}
            except Exception as e:
                results["psycopg2_pool"] = {"status": "failed", "error": str(e)}
        
        # Determine overall result
        all_success = all(r["status"] == "success" for r in results.values())
        
        return {
            "overall_status": "success" if all_success else "partial_failure",
            "connection_type": request.connection_type,
            "test_query": request.test_query,
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to test connections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test connections: {e}")

@router.put("/retry-config")
async def update_retry_configuration(config: RetryConfigRequest) -> Dict[str, Any]:
    """Update database retry configuration"""
    
    try:
        # Validate strategy
        try:
            strategy = RetryStrategy(config.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid retry strategy: {config.strategy}"
            )
        
        # Update configuration
        db_recovery.retry_config.max_retries = config.max_retries
        db_recovery.retry_config.initial_delay = config.initial_delay
        db_recovery.retry_config.max_delay = config.max_delay
        db_recovery.retry_config.backoff_multiplier = config.backoff_multiplier
        db_recovery.retry_config.strategy = strategy
        db_recovery.retry_config.timeout = config.timeout
        
        logger.info(f"Updated retry configuration: {config}")
        
        return {
            "message": "Retry configuration updated successfully",
            "config": {
                "max_retries": config.max_retries,
                "initial_delay": config.initial_delay,
                "max_delay": config.max_delay,
                "backoff_multiplier": config.backoff_multiplier,
                "strategy": config.strategy,
                "timeout": config.timeout
            },
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update retry config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update retry config: {e}")

@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker() -> Dict[str, Any]:
    """Reset circuit breaker state"""
    
    try:
        # Get current state
        was_open = db_recovery.circuit_breaker_open
        
        # Reset circuit breaker
        db_recovery.circuit_breaker_open = False
        db_recovery.circuit_breaker_failures = 0
        db_recovery.circuit_breaker_opened_at = None
        
        logger.info("Circuit breaker reset")
        
        return {
            "message": "Circuit breaker reset successfully",
            "was_open": was_open,
            "is_open": False,
            "reset_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset circuit breaker: {e}")

# Performance and Metrics Endpoints

@router.get("/metrics")
async def get_database_metrics(
    time_window_minutes: int = Query(default=5, ge=1, le=60)
) -> Dict[str, Any]:
    """Get database performance metrics"""
    
    try:
        # Get current stats
        stats = db_recovery.get_connection_stats()
        
        # Calculate additional metrics
        query_metrics = list(db_recovery.query_metrics)
        if query_metrics:
            # Filter by time window
            cutoff_time = datetime.now(timezone.utc).timestamp() - (time_window_minutes * 60)
            recent_queries = [
                q for q in query_metrics 
                if q['timestamp'].timestamp() > cutoff_time
            ]
            
            # Calculate percentiles for successful queries
            successful_times = [
                q['execution_time'] 
                for q in recent_queries 
                if q.get('success', False)
            ]
            
            if successful_times:
                successful_times.sort()
                p50_idx = int(len(successful_times) * 0.5)
                p95_idx = int(len(successful_times) * 0.95)
                p99_idx = int(len(successful_times) * 0.99)
                
                percentiles = {
                    "p50": successful_times[p50_idx] if p50_idx < len(successful_times) else 0,
                    "p95": successful_times[p95_idx] if p95_idx < len(successful_times) else 0,
                    "p99": successful_times[p99_idx] if p99_idx < len(successful_times) else 0
                }
            else:
                percentiles = {"p50": 0, "p95": 0, "p99": 0}
            
            # Error rate
            error_count = sum(1 for q in recent_queries if not q.get('success', False))
            error_rate = (error_count / len(recent_queries) * 100) if recent_queries else 0
        else:
            percentiles = {"p50": 0, "p95": 0, "p99": 0}
            error_rate = 0
        
        return {
            "time_window_minutes": time_window_minutes,
            "connection_metrics": stats["connections"],
            "query_metrics": {
                "total_queries": len(query_metrics),
                "success_rate": stats["performance"]["success_rate"],
                "error_rate": error_rate,
                "avg_execution_time": stats["performance"]["avg_execution_time"],
                "percentiles": percentiles
            },
            "health_metrics": {
                "uptime_percentage": 100.0 if stats["health"]["state"] == "connected" else 0.0,
                "consecutive_failures": stats["health"]["consecutive_failures"],
                "circuit_breaker_trips": 1 if stats["circuit_breaker"]["open"] else 0
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get database metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database metrics: {e}")

@router.get("/alerts")
async def get_database_alerts() -> Dict[str, Any]:
    """Get active database alerts"""
    
    try:
        stats = db_recovery.get_connection_stats()
        alerts = []
        
        # Check for connection failures
        if stats["health"]["consecutive_failures"] > 0:
            alerts.append({
                "id": "db_connection_failures",
                "severity": "high" if stats["health"]["consecutive_failures"] >= 3 else "medium",
                "message": f"Database experiencing {stats['health']['consecutive_failures']} consecutive failures",
                "details": stats["health"]["error_details"],
                "timestamp": stats["health"]["last_failure"]
            })
        
        # Check circuit breaker
        if stats["circuit_breaker"]["open"]:
            alerts.append({
                "id": "db_circuit_breaker_open",
                "severity": "critical",
                "message": "Database circuit breaker is open",
                "details": f"Opened after {stats['circuit_breaker']['failures']} failures",
                "timestamp": stats["circuit_breaker"]["opened_at"]
            })
        
        # Check connection pool saturation
        if stats["connections"]["total"] > 0:
            utilization = (stats["connections"]["active"] / stats["connections"]["total"]) * 100
            if utilization > 80:
                alerts.append({
                    "id": "db_pool_saturation",
                    "severity": "medium" if utilization < 90 else "high",
                    "message": f"Database connection pool {utilization:.1f}% utilized",
                    "details": f"{stats['connections']['active']} of {stats['connections']['total']} connections in use",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Check performance degradation
        if stats["performance"]["success_rate"] < 95:
            alerts.append({
                "id": "db_performance_degradation",
                "severity": "medium" if stats["performance"]["success_rate"] > 80 else "high",
                "message": f"Database success rate {stats['performance']['success_rate']:.1f}%",
                "details": "Query success rate below threshold",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a["severity"] == "critical"]),
            "high_alerts": len([a for a in alerts if a["severity"] == "high"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get database alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database alerts: {e}")

from sqlalchemy import text