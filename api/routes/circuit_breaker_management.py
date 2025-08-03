"""
Circuit Breaker Management API Routes

Provides endpoints for monitoring and managing circuit breakers for
external service calls.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.circuit_breaker import (
    circuit_manager, CircuitState, ServiceType, CircuitConfig
)
from ..services.external_service_client import (
    weaviate_client, ml_model_client, redis_client
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/circuit-breaker", tags=["circuit-breaker"])

# Request/Response Models

class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status response"""
    name: str
    service_type: str
    state: str
    failure_count: int
    success_count: int
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_rate: float
    avg_response_time: float
    total_calls: int

class CircuitConfigUpdate(BaseModel):
    """Circuit breaker configuration update request"""
    failure_threshold: int = Field(default=5, ge=1, le=100)
    success_threshold: int = Field(default=2, ge=1, le=50)
    timeout: float = Field(default=60.0, ge=1.0, le=600.0)
    half_open_max_calls: int = Field(default=3, ge=1, le=10)
    error_timeout: float = Field(default=30.0, ge=1.0, le=300.0)

class ServiceTestRequest(BaseModel):
    """Service test request"""
    service_name: str
    test_type: str = Field(default="health", pattern="^(health|performance|load)$")
    iterations: int = Field(default=1, ge=1, le=10)

# Status and Monitoring Endpoints

@router.get("/status")
async def get_all_circuit_breakers_status() -> Dict[str, Any]:
    """Get status of all circuit breakers"""
    
    try:
        all_status = circuit_manager.get_all_status()
        health_summary = circuit_manager.get_health_summary()
        
        return {
            "circuit_breakers": all_status,
            "summary": health_summary,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}")

@router.get("/status/{breaker_name}")
async def get_circuit_breaker_status(breaker_name: str) -> CircuitBreakerStatus:
    """Get status of specific circuit breaker"""
    
    breaker = circuit_manager.get_circuit_breaker(breaker_name)
    if not breaker:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{breaker_name}' not found")
    
    status = breaker.get_status()
    
    return CircuitBreakerStatus(
        name=status["name"],
        service_type=status["service_type"],
        state=status["state"],
        failure_count=status["failure_count"],
        success_count=status["success_count"],
        last_failure=status["last_failure"],
        last_success=status["last_success"],
        error_rate=status["error_rate"],
        avg_response_time=status["avg_response_time"],
        total_calls=status["total_calls"]
    )

@router.get("/health")
async def get_circuit_breaker_health() -> Dict[str, Any]:
    """Get overall circuit breaker system health"""
    
    try:
        health_summary = circuit_manager.get_health_summary()
        
        # Determine overall health status
        if health_summary["open_breakers"] == 0:
            status = "healthy"
        elif health_summary["open_breakers"] < health_summary["total_breakers"] / 2:
            status = "degraded"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": health_summary["health_score"],
            "total_services": health_summary["total_breakers"],
            "healthy_services": health_summary["closed_breakers"],
            "degraded_services": health_summary["half_open_breakers"],
            "failed_services": health_summary["open_breakers"],
            "total_calls": health_summary["total_calls"],
            "avg_error_rate": health_summary["avg_error_rate"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get circuit breaker health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health: {e}")

# Management Endpoints

@router.post("/reset/{breaker_name}")
async def reset_circuit_breaker(breaker_name: str) -> Dict[str, Any]:
    """Reset specific circuit breaker"""
    
    breaker = circuit_manager.get_circuit_breaker(breaker_name)
    if not breaker:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{breaker_name}' not found")
    
    try:
        # Get current state
        previous_state = breaker.health.state.value
        
        # Reset breaker
        await breaker.reset()
        
        return {
            "message": f"Circuit breaker '{breaker_name}' reset successfully",
            "previous_state": previous_state,
            "current_state": "closed",
            "reset_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker {breaker_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {e}")

@router.post("/reset-all")
async def reset_all_circuit_breakers() -> Dict[str, Any]:
    """Reset all circuit breakers"""
    
    try:
        # Get current states
        previous_states = {
            name: breaker.health.state.value
            for name, breaker in circuit_manager.circuit_breakers.items()
        }
        
        # Reset all
        await circuit_manager.reset_all()
        
        return {
            "message": "All circuit breakers reset successfully",
            "previous_states": previous_states,
            "reset_count": len(previous_states),
            "reset_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset all circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset all: {e}")

@router.post("/reset-by-type/{service_type}")
async def reset_circuit_breakers_by_type(service_type: str) -> Dict[str, Any]:
    """Reset circuit breakers by service type"""
    
    try:
        # Validate service type
        try:
            svc_type = ServiceType(service_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid service type: {service_type}"
            )
        
        # Count breakers to reset
        breakers_to_reset = [
            name for name, breaker in circuit_manager.circuit_breakers.items()
            if breaker.service_type == svc_type
        ]
        
        # Reset by type
        await circuit_manager.reset_by_type(svc_type)
        
        return {
            "message": f"Circuit breakers of type '{service_type}' reset successfully",
            "reset_breakers": breakers_to_reset,
            "reset_count": len(breakers_to_reset),
            "reset_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset circuit breakers by type {service_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset by type: {e}")

@router.put("/config/{breaker_name}")
async def update_circuit_breaker_config(
    breaker_name: str,
    config: CircuitConfigUpdate
) -> Dict[str, Any]:
    """Update circuit breaker configuration"""
    
    breaker = circuit_manager.get_circuit_breaker(breaker_name)
    if not breaker:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{breaker_name}' not found")
    
    try:
        # Update configuration
        breaker.config.failure_threshold = config.failure_threshold
        breaker.config.success_threshold = config.success_threshold
        breaker.config.timeout = config.timeout
        breaker.config.half_open_max_calls = config.half_open_max_calls
        breaker.config.error_timeout = config.error_timeout
        
        logger.info(f"Updated circuit breaker config for {breaker_name}: {config}")
        
        return {
            "message": f"Circuit breaker '{breaker_name}' configuration updated",
            "config": {
                "failure_threshold": config.failure_threshold,
                "success_threshold": config.success_threshold,
                "timeout": config.timeout,
                "half_open_max_calls": config.half_open_max_calls,
                "error_timeout": config.error_timeout
            },
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update circuit breaker config for {breaker_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {e}")

# Service Testing Endpoints

@router.post("/test-service")
async def test_external_service(request: ServiceTestRequest) -> Dict[str, Any]:
    """Test external service through circuit breaker"""
    
    results = {
        "service_name": request.service_name,
        "test_type": request.test_type,
        "iterations": request.iterations,
        "results": []
    }
    
    try:
        for i in range(request.iterations):
            start_time = datetime.now(timezone.utc)
            
            try:
                # Test different services
                if request.service_name == "weaviate_search":
                    # Test Weaviate search
                    result = await weaviate_client.get("/meta")
                    success = "version" in result
                    error = None
                elif request.service_name == "ml_inference":
                    # Test ML service
                    result = await ml_model_client.get("/health")
                    success = result.get("status") == "healthy"
                    error = None
                elif request.service_name == "redis_cache":
                    # Test Redis
                    result = await redis_client.get("test_key")
                    success = True  # No exception means success
                    error = None
                else:
                    raise ValueError(f"Unknown service: {request.service_name}")
                
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                results["results"].append({
                    "iteration": i + 1,
                    "success": success,
                    "duration": duration,
                    "error": error,
                    "timestamp": start_time.isoformat()
                })
                
            except Exception as e:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                results["results"].append({
                    "iteration": i + 1,
                    "success": False,
                    "duration": duration,
                    "error": str(e),
                    "timestamp": start_time.isoformat()
                })
            
            # Add delay between iterations for load testing
            if request.test_type == "load" and i < request.iterations - 1:
                await asyncio.sleep(0.1)
        
        # Calculate summary
        successful_tests = sum(1 for r in results["results"] if r["success"])
        avg_duration = sum(r["duration"] for r in results["results"]) / len(results["results"])
        
        results["summary"] = {
            "total_tests": request.iterations,
            "successful_tests": successful_tests,
            "failed_tests": request.iterations - successful_tests,
            "success_rate": (successful_tests / request.iterations) * 100,
            "avg_duration": avg_duration
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to test service {request.service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test service: {e}")

# Metrics and Analytics Endpoints

@router.get("/metrics")
async def get_circuit_breaker_metrics(
    time_window_minutes: int = Query(default=5, ge=1, le=60)
) -> Dict[str, Any]:
    """Get circuit breaker metrics"""
    
    try:
        all_status = circuit_manager.get_all_status()
        
        # Calculate metrics for each service
        service_metrics = {}
        for name, status in all_status.items():
            # Get service client stats if available
            client_stats = None
            if name == "weaviate_search":
                client_stats = weaviate_client.get_stats()
            elif name == "ml_inference":
                client_stats = ml_model_client.get_stats()
            
            service_metrics[name] = {
                "state": status["state"],
                "error_rate": status["error_rate"],
                "avg_response_time": status["avg_response_time"],
                "total_calls": status["total_calls"],
                "failure_count": status["failure_count"],
                "success_count": status["success_count"],
                "client_stats": client_stats
            }
        
        # Overall metrics
        total_calls = sum(m["total_calls"] for m in service_metrics.values())
        avg_error_rate = (
            sum(m["error_rate"] * m["total_calls"] for m in service_metrics.values()) / total_calls
            if total_calls > 0 else 0
        )
        
        return {
            "time_window_minutes": time_window_minutes,
            "service_metrics": service_metrics,
            "overall_metrics": {
                "total_calls": total_calls,
                "avg_error_rate": avg_error_rate,
                "services_monitored": len(service_metrics)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get circuit breaker metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")

@router.get("/alerts")
async def get_circuit_breaker_alerts() -> Dict[str, Any]:
    """Get active circuit breaker alerts"""
    
    try:
        all_status = circuit_manager.get_all_status()
        alerts = []
        
        for name, status in all_status.items():
            # Check for open circuit
            if status["state"] == "open":
                alerts.append({
                    "id": f"circuit_open_{name}",
                    "severity": "critical",
                    "service": name,
                    "message": f"Circuit breaker for {name} is OPEN",
                    "details": f"Failed after {status['failure_count']} consecutive failures",
                    "timestamp": status["last_failure"]
                })
            
            # Check for half-open circuit
            elif status["state"] == "half_open":
                alerts.append({
                    "id": f"circuit_half_open_{name}",
                    "severity": "warning",
                    "service": name,
                    "message": f"Circuit breaker for {name} is HALF-OPEN (testing recovery)",
                    "details": f"Success count: {status['success_count']}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Check for high error rate
            elif status["error_rate"] > 20:
                alerts.append({
                    "id": f"high_error_rate_{name}",
                    "severity": "medium",
                    "service": name,
                    "message": f"High error rate for {name}: {status['error_rate']:.1f}%",
                    "details": f"Total calls: {status['total_calls']}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a["severity"] == "critical"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get circuit breaker alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {e}")

import asyncio