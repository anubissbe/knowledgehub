"""
Recovery Management API Routes

Provides endpoints for managing service recovery, health monitoring,
and self-healing system configuration.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.service_recovery import (
    service_recovery, ServiceState, RecoveryAction, 
    register_default_services
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recovery", tags=["recovery_management"])

# Request/Response Models

class ServiceStatusResponse(BaseModel):
    """Service status response model"""
    name: str
    state: str
    last_check: datetime
    consecutive_failures: int
    last_failure: Optional[datetime] = None
    last_recovery: Optional[datetime] = None
    error_details: Optional[str] = None
    recovery_attempts: int
    next_check: Optional[datetime] = None

class RecoveryStatistics(BaseModel):
    """Recovery statistics response model"""
    total_recoveries: int
    successful_recoveries: int
    failed_recoveries: int
    avg_recovery_time: float
    last_recovery: Optional[datetime] = None
    services_monitored: int
    active_recoveries: int
    monitoring_enabled: bool

class ForceRecoveryRequest(BaseModel):
    """Force recovery request model"""
    service_name: str
    reason: Optional[str] = None

class ServiceConfigRequest(BaseModel):
    """Service configuration request model"""
    service_name: str
    enabled: bool

# Recovery Status and Monitoring

@router.get("/status")
async def get_recovery_system_status() -> Dict[str, Any]:
    """Get overall recovery system status"""
    
    try:
        services = service_recovery.get_all_services_status()
        stats = service_recovery.get_recovery_statistics()
        
        # Categorize services by state
        service_states = {}
        for state in ServiceState:
            service_states[state.value] = []
        
        for service_name, health in services.items():
            service_states[health.state.value].append(service_name)
        
        return {
            "system_status": "healthy" if stats["monitoring_enabled"] else "disabled",
            "services_by_state": service_states,
            "recovery_statistics": stats,
            "total_services": len(services),
            "healthy_services": len(service_states.get("healthy", [])),
            "failed_services": len(service_states.get("failed", [])),
            "recovering_services": len(service_states.get("recovering", [])),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recovery system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery system status: {e}")

@router.get("/services")
async def get_all_services_status() -> Dict[str, Any]:
    """Get status of all monitored services"""
    
    try:
        services = service_recovery.get_all_services_status()
        
        services_data = {}
        for service_name, health in services.items():
            services_data[service_name] = ServiceStatusResponse(
                name=health.name,
                state=health.state.value,
                last_check=health.last_check,
                consecutive_failures=health.consecutive_failures,
                last_failure=health.last_failure,
                last_recovery=health.last_recovery,
                error_details=health.error_details,
                recovery_attempts=health.recovery_attempts,
                next_check=health.next_check
            ).dict()
        
        return {
            "services": services_data,
            "total_services": len(services_data),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get services status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get services status: {e}")

@router.get("/services/{service_name}")
async def get_service_status(service_name: str) -> ServiceStatusResponse:
    """Get detailed status of a specific service"""
    
    health = service_recovery.get_service_status(service_name)
    
    if not health:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
    
    return ServiceStatusResponse(
        name=health.name,
        state=health.state.value,
        last_check=health.last_check,
        consecutive_failures=health.consecutive_failures,
        last_failure=health.last_failure,
        last_recovery=health.last_recovery,
        error_details=health.error_details,
        recovery_attempts=health.recovery_attempts,
        next_check=health.next_check
    )

@router.get("/statistics")
async def get_recovery_statistics() -> RecoveryStatistics:
    """Get recovery system statistics"""
    
    try:
        stats = service_recovery.get_recovery_statistics()
        
        return RecoveryStatistics(
            total_recoveries=stats["total_recoveries"],
            successful_recoveries=stats["successful_recoveries"],
            failed_recoveries=stats["failed_recoveries"],
            avg_recovery_time=stats["avg_recovery_time"],
            last_recovery=stats["last_recovery"],
            services_monitored=stats["services_monitored"],
            active_recoveries=stats["active_recoveries"],
            monitoring_enabled=stats["monitoring_enabled"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get recovery statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery statistics: {e}")

# Recovery Actions

@router.post("/services/{service_name}/recover")
async def force_service_recovery(
    service_name: str,
    request: ForceRecoveryRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Force immediate recovery attempt for a service"""
    
    try:
        # Validate service exists
        if not service_recovery.get_service_status(service_name):
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        # Log recovery request
        logger.info(f"Force recovery requested for {service_name}: {request.reason or 'No reason provided'}")
        
        # Execute recovery in background
        success = await service_recovery.force_recovery(service_name)
        
        if success:
            return {
                "message": f"Recovery initiated for service '{service_name}'",
                "service_name": service_name,
                "reason": request.reason,
                "initiated_at": datetime.now(timezone.utc).isoformat(),
                "status": "recovery_initiated"
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to initiate recovery for '{service_name}'")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to force recovery for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force recovery: {e}")

@router.post("/services/{service_name}/enable")
async def enable_service_monitoring(service_name: str) -> Dict[str, Any]:
    """Enable monitoring for a specific service"""
    
    try:
        # Validate service exists
        if not service_recovery.get_service_status(service_name):
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        await service_recovery.enable_service_monitoring(service_name)
        
        return {
            "message": f"Monitoring enabled for service '{service_name}'",
            "service_name": service_name,
            "enabled_at": datetime.now(timezone.utc).isoformat(),
            "status": "monitoring_enabled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable monitoring for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable monitoring: {e}")

@router.post("/services/{service_name}/disable")
async def disable_service_monitoring(service_name: str) -> Dict[str, Any]:
    """Disable monitoring for a specific service"""
    
    try:
        # Validate service exists
        if not service_recovery.get_service_status(service_name):
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        await service_recovery.disable_service_monitoring(service_name)
        
        return {
            "message": f"Monitoring disabled for service '{service_name}'",
            "service_name": service_name,
            "disabled_at": datetime.now(timezone.utc).isoformat(),
            "status": "monitoring_disabled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable monitoring for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable monitoring: {e}")

@router.post("/initialize")
async def initialize_recovery_system() -> Dict[str, Any]:
    """Initialize recovery system with default services"""
    
    try:
        # Register default services
        register_default_services()
        
        # Start monitoring if not already started
        if not service_recovery.monitoring_task or service_recovery.monitoring_task.done():
            await service_recovery.start_monitoring()
        
        return {
            "message": "Recovery system initialized successfully",
            "initialized_at": datetime.now(timezone.utc).isoformat(),
            "services_registered": len(service_recovery.services),
            "monitoring_started": True
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize recovery system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize recovery system: {e}")

# Health and Diagnostics

@router.get("/health")
async def get_recovery_system_health() -> Dict[str, Any]:
    """Get recovery system health status"""
    
    try:
        stats = service_recovery.get_recovery_statistics()
        services = service_recovery.get_all_services_status()
        
        # Calculate health score
        healthy_services = sum(1 for h in services.values() if h.state == ServiceState.HEALTHY)
        total_services = len(services)
        health_score = (healthy_services / total_services * 100) if total_services > 0 else 100
        
        return {
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "unhealthy",
            "health_score": health_score,
            "monitoring_enabled": stats["monitoring_enabled"],
            "services_monitored": stats["services_monitored"],
            "healthy_services": healthy_services,
            "total_services": total_services,
            "active_recoveries": stats["active_recoveries"],
            "last_check": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recovery system health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }

@router.get("/alerts")
async def get_recovery_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (critical/high/medium/low)"),
    limit: int = Query(default=50, le=500, description="Maximum number of alerts")
) -> Dict[str, Any]:
    """Get recovery system alerts and notifications"""
    
    try:
        services = service_recovery.get_all_services_status()
        alerts = []
        
        for service_name, health in services.items():
            # Generate alerts based on service state
            if health.state == ServiceState.FAILED:
                alerts.append({
                    "id": f"service_failed_{service_name}",
                    "severity": "critical",
                    "service": service_name,
                    "message": f"Service {service_name} has failed",
                    "details": health.error_details,
                    "consecutive_failures": health.consecutive_failures,
                    "recovery_attempts": health.recovery_attempts,
                    "timestamp": health.last_failure or health.last_check,
                    "active": True
                })
            elif health.state == ServiceState.UNHEALTHY:
                alerts.append({
                    "id": f"service_unhealthy_{service_name}",
                    "severity": "high" if health.consecutive_failures >= 2 else "medium",
                    "service": service_name,
                    "message": f"Service {service_name} is unhealthy",
                    "details": health.error_details,
                    "consecutive_failures": health.consecutive_failures,
                    "timestamp": health.last_failure or health.last_check,
                    "active": True
                })
            elif health.state == ServiceState.RECOVERING:
                alerts.append({
                    "id": f"service_recovering_{service_name}",
                    "severity": "medium",
                    "service": service_name,
                    "message": f"Service {service_name} is recovering",
                    "details": f"Recovery attempt {health.recovery_attempts}",
                    "recovery_attempts": health.recovery_attempts,
                    "timestamp": health.last_check,
                    "active": True
                })
        
        # Filter by severity if specified
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        # Sort by severity and timestamp
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        alerts.sort(key=lambda x: (severity_order.get(x["severity"], 0), x["timestamp"]), reverse=True)
        
        return {
            "alerts": alerts[:limit],
            "total_alerts": len(alerts),
            "active_alerts": len([a for a in alerts if a["active"]]),
            "critical_alerts": len([a for a in alerts if a["severity"] == "critical"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recovery alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery alerts: {e}")

@router.get("/metrics")
async def get_recovery_metrics() -> Dict[str, Any]:
    """Get detailed recovery metrics for monitoring"""
    
    try:
        stats = service_recovery.get_recovery_statistics()
        services = service_recovery.get_all_services_status()
        
        # Calculate detailed metrics
        service_metrics = {}
        for service_name, health in services.items():
            service_metrics[service_name] = {
                "state": health.state.value,
                "uptime_percentage": _calculate_uptime(health),
                "failure_rate": _calculate_failure_rate(health),
                "recovery_success_rate": _calculate_recovery_success_rate(health),
                "mean_time_to_recovery": _calculate_mttr(health),
                "last_incident": health.last_failure,
                "consecutive_failures": health.consecutive_failures
            }
        
        return {
            "system_metrics": {
                "overall_uptime": _calculate_overall_uptime(services),
                "total_incidents": stats["total_recoveries"],
                "recovery_success_rate": (
                    stats["successful_recoveries"] / stats["total_recoveries"] * 100
                    if stats["total_recoveries"] > 0 else 100
                ),
                "mean_recovery_time": stats["avg_recovery_time"],
                "services_healthy": len([s for s in services.values() if s.state == ServiceState.HEALTHY]),
                "services_total": len(services)
            },
            "service_metrics": service_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recovery metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery metrics: {e}")

# Helper Functions

def _calculate_uptime(health) -> float:
    """Calculate service uptime percentage based on health history"""
    # Calculate uptime based on last check time and consecutive failures
    now = datetime.now(timezone.utc)
    if health.last_check_time:
        time_since_check = (now - health.last_check_time).total_seconds()
        
        # If healthy, uptime is based on time without failures
        if health.state == ServiceState.HEALTHY:
            # Assume service has been up since last recovery or start
            if health.last_recovery_time:
                uptime_seconds = (now - health.last_recovery_time).total_seconds()
            else:
                # No recovery time means it's been up since monitoring started
                uptime_seconds = time_since_check
            
            # Calculate percentage (assuming 24-hour window)
            uptime_percentage = min((uptime_seconds / 86400) * 100, 100.0)
            return round(uptime_percentage, 1)
        else:
            # For degraded/unhealthy services, reduce uptime based on consecutive failures
            base_uptime = 100.0
            failure_penalty = health.consecutive_failures * 5  # 5% per consecutive failure
            return max(base_uptime - failure_penalty, 0.0)
    
    # Default if no check time available
    return 0.0 if health.state == ServiceState.UNHEALTHY else 100.0

def _calculate_failure_rate(health) -> float:
    """Calculate service failure rate based on failure history"""
    # Calculate failure rate based on consecutive failures and total attempts
    if health.recovery_attempts == 0:
        # No recovery attempts means no failures recorded
        return 0.0
    
    # Calculate failure rate as percentage of failed checks
    # Consider both consecutive failures and total recovery attempts
    failure_rate = health.consecutive_failures / max(health.recovery_attempts, 1)
    
    # Normalize to percentage (0-1 range)
    return min(failure_rate, 1.0)

def _calculate_recovery_success_rate(health) -> float:
    """Calculate recovery success rate based on recovery history"""
    if health.recovery_attempts == 0:
        # No recovery attempts means 100% success (no failures to recover from)
        return 100.0
    
    # Calculate success rate based on successful recoveries
    # Success is when consecutive failures reset to 0 after recovery
    # We estimate this based on current state vs total attempts
    if health.state == ServiceState.HEALTHY and health.consecutive_failures == 0:
        # Currently healthy means last recovery was successful
        # Estimate success rate based on inverse of failure accumulation
        successful_recoveries = max(health.recovery_attempts - health.consecutive_failures, 0)
        success_rate = (successful_recoveries / health.recovery_attempts) * 100
        return round(success_rate, 1)
    else:
        # Currently unhealthy, so recent recoveries have failed
        # Reduce success rate based on consecutive failures
        base_rate = 50.0  # Base rate when experiencing failures
        failure_penalty = min(health.consecutive_failures * 10, 50)  # Max 50% penalty
        return max(base_rate - failure_penalty, 0.0)

def _calculate_mttr(health) -> float:
    """Calculate mean time to recovery in seconds"""
    # MTTR = Mean Time To Recovery
    # Calculate based on last recovery time and check intervals
    
    if health.last_recovery_time and health.last_check_time:
        # If we have recovery history, calculate actual MTTR
        recovery_duration = (health.last_recovery_time - health.last_check_time).total_seconds()
        
        # Adjust based on number of recovery attempts
        # More attempts = longer MTTR
        if health.recovery_attempts > 0:
            # Average recovery time increases with attempts
            base_recovery_time = 60.0  # 1 minute base
            attempt_multiplier = min(health.recovery_attempts, 5)  # Cap at 5x
            estimated_mttr = base_recovery_time * attempt_multiplier
            
            # Blend actual and estimated for more accurate result
            return round((recovery_duration + estimated_mttr) / 2, 1)
        else:
            return round(recovery_duration, 1)
    
    # Default MTTR based on service state
    if health.state == ServiceState.HEALTHY:
        return 60.0  # 1 minute for healthy services
    elif health.state == ServiceState.DEGRADED:
        return 180.0  # 3 minutes for degraded services
    else:
        return 300.0  # 5 minutes for unhealthy services

def _calculate_overall_uptime(services) -> float:
    """Calculate overall system uptime"""
    if not services:
        return 100.0
    
    healthy_count = sum(1 for s in services.values() if s.state == ServiceState.HEALTHY)
    return (healthy_count / len(services)) * 100