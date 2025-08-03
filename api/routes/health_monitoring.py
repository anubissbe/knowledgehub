"""
Health Monitoring API Routes

Provides comprehensive health monitoring endpoints for production systems
with real-time status, metrics, alerts, and reporting capabilities.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func

from ..services.real_health_monitor import real_health_monitor, HealthStatus, ServiceType
from ..services.prometheus_metrics import prometheus_metrics
from ..models.health_check import (
    HealthCheck, SystemAlert, PerformanceMetric, UptimeRecord, 
    ServiceMetric, HealthReport, ServiceStatus, AlertSeverity
)
from ..dependencies import get_user_id
from ..database import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health_monitoring"])

# Request/Response Models

class HealthStatusResponse(BaseModel):
    """System health status response"""
    overall_status: str
    services: Dict[str, Dict[str, Any]]
    system_metrics: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    timestamp: str
    uptime_summary: Dict[str, Any]

class ServiceHealthResponse(BaseModel):
    """Individual service health response"""
    service_name: str
    service_type: str
    status: str
    response_time_ms: float
    metrics: List[Dict[str, Any]]
    error_message: Optional[str] = None
    last_check: str
    uptime_percentage: float
    consecutive_failures: int

class AlertResponse(BaseModel):
    """Alert response model"""
    id: str
    alert_id: str
    service_name: Optional[str]
    alert_type: str
    severity: str
    message: str
    acknowledged: bool
    resolved: bool
    count: int
    first_occurrence: str
    last_occurrence: str
    created_at: str

class MetricsRequest(BaseModel):
    """Metrics query request"""
    service_names: Optional[List[str]] = None
    metric_names: Optional[List[str]] = None
    time_window_hours: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week
    aggregation: str = Field(default="avg", pattern="^(avg|min|max|sum|count)$")

class AlertAcknowledgeRequest(BaseModel):
    """Alert acknowledgment request"""
    alert_ids: List[str]
    acknowledged_by: str
    notes: Optional[str] = None

class DashboardConfigRequest(BaseModel):
    """Dashboard configuration request"""
    dashboard_name: str
    services_monitored: List[str]
    refresh_interval: int = Field(default=30, ge=5, le=300)
    alert_thresholds: Dict[str, Dict[str, float]]
    is_public: bool = False

# Health Status Endpoints

@router.get("/status", response_model=HealthStatusResponse)
async def get_system_health(
    include_metrics: bool = Query(default=True, description="Include detailed metrics"),
    session: AsyncSession = Depends(get_db_session)
) -> HealthStatusResponse:
    """
    Get comprehensive system health status
    
    Returns the overall health status of all monitored services including:
    - Service statuses and response times
    - System metrics (CPU, memory, disk, network)
    - Active alerts
    - Uptime summary
    """
    
    try:
        # Get real-time health status
        system_health = await real_health_monitor.get_system_health()
        
        # Get uptime summary
        uptime_summary = await get_uptime_summary(session)
        
        # Format services data
        services = {}
        for service_name, service_health in system_health.services.items():
            services[service_name] = {
                "status": service_health.status.value,
                "response_time_ms": service_health.response_time_ms,
                "error_message": service_health.error_message,
                "last_check": service_health.last_check.isoformat(),
                "uptime_percentage": service_health.uptime_percentage,
                "consecutive_failures": service_health.consecutive_failures
            }
            
            if include_metrics:
                services[service_name]["metrics"] = [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "threshold_warning": m.threshold_warning,
                        "threshold_critical": m.threshold_critical,
                        "description": m.description,
                        "timestamp": m.timestamp.isoformat()
                    }
                    for m in service_health.metrics
                ]
        
        # Format system metrics
        system_metrics = []
        if include_metrics:
            system_metrics = [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "threshold_warning": m.threshold_warning,
                    "threshold_critical": m.threshold_critical,
                    "description": m.description,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in system_health.system_metrics
            ]
        
        return HealthStatusResponse(
            overall_status=system_health.overall_status.value,
            services=services,
            system_metrics=system_metrics,
            alerts=system_health.alerts,
            timestamp=system_health.timestamp.isoformat(),
            uptime_summary=uptime_summary
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.get("/services/{service_name}", response_model=ServiceHealthResponse)
async def get_service_health(
    service_name: str,
    include_history: bool = Query(default=False, description="Include recent health history")
) -> ServiceHealthResponse:
    """Get detailed health status for a specific service"""
    
    try:
        service_health = await real_health_monitor.check_service_health(service_name)
        
        response = ServiceHealthResponse(
            service_name=service_health.service_name,
            service_type=service_health.service_type.value,
            status=service_health.status.value,
            response_time_ms=service_health.response_time_ms,
            metrics=[
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "threshold_warning": m.threshold_warning,
                    "threshold_critical": m.threshold_critical,
                    "description": m.description,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in service_health.metrics
            ],
            error_message=service_health.error_message,
            last_check=service_health.last_check.isoformat(),
            uptime_percentage=service_health.uptime_percentage,
            consecutive_failures=service_health.consecutive_failures
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get service health for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Service health check failed: {e}")

@router.post("/check")
async def trigger_manual_health_check(
    service_name: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """Trigger a manual health check for all services or a specific service"""
    
    try:
        # Trigger the check in background
        background_tasks.add_task(
            _perform_manual_check, service_name
        )
        
        return {
            "message": f"Manual health check triggered for {service_name or 'all services'}",
            "triggered_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger manual health check: {e}")
        raise HTTPException(status_code=500, detail=f"Manual check failed: {e}")

# Metrics Endpoints

@router.post("/metrics/query")
async def query_metrics(
    request: MetricsRequest,
    session: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Query historical metrics with filtering and aggregation"""
    
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=request.time_window_hours)
        
        # Build query
        query = select(ServiceMetric).where(
            and_(
                ServiceMetric.timestamp >= start_time,
                ServiceMetric.timestamp <= end_time
            )
        )
        
        if request.service_names:
            query = query.where(ServiceMetric.service_name.in_(request.service_names))
        
        if request.metric_names:
            query = query.where(ServiceMetric.metric_name.in_(request.metric_names))
        
        query = query.order_by(ServiceMetric.timestamp.desc())
        
        result = await session.execute(query)
        metrics = result.scalars().all()
        
        # Group and aggregate metrics
        grouped_metrics = {}
        for metric in metrics:
            key = f"{metric.service_name}.{metric.metric_name}"
            if key not in grouped_metrics:
                grouped_metrics[key] = []
            grouped_metrics[key].append({
                "value": metric.metric_value,
                "timestamp": metric.timestamp.isoformat(),
                "unit": metric.metric_unit
            })
        
        return {
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": request.time_window_hours
            },
            "aggregation": request.aggregation,
            "metrics": grouped_metrics,
            "total_data_points": len(metrics)
        }
        
    except Exception as e:
        logger.error(f"Metrics query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics query failed: {e}")

@router.get("/metrics/realtime")
async def get_realtime_metrics(
    services: Optional[str] = Query(None, description="Comma-separated service names")
) -> Dict[str, Any]:
    """Get real-time metrics for specified services"""
    
    try:
        service_list = services.split(',') if services else None
        
        # Get current health status with metrics
        system_health = await real_health_monitor.get_system_health()
        
        realtime_metrics = {}
        
        for service_name, service_health in system_health.services.items():
            if service_list is None or service_name in service_list:
                realtime_metrics[service_name] = {
                    "status": service_health.status.value,
                    "response_time_ms": service_health.response_time_ms,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": m.value,
                            "unit": m.unit,
                            "timestamp": m.timestamp.isoformat()
                        }
                        for m in service_health.metrics
                    ]
                }
        
        # Add system metrics
        realtime_metrics["system"] = {
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in system_health.system_metrics
            ]
        }
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": realtime_metrics
        }
        
    except Exception as e:
        logger.error(f"Real-time metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Real-time metrics failed: {e}")

# Alerts Endpoints

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    service_name: Optional[str] = Query(None, description="Filter by service"),
    resolved: Optional[bool] = Query(None, description="Filter by resolution status"),
    limit: int = Query(default=100, le=1000),
    session: AsyncSession = Depends(get_db_session)
) -> List[AlertResponse]:
    """Get system alerts with filtering options"""
    
    try:
        query = select(SystemAlert)
        
        if severity:
            query = query.where(SystemAlert.severity == AlertSeverity(severity))
        
        if service_name:
            query = query.where(SystemAlert.service_name == service_name)
        
        if resolved is not None:
            query = query.where(SystemAlert.resolved == resolved)
        
        query = query.order_by(desc(SystemAlert.last_occurrence)).limit(limit)
        
        result = await session.execute(query)
        alerts = result.scalars().all()
        
        return [
            AlertResponse(
                id=str(alert.id),
                alert_id=alert.alert_id,
                service_name=alert.service_name,
                alert_type=alert.alert_type,
                severity=alert.severity.value,
                message=alert.message,
                acknowledged=alert.acknowledged,
                resolved=alert.resolved,
                count=alert.count,
                first_occurrence=alert.first_occurrence.isoformat(),
                last_occurrence=alert.last_occurrence.isoformat(),
                created_at=alert.created_at.isoformat()
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {e}")

@router.post("/alerts/acknowledge")
async def acknowledge_alerts(
    request: AlertAcknowledgeRequest,
    session: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Acknowledge one or more alerts"""
    
    try:
        acknowledged_count = 0
        
        for alert_id in request.alert_ids:
            # Update alert in database
            query = select(SystemAlert).where(SystemAlert.alert_id == alert_id)
            result = await session.execute(query)
            alert = result.scalar_one_or_none()
            
            if alert and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = request.acknowledged_by
                alert.acknowledged_at = datetime.now(timezone.utc)
                acknowledged_count += 1
        
        await session.commit()
        
        return {
            "message": f"Acknowledged {acknowledged_count} alerts",
            "acknowledged_by": request.acknowledged_by,
            "acknowledged_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to acknowledge alerts: {e}")
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alerts: {e}")

# Uptime and Reports

@router.get("/uptime/{service_name}")
async def get_service_uptime(
    service_name: str,
    days: int = Query(default=7, ge=1, le=90),
    session: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get uptime statistics for a service"""
    
    try:
        uptime_stats = await real_health_monitor.get_uptime_statistics(service_name, days)
        return uptime_stats
        
    except Exception as e:
        logger.error(f"Failed to get uptime for {service_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get uptime statistics: {e}")

@router.get("/reports/daily")
async def get_daily_health_report(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    session: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Generate daily health report"""
    
    try:
        if date:
            report_date = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        else:
            report_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate comprehensive daily report
        report = await _generate_daily_report(session, report_date)
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate daily report: {e}")

@router.get("/dashboard/config")
async def get_dashboard_configs(
    user_id: str = Depends(get_user_id),
    session: AsyncSession = Depends(get_db_session)
) -> List[Dict[str, Any]]:
    """Get dashboard configurations for user"""
    
    try:
        # This would query HealthDashboard model
        # For now, return default configuration
        return [
            {
                "id": "default",
                "dashboard_name": "System Overview",
                "services_monitored": ["postgresql", "redis", "weaviate", "knowledgehub_api"],
                "refresh_interval": 30,
                "is_public": True
            }
        ]
        
    except Exception as e:
        logger.error(f"Failed to get dashboard configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard configs: {e}")

# Monitoring Control

@router.post("/monitoring/start")
async def start_monitoring(
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start the health monitoring system"""
    
    try:
        background_tasks.add_task(real_health_monitor.start_monitoring)
        
        return {
            "message": "Health monitoring started",
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {e}")

@router.post("/monitoring/stop")
async def stop_monitoring() -> Dict[str, str]:
    """Stop the health monitoring system"""
    
    try:
        await real_health_monitor.stop_monitoring()
        
        return {
            "message": "Health monitoring stopped",
            "stopped_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {e}")

@router.get("/prometheus/metrics")
async def get_prometheus_metrics() -> str:
    """Get Prometheus metrics in standard format"""
    try:
        return prometheus_metrics.get_metrics()
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")

# Helper Functions

async def get_uptime_summary(session: AsyncSession) -> Dict[str, Any]:
    """Get overall uptime summary"""
    try:
        # Query recent uptime records
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        query = select(UptimeRecord).where(
            and_(
                UptimeRecord.date >= start_date,
                UptimeRecord.date <= end_date
            )
        )
        
        result = await session.execute(query)
        records = result.scalars().all()
        
        if not records:
            return {"overall_uptime": 100.0, "services_count": 0, "period_days": 7}
        
        # Calculate overall uptime
        total_uptime = sum(r.uptime_percentage for r in records)
        avg_uptime = total_uptime / len(records) if records else 100.0
        
        services = set(r.service_name for r in records)
        
        return {
            "overall_uptime": round(avg_uptime, 2),
            "services_count": len(services),
            "period_days": 7,
            "total_incidents": sum(r.failure_count for r in records),
            "avg_mttr_minutes": round(sum(r.mttr_seconds for r in records) / len(records) / 60, 2) if records else 0
        }
        
    except Exception as e:
        logger.warning(f"Failed to get uptime summary: {e}")
        return {"overall_uptime": 0.0, "services_count": 0, "period_days": 7}

async def _perform_manual_check(service_name: Optional[str] = None) -> None:
    """Perform manual health check"""
    try:
        await real_health_monitor.trigger_manual_check(service_name)
        logger.info(f"Manual health check completed for {service_name or 'all services'}")
    except Exception as e:
        logger.error(f"Manual health check failed: {e}")

async def _generate_daily_report(session: AsyncSession, report_date: datetime) -> Dict[str, Any]:
    """Generate comprehensive daily health report"""
    try:
        start_of_day = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        # Query health checks for the day
        query = select(HealthCheck).where(
            and_(
                HealthCheck.timestamp >= start_of_day,
                HealthCheck.timestamp < end_of_day
            )
        )
        
        result = await session.execute(query)
        health_checks = result.scalars().all()
        
        # Query alerts for the day
        alert_query = select(SystemAlert).where(
            and_(
                SystemAlert.created_at >= start_of_day,
                SystemAlert.created_at < end_of_day
            )
        )
        
        alert_result = await session.execute(alert_query)
        alerts = alert_result.scalars().all()
        
        # Generate report
        services_summary = {}
        for check in health_checks:
            if check.service_name not in services_summary:
                services_summary[check.service_name] = {
                    "total_checks": 0,
                    "healthy_checks": 0,
                    "response_times": []
                }
            
            services_summary[check.service_name]["total_checks"] += 1
            if check.status == ServiceStatus.HEALTHY:
                services_summary[check.service_name]["healthy_checks"] += 1
            services_summary[check.service_name]["response_times"].append(check.response_time_ms)
        
        # Calculate uptime for each service
        for service_name in services_summary:
            summary = services_summary[service_name]
            summary["uptime_percentage"] = (
                summary["healthy_checks"] / summary["total_checks"] * 100
                if summary["total_checks"] > 0 else 100
            )
            summary["avg_response_time"] = (
                sum(summary["response_times"]) / len(summary["response_times"])
                if summary["response_times"] else 0
            )
            summary["max_response_time"] = max(summary["response_times"]) if summary["response_times"] else 0
        
        # Alert summary
        alert_summary = {
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
            "warning_alerts": len([a for a in alerts if a.severity == AlertSeverity.WARNING]),
            "resolved_alerts": len([a for a in alerts if a.resolved])
        }
        
        return {
            "report_date": report_date.isoformat(),
            "services_summary": services_summary,
            "alert_summary": alert_summary,
            "overall_uptime": sum(s["uptime_percentage"] for s in services_summary.values()) / len(services_summary) if services_summary else 100,
            "total_health_checks": len(health_checks),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        return {"error": str(e)}