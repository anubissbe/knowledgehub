"""
Alert Management API Routes

Provides comprehensive alert management endpoints for receiving webhooks,
managing alerts, configuring notification channels, and viewing statistics.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from ..services.alert_service import real_alert_service, AlertPriority, AlertStatus
from ..models.health_check import SystemAlert, AlertSeverity
from ..dependencies import get_user_id
from ..database import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alert_management"])

# Request/Response Models

class AlertWebhookRequest(BaseModel):
    """AlertManager webhook request format"""
    version: str = "4"
    groupKey: str
    status: str
    receiver: str
    groupLabels: Dict[str, str]
    commonLabels: Dict[str, str]
    commonAnnotations: Dict[str, str]
    externalURL: str
    alerts: List[Dict[str, Any]]

class AlertAcknowledgeRequest(BaseModel):
    """Alert acknowledgment request"""
    alert_ids: List[str]
    acknowledged_by: str
    notes: Optional[str] = None

class AlertResolveRequest(BaseModel):
    """Alert resolution request"""
    alert_ids: List[str]
    resolved_by: str
    notes: Optional[str] = None

class NotificationChannelRequest(BaseModel):
    """Notification channel configuration"""
    name: str
    type: str = Field(pattern="^(email|slack|webhook|teams|sms)$")
    configuration: Dict[str, Any]
    priority_filter: List[str] = Field(default=["critical", "high"])
    enabled: bool = True

class AlertRuleRequest(BaseModel):
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    duration: int
    severity: str = Field(pattern="^(info|warning|critical)$")
    priority: str = Field(pattern="^(low|medium|high|critical)$")
    escalation_delay: int = 300
    auto_resolve: bool = True
    recovery_actions: List[str] = Field(default_factory=list)

class AlertResponse(BaseModel):
    """Alert response model"""
    alert_id: str
    name: str
    message: str
    severity: str
    priority: str
    status: str
    service_name: Optional[str]
    starts_at: str
    ends_at: Optional[str]
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[str]
    resolved: bool
    resolved_at: Optional[str]
    count: int
    labels: Dict[str, str]
    annotations: Dict[str, str]

class AlertStatisticsResponse(BaseModel):
    """Alert statistics response"""
    total_alerts: int
    active_alerts: int
    resolved_alerts: int
    acknowledged_alerts: int
    escalated_alerts: int
    auto_resolved_alerts: int
    active_by_priority: Dict[str, int]
    active_by_severity: Dict[str, int]
    pending_notifications: int
    configured_rules: int

# Webhook Endpoints

@router.post("/webhook")
async def receive_alertmanager_webhook(
    request: AlertWebhookRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Receive alerts from Prometheus AlertManager webhook
    
    This endpoint processes incoming alerts from AlertManager and triggers
    the appropriate notification and recovery workflows.
    """
    
    try:
        logger.info(f"Received AlertManager webhook with {len(request.alerts)} alerts")
        
        # Process webhook in background to ensure fast response
        background_tasks.add_task(
            _process_webhook_background,
            request.dict()
        )
        
        return {
            "status": "accepted",
            "message": f"Processing {len(request.alerts)} alerts",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to receive webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {e}")

@router.post("/webhook/test")
async def test_webhook() -> Dict[str, str]:
    """Test webhook endpoint for connectivity verification"""
    return {
        "status": "ok",
        "message": "Webhook endpoint is accessible",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Alert Management Endpoints

@router.get("/active", response_model=List[AlertResponse])
async def get_active_alerts(
    priority: Optional[str] = None,
    severity: Optional[str] = None,
    service: Optional[str] = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_db_session)
) -> List[AlertResponse]:
    """Get currently active alerts with filtering options"""
    
    try:
        query = select(SystemAlert).where(
            and_(
                SystemAlert.resolved == False,
                SystemAlert.acknowledged == False
            )
        )
        
        if priority:
            # Note: We'd need to add priority field to SystemAlert model
            pass
        
        if severity:
            query = query.where(SystemAlert.severity == AlertSeverity(severity))
        
        if service:
            query = query.where(SystemAlert.service_name == service)
        
        query = query.order_by(desc(SystemAlert.last_occurrence)).limit(limit)
        
        result = await session.execute(query)
        alerts = result.scalars().all()
        
        return [
            AlertResponse(
                alert_id=alert.alert_id,
                name=alert.alert_type,
                message=alert.message,
                severity=alert.severity.value,
                priority="high",  # Default, would come from alert service
                status="active",
                service_name=alert.service_name,
                starts_at=alert.first_occurrence.isoformat(),
                ends_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
                acknowledged=alert.acknowledged,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                resolved=alert.resolved,
                resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
                count=alert.count,
                labels=alert.details.get("labels", {}) if alert.details else {},
                annotations=alert.details.get("annotations", {}) if alert.details else {}
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {e}")

@router.get("/history")
async def get_alert_history(
    hours: int = 24,
    severity: Optional[str] = None,
    service: Optional[str] = None,
    limit: int = 500,
    session: AsyncSession = Depends(get_db_session)
) -> List[AlertResponse]:
    """Get alert history for the specified time period"""
    
    try:
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        query = select(SystemAlert).where(
            SystemAlert.created_at >= start_time
        )
        
        if severity:
            query = query.where(SystemAlert.severity == AlertSeverity(severity))
        
        if service:
            query = query.where(SystemAlert.service_name == service)
        
        query = query.order_by(desc(SystemAlert.created_at)).limit(limit)
        
        result = await session.execute(query)
        alerts = result.scalars().all()
        
        return [
            AlertResponse(
                alert_id=alert.alert_id,
                name=alert.alert_type,
                message=alert.message,
                severity=alert.severity.value,
                priority="medium",  # Default
                status="resolved" if alert.resolved else "active",
                service_name=alert.service_name,
                starts_at=alert.first_occurrence.isoformat(),
                ends_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
                acknowledged=alert.acknowledged,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                resolved=alert.resolved,
                resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
                count=alert.count,
                labels=alert.details.get("labels", {}) if alert.details else {},
                annotations=alert.details.get("annotations", {}) if alert.details else {}
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Failed to get alert history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert history: {e}")

@router.post("/acknowledge")
async def acknowledge_alerts(
    request: AlertAcknowledgeRequest
) -> Dict[str, Any]:
    """Acknowledge one or more alerts"""
    
    try:
        acknowledged_count = 0
        failed_alerts = []
        
        for alert_id in request.alert_ids:
            success = await real_alert_service.acknowledge_alert(
                alert_id=alert_id,
                acknowledged_by=request.acknowledged_by,
                notes=request.notes or ""
            )
            
            if success:
                acknowledged_count += 1
            else:
                failed_alerts.append(alert_id)
        
        return {
            "acknowledged_count": acknowledged_count,
            "failed_alerts": failed_alerts,
            "acknowledged_by": request.acknowledged_by,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to acknowledge alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alerts: {e}")

@router.post("/resolve")
async def resolve_alerts(
    request: AlertResolveRequest
) -> Dict[str, Any]:
    """Manually resolve one or more alerts"""
    
    try:
        resolved_count = 0
        failed_alerts = []
        
        for alert_id in request.alert_ids:
            success = await real_alert_service.resolve_alert(
                alert_id=alert_id,
                resolved_by=request.resolved_by,
                notes=request.notes or ""
            )
            
            if success:
                resolved_count += 1
            else:
                failed_alerts.append(alert_id)
        
        return {
            "resolved_count": resolved_count,
            "failed_alerts": failed_alerts,
            "resolved_by": request.resolved_by,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to resolve alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve alerts: {e}")

# Statistics and Monitoring

@router.get("/statistics", response_model=AlertStatisticsResponse)
async def get_alert_statistics() -> AlertStatisticsResponse:
    """Get comprehensive alert statistics"""
    
    try:
        stats = await real_alert_service.get_alert_statistics()
        
        return AlertStatisticsResponse(
            total_alerts=stats['total_alerts'],
            active_alerts=stats['active_alerts'],
            resolved_alerts=stats['resolved_alerts'],
            acknowledged_alerts=stats['acknowledged_alerts'],
            escalated_alerts=stats['escalated_alerts'],
            auto_resolved_alerts=stats['auto_resolved_alerts'],
            active_by_priority=stats['active_by_priority'],
            active_by_severity=stats['active_by_severity'],
            pending_notifications=stats['pending_notifications'],
            configured_rules=stats['configured_rules']
        )
        
    except Exception as e:
        logger.error(f"Failed to get alert statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")

@router.get("/summary")
async def get_alert_summary() -> Dict[str, Any]:
    """Get a quick summary of current alert status"""
    
    try:
        stats = await real_alert_service.get_alert_statistics()
        
        # Determine overall status
        critical_alerts = stats['active_by_priority'].get('critical', 0)
        high_alerts = stats['active_by_priority'].get('high', 0)
        
        if critical_alerts > 0:
            overall_status = "critical"
        elif high_alerts > 0:
            overall_status = "warning"
        elif stats['active_alerts'] > 0:
            overall_status = "attention"
        else:
            overall_status = "ok"
        
        return {
            "overall_status": overall_status,
            "active_alerts": stats['active_alerts'],
            "critical_alerts": critical_alerts,
            "high_priority_alerts": high_alerts,
            "pending_notifications": stats['pending_notifications'],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get alert summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {e}")

# Configuration Endpoints

@router.get("/rules")
async def get_alert_rules() -> Dict[str, Any]:
    """Get configured alert rules"""
    
    try:
        # Get rules from the real alert service
        rules = await real_alert_service.get_alert_rules()
        
        # Convert to response format
        rules_data = []
        for rule in rules:
            rules_data.append({
                "name": rule.name,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "duration": rule.duration,
                "severity": rule.severity.value,
                "priority": rule.priority.value,
                "escalation_delay": rule.escalation_delay,
                "auto_resolve": rule.auto_resolve,
                "recovery_actions": rule.recovery_actions
            })
        
        return {
            "rules": rules_data,
            "total_rules": len(rules_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get alert rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {e}")

@router.post("/rules")
async def create_alert_rule(
    request: AlertRuleRequest
) -> Dict[str, Any]:
    """Create a new alert rule"""
    
    try:
        # Create the rule using the real alert service
        rule_data = {
            "name": request.name,
            "condition": request.condition,
            "threshold": request.threshold,
            "duration": request.duration,
            "severity": request.severity,
            "priority": request.priority,
            "escalation_delay": request.escalation_delay,
            "auto_resolve": request.auto_resolve,
            "recovery_actions": request.recovery_actions
        }
        
        rule = await real_alert_service.create_alert_rule(rule_data)
        
        return {
            "message": f"Alert rule '{rule.name}' created successfully",
            "rule_id": rule.name,
            "rule": {
                "name": rule.name,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "duration": rule.duration,
                "severity": rule.severity.value,
                "priority": rule.priority.value,
                "escalation_delay": rule.escalation_delay,
                "auto_resolve": rule.auto_resolve,
                "recovery_actions": rule.recovery_actions
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create rule: {e}")

# System Control

@router.post("/system/start")
async def start_alert_system(
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start the alert processing system"""
    
    try:
        background_tasks.add_task(real_alert_service.start_processing)
        
        return {
            "message": "Alert processing system started",
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start alert system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start system: {e}")

@router.get("/health")
async def get_alert_system_health() -> Dict[str, Any]:
    """Get alert system health status"""
    
    try:
        stats = await real_alert_service.get_alert_statistics()
        
        return {
            "status": "healthy",
            "active_alerts": stats['active_alerts'],
            "processing_queue": stats['pending_notifications'],
            "last_check": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": 0  # Would track actual uptime
        }
        
    except Exception as e:
        logger.error(f"Failed to get alert system health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now(timezone.utc).isoformat()
        }

# Helper Functions

async def _process_webhook_background(webhook_data: Dict[str, Any]) -> None:
    """Process webhook data in background"""
    try:
        await real_alert_service.process_webhook_alert(webhook_data)
        logger.info("Webhook processing completed successfully")
    except Exception as e:
        logger.error(f"Background webhook processing failed: {e}")