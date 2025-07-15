"""
Security Monitoring API Endpoints

Provides API endpoints for security monitoring, threat analysis,
and incident response management.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from ..security.monitoring import (
    security_monitor,
    SecurityEventType,
    ThreatLevel,
    SecurityEvent,
    log_security_event
)
from ..security.dashboard import security_dashboard

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class SecurityStatsResponse(BaseModel):
    """Security statistics response model"""
    monitoring_status: str
    total_events_last_hour: int
    total_events_last_24h: int
    blocked_ips_count: int
    suspicious_ips_count: int
    event_types_24h: Dict[str, int]
    threat_levels_24h: Dict[str, int]
    top_threatening_ips: List[Dict[str, Any]]
    system_health: Dict[str, Any]


class ThreatIntelResponse(BaseModel):
    """Threat intelligence response model"""
    ip_address: str
    threat_score: int
    is_blocked: bool
    is_suspicious: bool
    event_count: int
    recent_events: List[Dict[str, Any]]
    risk_assessment: str


class SecurityEventRequest(BaseModel):
    """Manual security event logging request"""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_agent: str = ""
    endpoint: str
    method: str = "GET"
    description: str
    metadata: Optional[Dict[str, Any]] = None


class IPActionRequest(BaseModel):
    """IP action request model"""
    ip_address: str = Field(..., description="IP address to act on")
    reason: Optional[str] = Field(None, description="Reason for action")


class SecurityConfigResponse(BaseModel):
    """Security configuration response"""
    monitoring_enabled: bool
    auto_blocking_enabled: bool
    threat_detection_patterns: int
    retention_days: int
    alert_thresholds: Dict[str, Any]


# API key verification (simplified for now)
async def verify_admin_api_key(x_api_key: Optional[str] = None) -> str:
    """Verify admin API key for security endpoints"""
    # In production, this should validate against database/vault
    if not x_api_key or x_api_key != "admin":
        raise HTTPException(status_code=401, detail="Admin API key required")
    return x_api_key


@router.get("/security/monitoring/stats", response_model=SecurityStatsResponse)
async def get_security_stats(
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get comprehensive security monitoring statistics
    
    Returns detailed statistics about security events, threats,
    and system health over the last 24 hours.
    """
    try:
        stats = security_monitor.get_security_stats()
        return SecurityStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting security stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security statistics")


@router.get("/security/monitoring/events")
async def get_security_events(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return"),
    event_type: Optional[SecurityEventType] = Query(None, description="Filter by event type"),
    threat_level: Optional[ThreatLevel] = Query(None, description="Filter by threat level"),
    source_ip: Optional[str] = Query(None, description="Filter by source IP"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get recent security events with filtering options
    
    Returns a list of security events with optional filtering
    by event type, threat level, source IP, and time range.
    """
    try:
        # Filter events based on parameters
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_events = []
        
        for event in security_monitor.events:
            if event.timestamp < cutoff_time:
                continue
                
            if event_type and event.event_type != event_type:
                continue
                
            if threat_level and event.threat_level != threat_level:
                continue
                
            if source_ip and event.source_ip != source_ip:
                continue
            
            # Convert event to dict for JSON serialization
            event_dict = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "source_ip": event.source_ip,
                "user_agent": event.user_agent,
                "endpoint": event.endpoint,
                "method": event.method,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "origin": event.origin,
                "description": event.description,
                "metadata": event.metadata,
                "blocked": event.blocked
            }
            
            filtered_events.append(event_dict)
            
            if len(filtered_events) >= limit:
                break
        
        return {
            "events": filtered_events,
            "total_found": len(filtered_events),
            "filters_applied": {
                "event_type": event_type.value if event_type else None,
                "threat_level": threat_level.value if threat_level else None,
                "source_ip": source_ip,
                "hours_back": hours
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting security events: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security events")


@router.get("/security/monitoring/threats/{ip_address}", response_model=ThreatIntelResponse)
async def get_threat_intelligence(
    ip_address: str,
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get threat intelligence for a specific IP address
    
    Returns detailed threat analysis including risk assessment,
    event history, and current status for the specified IP.
    """
    try:
        # Get events for this IP
        ip_events = list(security_monitor.ip_events.get(ip_address, []))
        
        # Calculate threat score
        threat_score = 0
        for event in ip_events:
            if event.threat_level == ThreatLevel.CRITICAL:
                threat_score += 10
            elif event.threat_level == ThreatLevel.HIGH:
                threat_score += 5
            elif event.threat_level == ThreatLevel.MEDIUM:
                threat_score += 2
            else:
                threat_score += 1
        
        # Risk assessment
        if threat_score >= 50:
            risk_assessment = "CRITICAL - Immediate action required"
        elif threat_score >= 20:
            risk_assessment = "HIGH - Monitor closely"
        elif threat_score >= 10:
            risk_assessment = "MEDIUM - Potential threat"
        elif threat_score > 0:
            risk_assessment = "LOW - Minor activity"
        else:
            risk_assessment = "CLEAN - No threats detected"
        
        # Recent events (last 10)
        recent_events = []
        for event in ip_events[-10:]:
            recent_events.append({
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "endpoint": event.endpoint,
                "description": event.description,
                "blocked": event.blocked
            })
        
        return ThreatIntelResponse(
            ip_address=ip_address,
            threat_score=threat_score,
            is_blocked=security_monitor.is_ip_blocked(ip_address),
            is_suspicious=security_monitor.is_ip_suspicious(ip_address),
            event_count=len(ip_events),
            recent_events=recent_events,
            risk_assessment=risk_assessment
        )
        
    except Exception as e:
        logger.error(f"Error getting threat intelligence for {ip_address}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get threat intelligence")


@router.post("/security/monitoring/events")
async def log_manual_security_event(
    event_request: SecurityEventRequest,
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Manually log a security event
    
    Allows administrators to manually log security events
    for incidents detected by external systems or manual analysis.
    """
    try:
        await log_security_event(
            event_request.event_type,
            event_request.threat_level,
            event_request.source_ip,
            event_request.user_agent,
            event_request.endpoint,
            event_request.method,
            event_request.description,
            metadata=event_request.metadata
        )
        
        return {
            "success": True,
            "message": "Security event logged successfully",
            "event_type": event_request.event_type.value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error logging manual security event: {e}")
        raise HTTPException(status_code=500, detail="Failed to log security event")


@router.post("/security/monitoring/block-ip")
async def block_ip_address(
    request: IPActionRequest,
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Block an IP address
    
    Manually block an IP address from accessing the API.
    Blocked IPs will receive 403 responses for all requests.
    """
    try:
        security_monitor.block_ip(
            request.ip_address, 
            request.reason or "Manual block via API"
        )
        
        # Log the blocking action
        await log_security_event(
            SecurityEventType.API_ABUSE,
            ThreatLevel.HIGH,
            request.ip_address,
            "admin-action",
            "/security/monitoring/block-ip",
            "POST",
            f"IP manually blocked: {request.reason or 'No reason provided'}",
            blocked=True
        )
        
        return {
            "success": True,
            "message": f"IP {request.ip_address} has been blocked",
            "ip_address": request.ip_address,
            "reason": request.reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error blocking IP {request.ip_address}: {e}")
        raise HTTPException(status_code=500, detail="Failed to block IP address")


@router.post("/security/monitoring/unblock-ip")
async def unblock_ip_address(
    request: IPActionRequest,
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Unblock an IP address
    
    Remove an IP address from the blocked list,
    allowing it to access the API again.
    """
    try:
        security_monitor.unblock_ip(request.ip_address)
        
        # Log the unblocking action
        await log_security_event(
            SecurityEventType.AUTHENTICATION_SUCCESS,
            ThreatLevel.LOW,
            request.ip_address,
            "admin-action",
            "/security/monitoring/unblock-ip",
            "POST",
            f"IP manually unblocked: {request.reason or 'Admin action'}",
            blocked=False
        )
        
        return {
            "success": True,
            "message": f"IP {request.ip_address} has been unblocked",
            "ip_address": request.ip_address,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error unblocking IP {request.ip_address}: {e}")
        raise HTTPException(status_code=500, detail="Failed to unblock IP address")


@router.get("/security/monitoring/blocked-ips")
async def get_blocked_ips(
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get list of currently blocked IP addresses
    
    Returns a list of all currently blocked IPs with
    their blocking reasons and timestamps.
    """
    try:
        blocked_ips = list(security_monitor.blocked_ips)
        suspicious_ips = list(security_monitor.suspicious_ips)
        
        # Get additional info for each IP
        ip_details = []
        for ip in blocked_ips:
            ip_events = list(security_monitor.ip_events.get(ip, []))
            last_event = ip_events[-1] if ip_events else None
            
            ip_details.append({
                "ip_address": ip,
                "status": "blocked",
                "event_count": len(ip_events),
                "last_seen": last_event.timestamp.isoformat() if last_event else None,
                "last_event_type": last_event.event_type.value if last_event else None
            })
        
        return {
            "blocked_ips": ip_details,
            "suspicious_ips": list(suspicious_ips),
            "total_blocked": len(blocked_ips),
            "total_suspicious": len(suspicious_ips),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting blocked IPs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get blocked IP list")


@router.get("/security/monitoring/config", response_model=SecurityConfigResponse)
async def get_security_config(
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get current security monitoring configuration
    
    Returns the current configuration settings for
    the security monitoring system.
    """
    try:
        config = security_monitor.config
        
        return SecurityConfigResponse(
            monitoring_enabled=True,
            auto_blocking_enabled=True,
            threat_detection_patterns=len(security_monitor.config.get("suspicious_user_agents", [])),
            retention_days=30,  # Default retention
            alert_thresholds=config
        )
        
    except Exception as e:
        logger.error(f"Error getting security config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security configuration")


@router.post("/security/monitoring/cleanup")
async def cleanup_old_events(
    days_to_keep: int = Query(30, ge=1, le=365, description="Days of events to keep"),
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Clean up old security events and logs
    
    Remove security events older than the specified number of days
    to manage storage and improve performance.
    """
    try:
        await security_monitor.cleanup_old_events(days_to_keep)
        
        return {
            "success": True,
            "message": f"Cleaned up events older than {days_to_keep} days",
            "days_kept": days_to_keep,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up old events: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup old events")


@router.get("/security/monitoring/dashboard")
async def get_security_dashboard(
    hours_back: int = Query(24, ge=1, le=168, description="Hours of data to include"),
    api_key: str = Depends(verify_admin_api_key)
):
    """
    Get comprehensive security dashboard data
    
    Returns aggregated security data including threat analysis,
    geographic distribution, timeline data, and recommendations.
    """
    try:
        dashboard_data = security_dashboard.get_dashboard_data(hours_back)
        
        return {
            "dashboard_data": dashboard_data,
            "generated_at": datetime.now().isoformat(),
            "data_period_hours": hours_back
        }
        
    except Exception as e:
        logger.error(f"Error generating security dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate security dashboard")


@router.get("/security/monitoring/health")
async def security_monitoring_health():
    """
    Security monitoring system health check
    
    Returns the health status of the security monitoring system.
    No authentication required for health checks.
    """
    try:
        stats = security_monitor.get_security_stats()
        
        return {
            "status": "healthy",
            "monitoring": "active",
            "events_tracked": stats["system_health"]["events_in_memory"],
            "ips_tracked": stats["system_health"]["tracking_ips"],
            "blocked_ips": stats["blocked_ips_count"],
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Security monitoring health check failed: {e}")
        raise HTTPException(status_code=503, detail="Security monitoring system unhealthy")