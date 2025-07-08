"""
Rate Limiting and DDoS Protection Management API

Provides endpoints for monitoring and managing the advanced
rate limiting and DDoS protection system.
"""

from fastapi import APIRouter, Request, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..security.rate_limiting import (
    get_rate_limiter,
    RateLimitStrategy,
    ThreatLevel
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class RateLimitStatus(BaseModel):
    """Rate limiting system status"""
    enabled: bool = Field(..., description="Rate limiting enabled")
    strategy: str = Field(..., description="Current rate limiting strategy")
    adaptive_enabled: bool = Field(..., description="Adaptive rate limiting enabled")
    ddos_protection: bool = Field(..., description="DDoS protection enabled")
    global_stats: Dict[str, Any] = Field(..., description="Global statistics")


class ClientStats(BaseModel):
    """Client rate limiting statistics"""
    client_id: str = Field(..., description="Client identifier")
    requests_last_minute: int = Field(..., description="Requests in last minute")
    requests_last_hour: int = Field(..., description="Requests in last hour")
    requests_last_day: int = Field(..., description="Requests in last day")
    threat_score: float = Field(..., description="Current threat score")
    threat_level: str = Field(..., description="Assessed threat level")
    blocked_requests: int = Field(..., description="Number of blocked requests")
    first_seen: str = Field(..., description="First seen timestamp")
    last_request: str = Field(..., description="Last request timestamp")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    requests_per_minute: int = Field(100, description="Requests per minute limit")
    requests_per_hour: int = Field(2000, description="Requests per hour limit")
    requests_per_day: int = Field(20000, description="Requests per day limit")
    burst_limit: int = Field(20, description="Burst request limit")
    strategy: RateLimitStrategy = Field(RateLimitStrategy.SLIDING_WINDOW, description="Rate limiting strategy")
    enable_adaptive: bool = Field(True, description="Enable adaptive rate limiting")
    enable_ddos_protection: bool = Field(True, description="Enable DDoS protection")


class BlacklistRequest(BaseModel):
    """Request to blacklist/whitelist an IP"""
    ip_address: str = Field(..., description="IP address to blacklist/whitelist")
    duration: Optional[int] = Field(3600, description="Blacklist duration in seconds")
    reason: Optional[str] = Field(None, description="Reason for blacklisting")


# Simple API key verification for admin operations
async def verify_admin_key(x_api_key: Optional[str] = None) -> str:
    """Verify admin API key"""
    if not x_api_key or x_api_key != "admin":
        raise HTTPException(status_code=401, detail="Admin API key required")
    return x_api_key


@router.get("/security/rate-limiting/status", response_model=RateLimitStatus)
async def get_rate_limiting_status(
    request: Request,
    api_key: str = Depends(verify_admin_key)
):
    """
    Get rate limiting system status and configuration
    
    Returns current configuration, global statistics,
    and system health information.
    """
    try:
        rate_limiter = get_rate_limiter()
        global_stats = await rate_limiter.get_global_stats()
        
        return RateLimitStatus(
            enabled=True,
            strategy=rate_limiter.config.strategy.value,
            adaptive_enabled=rate_limiter.config.enable_adaptive,
            ddos_protection=rate_limiter.config.enable_ddos_protection,
            global_stats=global_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting rate limiting status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get rate limiting status")


@router.get("/security/rate-limiting/stats")
async def get_global_rate_limiting_stats(
    api_key: str = Depends(verify_admin_key)
):
    """
    Get global rate limiting statistics
    
    Returns comprehensive statistics about rate limiting
    performance, client behavior, and threat detection.
    """
    try:
        rate_limiter = get_rate_limiter()
        stats = await rate_limiter.get_global_stats()
        
        # Add additional calculated metrics
        stats.update({
            "system_health": "healthy" if stats["average_threat_score"] < 2.0 else "degraded",
            "load_level": "high" if stats["global_request_rate"] > 100 else "normal",
            "protection_effectiveness": "active" if stats["blacklisted_ips"] > 0 else "monitoring",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "rate_limiting_stats": stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting rate limiting statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get rate limiting statistics")


@router.get("/security/rate-limiting/clients")
async def list_active_clients(
    api_key: str = Depends(verify_admin_key),
    limit: int = Query(50, description="Maximum number of clients to return"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level")
):
    """
    List active clients with their rate limiting statistics
    
    Returns a list of clients currently being tracked
    by the rate limiting system.
    """
    try:
        rate_limiter = get_rate_limiter()
        
        # Get all client IDs from memory storage
        clients = []
        for client_id in list(rate_limiter.memory_storage.keys())[:limit]:
            try:
                stats = await rate_limiter.get_client_stats(client_id)
                
                # Determine threat level
                threat_score = stats["threat_score"]
                if threat_score >= 6.0:
                    client_threat_level = "critical"
                elif threat_score >= 4.0:
                    client_threat_level = "high"
                elif threat_score >= 2.0:
                    client_threat_level = "medium"
                else:
                    client_threat_level = "low"
                
                # Filter by threat level if specified
                if threat_level and client_threat_level != threat_level.lower():
                    continue
                
                client_info = ClientStats(
                    client_id=client_id,
                    requests_last_minute=stats["requests_last_minute"],
                    requests_last_hour=stats["requests_last_hour"],
                    requests_last_day=stats["requests_last_day"],
                    threat_score=stats["threat_score"],
                    threat_level=client_threat_level,
                    blocked_requests=stats["blocked_requests"],
                    first_seen=stats["first_seen"],
                    last_request=stats["last_request"]
                )
                clients.append(client_info)
                
            except Exception as e:
                logger.warning(f"Error getting stats for client {client_id}: {e}")
                continue
        
        # Sort by threat score (highest first)
        clients.sort(key=lambda x: x.threat_score, reverse=True)
        
        return {
            "clients": clients,
            "total_count": len(clients),
            "filtered_by": threat_level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing active clients: {e}")
        raise HTTPException(status_code=500, detail="Failed to list active clients")


@router.get("/security/rate-limiting/clients/{client_id}")
async def get_client_details(
    client_id: str,
    api_key: str = Depends(verify_admin_key)
):
    """
    Get detailed statistics for a specific client
    
    Returns comprehensive information about a client's
    request patterns and rate limiting status.
    """
    try:
        rate_limiter = get_rate_limiter()
        stats = await rate_limiter.get_client_stats(client_id)
        
        # Calculate additional metrics
        threat_score = stats["threat_score"]
        if threat_score >= 6.0:
            threat_level = "critical"
        elif threat_score >= 4.0:
            threat_level = "high"
        elif threat_score >= 2.0:
            threat_level = "medium"
        else:
            threat_level = "low"
        
        # Calculate request rate
        requests_per_second = stats["requests_last_minute"] / 60.0
        
        return {
            "client_details": {
                **stats,
                "threat_level": threat_level,
                "requests_per_second": round(requests_per_second, 2),
                "analysis": {
                    "high_frequency": stats["requests_last_minute"] > 60,
                    "persistent_activity": stats["requests_last_day"] > 1000,
                    "suspicious_behavior": stats["suspicious_patterns"] > 5,
                    "threat_detected": threat_score > 2.0
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting client details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get client details")


@router.post("/security/rate-limiting/blacklist")
async def blacklist_ip(
    blacklist_request: BlacklistRequest,
    api_key: str = Depends(verify_admin_key)
):
    """
    Manually blacklist an IP address
    
    Adds an IP address to the blacklist for a specified duration.
    Useful for blocking persistent threats manually.
    """
    try:
        rate_limiter = get_rate_limiter()
        
        # Add IP to blacklist
        from datetime import timedelta
        expiry_time = datetime.now() + timedelta(seconds=blacklist_request.duration)
        rate_limiter.blacklisted_ips[blacklist_request.ip_address] = expiry_time
        
        # Log the manual blacklist action
        from ..security.monitoring import log_security_event, SecurityEventType, ThreatLevel
        await log_security_event(
            SecurityEventType.MANUAL_BLOCK,
            ThreatLevel.HIGH,
            blacklist_request.ip_address,
            "Admin",
            "/admin/blacklist",
            "POST",
            f"Manually blacklisted: {blacklist_request.reason or 'No reason provided'}"
        )
        
        return {
            "success": True,
            "message": f"IP {blacklist_request.ip_address} blacklisted successfully",
            "ip_address": blacklist_request.ip_address,
            "duration": blacklist_request.duration,
            "expires_at": expiry_time.isoformat(),
            "reason": blacklist_request.reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error blacklisting IP: {e}")
        raise HTTPException(status_code=500, detail="Failed to blacklist IP")


@router.delete("/security/rate-limiting/blacklist/{ip_address}")
async def remove_ip_from_blacklist(
    ip_address: str,
    api_key: str = Depends(verify_admin_key)
):
    """
    Remove an IP address from the blacklist
    
    Removes an IP from the blacklist, allowing requests
    from that IP to be processed normally.
    """
    try:
        rate_limiter = get_rate_limiter()
        
        if ip_address in rate_limiter.blacklisted_ips:
            del rate_limiter.blacklisted_ips[ip_address]
            
            # Log the removal
            from ..security.monitoring import log_security_event, SecurityEventType, ThreatLevel
            await log_security_event(
                SecurityEventType.MANUAL_UNBLOCK,
                ThreatLevel.LOW,
                ip_address,
                "Admin",
                "/admin/unblock",
                "DELETE",
                "Manually removed from blacklist"
            )
            
            return {
                "success": True,
                "message": f"IP {ip_address} removed from blacklist",
                "ip_address": ip_address,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"IP {ip_address} not found in blacklist",
                "ip_address": ip_address,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error removing IP from blacklist: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove IP from blacklist")


@router.get("/security/rate-limiting/blacklist")
async def list_blacklisted_ips(
    api_key: str = Depends(verify_admin_key)
):
    """
    List all blacklisted IP addresses
    
    Returns a list of currently blacklisted IPs
    with their expiration times.
    """
    try:
        rate_limiter = get_rate_limiter()
        
        blacklisted_ips = []
        current_time = datetime.now()
        
        for ip, expiry in rate_limiter.blacklisted_ips.items():
            time_remaining = (expiry - current_time).total_seconds()
            blacklisted_ips.append({
                "ip_address": ip,
                "expires_at": expiry.isoformat(),
                "time_remaining": max(0, int(time_remaining)),
                "expired": time_remaining <= 0
            })
        
        # Sort by expiration time
        blacklisted_ips.sort(key=lambda x: x["expires_at"])
        
        return {
            "blacklisted_ips": blacklisted_ips,
            "total_count": len(blacklisted_ips),
            "active_count": len([ip for ip in blacklisted_ips if not ip["expired"]]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing blacklisted IPs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list blacklisted IPs")


@router.post("/security/rate-limiting/cleanup")
async def cleanup_rate_limiting_data(
    api_key: str = Depends(verify_admin_key)
):
    """
    Clean up expired rate limiting data
    
    Manually trigger cleanup of expired blacklist entries,
    old client metrics, and request timestamps.
    """
    try:
        rate_limiter = get_rate_limiter()
        
        # Get counts before cleanup
        blacklist_count_before = len(rate_limiter.blacklisted_ips)
        client_count_before = len(rate_limiter.memory_storage)
        
        # Perform cleanup
        await rate_limiter.cleanup_expired_data()
        
        # Get counts after cleanup
        blacklist_count_after = len(rate_limiter.blacklisted_ips)
        client_count_after = len(rate_limiter.memory_storage)
        
        # Calculate cleanup results
        blacklist_cleaned = blacklist_count_before - blacklist_count_after
        clients_cleaned = client_count_before - client_count_after
        
        return {
            "cleanup_completed": True,
            "blacklist_entries_removed": blacklist_cleaned,
            "client_records_removed": clients_cleaned,
            "remaining_blacklisted_ips": blacklist_count_after,
            "remaining_active_clients": client_count_after,
            "cleanup_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up rate limiting data: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup rate limiting data")


@router.get("/security/rate-limiting/test")
async def test_rate_limiting(
    request: Request,
    api_key: str = Depends(verify_admin_key)
):
    """
    Test rate limiting configuration
    
    Returns information about how the current request
    would be handled by the rate limiting system.
    """
    try:
        rate_limiter = get_rate_limiter()
        
        # Get client information
        client_id = rate_limiter._get_client_id(request)
        client_ip = rate_limiter._get_client_ip(request)
        
        # Get client stats
        stats = await rate_limiter.get_client_stats(client_id)
        
        # Test rate limit check (without actually applying it)
        allowed, error_response = await rate_limiter.check_rate_limit(request)
        
        # Determine what would happen
        if allowed:
            result = "allowed"
            message = "Request would be allowed"
        else:
            result = "blocked"
            message = "Request would be blocked"
        
        return {
            "test_result": {
                "client_id": client_id,
                "client_ip": client_ip,
                "result": result,
                "message": message,
                "client_stats": stats,
                "rate_limit_config": {
                    "strategy": rate_limiter.config.strategy.value,
                    "requests_per_minute": rate_limiter.config.requests_per_minute,
                    "adaptive_multiplier": rate_limiter.adaptive_multiplier,
                    "effective_limit": int(rate_limiter.config.requests_per_minute * rate_limiter.adaptive_multiplier)
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing rate limiting: {e}")
        raise HTTPException(status_code=500, detail="Failed to test rate limiting")


@router.get("/security/rate-limiting/health")
async def rate_limiting_health():
    """
    Rate limiting system health check
    
    Returns the health status of the rate limiting system.
    No authentication required for health checks.
    """
    try:
        rate_limiter = get_rate_limiter()
        global_stats = await rate_limiter.get_global_stats()
        
        # Determine health status
        health_status = "healthy"
        if global_stats["average_threat_score"] > 5.0:
            health_status = "critical"
        elif global_stats["average_threat_score"] > 3.0:
            health_status = "degraded"
        elif global_stats["blacklisted_ips"] > 100:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "rate_limiting": "active",
            "ddos_protection": "active" if rate_limiter.config.enable_ddos_protection else "disabled",
            "strategy": rate_limiter.config.strategy.value,
            "active_clients": global_stats["total_clients"],
            "blacklisted_ips": global_stats["blacklisted_ips"],
            "global_request_rate": global_stats["global_request_rate"],
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Rate limiting health check failed: {e}")
        raise HTTPException(status_code=503, detail="Rate limiting system unhealthy")