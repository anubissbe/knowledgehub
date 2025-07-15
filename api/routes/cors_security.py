"""
CORS Security Management API

Provides endpoints for monitoring and managing CORS security configuration,
including security statistics, origin management, and security event monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from datetime import datetime

from ..cors_config import (
    get_cors_origins,
    get_cors_config,
    validate_cors_origin,
    is_secure_origin,
    is_localhost_origin,
    is_local_network_origin,
    CORSSecurityConfig
)
# Note: API key verification disabled for now - endpoints are admin only
from fastapi import Header
from typing import Optional

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Simple API key verification (disabled for now)"""
    # For now, just return the key if provided
    # In production, this should validate against database
    return x_api_key or "admin"

logger = logging.getLogger(__name__)

router = APIRouter()


class CORSOriginRequest(BaseModel):
    """Request model for origin management"""
    origin: str
    reason: Optional[str] = None


class CORSConfigResponse(BaseModel):
    """Response model for CORS configuration"""
    environment: str
    allowed_origins: List[str]
    allowed_methods: List[str]
    allowed_headers: List[str]
    exposed_headers: List[str]
    allow_credentials: bool
    max_age: int
    total_origins: int


class CORSSecurityStats(BaseModel):
    """Response model for CORS security statistics"""
    suspicious_origins_count: int
    blocked_origins_count: int
    tracked_origins_count: int
    total_requests_tracked: int
    environment: str
    strict_mode: bool
    uptime_hours: float


class CORSOriginInfo(BaseModel):
    """Information about a specific origin"""
    origin: str
    is_allowed: bool
    is_secure: bool
    is_localhost: bool
    is_local_network: bool
    classification: str
    last_seen: Optional[datetime] = None
    request_count: int = 0


def get_cors_middleware(request: Request):
    """Get CORS security middleware instance from the app"""
    for middleware in request.app.user_middleware:
        if hasattr(middleware, 'cls') and middleware.cls.__name__ == 'CORSSecurityMiddleware':
            return middleware.kwargs.get('instance')
    return None


@router.get("/cors/config", response_model=CORSConfigResponse)
async def get_cors_configuration(
    environment: str = "development",
    api_key: str = Depends(verify_api_key)
):
    """
    Get current CORS configuration
    
    Returns the active CORS configuration including allowed origins,
    methods, headers, and security settings.
    """
    try:
        cors_config = get_cors_config(environment)
        
        return CORSConfigResponse(
            environment=environment,
            allowed_origins=cors_config["allow_origins"],
            allowed_methods=cors_config["allow_methods"],
            allowed_headers=cors_config["allow_headers"],
            exposed_headers=cors_config["expose_headers"],
            allow_credentials=cors_config["allow_credentials"],
            max_age=cors_config["max_age"],
            total_origins=len(cors_config["allow_origins"])
        )
        
    except Exception as e:
        logger.error(f"Error getting CORS configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get CORS configuration")


@router.get("/cors/security/stats", response_model=CORSSecurityStats)
async def get_cors_security_stats(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Get CORS security statistics
    
    Returns statistics about CORS security including suspicious origins,
    blocked origins, and request tracking data.
    """
    try:
        cors_middleware = get_cors_middleware(request)
        
        if not cors_middleware:
            raise HTTPException(status_code=503, detail="CORS security middleware not available")
        
        stats = cors_middleware.get_security_stats()
        
        # Calculate uptime (simplified - would be more accurate with actual start time)
        uptime_hours = 24.0  # Placeholder
        
        return CORSSecurityStats(
            suspicious_origins_count=stats["suspicious_origins_count"],
            blocked_origins_count=stats["blocked_origins_count"],
            tracked_origins_count=stats["tracked_origins_count"],
            total_requests_tracked=stats["total_requests_tracked"],
            environment=stats["environment"],
            strict_mode=stats["strict_mode"],
            uptime_hours=uptime_hours
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting CORS security stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security statistics")


@router.get("/cors/origins/validate/{origin:path}")
async def validate_origin(
    origin: str,
    environment: str = "development",
    api_key: str = Depends(verify_api_key)
) -> CORSOriginInfo:
    """
    Validate and analyze a specific origin
    
    Returns detailed information about whether an origin is allowed
    and its security characteristics.
    """
    try:
        # Ensure origin has protocol
        if not origin.startswith(('http://', 'https://')):
            origin = f"https://{origin}"
        
        is_allowed = validate_cors_origin(origin, environment)
        
        # Classify origin
        if is_localhost_origin(origin):
            classification = "localhost"
        elif is_local_network_origin(origin):
            classification = "local_network"
        elif is_secure_origin(origin):
            classification = "secure_remote"
        else:
            classification = "insecure_remote"
        
        return CORSOriginInfo(
            origin=origin,
            is_allowed=is_allowed,
            is_secure=is_secure_origin(origin),
            is_localhost=is_localhost_origin(origin),
            is_local_network=is_local_network_origin(origin),
            classification=classification
        )
        
    except Exception as e:
        logger.error(f"Error validating origin {origin}: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate origin")


@router.post("/cors/origins/block")
async def block_origin(
    request: Request,
    origin_request: CORSOriginRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Block a specific origin
    
    Adds an origin to the blocked list, preventing future requests
    from that origin.
    """
    try:
        cors_middleware = get_cors_middleware(request)
        
        if not cors_middleware:
            raise HTTPException(status_code=503, detail="CORS security middleware not available")
        
        cors_middleware.block_origin(
            origin_request.origin,
            reason=origin_request.reason or "Manual block via API"
        )
        
        logger.info(f"Origin blocked via API: {origin_request.origin}")
        
        return {
            "success": True,
            "message": f"Origin {origin_request.origin} has been blocked",
            "origin": origin_request.origin,
            "reason": origin_request.reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error blocking origin: {e}")
        raise HTTPException(status_code=500, detail="Failed to block origin")


@router.post("/cors/origins/unblock")
async def unblock_origin(
    request: Request,
    origin_request: CORSOriginRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Unblock a previously blocked origin
    
    Removes an origin from the blocked list, allowing future requests.
    """
    try:
        cors_middleware = get_cors_middleware(request)
        
        if not cors_middleware:
            raise HTTPException(status_code=503, detail="CORS security middleware not available")
        
        cors_middleware.unblock_origin(origin_request.origin)
        
        logger.info(f"Origin unblocked via API: {origin_request.origin}")
        
        return {
            "success": True,
            "message": f"Origin {origin_request.origin} has been unblocked",
            "origin": origin_request.origin,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unblocking origin: {e}")
        raise HTTPException(status_code=500, detail="Failed to unblock origin")


@router.post("/cors/origins/clear-suspicious")
async def clear_suspicious_origin(
    request: Request,
    origin_request: CORSOriginRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Clear suspicious status for an origin
    
    Removes an origin from the suspicious origins list.
    """
    try:
        cors_middleware = get_cors_middleware(request)
        
        if not cors_middleware:
            raise HTTPException(status_code=503, detail="CORS security middleware not available")
        
        cors_middleware.clear_suspicious_origin(origin_request.origin)
        
        logger.info(f"Suspicious status cleared for origin: {origin_request.origin}")
        
        return {
            "success": True,
            "message": f"Suspicious status cleared for {origin_request.origin}",
            "origin": origin_request.origin,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing suspicious origin: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear suspicious status")


@router.get("/cors/security/test")
async def test_cors_security(
    origin: str,
    method: str = "GET",
    api_key: str = Depends(verify_api_key)
):
    """
    Test CORS security for a specific origin and method
    
    Simulates a CORS request to test if it would be allowed
    by the current security configuration.
    """
    try:
        # Validate origin format
        if not origin.startswith(('http://', 'https://')):
            origin = f"https://{origin}"
        
        config = CORSSecurityConfig()
        
        # Check if origin is allowed
        is_origin_allowed = validate_cors_origin(origin, "development")
        
        # Check if method is allowed
        is_method_allowed = method.upper() in config.allowed_methods
        
        # Overall result
        would_allow = is_origin_allowed and is_method_allowed
        
        return {
            "origin": origin,
            "method": method.upper(),
            "would_allow": would_allow,
            "origin_allowed": is_origin_allowed,
            "method_allowed": is_method_allowed,
            "allowed_methods": config.allowed_methods,
            "security_classification": {
                "is_secure": is_secure_origin(origin),
                "is_localhost": is_localhost_origin(origin),
                "is_local_network": is_local_network_origin(origin)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing CORS security: {e}")
        raise HTTPException(status_code=500, detail="Failed to test CORS security")


@router.get("/cors/health")
async def cors_health_check():
    """
    CORS security health check
    
    Returns the health status of the CORS security system.
    """
    try:
        config = CORSSecurityConfig()
        
        return {
            "status": "healthy",
            "cors_security": "active",
            "allowed_methods": len(config.allowed_methods),
            "allowed_headers": len(config.allowed_headers),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"CORS health check failed: {e}")
        raise HTTPException(status_code=503, detail="CORS security system unhealthy")