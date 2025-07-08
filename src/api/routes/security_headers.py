"""
Security Headers and CSRF Management API

Provides endpoints for managing security headers configuration,
CSRF token generation, and security header monitoring.
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..security.headers import (
    security_headers_manager,
    SecurityHeaderLevel,
    CSRFConfig,
    get_csrf_protection_stats,
    cleanup_expired_csrf_tokens
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class SecurityHeadersConfig(BaseModel):
    """Security headers configuration model"""
    level: SecurityHeaderLevel = Field(..., description="Security header level")
    csrf_enabled: bool = Field(True, description="Enable CSRF protection")
    csrf_token_lifetime: int = Field(3600, description="CSRF token lifetime in seconds")
    trusted_origins: List[str] = Field([], description="Trusted origins for CSRF")


class CSRFTokenResponse(BaseModel):
    """CSRF token response model"""
    csrf_token: str = Field(..., description="CSRF token")
    expires_at: datetime = Field(..., description="Token expiration time")
    session_id: str = Field(..., description="Session identifier")


class SecurityHeadersStatus(BaseModel):
    """Security headers status response"""
    security_level: str = Field(..., description="Current security level")
    csrf_protection: Dict[str, Any] = Field(..., description="CSRF protection status")
    headers_applied: Dict[str, str] = Field(..., description="Applied security headers")
    csp_policy: str = Field(..., description="Content Security Policy")
    feature_policy: str = Field(..., description="Feature/Permissions Policy")


class TrustedOriginRequest(BaseModel):
    """Request to add/remove trusted origin"""
    origin: str = Field(..., description="Origin URL")
    reason: Optional[str] = Field(None, description="Reason for change")


# Simple API key verification for demo
async def verify_admin_key(x_api_key: Optional[str] = None) -> str:
    """Verify admin API key"""
    if not x_api_key or x_api_key != "admin":
        raise HTTPException(status_code=401, detail="Admin API key required")
    return x_api_key


@router.get("/security/headers/status", response_model=SecurityHeadersStatus)
async def get_security_headers_status(
    request: Request,
    api_key: str = Depends(verify_admin_key)
):
    """
    Get current security headers status and configuration
    
    Returns detailed information about security headers,
    CSRF protection status, and applied policies.
    """
    try:
        # Get CSRF protection statistics
        csrf_stats = get_csrf_protection_stats()
        
        # Get security headers configuration
        headers_config = security_headers_manager.security_headers
        
        # Build CSP policy string
        csp_policy = security_headers_manager._build_csp_header(request)
        
        # Build feature policy string
        feature_policy = security_headers_manager._build_feature_policy_header()
        
        return SecurityHeadersStatus(
            security_level=security_headers_manager.level.value,
            csrf_protection=csrf_stats,
            headers_applied=headers_config,
            csp_policy=csp_policy,
            feature_policy=feature_policy
        )
        
    except Exception as e:
        logger.error(f"Error getting security headers status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get security headers status")


@router.post("/security/headers/csrf/token", response_model=CSRFTokenResponse)
async def generate_csrf_token(
    request: Request,
    api_key: str = Depends(verify_admin_key)
):
    """
    Generate a new CSRF token
    
    Creates a new CSRF token for the current session
    and returns token details.
    """
    try:
        # Generate CSRF token
        csrf_token = security_headers_manager.generate_csrf_token(request)
        
        # Get token expiration time
        expires_at = datetime.fromtimestamp(
            security_headers_manager.csrf_tokens[csrf_token]["timestamp"] + 
            security_headers_manager.csrf_config.token_lifetime
        )
        
        # Get session ID
        session_id = security_headers_manager._get_session_id(request)
        
        return CSRFTokenResponse(
            csrf_token=csrf_token,
            expires_at=expires_at,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error generating CSRF token: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate CSRF token")


class CSRFValidationRequest(BaseModel):
    """CSRF token validation request model"""
    csrf_token: str = Field(..., description="CSRF token to validate")


@router.post("/security/headers/csrf/validate")
async def validate_csrf_token(
    request: Request,
    validation_request: CSRFValidationRequest,
    api_key: str = Depends(verify_admin_key)
):
    """
    Validate a CSRF token
    
    Checks if the provided CSRF token is valid
    for the current session.
    """
    try:
        # Validate the token
        is_valid = security_headers_manager.validate_csrf_token(validation_request.csrf_token, request)
        
        # Get token info if valid
        token_info = None
        if is_valid and validation_request.csrf_token in security_headers_manager.csrf_tokens:
            token_data = security_headers_manager.csrf_tokens[validation_request.csrf_token]
            token_info = {
                "created_at": token_data["created_at"].isoformat(),
                "session_id": token_data["session_id"],
                "used": token_data.get("used", False)
            }
        
        return {
            "valid": is_valid,
            "token": validation_request.csrf_token,
            "token_info": token_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error validating CSRF token: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate CSRF token")


@router.get("/security/headers/csrf/stats")
async def get_csrf_statistics(
    api_key: str = Depends(verify_admin_key)
):
    """
    Get CSRF protection statistics
    
    Returns detailed statistics about CSRF token usage,
    expiration, and protection effectiveness.
    """
    try:
        stats = get_csrf_protection_stats()
        
        # Add additional statistics
        stats.update({
            "token_lifetime_hours": stats["token_lifetime"] / 3600,
            "protection_active": stats["csrf_enabled"],
            "cleanup_needed": stats["expired_tokens"] > 0
        })
        
        return {
            "csrf_statistics": stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting CSRF statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get CSRF statistics")


@router.post("/security/headers/csrf/cleanup")
async def cleanup_csrf_tokens(
    api_key: str = Depends(verify_admin_key)
):
    """
    Clean up expired CSRF tokens
    
    Removes expired and used CSRF tokens from memory
    to optimize performance.
    """
    try:
        # Get stats before cleanup
        stats_before = get_csrf_protection_stats()
        
        # Perform cleanup
        cleanup_expired_csrf_tokens()
        
        # Get stats after cleanup
        stats_after = get_csrf_protection_stats()
        
        # Calculate cleanup results
        tokens_removed = stats_before["active_tokens"] - stats_after["active_tokens"]
        
        return {
            "cleanup_completed": True,
            "tokens_removed": tokens_removed,
            "tokens_remaining": stats_after["active_tokens"],
            "cleanup_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up CSRF tokens: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup CSRF tokens")


@router.post("/security/headers/csrf/trusted-origins")
async def add_trusted_origin(
    request_data: TrustedOriginRequest,
    api_key: str = Depends(verify_admin_key)
):
    """
    Add trusted origin for CSRF protection
    
    Adds an origin to the trusted origins list,
    allowing CSRF requests from that origin.
    """
    try:
        # Validate origin format
        if not request_data.origin.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid origin format")
        
        # Add trusted origin
        security_headers_manager.add_trusted_origin(request_data.origin)
        
        return {
            "success": True,
            "message": f"Added trusted origin: {request_data.origin}",
            "origin": request_data.origin,
            "reason": request_data.reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding trusted origin: {e}")
        raise HTTPException(status_code=500, detail="Failed to add trusted origin")


@router.delete("/security/headers/csrf/trusted-origins")
async def remove_trusted_origin(
    request_data: TrustedOriginRequest,
    api_key: str = Depends(verify_admin_key)
):
    """
    Remove trusted origin from CSRF protection
    
    Removes an origin from the trusted origins list,
    disallowing CSRF requests from that origin.
    """
    try:
        # Remove trusted origin
        security_headers_manager.remove_trusted_origin(request_data.origin)
        
        return {
            "success": True,
            "message": f"Removed trusted origin: {request_data.origin}",
            "origin": request_data.origin,
            "reason": request_data.reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error removing trusted origin: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove trusted origin")


@router.get("/security/headers/csrf/trusted-origins")
async def list_trusted_origins(
    api_key: str = Depends(verify_admin_key)
):
    """
    List all trusted origins for CSRF protection
    
    Returns a list of all origins that are trusted
    for CSRF token validation.
    """
    try:
        trusted_origins = list(security_headers_manager.csrf_config.trusted_origins)
        
        return {
            "trusted_origins": trusted_origins,
            "total_count": len(trusted_origins),
            "csrf_enabled": security_headers_manager.csrf_config.enabled,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing trusted origins: {e}")
        raise HTTPException(status_code=500, detail="Failed to list trusted origins")


@router.get("/security/headers/test")
async def test_security_headers(
    request: Request,
    api_key: str = Depends(verify_admin_key)
):
    """
    Test security headers configuration
    
    Returns information about which security headers
    would be applied to the current request.
    """
    try:
        # Create test response
        test_response = JSONResponse(content={"test": "response"})
        
        # Apply security headers
        test_response = security_headers_manager.apply_security_headers(test_response, request)
        
        # Extract applied headers
        applied_headers = dict(test_response.headers)
        
        # Check CSRF requirements
        csrf_required = security_headers_manager.require_csrf_token(request)
        
        # Generate CSP nonce if needed
        csp_nonce = None
        if security_headers_manager.level in [SecurityHeaderLevel.STRICT, SecurityHeaderLevel.MODERATE]:
            csp_nonce = security_headers_manager.generate_csp_nonce()
        
        return {
            "request_info": {
                "method": request.method,
                "path": str(request.url.path),
                "headers": dict(request.headers),
                "csrf_required": csrf_required
            },
            "security_headers": applied_headers,
            "csp_nonce": csp_nonce,
            "security_level": security_headers_manager.level.value,
            "csrf_config": {
                "enabled": security_headers_manager.csrf_config.enabled,
                "same_site": security_headers_manager.csrf_config.same_site,
                "secure": security_headers_manager.csrf_config.secure
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing security headers: {e}")
        raise HTTPException(status_code=500, detail="Failed to test security headers")


@router.get("/security/headers/health")
async def security_headers_health():
    """
    Security headers system health check
    
    Returns the health status of the security headers system.
    No authentication required for health checks.
    """
    try:
        csrf_stats = get_csrf_protection_stats()
        
        return {
            "status": "healthy",
            "security_headers": "active",
            "csrf_protection": "active" if csrf_stats["csrf_enabled"] else "disabled",
            "security_level": security_headers_manager.level.value,
            "active_tokens": csrf_stats["active_tokens"],
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Security headers health check failed: {e}")
        raise HTTPException(status_code=503, detail="Security headers system unhealthy")