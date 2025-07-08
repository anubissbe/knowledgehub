"""
Security Headers Middleware

Applies comprehensive HTTP security headers and CSRF protection
to all responses to prevent various web-based attacks.
"""

import logging
import time
from typing import Dict, Any, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException

from ..security.headers import (
    SecurityHeadersManager,
    SecurityHeaderLevel,
    CSRFConfig,
    get_security_headers_for_response,
    validate_csrf_token_for_request
)
from ..security.monitoring import log_security_event, SecurityEventType, ThreatLevel

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for applying security headers and CSRF protection
    
    Features:
    - Comprehensive HTTP security headers
    - Content Security Policy (CSP) with nonce support
    - CSRF token generation and validation
    - Feature Policy/Permissions Policy
    - Configurable security levels
    - Security event logging
    """
    
    def __init__(self, app, 
                 security_level: SecurityHeaderLevel = SecurityHeaderLevel.MODERATE,
                 csrf_enabled: bool = True,
                 environment: str = "development"):
        super().__init__(app)
        self.environment = environment
        self.security_level = security_level
        
        # Configure CSRF based on environment
        csrf_config = CSRFConfig(
            enabled=csrf_enabled,
            secure=environment == "production",
            same_site="Strict" if environment == "production" else "Lax",
            require_referer=environment == "production",
            trusted_origins=self._get_trusted_origins()
        )
        
        # Initialize security headers manager
        self.headers_manager = SecurityHeadersManager(
            level=security_level,
            csrf_config=csrf_config
        )
        
        # Endpoints that don't need CSRF protection
        self.csrf_exempt_paths = {
            "/health",
            "/metrics",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
            "/api/security/monitoring/health"
        }
        
        # API endpoints (typically use API key auth, not CSRF)
        self.api_paths = {
            "/api/v1/",
            "/api/security/",
            "/api/memory/"
        }
        
        # Cleanup interval for expired tokens
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
        logger.info(f"Security headers middleware initialized with {security_level.value} level")
    
    def _get_trusted_origins(self) -> set:
        """Get trusted origins for CSRF protection"""
        trusted_origins = set()
        
        if self.environment == "development":
            trusted_origins.update({
                "http://localhost:3000",
                "http://localhost:3102",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3102",
                "http://192.168.1.25:3000",
                "http://192.168.1.24:3000"
            })
        else:
            trusted_origins.update({
                "https://knowledgehub.example.com",
                "https://api.knowledgehub.example.com",
                "https://app.knowledgehub.example.com"
            })
        
        return trusted_origins
    
    async def dispatch(self, request: Request, call_next):
        """Process request with security headers and CSRF protection"""
        
        # Periodic cleanup of expired tokens
        await self._periodic_cleanup()
        
        # Generate CSP nonce for inline scripts
        if self.security_level in [SecurityHeaderLevel.STRICT, SecurityHeaderLevel.MODERATE]:
            csp_nonce = self.headers_manager.generate_csp_nonce()
            request.state.csp_nonce = csp_nonce
        
        # Check CSRF protection for state-changing requests
        if await self._requires_csrf_protection(request):
            csrf_valid = await self._validate_csrf_token(request)
            if not csrf_valid:
                await self._log_csrf_violation(request)
                return self._create_csrf_error_response()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Apply security headers to response
            response = await self._apply_security_headers(response, request)
            
            return response
            
        except Exception as e:
            logger.error(f"Security headers middleware error: {e}")
            # Apply security headers even to error responses
            error_response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
            return await self._apply_security_headers(error_response, request)
    
    async def _requires_csrf_protection(self, request: Request) -> bool:
        """Check if request requires CSRF protection"""
        
        # Skip CSRF for exempt paths
        path = str(request.url.path)
        if any(exempt_path in path for exempt_path in self.csrf_exempt_paths):
            return False
        
        # Skip CSRF for API endpoints with API key
        if request.headers.get("X-API-Key"):
            return False
        
        # Skip CSRF for GET, HEAD, OPTIONS requests
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return False
        
        # Skip CSRF for JSON API requests with proper headers
        content_type = request.headers.get("content-type", "").lower()
        x_requested_with = request.headers.get("X-Requested-With")
        
        # Debug logging for CSRF checks
        if self.environment == "development":
            logger.debug(f"CSRF Check - Path: {path}, Method: {request.method}")
            logger.debug(f"CSRF Check - Content-Type: {content_type}")
            logger.debug(f"CSRF Check - X-Requested-With: {x_requested_with}")
        
        if (content_type.startswith("application/json") and 
            x_requested_with == "XMLHttpRequest"):
            return False
        
        # Require CSRF for state-changing requests
        return request.method in ["POST", "PUT", "PATCH", "DELETE"]
    
    async def _validate_csrf_token(self, request: Request) -> bool:
        """Validate CSRF token for request"""
        
        try:
            return self.headers_manager.validate_csrf_request(request)
        except Exception as e:
            logger.error(f"CSRF validation error: {e}")
            return False
    
    async def _apply_security_headers(self, response: Response, request: Request) -> Response:
        """Apply security headers to response"""
        
        try:
            # Apply all security headers
            response = self.headers_manager.apply_security_headers(response, request)
            
            # Add custom headers based on request
            await self._add_custom_headers(response, request)
            
            return response
            
        except Exception as e:
            logger.error(f"Error applying security headers: {e}")
            return response
    
    async def _add_custom_headers(self, response: Response, request: Request):
        """Add custom security headers based on request context"""
        
        # Add security headers for API responses
        if any(api_path in str(request.url.path) for api_path in self.api_paths):
            response.headers["X-API-Version"] = "1.0.0"
            response.headers["X-Rate-Limit-Policy"] = "100/minute"
        
        # Add CSP nonce to response if available
        if hasattr(request.state, 'csp_nonce'):
            response.headers["X-CSP-Nonce"] = request.state.csp_nonce
        
        # Add timing information in development
        if self.environment == "development":
            response.headers["X-Security-Level"] = self.security_level.value
            response.headers["X-CSRF-Enabled"] = str(self.headers_manager.csrf_config.enabled)
    
    async def _log_csrf_violation(self, request: Request):
        """Log CSRF protection violation"""
        
        source_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Log detailed information about the failed request
        logger.warning(f"CSRF validation failed for {request.method} {request.url.path}")
        logger.warning(f"Headers: Content-Type={request.headers.get('content-type')}, X-Requested-With={request.headers.get('X-Requested-With')}")
        logger.warning(f"Origin: {request.headers.get('origin')}, Referer: {request.headers.get('referer')}")
        
        await log_security_event(
            SecurityEventType.CORS_VIOLATION,  # Using CORS_VIOLATION for now, could add CSRF_VIOLATION
            ThreatLevel.MEDIUM,
            source_ip,
            user_agent,
            str(request.url.path),
            request.method,
            "CSRF token validation failed",
            origin=request.headers.get("origin")
        )
    
    def _create_csrf_error_response(self) -> JSONResponse:
        """Create CSRF error response"""
        
        return JSONResponse(
            status_code=403,
            content={
                "error": "CSRF_PROTECTION_FAILED",
                "message": "CSRF token validation failed",
                "code": "CSRF_INVALID"
            }
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else "unknown"
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired tokens"""
        
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.headers_manager.cleanup_expired_tokens()
            self.last_cleanup = current_time