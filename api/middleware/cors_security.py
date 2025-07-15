"""
Enhanced CORS Security Middleware

This middleware provides additional CORS security features beyond the basic
FastAPI CORSMiddleware, including origin validation, security logging,
and attack detection.
"""

import logging
import time
from typing import Set, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta

from ..cors_config import (
    validate_cors_origin,
    is_secure_origin,
    is_localhost_origin,
    is_local_network_origin,
    get_environment_config,
    get_cors_security_headers
)

logger = logging.getLogger(__name__)


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS Security Middleware
    
    Features:
    - Origin validation with environment-aware rules
    - Suspicious activity detection and logging
    - Rate limiting for preflight requests
    - Security headers injection
    - Attack pattern detection
    """
    
    def __init__(self, app, environment: str = "development"):
        super().__init__(app)
        self.environment = environment
        self.config = get_environment_config(environment)
        
        # Attack detection
        self.suspicious_origins: Set[str] = set()
        self.origin_request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.blocked_origins: Set[str] = set()
        
        # Rate limiting for preflight requests
        self.preflight_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.max_preflight_per_minute = 30
        
        # Security patterns to detect
        self.suspicious_patterns = [
            r'<script',           # XSS attempts
            r'javascript:',       # JavaScript injection
            r'data:text/html',    # Data URL attacks
            r'vbscript:',         # VBScript injection
            r'\.\./',             # Path traversal
            r'file://',           # File protocol
            r'ftp://',            # FTP protocol
            r'ldap://',           # LDAP protocol
        ]
        
        logger.info(f"CORS Security Middleware initialized for {environment} environment")
        
    async def dispatch(self, request: Request, call_next):
        """Process request with enhanced CORS security"""
        start_time = time.time()
        origin = request.headers.get("origin")
        
        # Process CORS request
        cors_result = await self._process_cors_request(request, origin)
        if cors_result:
            return cors_result
        
        # Continue with normal request processing
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response, origin)
        
        # Log processing time for monitoring
        processing_time = time.time() - start_time
        if processing_time > 1.0:  # Log slow requests
            logger.warning(f"Slow CORS request: {processing_time:.2f}s for {request.url}")
        
        return response
    
    async def _process_cors_request(self, request: Request, origin: Optional[str]) -> Optional[Response]:
        """Process CORS-related security checks"""
        
        # Skip CORS processing for performance endpoints (avoid timeouts)
        if str(request.url.path).startswith('/api/performance/'):
            return None
        
        # Skip CORS processing for memory endpoints (avoid timeouts)
        if str(request.url.path).startswith('/api/memory/') or str(request.url.path).startswith('/api/v1/memories/'):
            return None
        
        # Skip CORS processing for search endpoints (avoid timeouts)
        if str(request.url.path).startswith('/api/v1/search'):
            return None
        
        # Skip CORS processing for same-origin requests
        if not origin:
            return None
        
        # Validate origin format
        if not self._is_valid_origin_format(origin):
            logger.warning(f"Invalid origin format: {origin}")
            await self._log_security_event("invalid_origin_format", origin, request)
            return self._create_cors_error_response("Invalid origin format")
        
        # Check for suspicious patterns
        if self._contains_suspicious_patterns(origin):
            logger.warning(f"Suspicious origin pattern detected: {origin}")
            await self._log_security_event("suspicious_origin_pattern", origin, request)
            self.suspicious_origins.add(origin)
            return self._create_cors_error_response("Suspicious origin detected")
        
        # Check if origin is blocked
        if origin in self.blocked_origins:
            logger.warning(f"Blocked origin attempted access: {origin}")
            await self._log_security_event("blocked_origin_access", origin, request)
            return self._create_cors_error_response("Origin blocked")
        
        # Validate origin against allowed list
        if not validate_cors_origin(origin, self.environment):
            logger.warning(f"Unauthorized origin: {origin}")
            await self._log_security_event("unauthorized_origin", origin, request)
            
            # In strict mode, block the request
            if self.config.get("strict_mode", False):
                return self._create_cors_error_response("Origin not allowed")
        
        # Rate limit preflight requests
        if request.method == "OPTIONS":
            if self._is_preflight_rate_limited(origin):
                logger.warning(f"Preflight rate limit exceeded for origin: {origin}")
                await self._log_security_event("preflight_rate_limit", origin, request)
                return self._create_cors_error_response("Rate limit exceeded")
        
        # Track request for analysis
        self._track_origin_request(origin, request)
        
        # Check for potential attacks
        await self._detect_potential_attacks(origin, request)
        
        return None  # Continue with normal processing
    
    def _is_valid_origin_format(self, origin: str) -> bool:
        """Validate origin format according to RFC 6454"""
        if not origin or len(origin) > 2048:  # Reasonable length limit
            return False
        
        # Basic format validation
        origin_pattern = r'^https?://[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*(:([1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5]))?$'
        
        return bool(re.match(origin_pattern, origin))
    
    def _contains_suspicious_patterns(self, origin: str) -> bool:
        """Check if origin contains suspicious patterns"""
        origin_lower = origin.lower()
        return any(re.search(pattern, origin_lower, re.IGNORECASE) 
                  for pattern in self.suspicious_patterns)
    
    def _is_preflight_rate_limited(self, origin: str) -> bool:
        """Check if preflight requests from origin are rate limited"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        preflight_queue = self.preflight_requests[origin]
        while preflight_queue and preflight_queue[0] < minute_ago:
            preflight_queue.popleft()
        
        # Check rate limit
        if len(preflight_queue) >= self.max_preflight_per_minute:
            return True
        
        # Add current request
        preflight_queue.append(now)
        return False
    
    def _track_origin_request(self, origin: str, request: Request):
        """Track origin requests for analysis"""
        now = datetime.now()
        request_info = {
            "timestamp": now,
            "method": request.method,
            "path": str(request.url.path),
            "user_agent": request.headers.get("user-agent", ""),
        }
        
        self.origin_request_counts[origin].append(request_info)
    
    async def _detect_potential_attacks(self, origin: str, request: Request):
        """Detect potential CORS-based attacks"""
        request_queue = self.origin_request_counts[origin]
        
        if len(request_queue) < 10:  # Need sufficient data
            return
        
        # Check for rapid requests (potential DoS)
        recent_requests = [req for req in request_queue 
                          if req["timestamp"] > datetime.now() - timedelta(minutes=1)]
        
        if len(recent_requests) > 50:  # More than 50 requests per minute
            logger.warning(f"Potential DoS attack from origin: {origin}")
            await self._log_security_event("potential_dos_attack", origin, request)
            self.suspicious_origins.add(origin)
        
        # Check for scanning behavior (many different paths)
        recent_paths = set(req["path"] for req in recent_requests)
        if len(recent_paths) > 20:  # More than 20 different paths
            logger.warning(f"Potential scanning behavior from origin: {origin}")
            await self._log_security_event("potential_scanning", origin, request)
            self.suspicious_origins.add(origin)
    
    def _add_security_headers(self, response: Response, origin: Optional[str]):
        """Add security headers to response"""
        security_headers = get_cors_security_headers()
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        # Add origin-specific headers
        if origin:
            # Vary header for caching
            response.headers["Vary"] = "Origin"
            
            # Content Security Policy
            if is_localhost_origin(origin) or is_local_network_origin(origin):
                # More relaxed CSP for development
                csp = "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; connect-src 'self' ws: wss:;"
            else:
                # Strict CSP for production
                csp = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self';"
            
            response.headers["Content-Security-Policy"] = csp
    
    def _create_cors_error_response(self, message: str) -> JSONResponse:
        """Create standardized CORS error response"""
        return JSONResponse(
            status_code=403,
            content={
                "error": "CORS_SECURITY_VIOLATION",
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            headers=get_cors_security_headers()
        )
    
    async def _log_security_event(self, event_type: str, origin: str, request: Request):
        """Log security events for monitoring and analysis"""
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "origin": origin,
            "method": request.method,
            "path": str(request.url.path),
            "user_agent": request.headers.get("user-agent", ""),
            "ip_address": request.client.host if request.client else "unknown",
            "headers": dict(request.headers)
        }
        
        # Log to security log
        security_logger = logging.getLogger("security.cors")
        security_logger.warning(f"CORS Security Event: {event_type}", extra=event_data)
        
        # In production, could send to SIEM or security monitoring system
        if self.environment == "production":
            await self._send_to_security_monitoring(event_data)
    
    async def _send_to_security_monitoring(self, event_data: Dict[str, Any]):
        """Send security events to monitoring system (placeholder)"""
        # This would integrate with your security monitoring system
        # Examples: Splunk, ELK Stack, Azure Sentinel, etc.
        pass
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics for monitoring dashboard"""
        return {
            "suspicious_origins_count": len(self.suspicious_origins),
            "blocked_origins_count": len(self.blocked_origins),
            "tracked_origins_count": len(self.origin_request_counts),
            "total_requests_tracked": sum(len(queue) for queue in self.origin_request_counts.values()),
            "environment": self.environment,
            "strict_mode": self.config.get("strict_mode", False)
        }
    
    def block_origin(self, origin: str, reason: str = "Manual block"):
        """Manually block an origin"""
        self.blocked_origins.add(origin)
        logger.warning(f"Origin manually blocked: {origin}, Reason: {reason}")
    
    def unblock_origin(self, origin: str):
        """Unblock a previously blocked origin"""
        self.blocked_origins.discard(origin)
        logger.info(f"Origin unblocked: {origin}")
    
    def clear_suspicious_origin(self, origin: str):
        """Clear suspicious status for an origin"""
        self.suspicious_origins.discard(origin)
        logger.info(f"Suspicious status cleared for origin: {origin}")