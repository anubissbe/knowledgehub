"""
Advanced Rate Limiting Middleware

Enhanced rate limiting middleware with DDoS protection,
adaptive limits, and multiple rate limiting strategies.
"""

import time
import logging
from typing import Optional, Set
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from ..security.rate_limiting import (
    AdvancedRateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    get_rate_limiter
)
from ..security.monitoring import log_security_event, SecurityEventType, ThreatLevel

logger = logging.getLogger(__name__)


class AdvancedRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with DDoS protection
    
    Features:
    - Multiple rate limiting strategies (sliding window, token bucket, etc.)
    - Adaptive rate limiting based on server load
    - DDoS threat assessment and mitigation
    - IP blacklisting for persistent threats
    - Redis backend support for distributed rate limiting
    - Comprehensive request pattern analysis
    """
    
    def __init__(self, app,
                 requests_per_minute: int = 100,
                 requests_per_hour: int = 2000,
                 requests_per_day: int = 20000,
                 burst_limit: int = 20,
                 strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
                 enable_adaptive: bool = True,
                 enable_ddos_protection: bool = True,
                 redis_client: Optional[object] = None):
        super().__init__(app)
        
        # Configuration
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            requests_per_day=requests_per_day,
            burst_limit=burst_limit,
            strategy=strategy,
            enable_adaptive=enable_adaptive,
            enable_ddos_protection=enable_ddos_protection
        )
        
        # Initialize rate limiter
        self.rate_limiter = AdvancedRateLimiter(self.config, redis_client)
        
        # Exempt paths (no rate limiting)
        self.exempt_paths: Set[str] = {
            "/health",
            "/",
            "/metrics",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json"
        }
        
        # API paths with stricter limits
        self.api_paths: Set[str] = {
            "/api/v1/",
            "/api/security/",
            "/api/memory/"
        }
        
        # WebSocket paths (different handling)
        self.websocket_paths: Set[str] = {
            "/ws/"
        }
        
        # Cleanup interval
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        logger.info(f"Advanced rate limiting middleware initialized with {strategy.value} strategy")
    
    def _should_exempt(self, request: Request) -> bool:
        """Check if request should be exempt from rate limiting"""
        path = str(request.url.path)
        
        # Check exact matches
        if path in self.exempt_paths:
            return True
        
        # Check WebSocket paths
        if any(ws_path in path for ws_path in self.websocket_paths):
            return True
        
        # Check for preflight requests
        if request.method == "OPTIONS":
            return True
        
        # Check for health check endpoints
        if "health" in path.lower():
            return True
        
        return False
    
    def _is_api_endpoint(self, request: Request) -> bool:
        """Check if request is to an API endpoint"""
        path = str(request.url.path)
        return any(api_path in path for api_path in self.api_paths)
    
    async def _add_rate_limit_headers(self, response: Response, request: Request):
        """Add rate limit headers to response"""
        try:
            # Get current rate limit status
            client_id = self.rate_limiter._get_client_id(request)
            stats = await self.rate_limiter.get_client_stats(client_id)
            
            # Calculate remaining requests
            remaining = max(0, self.config.requests_per_minute - stats["requests_last_minute"])
            
            # Add standard rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
            
            # Add advanced headers
            response.headers["X-RateLimit-Strategy"] = self.config.strategy.value
            response.headers["X-RateLimit-Adaptive"] = str(self.config.enable_adaptive)
            
            # Add threat level if applicable
            if stats["threat_score"] > 0:
                if stats["threat_score"] >= 6.0:
                    threat_level = "critical"
                elif stats["threat_score"] >= 4.0:
                    threat_level = "high"
                elif stats["threat_score"] >= 2.0:
                    threat_level = "medium"
                else:
                    threat_level = "low"
                
                response.headers["X-Threat-Level"] = threat_level
                response.headers["X-Threat-Score"] = str(round(stats["threat_score"], 2))
            
        except Exception as e:
            logger.error(f"Error adding rate limit headers: {e}")
    
    async def _periodic_cleanup(self):
        """Perform periodic cleanup of expired data"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            try:
                await self.rate_limiter.cleanup_expired_data()
                self.last_cleanup = current_time
                logger.debug("Rate limiter cleanup completed")
            except Exception as e:
                logger.error(f"Error during rate limiter cleanup: {e}")
    
    async def dispatch(self, request: Request, call_next):
        """Apply advanced rate limiting"""
        
        # Skip rate limiting for exempt paths
        if self._should_exempt(request):
            return await call_next(request)
        
        # Periodic cleanup
        await self._periodic_cleanup()
        
        # Check rate limit
        try:
            allowed, error_response = await self.rate_limiter.check_rate_limit(request)
            
            if not allowed:
                # Request was rate limited
                logger.warning(f"Rate limited request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
                return error_response
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to successful responses
            await self._add_rate_limit_headers(response, request)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            
            # In case of error, allow request but log the issue
            try:
                response = await call_next(request)
                return response
            except Exception as inner_e:
                logger.error(f"Request processing error after rate limit failure: {inner_e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal server error",
                        "message": "Request processing failed"
                    }
                )


class DDoSProtectionMiddleware(BaseHTTPMiddleware):
    """
    Specialized DDoS protection middleware
    
    Focuses specifically on DDoS detection and mitigation,
    working alongside the rate limiting middleware.
    """
    
    def __init__(self, app,
                 enable_protection: bool = True,
                 protection_threshold: int = 1000,
                 blacklist_duration: int = 3600):
        super().__init__(app)
        self.enable_protection = enable_protection
        self.protection_threshold = protection_threshold
        self.blacklist_duration = blacklist_duration
        
        # DDoS detection metrics
        self.request_patterns = {}
        self.suspicious_ips = set()
        
        logger.info(f"DDoS protection middleware initialized: {'enabled' if enable_protection else 'disabled'}")
    
    def _analyze_request_pattern(self, request: Request) -> bool:
        """Analyze request for DDoS patterns"""
        if not self.enable_protection:
            return False
        
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "")
        
        # Check for known attack patterns
        attack_patterns = [
            # SQL injection attempts
            "union select", "' or 1=1", "drop table",
            # XSS attempts
            "<script>", "javascript:", "onerror=",
            # Command injection
            "; cat /etc/passwd", "| nc ", "&& wget",
            # Path traversal
            "../", "..\\", "%2e%2e%2f"
        ]
        
        request_data = f"{request.url.path} {request.url.query} {user_agent}".lower()
        
        for pattern in attack_patterns:
            if pattern in request_data:
                logger.warning(f"Attack pattern detected from {client_ip}: {pattern}")
                return True
        
        return False
    
    async def dispatch(self, request: Request, call_next):
        """Apply DDoS protection"""
        
        # Analyze request for attack patterns
        if self._analyze_request_pattern(request):
            client_ip = request.client.host if request.client else "unknown"
            
            # Log security event
            await log_security_event(
                SecurityEventType.MALICIOUS_REQUEST,
                ThreatLevel.HIGH,
                client_ip,
                request.headers.get("User-Agent", ""),
                str(request.url.path),
                request.method,
                "DDoS attack pattern detected"
            )
            
            return JSONResponse(
                status_code=403,
                content={
                    "error": "MALICIOUS_REQUEST_DETECTED",
                    "message": "Request blocked due to suspicious patterns",
                    "code": "DDOS_PROTECTION"
                }
            )
        
        # Continue with normal processing
        return await call_next(request)