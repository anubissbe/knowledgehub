"""
Consolidated Middleware Module
Combines common middleware functions to reduce duplication.
"""

from api.shared import *
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time

class ConsolidatedMiddleware(BaseHTTPMiddleware):
    """Consolidated middleware combining common functionality"""
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch with consolidated functionality"""
        start_time = time.time()
        
        # Security headers
        response = await self._security_middleware(request, call_next)
        
        # Performance tracking
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    async def _security_middleware(self, request: Request, call_next) -> Response:
        """Consolidated security middleware"""
        # Add security headers
        response = await call_next(request)
        
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
            
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP with proxy support"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        forwarded = request.headers.get("X-Forwarded")
        if forwarded:
            return forwarded
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        return request.client.host if request.client else "unknown"
    
    def _get_route_pattern(self, request: Request) -> str:
        """Get route pattern for the request"""
        if hasattr(request, "route") and hasattr(request.route, "path"):
            return request.route.path
        return request.url.path

class ValidationMiddleware:
    """Consolidated validation middleware"""
    
    @staticmethod
    def validate_request_size(max_size: int = 10 * 1024 * 1024):  # 10MB default
        """Validate request content length"""
        def decorator(func):
            async def wrapper(request: Request, *args, **kwargs):
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request too large. Maximum size: {max_size} bytes"
                    )
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def validate_content_type(allowed_types: List[str]):
        """Validate request content type"""
        def decorator(func):
            async def wrapper(request: Request, *args, **kwargs):
                content_type = request.headers.get("content-type", "")
                if not any(allowed_type in content_type for allowed_type in allowed_types):
                    raise HTTPException(
                        status_code=415,
                        detail=f"Unsupported content type. Allowed: {allowed_types}"
                    )
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator

class RateLimitMiddleware:
    """Consolidated rate limiting middleware"""
    
    def __init__(self):
        self._requests = {}
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()
    
    def limit_requests(self, max_requests: int = 100, window_seconds: int = 60):
        """Rate limit decorator"""
        def decorator(func):
            async def wrapper(request: Request, *args, **kwargs):
                client_ip = self._get_client_ip(request)
                current_time = time.time()
                
                # Periodic cleanup
                if current_time - self._last_cleanup > self._cleanup_interval:
                    await self._cleanup_old_requests()
                    self._last_cleanup = current_time
                
                # Check rate limit
                if not await self._check_rate_limit(client_ip, max_requests, window_seconds):
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded"
                    )
                
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator
    
    async def _check_rate_limit(self, client_ip: str, max_requests: int, window_seconds: int) -> bool:
        """Check if client is within rate limit"""
        current_time = time.time()
        window_start = current_time - window_seconds
        
        if client_ip not in self._requests:
            self._requests[client_ip] = []
        
        # Remove old requests
        self._requests[client_ip] = [
            req_time for req_time in self._requests[client_ip] 
            if req_time > window_start
        ]
        
        # Check limit
        if len(self._requests[client_ip]) >= max_requests:
            return False
        
        # Add current request
        self._requests[client_ip].append(current_time)
        return True
    
    async def _cleanup_old_requests(self):
        """Clean up old request tracking data"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour
        
        for client_ip in list(self._requests.keys()):
            self._requests[client_ip] = [
                req_time for req_time in self._requests[client_ip]
                if req_time > cutoff_time
            ]
            
            if not self._requests[client_ip]:
                del self._requests[client_ip]
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP (consolidated implementation)"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"

# Global middleware instances
rate_limiter = RateLimitMiddleware()
