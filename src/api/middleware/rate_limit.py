"""Rate limiting middleware"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import time
from collections import defaultdict
import asyncio
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.cleanup_interval = 60  # Clean up old entries every minute
        self._cleanup_task = None
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting"""
        # Skip rate limiting for health checks and WebSocket endpoints
        if request.url.path in ["/health", "/"] or request.url.path.startswith("/ws"):
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client and hasattr(request.client, 'host') else "unknown"
        
        # Current timestamp
        now = time.time()
        
        # Clean up old requests
        minute_ago = now - 60
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute"
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(minute_ago + 60))
                }
            )
        
        # Record request
        self.requests[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.requests_per_minute - len(self.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(minute_ago + 60))
        
        return response
    
    async def cleanup_old_entries(self):
        """Periodically clean up old request entries"""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            now = time.time()
            minute_ago = now - 60
            
            # Clean up old entries
            for client_ip in list(self.requests.keys()):
                self.requests[client_ip] = [
                    req_time for req_time in self.requests[client_ip]
                    if req_time > minute_ago
                ]
                
                # Remove empty entries
                if not self.requests[client_ip]:
                    del self.requests[client_ip]