"""Authentication middleware"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware"""
    
    # Paths that don't require authentication
    EXEMPT_PATHS = [
        "/",
        "/health",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/ws"  # WebSocket connections handle auth separately
    ]
    
    async def dispatch(self, request: Request, call_next):
        """Check API key for protected endpoints"""
        # Allow CORS preflight OPTIONS requests to pass through
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Skip auth for exempt paths and WebSocket endpoints
        if request.url.path in self.EXEMPT_PATHS or request.url.path.startswith("/ws"):
            return await call_next(request)
        
        # Skip auth in development mode - make this more permissive for dev
        if settings.DEBUG or settings.APP_ENV == "development":
            logger.info(f"Bypassing auth for development mode: DEBUG={settings.DEBUG}, APP_ENV={settings.APP_ENV}")
            return await call_next(request)
        
        # Check for API key (both X-API-Key header and Authorization Bearer)
        api_key = request.headers.get(settings.API_KEY_HEADER)
        auth_header = request.headers.get("Authorization")
        
        # Extract bearer token if present
        if not api_key and auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header.split(" ")[1]
        
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "API key required"}
            )
        
        # TODO: Implement proper API key validation
        # For now, accept a development key or any bearer token in dev mode
        if api_key not in ["dev-api-key-123"] and not settings.DEBUG:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )
        
        # Process request
        response = await call_next(request)
        return response