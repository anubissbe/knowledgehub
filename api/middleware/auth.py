"""Secure Authentication middleware"""

import hashlib
import hmac
import time
import logging
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session

from ..config import settings
from ..models import get_db
from ..models.auth import APIKey
from ..security.sanitization import InputSanitizer

logger = logging.getLogger(__name__)


class SecureAuthMiddleware(BaseHTTPMiddleware):
    """Secure API key authentication middleware with database validation"""
    
    # Paths that don't require authentication
    EXEMPT_PATHS = [
        "/",
        "/health",
        "/api/docs", 
        "/api/redoc",
        "/api/openapi.json",
        "/metrics",
        "/api/auth/status",
        "/api/security/monitoring/health",  # Security monitoring health check
        "/api/security/headers/health",     # Security headers health check
        "/api/security/rate-limiting/health",  # Rate limiting health check
        "/api/persistent-context/health",  # Persistent context health check
        "/api/performance/health",  # Performance optimization health check
        "/api/performance/enable",  # Temporary: allow enable endpoint for testing
        "/api/performance/disable",  # Temporary: allow disable endpoint for testing
        "/api/performance/optimize",  # Temporary: allow optimize endpoint for testing
        "/api/performance/cache/clear",  # Temporary: allow cache clear for testing
        "/api/performance/metric",  # Temporary: allow metric recording for testing
        "/api/performance/database/optimize",  # Database optimization endpoint
        "/api/v1/workflow/health",  # Workflow integration health check
        "/api/v1/sync/health",  # Memory sync health check
        "/api/v1/sync/count",  # Memory sync count check
        # Claude working endpoints
        "/api/claude-working/health",
        "/api/claude-working/initialize",
        "/api/claude-working/session/continue",
        "/api/claude-working/session/handoff",
        "/api/claude-working/project/detect",
        "/api/claude-working/error/record",
        "/api/claude-working/error/similar",
        "/api/claude-working/task/predict",
        # Claude sync endpoints
        "/api/claude-sync/health",
        "/api/claude-sync/test-db",
        "/api/claude-sync/initialize",
        "/api/claude-sync/session/continue",
        "/api/claude-sync/error/record",
        "/api/claude-sync/test-simple",
        "/api/claude-sync/test-project",
        "/api/claude-sync/session/handoff",
        "/api/claude-sync/error/similar",
        "/api/claude-sync/task/predict",
        # Claude auto endpoints
        "/api/claude-auto/health",
        "/api/claude-auto/session/start",
        "/api/claude-auto/session/handoff",
        "/api/claude-auto/session/end",
        "/api/claude-auto/session/current",
        "/api/claude-auto/error/record",
        "/api/claude-auto/error/similar",
        "/api/claude-auto/tasks/predict",
        "/api/claude-auto/memory/stats",
        # Project context endpoints
        "/api/project-context/health",
        "/api/project-context/switch",
        "/api/project-context/current",
        "/api/project-context/list",
        "/api/project-context/preference",
        "/api/project-context/pattern",
        "/api/project-context/memory",
        "/api/project-context/memories",
        "/api/project-context/auto-detect",
        "/api/project-context/clear-active",
        # Mistake learning endpoints
        "/api/mistake-learning/health",
        "/api/mistake-learning/track",
        "/api/mistake-learning/check-action",
        "/api/mistake-learning/lessons",
        "/api/mistake-learning/patterns",
        "/api/mistake-learning/report",
        "/api/mistake-learning/learn-pattern",
        "/api/mistake-learning/reset-patterns",
        # Proactive assistant endpoints
        "/api/proactive/health",
        "/api/proactive/analyze",
        "/api/proactive/brief",
        "/api/proactive/incomplete-tasks",
        "/api/proactive/predictions",
        "/api/proactive/reminders", 
        "/api/proactive/check-interrupt",
        "/api/proactive/context",
        "/api/proactive/suggestions",
        # Decision reasoning endpoints
        "/api/decisions/health",
        "/api/decisions/record",
        "/api/decisions/explain",
        "/api/decisions/update-outcome",
        "/api/decisions/similar",
        "/api/decisions/suggest",
        "/api/decisions/confidence-report",
        "/api/decisions/categories",
        "/api/decisions/patterns",
        "/api/decisions/search",
        # Code evolution endpoints
        "/api/code-evolution/health",
        "/api/code-evolution/track-change",
        "/api/code-evolution/history",
        "/api/code-evolution/compare",
        "/api/code-evolution/suggest-refactoring",
        "/api/code-evolution/update-impact",
        "/api/code-evolution/patterns/analytics",
        "/api/code-evolution/patterns/learned",
        "/api/code-evolution/suggestions/file",
        "/api/code-evolution/trends",
        "/api/code-evolution/upload-diff",
        "/api/code-evolution/search",
        # Performance metrics endpoints
        "/api/performance/health",
        "/api/performance/track",
        "/api/performance/track-batch",
        "/api/performance/report",
        "/api/performance/predict",
        "/api/performance/patterns",
        "/api/performance/optimization-history",
        "/api/performance/categories",
        "/api/performance/thresholds",
        "/api/performance/optimization-strategies",
        "/api/performance/benchmark",
        "/api/performance/recommendations",
        "/api/performance/trends",
        # Claude workflow integration endpoints
        "/api/claude-workflow/health",
        "/api/claude-workflow/capture/conversation",
        "/api/claude-workflow/capture/terminal",
        "/api/claude-workflow/capture/tool-usage",
        "/api/claude-workflow/save/discovery",
        "/api/claude-workflow/extract/insights",
        "/api/claude-workflow/stats",
        "/api/claude-workflow/capture/batch"
    ]
    
    # Claude endpoints (temporary for testing)
    CLAUDE_PATHS = [
        "/api/claude/test",
        "/api/claude/test-detect",
        "/api/claude/initialize",
        "/api/claude/session/continue",
        "/api/claude/session/handoff",
        "/api/claude/project/detect",
        "/api/claude/error/record",
        "/api/claude/error/similar",
        "/api/claude/task/predict"
    ]
    
    # Development paths (only exempt in development mode)
    DEV_EXEMPT_PATHS = [
        "/api/auth/setup",  # Initial setup endpoint
        "/api/system/status",  # System status in dev
        "/api/v1/sources/",  # Temporarily allow sources access
        "/api/v1/jobs/",  # Temporarily allow jobs access
        "/api/v1/analytics/",  # Temporarily allow analytics access
        "/api/v1/health",  # Health check
        "/api/v1/memories/",  # Temporarily allow memories access
        "/api/v1/sync/",  # Allow memory sync access</        
        "/api/v1/search",  # Allow search access
        "/api/v1/documents/",  # Allow documents access
        "/api/v1/chunks/",  # Allow chunks access
        "/api/v1/admin/",  # Admin dashboard endpoints
        "/api/v1/diagrams/",  # Threat model diagrams
        "/ws/",  # WebSocket connections
        "/api/memory/",  # Memory system endpoints
        "/api/performance/"  # Performance optimization endpoints (for testing)
    ]
    
    def __init__(self, app):
        super().__init__(app)
        self.api_key_cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
    
    def _hash_api_key(self, api_key: str) -> str:
        """Securely hash API key using HMAC-SHA256"""
        return hmac.new(
            settings.SECRET_KEY.encode(),
            api_key.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _get_cached_key(self, key_hash: str) -> Optional[dict]:
        """Get API key from cache if valid"""
        if key_hash in self.api_key_cache:
            cached_data, timestamp = self.api_key_cache[key_hash]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                del self.api_key_cache[key_hash]
        return None
    
    def _cache_key(self, key_hash: str, key_data: dict):
        """Cache API key data"""
        self.api_key_cache[key_hash] = (key_data, time.time())
    
    async def _validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key against database"""
        try:
            # Sanitize input
            api_key = InputSanitizer.sanitize_text(api_key, max_length=512, allow_html=False)
            
            # Hash the API key
            key_hash = self._hash_api_key(api_key)
            
            # Check cache first
            cached_key = self._get_cached_key(key_hash)
            if cached_key:
                return cached_key
            
            # Query database
            db: Session = next(get_db())
            try:
                api_key_obj = db.query(APIKey).filter(
                    APIKey.key_hash == key_hash,
                    APIKey.is_active == True
                ).first()
                
                if api_key_obj and api_key_obj.is_valid():
                    # Update last used timestamp
                    api_key_obj.update_last_used()
                    db.commit()
                    
                    key_data = {
                        "id": str(api_key_obj.id),
                        "name": api_key_obj.name,
                        "permissions": api_key_obj.permissions
                    }
                    
                    # Cache the result
                    self._cache_key(key_hash, key_data)
                    return key_data
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"API key validation error: {e}")
        
        return None
    
    async def dispatch(self, request: Request, call_next):
        """Secure API key validation for protected endpoints"""
        # Allow CORS preflight OPTIONS requests
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Skip auth for public paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Check Claude working project context endpoint
        if request.url.path.startswith("/api/claude-working/project/") and request.url.path.endswith("/context"):
            return await call_next(request)
        
        # Check project context conventions endpoint
        if request.url.path.startswith("/api/project-context/conventions/"):
            return await call_next(request)
        
        # Check mistake learning similar mistakes endpoint
        if request.url.path.startswith("/api/mistake-learning/similar-mistakes/"):
            return await call_next(request)
        
        # Check decision explanation endpoint with dynamic ID
        if request.url.path.startswith("/api/decisions/explain/"):
            return await call_next(request)
        
        # Check decision patterns endpoint with dynamic category
        if request.url.path.startswith("/api/decisions/patterns/"):
            return await call_next(request)
        
        # Check code evolution compare endpoint with dynamic ID
        if request.url.path.startswith("/api/code-evolution/compare/"):
            return await call_next(request)
        
        # Skip auth for Claude paths (temporary)
        if request.url.path in self.CLAUDE_PATHS:
            logger.info(f"Allowing Claude path: {request.url.path}")
            return await call_next(request)
        
        # In development mode, allow additional dev paths
        if settings.APP_ENV == "development":
            for dev_path in self.DEV_EXEMPT_PATHS:
                if request.url.path.startswith(dev_path.rstrip('/')):
                    logger.info(f"Allowing dev path: {request.url.path}")
                    return await call_next(request)
        
        # Extract API key from headers
        api_key = request.headers.get(settings.API_KEY_HEADER)
        auth_header = request.headers.get("Authorization")
        
        # Support Bearer token format
        if not api_key and auth_header:
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove "Bearer " prefix
            elif auth_header.startswith("ApiKey "):
                api_key = auth_header[7:]  # Remove "ApiKey " prefix
        
        if not api_key:
            logger.warning(f"Missing API key for {request.method} {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication required",
                    "message": "API key must be provided in X-API-Key header or Authorization header"
                }
            )
        
        # Validate API key
        api_key_data = await self._validate_api_key(api_key)
        if not api_key_data:
            logger.warning(f"Invalid API key for {request.method} {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid API key",
                    "message": "The provided API key is invalid or expired"
                }
            )
        
        # Add authentication info to request state
        request.state.api_key = api_key_data
        request.state.authenticated = True
        
        # Process request
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )