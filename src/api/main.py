"""Main FastAPI application"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import time
import os
from typing import Dict, Any

from .routers import sources, search, jobs, websocket, memories, chunks, documents, scheduler
try:
    from .routes import analytics_simple as analytics
except ImportError:
    # Fallback if psutil not available
    from .routes import analytics_fixed as analytics
from .routes import auth, cors_security, security_monitoring, security_headers, rate_limiting
from .services.startup import initialize_services, shutdown_services
from .middleware.auth import SecureAuthMiddleware
from .middleware.advanced_rate_limit import AdvancedRateLimitMiddleware, DDoSProtectionMiddleware
from .security.rate_limiting import RateLimitStrategy
from .middleware.security import ContentValidationMiddleware
from .middleware.security_headers import SecurityHeadersMiddleware as SecureHeadersMiddleware
from .middleware.session_tracking import SessionTrackingMiddleware
from .middleware.security_monitoring import SecurityMonitoringMiddleware
from .middleware.validation import ValidationMiddleware
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting AI Knowledge Hub API...")
    await initialize_services()
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Knowledge Hub API...")
    await shutdown_services()
    logger.info("API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent documentation indexing and knowledge management system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure Secure CORS
from .cors_config import get_cors_config, log_cors_security_info
from .middleware.cors_security import CORSSecurityMiddleware

# Log CORS security configuration
log_cors_security_info()

# Get secure CORS configuration
cors_config = get_cors_config(
    environment=settings.APP_ENV,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS
)

# Add FastAPI CORS middleware with secure configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
    expose_headers=cors_config["expose_headers"],
    max_age=cors_config["max_age"]
)

# Add custom middleware (order matters - last added runs first)
app.add_middleware(SecureHeadersMiddleware)
app.add_middleware(ContentValidationMiddleware)
# Add advanced rate limiting and DDoS protection middleware
app.add_middleware(
    AdvancedRateLimitMiddleware,
    requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    requests_per_hour=settings.RATE_LIMIT_REQUESTS_PER_MINUTE * 60,
    requests_per_day=settings.RATE_LIMIT_REQUESTS_PER_MINUTE * 60 * 24,
    burst_limit=settings.RATE_LIMIT_REQUESTS_PER_MINUTE // 3,
    strategy=RateLimitStrategy.SLIDING_WINDOW,
    enable_adaptive=True,
    enable_ddos_protection=True
)

# Add DDoS protection middleware
app.add_middleware(
    DDoSProtectionMiddleware,
    enable_protection=True,
    protection_threshold=1000,
    blacklist_duration=3600
)
app.add_middleware(SecureAuthMiddleware)

# Add enhanced CORS security middleware
app.add_middleware(CORSSecurityMiddleware, environment=settings.APP_ENV)

# Add security monitoring middleware
app.add_middleware(SecurityMonitoringMiddleware, environment=settings.APP_ENV)

# Add input validation middleware
from .security.validation import ValidationLevel
validation_level = ValidationLevel.STRICT if settings.APP_ENV == "production" else ValidationLevel.MODERATE
app.add_middleware(ValidationMiddleware, validation_level=validation_level)

# Add security headers middleware
from .security.headers import SecurityHeaderLevel
security_level = SecurityHeaderLevel.STRICT if settings.APP_ENV == "production" else SecurityHeaderLevel.MODERATE
app.add_middleware(SecureHeadersMiddleware, security_level=security_level, csrf_enabled=True, environment=settings.APP_ENV)

# Initialize session tracking middleware
try:
    from .memory_system.core.session_manager import SessionManager
    from .models import get_db
    
    # Create a session manager factory for the middleware
    def get_session_manager():
        db = next(get_db())
        return SessionManager(db)
    
    app.add_middleware(SessionTrackingMiddleware, session_manager_factory=get_session_manager)
    logger.info("Session tracking middleware initialized successfully")
except ImportError as e:
    logger.warning(f"Session tracking middleware not available: {e}")
except Exception as e:
    logger.error(f"Failed to initialize session tracking middleware: {e}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Time: {process_time:.3f}s"
    )
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-API-Version"] = "1.0.0"
    
    return response


# Include routers
app.include_router(auth.router)  # Authentication endpoints
app.include_router(cors_security.router, prefix="/api/security/cors", tags=["security"])  # CORS security management
app.include_router(security_monitoring.router, prefix="/api", tags=["security"])  # Security monitoring
app.include_router(security_headers.router, prefix="/api", tags=["security"])  # Security headers
app.include_router(rate_limiting.router, prefix="/api", tags=["security"])  # Rate limiting and DDoS protection
app.include_router(sources.router, prefix="/api/v1/sources", tags=["sources"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(chunks.router, prefix="/api/v1/chunks", tags=["chunks"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])
app.include_router(scheduler.router, prefix="/api/v1/scheduler", tags=["scheduler"])
app.include_router(analytics.router)
# WebSocket router
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Unified search router
try:
    from .routers import unified_search
    app.include_router(unified_search.router, prefix="/api/v1/search", tags=["unified-search"])
    logger.info("Unified search router integrated successfully")
except ImportError as e:
    logger.warning(f"Unified search router not available: {e}")

# Memory system routers
try:
    from .memory_system.api.routers import session as memory_session
    from .memory_system.api.routers import memory as memory_router
    from .memory_system.api.routers import vector_search
    from .memory_system.api.routers import context as context_router
    from .memory_system.api.routers import session_admin
    from .memory_system.api.routers import session_linking
    from .memory_system.api.routers import session_lifecycle
    from .memory_system.api.routers import text_processing
    from .memory_system.api.routers import entity_extraction
    from .memory_system.api.routers import fact_extraction
    from .memory_system.api.routers import importance_scoring
    from .memory_system.api.routers import context_compression
    app.include_router(memory_session.router, prefix="/api/memory/session", tags=["memory-session"])
    app.include_router(memory_router.router, prefix="/api/memory/memories", tags=["memory"])
    app.include_router(vector_search.router, prefix="/api/memory/vector", tags=["memory-vector"])
    app.include_router(context_router.router, prefix="/api/memory/context", tags=["memory-context"])
    app.include_router(session_admin.router, prefix="/api/memory/admin", tags=["memory-admin"])
    app.include_router(session_linking.router, prefix="/api/memory/linking", tags=["memory-linking"])
    app.include_router(session_lifecycle.router, prefix="/api/memory", tags=["memory-lifecycle"])
    app.include_router(text_processing.router, prefix="/api/memory/text", tags=["memory-text"])
    app.include_router(entity_extraction.router, prefix="/api/memory/entities", tags=["memory-entities"])
    app.include_router(fact_extraction.router, prefix="/api/memory/facts", tags=["memory-facts"])
    app.include_router(importance_scoring.router, prefix="/api/memory/importance", tags=["memory-importance"])
    app.include_router(context_compression.router, prefix="/api/memory/compression", tags=["memory-compression"])
    logger.info("Memory system with context injection, text processing, entity extraction, fact extraction, importance scoring and context compression integrated successfully")
except ImportError as e:
    logger.warning(f"Memory system not available: {e}")


@app.get("/api", tags=["root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "operational",
        "environment": settings.APP_ENV,
        "endpoints": {
            "documentation": "/api/docs",
            "openapi": "/api/openapi.json",
            "health": "/health",
            "sources": "/api/v1/sources",
            "search": "/api/v1/search",
            "unified_search": "/api/v1/search/unified",
            "search_suggestions": "/api/v1/search/suggest",
            "jobs": "/api/v1/jobs",
            "memories": "/api/v1/memories",
            "websocket": "/ws",
            "memory_system": {
                "sessions": "/api/memory/session",
                "memory_storage": "/api/memory/memories",
                "vector_search": "/api/memory/vector",
                "context_injection": "/api/memory/context",
                "administration": "/api/memory/admin",
                "session_linking": "/api/memory/linking",
                "text_processing": "/api/memory/text",
                "entity_extraction": "/api/memory/entities",
                "fact_extraction": "/api/memory/facts",
                "importance_scoring": "/api/memory/importance"
            },
            "security": {
                "cors_config": "/api/security/cors/config",
                "cors_stats": "/api/security/cors/security/stats",
                "cors_health": "/api/security/cors/health",
                "monitoring_stats": "/api/security/monitoring/stats",
                "monitoring_events": "/api/security/monitoring/events",
                "monitoring_health": "/api/security/monitoring/health",
                "headers_status": "/api/security/headers/status",
                "csrf_token": "/api/security/headers/csrf/token",
                "headers_health": "/api/security/headers/health",
                "rate_limiting_status": "/api/security/rate-limiting/status",
                "rate_limiting_stats": "/api/security/rate-limiting/stats",
                "rate_limiting_health": "/api/security/rate-limiting/health",
                "blacklist_management": "/api/security/rate-limiting/blacklist",
                "active_clients": "/api/security/rate-limiting/clients"
            }
        }
    }

# Mount static files for frontend (check if directory exists)
# Note: This should be done AFTER all routes are registered
frontend_dist_path = os.path.join(os.path.dirname(__file__), "..", "web-ui", "dist")
if os.path.exists(frontend_dist_path):
    # Don't mount on root "/" to avoid conflicts with WebSocket routes
    # Instead, handle static files through a specific route
    pass


@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "api": "operational",
            "database": "checking",
            "redis": "checking",
            "weaviate": "checking"
        }
    }
    
    # Check database
    try:
        from .models import get_db
        from sqlalchemy import text
        
        db = next(get_db())
        db.execute(text("SELECT 1"))
        health_status["services"]["database"] = "operational"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["services"]["database"] = "error"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        from .services.cache import redis_client
        if await redis_client.ping():
            health_status["services"]["redis"] = "operational"
        else:
            health_status["services"]["redis"] = "error"
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["services"]["redis"] = "error"
        health_status["status"] = "degraded"
    
    # Check Weaviate
    try:
        from .services.vector_store import vector_store
        if await vector_store.health_check():
            health_status["services"]["weaviate"] = "operational"
        else:
            health_status["services"]["weaviate"] = "error"
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Weaviate health check failed: {e}")
        health_status["services"]["weaviate"] = "error"
        health_status["status"] = "degraded"
    
    return health_status


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Don't expose internal errors in production
    if settings.DEBUG:
        error_detail = str(exc)
    else:
        error_detail = "An internal error occurred"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": error_detail,
            "path": str(request.url.path)
        }
    )