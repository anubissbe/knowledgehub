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
from .services.startup import initialize_services, shutdown_services
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.security import SecurityHeadersMiddleware, ContentValidationMiddleware
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware (order matters - last added runs first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(ContentValidationMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE)
app.add_middleware(AuthMiddleware)


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
            "jobs": "/api/v1/jobs",
            "memories": "/api/v1/memories",
            "websocket": "/ws"
        }
    }

# Mount static files for frontend (check if directory exists) - this must be last
# BUT exclude WebSocket paths to prevent assertion errors
frontend_dist_path = os.path.join(os.path.dirname(__file__), "..", "web-ui", "dist")
if os.path.exists(frontend_dist_path):
    # Create a custom static files handler that excludes WebSocket paths
    from starlette.types import Scope, Receive, Send
    
    class SelectiveStaticFiles(StaticFiles):
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            # Skip static file handling for WebSocket paths
            if scope["type"] == "websocket" or scope["path"].startswith("/ws"):
                # Let the WebSocket handler take over
                await self.app(scope, receive, send)
                return
            # For HTTP requests, use normal static file handling
            await super().__call__(scope, receive, send)
    
    app.mount("/", SelectiveStaticFiles(directory=frontend_dist_path, html=True), name="static")


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