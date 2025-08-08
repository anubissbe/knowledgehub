"""Main FastAPI application"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import logging
import time
import os
from typing import Dict, Any, Optional

# Set up basic logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# ROUTER IMPORTS - Organized by Category
# =====================================================

# Core routers (always available)
from .routers import (
    sources, search, jobs, websocket, memories, chunks, documents, 
    scheduler, project_timeline, admin, exports, claude_simple, 
    claude_working, claude_sync, internal, scraping_status, 
    ai_features, activity, memory_stats, hybrid_memory, tasks
)

# AI Intelligence Features (always available)
from .routers import (
    claude_auto, project_context, mistake_learning, proactive,
    decision_reasoning, code_evolution, performance_metrics,
    claude_workflow, realtime_learning, pattern_recognition,
    claude_integration
)

# Analytics router with fallback
try:
    from .routers import analytics
except ImportError:
    # Fallback if psutil not available
    from .routers import analytics_fixed as analytics

# RAG System with fallback
try:
    from .routers import rag
    RAG_ROUTER = rag
    logger.info("Full RAG router imported successfully")
except ImportError as e:
    logger.warning(f"Full RAG router not available: {e}. Using simple RAG router.")
    from .routers import rag_simple as RAG_ROUTER

# =====================================================
# CONDITIONAL ROUTER IMPORTS
# =====================================================

def safe_import_router(module_name, variable_name=None, description=None):
    """Safely import router and return (router, availability_flag)"""
    try:
        router_module = __import__(f".routers.{module_name}", fromlist=[module_name], level=1)
        if description:
            logger.info(f"{description} router imported successfully")
        return router_module, True
    except ImportError as e:
        if description:
            logger.warning(f"{description} router not available: {e}")
        return None, False

# Advanced Features
semantic_analysis, SEMANTIC_ANALYSIS_AVAILABLE = safe_import_router("semantic_analysis", description="Semantic analysis")
rag_advanced, RAG_ADVANCED_AVAILABLE = safe_import_router("rag_advanced", description="Advanced RAG")
unified_search, UNIFIED_SEARCH_AVAILABLE = safe_import_router("unified_search", description="Unified search")
graphrag, GRAPHRAG_AVAILABLE = safe_import_router("graphrag", description="GraphRAG")

# Enterprise Features
code_embeddings, CODE_EMBEDDINGS_AVAILABLE = safe_import_router("code_embeddings", description="Code embeddings")
knowledge_graph, KNOWLEDGE_GRAPH_AVAILABLE = safe_import_router("knowledge_graph", description="Knowledge graph")
timescale_analytics, TIME_SERIES_AVAILABLE = safe_import_router("timescale_analytics", description="TimescaleDB analytics")
object_storage, OBJECT_STORAGE_AVAILABLE = safe_import_router("object_storage", description="Object storage")

# Session and Error Management
session_management, SESSION_MANAGEMENT_AVAILABLE = safe_import_router("session_management", description="Session management")
error_tracking, ERROR_TRACKING_AVAILABLE = safe_import_router("error_tracking", description="Error tracking")
enhanced_decisions, ENHANCED_DECISIONS_AVAILABLE = safe_import_router("enhanced_decisions", description="Enhanced decisions")

# Workflow and Integration
workflow_integration, WORKFLOW_ROUTER_AVAILABLE = safe_import_router("workflow_integration", description="Workflow integration")
memory_sync, MEMORY_SYNC_ROUTER_AVAILABLE = safe_import_router("memory_sync", description="Memory sync")
from .routes import auth, cors_security, security_monitoring, security_headers, rate_limiting, persistent_context_simple
from .services.startup import initialize_services, shutdown_services
from .services.real_startup_service import startup_handler, shutdown_handler, get_real_services_health, get_real_services_metrics
from .middleware.auth import SecureAuthMiddleware
from .middleware.advanced_rate_limit import AdvancedRateLimitMiddleware, DDoSProtectionMiddleware
from .security.rate_limiting import RateLimitStrategy
from .middleware.security import ContentValidationMiddleware
from .middleware.security_headers import SecurityHeadersMiddleware as SecureHeadersMiddleware
from .middleware.session_tracking import SessionTrackingMiddleware
from .middleware.security_monitoring import SecurityMonitoringMiddleware
from .middleware.validation import ValidationMiddleware
from .middleware.prometheus_middleware import PrometheusMiddleware
from .middleware.tracing_middleware import TracingMiddleware
from .services.prometheus_metrics import prometheus_metrics
from .services.opentelemetry_tracing import otel_tracing
from .services.service_recovery import service_recovery, register_default_services
from .config import settings

# Reconfigure logging with settings
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting AI Knowledge Hub API...")
    await initialize_services()
    
    # Start real AI and WebSocket services
    try:
        await startup_handler()
        logger.info("Real AI and WebSocket services started")
    except Exception as e:
        logger.error(f"Failed to start real services: {e}")
        # Continue with basic services
    
    # Start Prometheus metrics collection
    try:
        await prometheus_metrics.start_collection()
        logger.info("Prometheus metrics collection started")
    except Exception as e:
        logger.error(f"Failed to start metrics collection: {e}")
        # Continue without metrics
    
    # Start alert processing system
    try:
        from .services.alert_service import real_alert_service
        # Start in background to not block startup
        asyncio.create_task(real_alert_service.start_processing())
        logger.info("Alert processing system started")
    except Exception as e:
        logger.error(f"Failed to start alert processing: {e}")
        # Continue without alerting
    
    # Disable service recovery system temporarily to fix performance issues
    logger.info("Service recovery system disabled to improve API performance")
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Knowledge Hub API...")
    
    # Stop real services first
    try:
        await shutdown_handler()
        logger.info("Real AI and WebSocket services stopped")
    except Exception as e:
        logger.error(f"Error stopping real services: {e}")
    
    # Stop Prometheus metrics collection
    try:
        await prometheus_metrics.stop_collection()
        logger.info("Prometheus metrics collection stopped")
    except Exception as e:
        logger.error(f"Error stopping metrics collection: {e}")
    
    # Shutdown OpenTelemetry tracing
    try:
        otel_tracing.shutdown()
        logger.info("OpenTelemetry tracing shutdown completed")
    except Exception as e:
        logger.error(f"Error shutting down tracing: {e}")
    
    # Shutdown service recovery system
    # Service recovery was disabled, no shutdown needed
    logger.info("Service recovery was disabled - no shutdown required")
    
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
# Remove duplicate SecureHeadersMiddleware - it's added again below with proper config
# app.add_middleware(SecureHeadersMiddleware)
# Temporarily disable ContentValidationMiddleware to debug timeouts
# app.add_middleware(ContentValidationMiddleware)
# Add advanced rate limiting and DDoS protection middleware
# Temporarily disabled for development to prevent blocking UI requests
# app.add_middleware(
#     AdvancedRateLimitMiddleware,
#     requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
#     requests_per_hour=settings.RATE_LIMIT_REQUESTS_PER_MINUTE * 60,
#     requests_per_day=settings.RATE_LIMIT_REQUESTS_PER_MINUTE * 60 * 24,
#     burst_limit=settings.RATE_LIMIT_REQUESTS_PER_MINUTE // 3,
#     strategy=RateLimitStrategy.SLIDING_WINDOW,
#     enable_adaptive=True,
#     enable_ddos_protection=True
# )

# Add DDoS protection middleware
# Temporarily disabled for development to prevent blocking UI requests
# app.add_middleware(
#     DDoSProtectionMiddleware,
#     enable_protection=True,
#     protection_threshold=1000,
#     blacklist_duration=3600
# )
app.add_middleware(SecureAuthMiddleware)

# Add enhanced CORS security middleware
app.add_middleware(CORSSecurityMiddleware, environment=settings.APP_ENV)

# Add Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)

# Add OpenTelemetry tracing middleware
app.add_middleware(TracingMiddleware)

# Add database recovery middleware
try:
    from .middleware.database_recovery_middleware import DatabaseRecoveryMiddleware, DatabaseConnectionPoolMiddleware
    app.add_middleware(DatabaseRecoveryMiddleware, enable_circuit_breaker=True, read_only_fallback=True)
    app.add_middleware(DatabaseConnectionPoolMiddleware)
    logger.info("Database recovery middleware initialized")
except ImportError as e:
    logger.warning(f"Database recovery middleware not available: {e}")

# Add security monitoring middleware
app.add_middleware(SecurityMonitoringMiddleware, environment=settings.APP_ENV)

# Add input validation middleware
from .security.validation import ValidationLevel
validation_level = ValidationLevel.STRICT if settings.APP_ENV == "production" else ValidationLevel.MODERATE
app.add_middleware(ValidationMiddleware, validation_level=validation_level)

# Add security headers middleware
from .security.headers import SecurityHeaderLevel
security_level = SecurityHeaderLevel.STRICT if settings.APP_ENV == "production" else SecurityHeaderLevel.MODERATE
app.add_middleware(SecureHeadersMiddleware, security_level=security_level, csrf_enabled=False, environment=settings.APP_ENV)

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


# =====================================================
# ROUTER REGISTRATION - Organized by Category
# =====================================================

# Security and Authentication
app.include_router(auth.router)
app.include_router(cors_security.router, prefix="/api/security/cors", tags=["security"])
app.include_router(security_monitoring.router, prefix="/api", tags=["security"])
app.include_router(security_headers.router, prefix="/api", tags=["security"])
app.include_router(rate_limiting.router, prefix="/api", tags=["security"])

# Core API Endpoints
app.include_router(sources.router, prefix="/api/v1/sources", tags=["sources"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(tasks.router, prefix="", tags=["tasks"])
app.include_router(chunks.router, prefix="/api/v1/chunks", tags=["chunks"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])
app.include_router(scheduler.router, prefix="/api/v1/scheduler", tags=["scheduler"])
app.include_router(project_timeline.router, prefix="/api/v1", tags=["project-timeline"])
app.include_router(admin.router, prefix="/api/v1", tags=["admin"])
app.include_router(exports.router, prefix="/api/v1", tags=["exports"])
app.include_router(internal.router, prefix="/api/internal", tags=["internal"])
app.include_router(scraping_status.router, prefix="/api/v1/scraping", tags=["scraping"])

# Memory Systems
app.include_router(persistent_context_simple.router, prefix="/api/persistent-context", tags=["memory"])
app.include_router(memory_stats.router, tags=["memory-stats"])
app.include_router(hybrid_memory.router, prefix="/api/hybrid", tags=["hybrid-memory"])

# AI Intelligence Features (always available)
app.include_router(claude_simple.router, tags=["claude-enhancements"])
app.include_router(claude_working.router, tags=["claude-working"])
app.include_router(claude_sync.router, tags=["claude-sync"])
app.include_router(ai_features.router, tags=["ai-features"])
app.include_router(activity.router, tags=["activity"])
app.include_router(claude_auto.router, tags=["claude-auto"])
app.include_router(project_context.router, tags=["project-context"])
app.include_router(mistake_learning.router, tags=["mistake-learning"])
app.include_router(proactive.router, tags=["proactive"])
app.include_router(decision_reasoning.router, tags=["decision-reasoning"])
app.include_router(code_evolution.router, tags=["code-evolution"])
app.include_router(performance_metrics.router, tags=["performance-metrics"])
app.include_router(claude_workflow.router, tags=["claude-workflow"])
app.include_router(realtime_learning.router, tags=["realtime-learning"])
app.include_router(pattern_recognition.router, tags=["pattern-recognition"])
app.include_router(claude_integration.router, tags=["claude-integration"])

# RAG Systems
app.include_router(RAG_ROUTER.router, tags=["rag"])

# Analytics
app.include_router(analytics.router, prefix="/api", tags=["analytics"])

# Conditional Routers - Advanced Features
if SEMANTIC_ANALYSIS_AVAILABLE:
    app.include_router(semantic_analysis.router, tags=["semantic-analysis"])
    logger.info("Semantic analysis router registered")

if RAG_ADVANCED_AVAILABLE:
    app.include_router(rag_advanced.router, tags=["rag-advanced"])
    logger.info("Advanced RAG router registered")

if UNIFIED_SEARCH_AVAILABLE:
    app.include_router(unified_search.router, prefix="/api/v1/search", tags=["unified-search"])
    logger.info("Unified search router registered")

if GRAPHRAG_AVAILABLE:
    app.include_router(graphrag.router, tags=["graphrag"])
    logger.info("GraphRAG router registered")

# Conditional Routers - Enterprise Features
if CODE_EMBEDDINGS_AVAILABLE:
    app.include_router(code_embeddings.router, tags=["code-embeddings"])
    logger.info("Code embeddings router registered")

if KNOWLEDGE_GRAPH_AVAILABLE:
    app.include_router(knowledge_graph.router, tags=["knowledge-graph"])
    logger.info("Knowledge graph router registered")

if TIME_SERIES_AVAILABLE:
    app.include_router(timescale_analytics.router, tags=["timescale-analytics"])
    logger.info("TimescaleDB analytics router registered")

if OBJECT_STORAGE_AVAILABLE:
    app.include_router(object_storage.router, tags=["object-storage"])
    logger.info("Object storage router registered")

# Conditional Routers - Session and Error Management
if SESSION_MANAGEMENT_AVAILABLE:
    app.include_router(session_management.router, tags=["session-management"])
    logger.info("Session management router registered")

if ERROR_TRACKING_AVAILABLE:
    app.include_router(error_tracking.router, tags=["error-tracking"])
    logger.info("Error tracking router registered")

if ENHANCED_DECISIONS_AVAILABLE:
    app.include_router(enhanced_decisions.router, prefix="/api/enhanced", tags=["decision-recording"])
    logger.info("Enhanced decisions router registered")

# Conditional Routers - Workflow and Integration
if WORKFLOW_ROUTER_AVAILABLE:
    app.include_router(workflow_integration.router, prefix="/api/v1", tags=["workflow"])
    logger.info("Workflow integration router registered")

if MEMORY_SYNC_ROUTER_AVAILABLE:
    app.include_router(memory_sync.router, prefix="/api/v1", tags=["memory-sync"])
    logger.info("Memory sync router registered")
# Additional Conditional Routers - Extended Features
# LlamaIndex RAG System (Enterprise RAG with mathematical optimizations)
llamaindex_rag, LLAMAINDEX_AVAILABLE = safe_import_router("llamaindex_rag", description="LlamaIndex RAG")
if LLAMAINDEX_AVAILABLE:
    app.include_router(llamaindex_rag.router, tags=["llamaindex-rag"])
    logger.info("LlamaIndex RAG router registered")

# Zep Memory System (Conversational Memory)
zep_memory, ZEP_AVAILABLE = safe_import_router("zep_memory", description="Zep memory")
if ZEP_AVAILABLE:
    app.include_router(zep_memory.router, tags=["zep-memory"])
    logger.info("Zep memory router registered")

# Multi-Agent System (Complex query processing with specialized agents)
multi_agent, MULTI_AGENT_AVAILABLE = safe_import_router("multi_agent", description="Multi-agent")
if MULTI_AGENT_AVAILABLE:
    app.include_router(multi_agent.router, tags=["multi-agent"])
    logger.info("Multi-agent router registered")

# Real AI and WebSocket endpoints
@app.get("/api/real-services/health")
async def real_services_health():
    """Get health status of real AI and WebSocket services"""
    try:
        return await get_real_services_health()
    except Exception as e:
        logger.error(f"Real services health check failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/api/real-services/metrics")
async def real_services_metrics():
    """Get metrics from real AI and WebSocket services"""
    try:
        return await get_real_services_metrics()
    except Exception as e:
        logger.error(f"Real services metrics failed: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/api/ai/analyze-error")
async def analyze_error_endpoint(request: Request):
    """Analyze error patterns using real AI intelligence"""
    try:
        from .services.real_ai_intelligence import real_ai_intelligence
        
        data = await request.json()
        result = await real_ai_intelligence.analyze_error_patterns(
            error_text=data["error_text"],
            error_type=data["error_type"],
            context=data.get("context", {}),
            user_id=data["user_id"]
        )
        
        return {
            "pattern_type": result.pattern_type,
            "pattern_name": result.pattern_name,
            "confidence": result.confidence,
            "recommendations": result.recommendations,
            "evidence_count": len(result.evidence),
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        return {"error": str(e)}

@app.post("/api/ai/predict-tasks")
async def predict_tasks_endpoint(request: Request):
    """Predict next tasks using real AI intelligence"""
    try:
        from .services.real_ai_intelligence import real_ai_intelligence
        
        data = await request.json()
        predictions = await real_ai_intelligence.predict_next_tasks(
            user_id=data["user_id"],
            session_id=data["session_id"],
            current_context=data.get("context", {}),
            project_id=data.get("project_id")
        )
        
        return {
            "predictions": [
                {
                    "prediction_type": p.prediction_type,
                    "prediction": p.prediction,
                    "confidence": p.confidence,
                    "reasoning": p.reasoning,
                    "alternatives": p.alternatives
                }
                for p in predictions
            ]
        }
    except Exception as e:
        logger.error(f"Task prediction failed: {e}")
        return {"error": str(e)}

@app.post("/api/ai/analyze-decision")
async def analyze_decision_endpoint(request: Request):
    """Analyze decision patterns using real AI intelligence"""
    try:
        from .services.real_ai_intelligence import real_ai_intelligence
        
        data = await request.json()
        result = await real_ai_intelligence.analyze_decision_patterns(
            decision_context=data["decision_context"],
            user_id=data["user_id"],
            session_id=data["session_id"]
        )
        
        return {
            "pattern_type": result.pattern_type,
            "pattern_name": result.pattern_name,
            "confidence": result.confidence,
            "recommendations": result.recommendations,
            "evidence_count": len(result.evidence),
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"Decision analysis failed: {e}")
        return {"error": str(e)}

@app.post("/api/ai/performance-insights")
async def performance_insights_endpoint(request: Request):
    """Generate performance insights using real AI intelligence"""
    try:
        from .services.real_ai_intelligence import real_ai_intelligence
        
        data = await request.json()
        insights = await real_ai_intelligence.generate_performance_insights(
            metrics=data["metrics"],
            context=data.get("context", {})
        )
        
        return {
            "insights": [
                {
                    "insight_type": i.insight_type,
                    "description": i.description,
                    "impact_score": i.impact_score,
                    "actionable_steps": i.actionable_steps,
                    "evidence_count": i.evidence_count
                }
                for i in insights
            ]
        }
    except Exception as e:
        logger.error(f"Performance insights failed: {e}")
        return {"error": str(e)}

# WebSocket router
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Routes-based routers (from .routes)
def safe_import_route(module_name, description=None):
    """Safely import route and return (router, availability_flag)"""
    try:
        route_module = __import__(f".routes.{module_name}", fromlist=[module_name], level=1)
        if description:
            logger.info(f"{description} route imported successfully")
        return route_module, True
    except ImportError as e:
        if description:
            logger.warning(f"{description} route not available: {e}")
        return None, False

# Performance and Monitoring Routes
performance, PERFORMANCE_AVAILABLE = safe_import_route("performance", description="Performance optimization")
if PERFORMANCE_AVAILABLE:
    app.include_router(performance.router, prefix="/api", tags=["performance"])

copilot_enhancement, COPILOT_AVAILABLE = safe_import_route("copilot_enhancement", description="GitHub Copilot Enhancement")
if COPILOT_AVAILABLE:
    app.include_router(copilot_enhancement.router, tags=["copilot-enhancement"])

health_monitoring, HEALTH_MONITORING_AVAILABLE = safe_import_route("health_monitoring", description="Health Monitoring")
if HEALTH_MONITORING_AVAILABLE:
    app.include_router(health_monitoring.router, tags=["health-monitoring"])

alert_management, ALERT_MANAGEMENT_AVAILABLE = safe_import_route("alert_management", description="Alert Management")
if ALERT_MANAGEMENT_AVAILABLE:
    app.include_router(alert_management.router, tags=["alert-management"])

tracing_management, TRACING_MANAGEMENT_AVAILABLE = safe_import_route("tracing_management", description="Tracing Management")
if TRACING_MANAGEMENT_AVAILABLE:
    app.include_router(tracing_management.router, tags=["tracing-management"])

recovery_management, RECOVERY_MANAGEMENT_AVAILABLE = safe_import_route("recovery_management", description="Recovery Management")
if RECOVERY_MANAGEMENT_AVAILABLE:
    app.include_router(recovery_management.router, tags=["recovery-management"])

database_recovery, DATABASE_RECOVERY_AVAILABLE = safe_import_route("database_recovery", description="Database Recovery")
if DATABASE_RECOVERY_AVAILABLE:
    app.include_router(database_recovery.router, tags=["database-recovery"])

circuit_breaker_management, CIRCUIT_BREAKER_AVAILABLE = safe_import_route("circuit_breaker_management", description="Circuit Breaker Management")
if CIRCUIT_BREAKER_AVAILABLE:
    app.include_router(circuit_breaker_management.router, tags=["circuit-breaker"])

# Additional router imports
background_jobs, BACKGROUND_JOBS_AVAILABLE = safe_import_router("background_jobs", description="Background Jobs")
if BACKGROUND_JOBS_AVAILABLE:
    app.include_router(background_jobs.router, tags=["background-jobs"])

monitoring, MONITORING_AVAILABLE = safe_import_router("monitoring", description="Monitoring")
if MONITORING_AVAILABLE:
    app.include_router(monitoring.router, tags=["monitoring"])

public_search, PUBLIC_SEARCH_AVAILABLE = safe_import_router("public_search", description="Public search")
if PUBLIC_SEARCH_AVAILABLE:
    app.include_router(public_search.router, tags=["public"])

# Memory system routers (complex import)
try:
    from .memory_system.api.routers import (
        session as memory_session,
        memory as memory_router,
        vector_search,
        context as context_router,
        session_admin,
        session_linking,
        session_lifecycle,
        text_processing,
        entity_extraction,
        fact_extraction,
        importance_scoring,
        context_compression,
        persistent_context
    )
    
    # Register all memory system routers
    memory_routers = [
        (memory_session.router, "/api/memory/session", "memory-session"),
        (memory_router.router, "/api/memory/memories", "memory"),
        (vector_search.router, "/api/memory/vector", "memory-vector"),
        (context_router.router, "/api/memory/context", "memory-context"),
        (session_admin.router, "/api/memory/admin", "memory-admin"),
        (session_linking.router, "/api/memory/linking", "memory-linking"),
        (session_lifecycle.router, "/api/memory", "memory-lifecycle"),
        (text_processing.router, "/api/memory/text", "memory-text"),
        (entity_extraction.router, "/api/memory/entities", "memory-entities"),
        (fact_extraction.router, "/api/memory/facts", "memory-facts"),
        (importance_scoring.router, "/api/memory/importance", "memory-importance"),
        (context_compression.router, "/api/memory/compression", "memory-compression"),
        (persistent_context.router, "/api/memory/persistent", "memory-persistent"),
    ]
    
    for router, prefix, tag in memory_routers:
        app.include_router(router, prefix=prefix, tags=[tag])
    
    logger.info("Memory system routers registered successfully")
except ImportError as e:
    logger.warning(f"Memory system not available: {e}")

# Learning system routers
learning_working, LEARNING_AVAILABLE = safe_import_router("learning_working", description="Learning system")
if LEARNING_AVAILABLE:
    app.include_router(learning_working.router, tags=["learning"])

cross_session_learning, CROSS_SESSION_LEARNING_AVAILABLE = safe_import_router("cross_session_learning", description="Cross-session learning")
if CROSS_SESSION_LEARNING_AVAILABLE:
    app.include_router(cross_session_learning.router, prefix="/api", tags=["cross-session-learning"])


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
                "importance_scoring": "/api/memory/importance",
                "persistent_context": "/api/memory/persistent",
                "context_query": "/api/memory/persistent/context/query",
                "context_summary": "/api/memory/persistent/context/summary",
                "context_analytics": "/api/memory/persistent/context/analytics",
                "persistent_context_health": "/api/persistent-context/health",
                "persistent_context_status": "/api/persistent-context/status"
            },
            "learning_system": {
                "learn": "/api/learning/learn",
                "feedback": "/api/learning/feedback",
                "outcomes": "/api/learning/outcomes",
                "patterns": "/api/learning/patterns/search",
                "adapt": "/api/learning/adapt",
                "analytics": "/api/learning/analytics",
                "success_rate": "/api/learning/success-rate",
                "success_patterns": "/api/learning/patterns/success",
                "failure_patterns": "/api/learning/patterns/failure"
            },
            "cross_session_learning": {
                "sessions": "/api/cross-session-learning/sessions",
                "active_sessions": "/api/cross-session-learning/sessions/active",
                "continue_session": "/api/cross-session-learning/sessions/{session_id}/continue",
                "pause_session": "/api/cross-session-learning/sessions/{session_id}/pause",
                "complete_session": "/api/cross-session-learning/sessions/{session_id}/complete",
                "knowledge_transfer": "/api/cross-session-learning/knowledge-transfer",
                "learning_history": "/api/cross-session-learning/history",
                "cross_session_patterns": "/api/cross-session-learning/patterns/cross-session",
                "pattern_analysis": "/api/cross-session-learning/analysis",
                "recurring_patterns": "/api/cross-session-learning/patterns/recurring",
                "pattern_clusters": "/api/cross-session-learning/patterns/clusters",
                "learning_progression": "/api/cross-session-learning/progression",
                "outcome_predictions": "/api/cross-session-learning/predictions"
            },
            "decision_reasoning": {
                "record_decision": "/api/decisions/record",
                "explain_decision": "/api/decisions/explain/{decision_id}",
                "update_outcome": "/api/decisions/update-outcome",
                "find_similar": "/api/decisions/similar",
                "suggest_decision": "/api/decisions/suggest",
                "confidence_report": "/api/decisions/confidence-report",
                "categories": "/api/decisions/categories",
                "reasoning_patterns": "/api/decisions/patterns/{category}",
                "search_decisions": "/api/decisions/search"
            },
            "code_evolution": {
                "track_change": "/api/code-evolution/track-change",
                "evolution_history": "/api/code-evolution/history",
                "compare_versions": "/api/code-evolution/compare/{change_id}",
                "refactoring_suggestions": "/api/code-evolution/suggest-refactoring",
                "update_impact": "/api/code-evolution/update-impact",
                "pattern_analytics": "/api/code-evolution/patterns/analytics",
                "learned_patterns": "/api/code-evolution/patterns/learned",
                "analyze_file": "/api/code-evolution/suggestions/file",
                "evolution_trends": "/api/code-evolution/trends",
                "upload_diff": "/api/code-evolution/upload-diff",
                "search_records": "/api/code-evolution/search"
            },
            "performance_metrics": {
                "track_performance": "/api/performance/track",
                "track_batch": "/api/performance/track-batch",
                "performance_report": "/api/performance/report",
                "predict_performance": "/api/performance/predict",
                "analyze_patterns": "/api/performance/patterns",
                "optimization_history": "/api/performance/optimization-history",
                "command_categories": "/api/performance/categories",
                "performance_thresholds": "/api/performance/thresholds",
                "optimization_strategies": "/api/performance/optimization-strategies",
                "benchmark_command": "/api/performance/benchmark",
                "recommendations": "/api/performance/recommendations",
                "performance_trends": "/api/performance/trends"
            },
            "project_timeline": {
                "timelines": "/api/v1/project-timeline/timelines",
                "milestones": "/api/v1/project-timeline/milestones",
                "progress_analysis": "/api/v1/project-timeline/timelines/{timeline_id}/progress/analysis",
                "velocity_tracking": "/api/v1/project-timeline/timelines/{timeline_id}/velocity",
                "insights": "/api/v1/project-timeline/timelines/{timeline_id}/insights",
                "milestone_detection": "/api/v1/project-timeline/timelines/{timeline_id}/milestones/detect",
                "progress_snapshots": "/api/v1/project-timeline/timelines/{timeline_id}/snapshots"
            },
            "admin": {
                "dashboard": "/api/v1/admin/dashboard",
                "system_overview": "/api/v1/admin/system/overview",
                "user_management": "/api/v1/admin/users/management",
                "user_manage": "/api/v1/admin/users/manage",
                "system_configuration": "/api/v1/admin/system/configuration",
                "system_command": "/api/v1/admin/system/command",
                "system_health_detailed": "/api/v1/admin/system/health/detailed",
                "advanced_analytics": "/api/v1/admin/analytics/advanced",
                "system_logs": "/api/v1/admin/logs/system"
            },
            "error_tracking": {
                "record_error": "/api/errors/occurrences",
                "provide_feedback": "/api/errors/feedback",
                "get_predictions": "/api/errors/predictions",
                "error_analytics": "/api/errors/analytics",
                "list_patterns": "/api/errors/patterns",
                "pattern_details": "/api/errors/patterns/{pattern_id}",
                "create_solution": "/api/errors/solutions",
                "analyzer_start": "/api/errors/analyzer/start",
                "analyzer_stop": "/api/errors/analyzer/stop",
                "health": "/api/errors/health"
            },
            "decision_recording": {
                "record_decision": "/api/enhanced/decisions/record",
                "get_recommendations": "/api/enhanced/decisions/recommend",
                "record_outcome": "/api/enhanced/decisions/outcomes",
                "provide_feedback": "/api/enhanced/decisions/feedback",
                "decision_tree": "/api/enhanced/decisions/tree/{decision_id}",
                "analyze_impact": "/api/enhanced/decisions/impact/{decision_id}",
                "analytics": "/api/enhanced/decisions/analytics",
                "trends": "/api/enhanced/decisions/trends",
                "success_factors": "/api/enhanced/decisions/success-factors",
                "insights": "/api/enhanced/decisions/insights",
                "predict_outcome": "/api/enhanced/decisions/predict",
                "search": "/api/enhanced/decisions/search",
                "patterns": "/api/enhanced/decisions/patterns",
                "health": "/api/enhanced/decisions/health"
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


@app.get("/docs", tags=["docs"])
async def docs_redirect():
    """Redirect /docs to /api/docs for compatibility with README badge"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/docs", status_code=301)


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


# =====================================================
# SPECIALIZED FEATURE ROUTERS - End of File
# =====================================================

# FPGA Workflow Optimization System - Phase 2.3
fpga_workflow, FPGA_WORKFLOW_AVAILABLE = safe_import_router("fpga_workflow", description="FPGA Workflow Optimization")
if FPGA_WORKFLOW_AVAILABLE:
    app.include_router(fpga_workflow.router, tags=["fpga-workflow"])

# Cross-Domain Knowledge Synthesis System - Phase 2.4
cross_domain_synthesis, CROSS_DOMAIN_AVAILABLE = safe_import_router("cross_domain_synthesis", description="Cross-Domain Knowledge Synthesis")
if CROSS_DOMAIN_AVAILABLE:
    app.include_router(cross_domain_synthesis.router, tags=["cross-domain-synthesis"])

# Enterprise Features System - Multi-tenant, scaling, security
enterprise, ENTERPRISE_AVAILABLE = safe_import_router("enterprise", description="Enterprise Features")
if ENTERPRISE_AVAILABLE:
    app.include_router(enterprise.router, tags=["enterprise"])


