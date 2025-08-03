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

from .routers import sources, search, jobs, websocket, memories, chunks, documents, scheduler, project_timeline, admin, exports, claude_simple, claude_working, claude_sync, internal, scraping_status, ai_features, activity, memory_stats, hybrid_memory, tasks

# Try to import RAG router with fallback
try:
    from .routers import rag
    RAG_ROUTER = rag
    logger.info("Full RAG router imported successfully")
except ImportError as e:
    logger.warning(f"Full RAG router not available: {e}. Using simple RAG router.")
    from .routers import rag_simple as RAG_ROUTER
from .routers import (
    claude_auto,
    project_context,
    mistake_learning,
    proactive,
    decision_reasoning,
    code_evolution,
    performance_metrics,
    claude_workflow,
    realtime_learning,
    pattern_recognition,
    claude_integration
)

# Try to import unified search router
try:
    from .routers import unified_search
    UNIFIED_SEARCH_AVAILABLE = True
    logger.info("Unified search router imported successfully")
except ImportError as e:
    logger.warning(f"Unified search router not available: {e}")
    UNIFIED_SEARCH_AVAILABLE = False

# Try to import new services, gracefully handle import errors
try:
    from .routers import code_embeddings
    CODE_EMBEDDINGS_AVAILABLE = True
    logger.info("Code embeddings router imported successfully")
except ImportError as e:
    logger.warning(f"Code embeddings router not available: {e}")
    CODE_EMBEDDINGS_AVAILABLE = False

try:
    from .routers import knowledge_graph
    KNOWLEDGE_GRAPH_AVAILABLE = True
    logger.info("Knowledge graph router imported successfully")
except ImportError as e:
    logger.warning(f"Knowledge graph router not available: {e}")
    KNOWLEDGE_GRAPH_AVAILABLE = False

try:
    from .routers import timescale_analytics
    TIME_SERIES_AVAILABLE = True
    logger.info("TimescaleDB analytics router imported successfully")
except ImportError as e:
    logger.warning(f"TimescaleDB analytics router not available: {e}")
    TIME_SERIES_AVAILABLE = False

try:
    from .routers import object_storage
    OBJECT_STORAGE_AVAILABLE = True
    logger.info("Object storage router imported successfully")
except ImportError as e:
    logger.warning(f"Object storage router not available: {e}")
    OBJECT_STORAGE_AVAILABLE = False

try:
    from .routers import session_management
    SESSION_MANAGEMENT_AVAILABLE = True
    logger.info("Session management router imported successfully")
except ImportError as e:
    logger.warning(f"Session management router not available: {e}")
    SESSION_MANAGEMENT_AVAILABLE = False

try:
    from .routers import error_tracking
    ERROR_TRACKING_AVAILABLE = True
    logger.info("Error tracking router imported successfully")
except ImportError as e:
    logger.warning(f"Error tracking router not available: {e}")
    ERROR_TRACKING_AVAILABLE = False

try:
    from .routers import enhanced_decisions
    ENHANCED_DECISIONS_AVAILABLE = True
    logger.info("Enhanced decisions router imported successfully")
except ImportError as e:
    logger.warning(f"Enhanced decisions router not available: {e}")
    ENHANCED_DECISIONS_AVAILABLE = False

try:
    from .routers import workflow_integration
    WORKFLOW_ROUTER_AVAILABLE = True
    logger.info("Workflow integration router imported successfully")
except ImportError as e:
    logger.warning(f"Workflow integration router not available: {e}")
    WORKFLOW_ROUTER_AVAILABLE = False

try:
    from .routers import memory_sync
    MEMORY_SYNC_ROUTER_AVAILABLE = True
    logger.info("Memory sync router imported successfully")
except ImportError as e:
    logger.warning(f"Memory sync router not available: {e}")
    MEMORY_SYNC_ROUTER_AVAILABLE = False
try:
    from .routes import analytics
except ImportError:
    # Fallback if psutil not available
    from .routes import analytics_fixed as analytics
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


# Include routers
app.include_router(auth.router)  # Authentication endpoints
app.include_router(cors_security.router, prefix="/api/security/cors", tags=["security"])  # CORS security management
app.include_router(security_monitoring.router, prefix="/api", tags=["security"])  # Security monitoring
app.include_router(security_headers.router, prefix="/api", tags=["security"])  # Security headers
app.include_router(rate_limiting.router, prefix="/api", tags=["security"])  # Rate limiting and DDoS protection
app.include_router(persistent_context_simple.router, prefix="/api/persistent-context", tags=["memory"])  # Persistent context
app.include_router(sources.router, prefix="/api/v1/sources", tags=["sources"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])

# Unified Search router
if UNIFIED_SEARCH_AVAILABLE:
    app.include_router(unified_search.router, prefix="/api/v1/search", tags=["unified-search"])
    logger.info("Unified search router added successfully")
else:
    logger.warning("Unified search router not available")
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(tasks.router, prefix="", tags=["tasks"])
app.include_router(chunks.router, prefix="/api/v1/chunks", tags=["chunks"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])
app.include_router(scheduler.router, prefix="/api/v1/scheduler", tags=["scheduler"])
app.include_router(project_timeline.router, prefix="/api/v1", tags=["project-timeline"])
app.include_router(admin.router, prefix="/api/v1", tags=["admin"])
app.include_router(exports.router, prefix="/api/v1", tags=["exports"])
if WORKFLOW_ROUTER_AVAILABLE:
    app.include_router(workflow_integration.router, prefix="/api/v1", tags=["workflow"])
    logger.info("Workflow integration router added successfully")
else:
    logger.warning("Workflow integration router not available")

if MEMORY_SYNC_ROUTER_AVAILABLE:
    app.include_router(memory_sync.router, prefix="/api/v1", tags=["memory-sync"])
    logger.info("Memory sync router added successfully")
else:
    logger.warning("Memory sync router not available")
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(claude_simple.router, tags=["claude-enhancements"])
app.include_router(claude_working.router, tags=["claude-working"])
app.include_router(claude_sync.router, tags=["claude-sync"])
app.include_router(internal.router, prefix="/api/internal", tags=["internal"])
app.include_router(scraping_status.router, prefix="/api/v1/scraping", tags=["scraping"])

# AI Features Summary
app.include_router(ai_features.router, tags=["ai-features"])

# Activity Tracking
app.include_router(activity.router, tags=["activity"])

# Memory Stats - UI compatibility endpoint
app.include_router(memory_stats.router, tags=["memory-stats"])

# Hybrid Memory System - Nova-style local + distributed
app.include_router(hybrid_memory.router, prefix="/api/hybrid", tags=["hybrid-memory"])

# RAG (Retrieval-Augmented Generation) System
app.include_router(RAG_ROUTER.router, tags=["rag"])

# Zep Memory System (Conversational Memory)
try:
    from .routers import zep_memory
    app.include_router(zep_memory.router, tags=["zep-memory"])
    logger.info("Zep memory router integrated successfully")
except ImportError as e:
    logger.warning(f"Zep memory router not available: {e}")

# Multi-Agent System (Complex query processing with specialized agents)
try:
    from .routers import multi_agent
    app.include_router(multi_agent.router, tags=["multi-agent"])
    logger.info("Multi-agent router integrated successfully")
except ImportError as e:
    logger.warning(f"Multi-agent router not available: {e}")

# Claude Auto - Automatic session management
try:
    from .routers import claude_auto
    app.include_router(claude_auto.router, tags=["claude-auto"])
    logger.info("Claude Auto router integrated successfully")
except ImportError as e:
    logger.warning(f"Claude Auto router not available: {e}")

# Project Context - Per-project isolation
try:
    from .routers import project_context
    app.include_router(project_context.router, tags=["project-context"])
    logger.info("Project Context router integrated successfully")
except ImportError as e:
    logger.warning(f"Project Context router not available: {e}")

# Mistake Learning - Learn from errors to prevent repetition
try:
    from .routers import mistake_learning
    app.include_router(mistake_learning.router, tags=["mistake-learning"])
    logger.info("Mistake Learning router integrated successfully")
except ImportError as e:
    logger.warning(f"Mistake Learning router not available: {e}")

# Proactive Assistant - Anticipate needs and provide assistance
try:
    from .routers import proactive
    app.include_router(proactive.router, tags=["proactive"])
    logger.info("Proactive Assistant router integrated successfully")
except ImportError as e:
    logger.warning(f"Proactive Assistant router not available: {e}")

# Decision Reasoning - Track decisions, alternatives, and reasoning with confidence scores
try:
    from .routers import decision_reasoning
    app.include_router(decision_reasoning.router, tags=["decision-reasoning"])
    logger.info("Decision Reasoning router integrated successfully")
except ImportError as e:
    logger.warning(f"Decision Reasoning router not available: {e}")

# Code Evolution - Track code changes, refactoring patterns, and improvements over time
try:
    from .routers import code_evolution
    app.include_router(code_evolution.router, tags=["code-evolution"])
    logger.info("Code Evolution router integrated successfully")
except ImportError as e:
    logger.warning(f"Code Evolution router not available: {e}")

# Performance Metrics - Track command execution patterns, success rates, and optimize performance
try:
    from .routers import performance_metrics
    app.include_router(performance_metrics.router, tags=["performance-metrics"])
    logger.info("Performance Metrics router integrated successfully")
except ImportError as e:
    logger.warning(f"Performance Metrics router not available: {e}")

# Claude Workflow Integration - Automatic memory capture and context extraction
try:
    from .routers import claude_workflow
    app.include_router(claude_workflow.router, tags=["claude-workflow"])
    logger.info("Claude Workflow Integration router integrated successfully")
except ImportError as e:
    logger.warning(f"Claude Workflow Integration router not available: {e}")

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

# Performance optimization router
try:
    from .routes import performance
    app.include_router(performance.router, prefix="/api", tags=["performance"])
    logger.info("Performance optimization router integrated successfully")
except ImportError as e:
    logger.warning(f"Performance optimization router not available: {e}")

# Real-time Learning Pipeline router
try:
    app.include_router(realtime_learning.router, tags=["realtime-learning"])
    logger.info("Real-time Learning Pipeline router integrated successfully")
except ImportError as e:
    logger.warning(f"Real-time Learning Pipeline router not available: {e}")

# Pattern Recognition router
try:
    app.include_router(pattern_recognition.router, tags=["pattern-recognition"])
    logger.info("Pattern Recognition router integrated successfully")
except ImportError as e:
    logger.warning(f"Pattern Recognition router not available: {e}")

# Background Jobs router
try:
    from .routers import background_jobs
    app.include_router(background_jobs.router, tags=["background-jobs"])
    logger.info("Background Jobs router integrated successfully")
except ImportError as e:
    logger.warning(f"Background Jobs router not available: {e}")

# Monitoring router
try:
    from .routers import monitoring
    app.include_router(monitoring.router, tags=["monitoring"])
    logger.info("Monitoring router integrated successfully")
except ImportError as e:
    logger.warning(f"Monitoring router not available: {e}")

# Claude Code Integration router
try:
    app.include_router(claude_integration.router, tags=["claude-integration"])
    logger.info("Claude Code Integration router integrated successfully")
except ImportError as e:
    logger.warning(f"Claude Code Integration router not available: {e}")

# GitHub Copilot Enhancement router
try:
    from .routes import copilot_enhancement
    app.include_router(copilot_enhancement.router, tags=["copilot-enhancement"])
    logger.info("GitHub Copilot Enhancement router integrated successfully")
except ImportError as e:
    logger.warning(f"GitHub Copilot Enhancement router not available: {e}")

# Code embeddings router
if CODE_EMBEDDINGS_AVAILABLE:
    app.include_router(code_embeddings.router, tags=["code-embeddings"])
    logger.info("Code embeddings router added successfully")

# Knowledge graph router
if KNOWLEDGE_GRAPH_AVAILABLE:
    app.include_router(knowledge_graph.router, tags=["knowledge-graph"])
    logger.info("Knowledge graph router added successfully")

# TimescaleDB analytics router
if TIME_SERIES_AVAILABLE:
    app.include_router(timescale_analytics.router, tags=["timescale-analytics"])
    logger.info("TimescaleDB analytics router added successfully")

# Object storage router
if OBJECT_STORAGE_AVAILABLE:
    app.include_router(object_storage.router, tags=["object-storage"])
    logger.info("Object storage router added successfully")

# Session management router
if SESSION_MANAGEMENT_AVAILABLE:
    app.include_router(session_management.router, tags=["session-management"])
    logger.info("Session management router added successfully")

# Error tracking router
if ERROR_TRACKING_AVAILABLE:
    app.include_router(error_tracking.router, tags=["error-tracking"])
    logger.info("Error tracking router added successfully")

# Enhanced decisions router
if ENHANCED_DECISIONS_AVAILABLE:
    app.include_router(enhanced_decisions.router, prefix="/api/enhanced", tags=["decision-recording"])
    logger.info("Enhanced decisions router added successfully")

# Health Monitoring - Production health checks and monitoring
try:
    from .routes import health_monitoring
    app.include_router(health_monitoring.router, tags=["health-monitoring"])
    logger.info("Health Monitoring router integrated successfully")
except ImportError as e:
    logger.warning(f"Health Monitoring router not available: {e}")

# Alert Management - Alert processing and notification system
try:
    from .routes import alert_management
    app.include_router(alert_management.router, tags=["alert-management"])
    logger.info("Alert Management router integrated successfully")
except ImportError as e:
    logger.warning(f"Alert Management router not available: {e}")

# Tracing Management - Distributed tracing and performance analysis
try:
    from .routes import tracing_management
    app.include_router(tracing_management.router, tags=["tracing-management"])
    logger.info("Tracing Management router integrated successfully")
except ImportError as e:
    logger.warning(f"Tracing Management router not available: {e}")

# Recovery Management - Service recovery and self-healing system
try:
    from .routes import recovery_management
    app.include_router(recovery_management.router, tags=["recovery-management"])
    logger.info("Recovery Management router integrated successfully")
except ImportError as e:
    logger.warning(f"Recovery Management router not available: {e}")

# Database Recovery - Database connection recovery and retry logic
try:
    from .routes import database_recovery
    app.include_router(database_recovery.router, tags=["database-recovery"])
    logger.info("Database Recovery router integrated successfully")
except ImportError as e:
    logger.warning(f"Database Recovery router not available: {e}")

# Circuit Breaker Management - Circuit breaker patterns for external services
try:
    from .routes import circuit_breaker_management
    app.include_router(circuit_breaker_management.router, tags=["circuit-breaker"])
    logger.info("Circuit Breaker Management router integrated successfully")
except ImportError as e:
    logger.warning(f"Circuit Breaker Management router not available: {e}")

# WebSocket router
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Unified search router
try:
    from .routers import unified_search
    app.include_router(unified_search.router, prefix="/api/v1/search", tags=["unified-search"])
    logger.info("Unified search router integrated successfully")
except ImportError as e:
    logger.warning(f"Unified search router not available: {e}")

# Public search router (no auth required)
try:
    from .routers import public_search
    app.include_router(public_search.router, tags=["public"])
    logger.info("Public search router integrated successfully")
except ImportError as e:
    logger.warning(f"Public search router not available: {e}")

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
    from .memory_system.api.routers import persistent_context
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
    app.include_router(persistent_context.router, prefix="/api/memory/persistent", tags=["memory-persistent"])
    logger.info("Memory system with context injection, text processing, entity extraction, fact extraction, importance scoring, context compression and persistent context integrated successfully")
except ImportError as e:
    logger.warning(f"Memory system not available: {e}")

# Learning system router
try:
    # Using working learning router to avoid Pydantic SQLAlchemy issues
    from .routers import learning_working
    app.include_router(learning_working.router, tags=["learning"])
    logger.info("Learning system integrated successfully")
except ImportError as e:
    logger.warning(f"Learning system not available: {e}")

# Cross-session learning router
try:
    from .routers import cross_session_learning
    app.include_router(cross_session_learning.router, prefix="/api", tags=["cross-session-learning"])
    logger.info("Cross-session learning system integrated successfully")
except ImportError as e:
    logger.warning(f"Cross-session learning system not available: {e}")


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