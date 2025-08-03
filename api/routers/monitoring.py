"""
Monitoring and Health Check Router
Provides comprehensive system monitoring and health status
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
import psutil
import logging

from ..dependencies import get_db
from ..models import MemoryItem, Document, DocumentChunk, MistakeTracking
from ..services.cache import get_cache_service
from ..services.background_jobs import get_job_manager
from ..services.pattern_workers import get_pattern_workers
from ..services.realtime_learning_pipeline import get_learning_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Comprehensive health check of all system components
    
    Returns detailed status of:
    - Database connections
    - Cache services
    - Background jobs
    - AI features
    - System resources
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
        "metrics": {}
    }
    
    # Database health
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_count = {
            "memories": db.query(MemoryItem).count(),
            "documents": db.query(Document).count(),
            "chunks": db.query(DocumentChunk).count(),
            "mistakes": db.query(MistakeTracking).count()
        }
        health_status["components"]["database"] = {
            "status": "healthy",
            "counts": db_count
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Cache health
    try:
        cache = await get_cache_service()
        if cache and cache.client:
            await cache.client.ping()
            health_status["components"]["cache"] = {"status": "healthy"}
        else:
            health_status["components"]["cache"] = {"status": "unavailable"}
    except Exception as e:
        health_status["components"]["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Background jobs health
    try:
        job_manager = await get_job_manager()
        job_status = job_manager.get_job_status()
        health_status["components"]["background_jobs"] = {
            "status": "healthy" if job_status["scheduler_running"] else "stopped",
            "job_count": job_status["total_jobs"]
        }
    except Exception as e:
        health_status["components"]["background_jobs"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Pattern workers health
    try:
        pattern_workers = await get_pattern_workers()
        worker_status = pattern_workers.get_status()
        health_status["components"]["pattern_workers"] = {
            "status": "healthy" if worker_status["running"] else "stopped",
            "worker_count": worker_status["worker_count"]
        }
    except Exception as e:
        health_status["components"]["pattern_workers"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Real-time pipeline health
    try:
        pipeline = await get_learning_pipeline()
        await pipeline.redis_client.ping()
        health_status["components"]["realtime_pipeline"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["realtime_pipeline"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # System resources
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status["metrics"]["system"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
        
        # Warn if resources are high
        if cpu_percent > 80 or memory.percent > 80 or disk.percent > 90:
            health_status["status"] = "warning"
            
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
    
    return health_status


@router.get("/metrics")
async def get_system_metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get current system metrics and statistics"""
    
    # Database metrics
    since_24h = datetime.utcnow() - timedelta(hours=24)
    since_1h = datetime.utcnow() - timedelta(hours=1)
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "database": {
            "total_memories": db.query(MemoryItem).count(),
            "memories_24h": db.query(MemoryItem).filter(MemoryItem.created_at >= since_24h).count(),
            "memories_1h": db.query(MemoryItem).filter(MemoryItem.created_at >= since_1h).count(),
            "total_documents": db.query(Document).count(),
            "total_chunks": db.query(DocumentChunk).count(),
            "total_mistakes": db.query(MistakeTracking).count(),
            "unresolved_mistakes": db.query(MistakeTracking).filter(MistakeTracking.resolved == False).count()
        },
        "performance": {
            "avg_chunk_size": db.query(func.avg(func.length(DocumentChunk.content))).scalar() or 0,
            "avg_memory_size": db.query(func.avg(func.length(MemoryItem.content))).scalar() or 0
        }
    }
    
    # Add cache metrics if available
    try:
        cache = await get_cache_service()
        if cache and hasattr(cache, 'get_stats'):
            metrics["cache"] = await cache.get_stats()
    except:
        pass
    
    # Add system metrics
    try:
        metrics["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids())
        }
    except:
        pass
    
    return metrics


@router.get("/ai-features/status")
async def get_ai_features_status() -> Dict[str, Any]:
    """Get status of all AI Intelligence features"""
    
    features = {
        "session_continuity": {
            "name": "Session Continuity & Context Restoration",
            "status": "partial",
            "working_endpoints": ["/api/claude-auto/memory/stats", "/api/claude-auto/session/current"],
            "missing_endpoints": ["/api/claude-auto/session/start", "/api/claude-auto/session/handoff"]
        },
        "project_context": {
            "name": "Project-Level Context Management",
            "status": "partial",
            "working_endpoints": ["/api/project-context/current", "/api/project-context/list"],
            "missing_endpoints": ["/api/project-context/auto-detect", "/api/project-context/memory"]
        },
        "mistake_learning": {
            "name": "Mistake Learning & Prevention",
            "status": "partial",
            "working_endpoints": ["/api/mistake-learning/lessons", "/api/mistake-learning/patterns"],
            "missing_endpoints": ["/api/mistake-learning/track", "/api/mistake-learning/search"]
        },
        "proactive_assistance": {
            "name": "Proactive Task Assistance",
            "status": "working",
            "working_endpoints": [
                "/api/proactive/health",
                "/api/proactive/analyze", 
                "/api/proactive/brief",
                "/api/proactive/incomplete-tasks",
                "/api/proactive/predictions", 
                "/api/proactive/suggestions",
                "/api/proactive/reminders",
                "/api/proactive/check-interrupt",
                "/api/proactive/context"
            ],
            "missing_endpoints": []
        },
        "decision_reasoning": {
            "name": "Decision Recording & Reasoning",
            "status": "working",
            "working_endpoints": ["/api/decisions/record", "/api/decisions/explain/{id}", "/api/decisions/search"],
            "missing_endpoints": []
        },
        "code_evolution": {
            "name": "Code Evolution Tracking",
            "status": "partial",
            "working_endpoints": ["/api/code-evolution/history", "/api/code-evolution/patterns/analytics"],
            "missing_endpoints": ["/api/code-evolution/track"]
        },
        "performance_intelligence": {
            "name": "Performance Intelligence",
            "status": "partial",
            "working_endpoints": ["/api/performance/stats", "/api/performance/recommendations"],
            "missing_endpoints": ["/api/performance/track"]
        },
        "pattern_recognition": {
            "name": "Pattern Recognition & Learning",
            "status": "partial",
            "working_endpoints": ["/api/patterns/analyze"],
            "missing_endpoints": ["/api/patterns/user/{user_id}", "/api/patterns/recent"]
        }
    }
    
    # Calculate overall completion
    total_features = len(features)
    working_features = sum(1 for f in features.values() if f["status"] == "working")
    partial_features = sum(1 for f in features.values() if f["status"] == "partial")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": {
            "total_features": total_features,
            "working": working_features,
            "partial": partial_features,
            "not_implemented": total_features - working_features - partial_features,
            "completion_percentage": round((working_features + partial_features * 0.5) / total_features * 100, 1)
        },
        "features": features
    }


@router.get("/alerts")
async def get_system_alerts(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Get current system alerts and warnings"""
    
    alerts = []
    
    # Check for high error rate
    recent_errors = db.query(MistakeTracking).filter(
        MistakeTracking.created_at >= datetime.utcnow() - timedelta(hours=1),
        MistakeTracking.resolved == False
    ).count()
    
    if recent_errors > 10:
        alerts.append({
            "level": "warning",
            "category": "errors",
            "message": f"High error rate: {recent_errors} unresolved errors in the last hour",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Check system resources
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        if cpu_percent > 80:
            alerts.append({
                "level": "warning",
                "category": "performance",
                "message": f"High CPU usage: {cpu_percent}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if memory_percent > 80:
            alerts.append({
                "level": "warning",
                "category": "performance",
                "message": f"High memory usage: {memory_percent}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        if disk_percent > 90:
            alerts.append({
                "level": "critical",
                "category": "storage",
                "message": f"Low disk space: {100 - disk_percent}% free",
                "timestamp": datetime.utcnow().isoformat()
            })
    except:
        pass
    
    # Check database size
    large_table_threshold = 100000
    if db.query(MemoryItem).count() > large_table_threshold:
        alerts.append({
            "level": "info",
            "category": "database",
            "message": f"Memory table has over {large_table_threshold} records",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return alerts


@router.get("/dashboard")
async def get_monitoring_dashboard(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard data"""
    
    # Combine all monitoring data
    health = await detailed_health_check(db)
    metrics = await get_system_metrics(db)
    ai_status = await get_ai_features_status()
    alerts = await get_system_alerts(db)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "health": health,
        "metrics": metrics,
        "ai_features": ai_status,
        "alerts": alerts,
        "summary": {
            "system_status": health["status"],
            "ai_completion": ai_status["overall_status"]["completion_percentage"],
            "active_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a["level"] == "critical")
        }
    }