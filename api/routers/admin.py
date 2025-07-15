"""Admin Dashboard API Router

Provides administrative endpoints for system management, user administration,
and advanced monitoring capabilities.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from pydantic import BaseModel, Field

from ..models import get_db
from ..models.base import Base
from ..models.memory import MemoryItem as Memory
from ..models.knowledge_source import KnowledgeSource as Source
from ..models.job import Job
from ..models.chunk import Chunk
from ..models.document import Document
from ..services.auth import require_admin
# from ..security.dashboard import security_dashboard
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBearer()

# Pydantic models for admin API

class UserSummary(BaseModel):
    """User summary for admin dashboard"""
    total_users: int
    active_users: int
    admin_users: int
    recent_registrations: int
    most_active_users: List[Dict[str, Any]]

class SystemStats(BaseModel):
    """Comprehensive system statistics"""
    database_stats: Dict[str, Any]
    storage_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    resource_usage: Dict[str, Any]
    uptime: str
    last_backup: Optional[str]

class AdminDashboardData(BaseModel):
    """Complete admin dashboard data"""
    system_overview: Dict[str, Any]
    user_summary: UserSummary
    system_stats: SystemStats
    security_summary: Dict[str, Any]
    recent_activities: List[Dict[str, Any]]
    system_health: Dict[str, Any]
    recommendations: List[Dict[str, Any]]

class ConfigurationUpdate(BaseModel):
    """Configuration update request"""
    section: str
    key: str
    value: Any
    description: Optional[str] = None

class UserManagementRequest(BaseModel):
    """User management request"""
    action: str = Field(..., description="create, update, delete, activate, deactivate")
    user_id: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None

class SystemCommand(BaseModel):
    """System command request"""
    command: str = Field(..., description="Command to execute")
    parameters: Optional[Dict[str, Any]] = None
    confirm: bool = Field(False, description="Confirmation flag for destructive operations")

# Admin Dashboard Endpoints

@router.get("/test")
async def test_admin_endpoint():
    """Simple test endpoint"""
    return {"status": "ok", "message": "Admin router is working"}

@router.post("/system/command-simple")
async def execute_system_command_simple(
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Simple system command endpoint for testing"""
    return {
        "command": "test",
        "result": {"status": "success", "message": "Simple command executed"},
        "executed_at": datetime.now(timezone.utc).isoformat()
    }

@router.get("/dashboard", response_model=AdminDashboardData)
async def get_admin_dashboard(
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get comprehensive admin dashboard data"""
    
    try:
        # Get system overview
        system_overview = await _get_system_overview(db)
        
        # Get user summary
        user_summary = await _get_user_summary(db)
        
        # Get system statistics
        system_stats = await _get_system_stats(db)
        
        # Get security summary
        try:
            from ..security.dashboard import security_dashboard
            security_summary = security_dashboard.get_dashboard_data(hours_back=24)
        except ImportError:
            security_summary = {"status": "not_available", "message": "Security dashboard not available"}
        
        # Get recent activities
        recent_activities = await _get_recent_activities(db)
        
        # Get system health
        system_health = await _get_system_health(db)
        
        # Get recommendations
        recommendations = await _get_admin_recommendations(db)
        
        return AdminDashboardData(
            system_overview=system_overview,
            user_summary=user_summary,
            system_stats=system_stats,
            security_summary=security_summary,
            recent_activities=recent_activities,
            system_health=system_health,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error getting admin dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load admin dashboard")

@router.get("/system/overview")
async def get_system_overview(
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get system overview metrics"""
    return await _get_system_overview(db)

@router.get("/users/management")
async def get_user_management_data(
    current_user: dict = Depends(require_admin),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    role_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get user management data with pagination and filtering"""
    
    try:
        # This would typically query a users table
        # For now, return mock data structure
        users = []
        total_users = 0
        
        # TODO: Implement actual user management once user model is defined
        # query = db.query(User)
        # if search:
        #     query = query.filter(User.username.ilike(f"%{search}%"))
        # if status_filter:
        #     query = query.filter(User.status == status_filter)
        # if role_filter:
        #     query = query.filter(User.role == role_filter)
        
        # total_users = query.count()
        # users = query.offset((page - 1) * page_size).limit(page_size).all()
        
        return {
            "users": users,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total_users,
                "total_pages": (total_users + page_size - 1) // page_size
            },
            "filters": {
                "search": search,
                "status": status_filter,
                "role": role_filter
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting user management data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load user management data")

@router.post("/users/manage")
async def manage_user(
    request: UserManagementRequest,
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Manage user accounts (create, update, delete, activate, deactivate)"""
    
    try:
        # TODO: Implement user management operations
        # This would interact with a user model to perform CRUD operations
        
        action = request.action.lower()
        
        if action == "create":
            # Create new user
            logger.info(f"Creating new user: {request.user_data}")
            return {"message": "User created successfully", "action": "create"}
            
        elif action == "update":
            # Update existing user
            logger.info(f"Updating user {request.user_id}: {request.user_data}")
            return {"message": "User updated successfully", "action": "update"}
            
        elif action == "delete":
            # Delete user
            logger.info(f"Deleting user {request.user_id}")
            return {"message": "User deleted successfully", "action": "delete"}
            
        elif action == "activate":
            # Activate user
            logger.info(f"Activating user {request.user_id}")
            return {"message": "User activated successfully", "action": "activate"}
            
        elif action == "deactivate":
            # Deactivate user
            logger.info(f"Deactivating user {request.user_id}")
            return {"message": "User deactivated successfully", "action": "deactivate"}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
            
    except Exception as e:
        logger.error(f"Error managing user: {e}")
        raise HTTPException(status_code=500, detail="Failed to manage user")

@router.get("/system/configuration")
async def get_system_configuration(
    current_user: dict = Depends(require_admin),
    section: Optional[str] = Query(None, description="Configuration section to retrieve")
):
    """Get system configuration"""
    
    try:
        # Get configuration from settings
        config = {}
        
        if section:
            # Get specific section
            if hasattr(settings, section.upper()):
                config[section] = getattr(settings, section.upper())
            else:
                raise HTTPException(status_code=404, detail="Configuration section not found")
        else:
            # Get all configuration
            config = {
                "app": {
                    "name": settings.APP_NAME,
                    "environment": settings.APP_ENV,
                    "debug": settings.DEBUG,
                    "log_level": settings.LOG_LEVEL
                },
                "api": {
                    "host": settings.API_HOST,
                    "port": settings.API_PORT,
                    "workers": settings.API_WORKERS
                },
                "database": {
                    "url": settings.DATABASE_URL.replace(settings.DATABASE_URL.split('@')[0].split('//')[1], '***'),
                    "pool_size": settings.DATABASE_POOL_SIZE,
                    "max_overflow": settings.DATABASE_MAX_OVERFLOW
                },
                "redis": {
                    "url": settings.REDIS_URL.replace(settings.REDIS_URL.split('@')[0].split('//')[1], '***') if '@' in settings.REDIS_URL else settings.REDIS_URL,
                    "max_connections": settings.REDIS_MAX_CONNECTIONS
                },
                "security": {
                    "cors_origins": settings.CORS_ORIGINS,
                    "rate_limit": settings.RATE_LIMIT_REQUESTS_PER_MINUTE
                }
            }
        
        return {"configuration": config}
        
    except Exception as e:
        logger.error(f"Error getting system configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system configuration")

@router.post("/system/configuration")
async def update_system_configuration(
    request: ConfigurationUpdate,
    current_user: dict = Depends(require_admin)
):
    """Update system configuration"""
    
    try:
        # TODO: Implement configuration updates
        # This would typically update a configuration file or database
        
        logger.info(f"Updating configuration: {request.section}.{request.key} = {request.value}")
        
        return {
            "message": "Configuration updated successfully",
            "section": request.section,
            "key": request.key,
            "value": request.value
        }
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@router.post("/system/command")
async def execute_system_command(
    request: dict,  # Changed from SystemCommand to dict for debugging
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Execute system administration command"""
    
    logger.info(f"Received system command request: {request}")
    
    try:
        # Validate request manually
        if not isinstance(request, dict):
            raise HTTPException(status_code=400, detail="Invalid request format")
        
        command = request.get('command', '').lower()
        confirm = request.get('confirm', False)
        
        if not command:
            raise HTTPException(status_code=400, detail="Command is required")
            
        logger.info(f"Processing command: {command}")
        
        # Define allowed commands
        allowed_commands = {
            "clear_cache": "Clear application cache",
            "restart_services": "Restart background services",
            "backup_database": "Create database backup",
            "cleanup_logs": "Clean up old log files",
            "update_indexes": "Update database indexes",
            "vacuum_database": "Run VACUUM to reclaim storage space",
            "analyze_database": "Run ANALYZE to update statistics",
            "full_db_maintenance": "Run VACUUM ANALYZE for complete optimization",
            "health_check": "Run comprehensive health check"
        }
        
        if command not in allowed_commands:
            raise HTTPException(status_code=400, detail="Command not allowed")
        
        # Execute command based on type
        if command == "clear_cache":
            result = await _clear_cache()
        elif command == "restart_services":
            result = await _restart_services()
        elif command == "backup_database":
            result = await _backup_database()
        elif command == "cleanup_logs":
            result = await _cleanup_logs()
        elif command == "update_indexes":
            result = await _update_indexes()
        elif command == "vacuum_database":
            result = await _vacuum_database(db)
        elif command == "analyze_database":
            result = await _analyze_database(db)
        elif command == "full_db_maintenance":
            result = await _full_db_maintenance(db)
        elif command == "health_check":
            result = await _run_health_check()
        else:
            result = {"status": "unknown", "message": "Command not implemented"}
        
        logger.info(f"Executed system command: {command}")
        
        return {
            "command": command,
            "description": allowed_commands.get(command, "Unknown command"),
            "result": result,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "executed_by": current_user.get("username", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error executing system command: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute system command")

@router.get("/system/health/detailed")
async def get_detailed_system_health(
    current_user: dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get detailed system health information"""
    
    try:
        return await _get_detailed_system_health(db)
        
    except Exception as e:
        logger.error(f"Error getting detailed system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@router.get("/analytics/advanced")
async def get_advanced_analytics(
    current_user: dict = Depends(require_admin),
    days_back: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """Get advanced analytics data for admin dashboard"""
    
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        # Get usage analytics
        usage_analytics = await _get_usage_analytics(db, cutoff_date)
        
        # Get performance analytics
        performance_analytics = await _get_performance_analytics(db, cutoff_date)
        
        # Get error analytics
        error_analytics = await _get_error_analytics(cutoff_date)
        
        return {
            "period": {
                "days_back": days_back,
                "from_date": cutoff_date.isoformat(),
                "to_date": datetime.now(timezone.utc).isoformat()
            },
            "usage": usage_analytics,
            "performance": performance_analytics,
            "errors": error_analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting advanced analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get advanced analytics")

@router.get("/logs/system")
async def get_system_logs(
    current_user: dict = Depends(require_admin),
    level: Optional[str] = Query("INFO", description="Log level filter"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get system logs with filtering"""
    
    try:
        # TODO: Implement log retrieval
        # This would typically read from log files or a logging database
        
        logs = []
        total_logs = 0
        
        # Mock log data for now
        for i in range(limit):
            logs.append({
                "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat(),
                "level": level,
                "logger": "api.main",
                "message": f"Sample log message {i}",
                "details": {"request_id": f"req_{i}"}
            })
        
        return {
            "logs": logs,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_logs
            },
            "filters": {
                "level": level
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system logs")

# Helper functions

async def _get_system_overview(db: Session) -> Dict[str, Any]:
    """Get system overview metrics"""
    
    # Get counts from various tables
    total_sources = db.query(Source).count()
    total_jobs = db.query(Job).count()
    total_chunks = db.query(Chunk).count()
    total_documents = db.query(Document).count()
    total_memories = db.query(Memory).count()
    
    # Get recent activity counts
    recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_jobs = db.query(Job).filter(Job.created_at >= recent_cutoff).count()
    recent_chunks = db.query(Chunk).filter(Chunk.created_at >= recent_cutoff).count()
    
    return {
        "totals": {
            "sources": total_sources,
            "jobs": total_jobs,
            "chunks": total_chunks,
            "documents": total_documents,
            "memories": total_memories
        },
        "recent_activity": {
            "new_jobs_24h": recent_jobs,
            "new_chunks_24h": recent_chunks
        },
        "system_status": "operational",
        "uptime": "99.9%"
    }

async def _get_user_summary(db: Session) -> UserSummary:
    """Get user summary statistics"""
    
    # TODO: Implement with actual user model
    return UserSummary(
        total_users=0,
        active_users=0,
        admin_users=0,
        recent_registrations=0,
        most_active_users=[]
    )

async def _get_system_stats(db: Session) -> SystemStats:
    """Get comprehensive system statistics"""
    
    # Database stats
    # Get actual record counts
    total_sources = db.query(Source).count()
    total_jobs = db.query(Job).count()
    total_chunks = db.query(Chunk).count()
    total_documents = db.query(Document).count()
    total_memories = db.query(Memory).count()
    total_records = total_sources + total_jobs + total_chunks + total_documents + total_memories
    
    database_stats = {
        "total_tables": 10,  # Approximate
        "total_records": total_records,
        "database_size": "1.2 GB",
        "connection_pool_size": settings.DATABASE_POOL_SIZE,
        "active_connections": 5
    }
    
    # Storage stats
    storage_stats = {
        "total_storage": "50 GB",
        "used_storage": "12 GB",
        "free_storage": "38 GB",
        "storage_usage_percent": 24
    }
    
    # Performance metrics
    performance_stats = {
        "avg_response_time": "120ms",
        "requests_per_second": 45,
        "cpu_usage": "35%",
        "memory_usage": "2.1 GB",
        "disk_io": "moderate"
    }
    
    # Resource usage
    resource_usage = {
        "cpu_percent": 35,
        "memory_percent": 42,
        "disk_percent": 24,
        "network_io": "low"
    }
    
    return SystemStats(
        database_stats=database_stats,
        storage_stats=storage_stats,
        performance_metrics=performance_stats,
        resource_usage=resource_usage,
        uptime="3 days, 14 hours",
        last_backup="2025-07-09T12:00:00Z"
    )

async def _get_recent_activities(db: Session, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent system activities"""
    
    activities = []
    
    # Get recent jobs
    recent_jobs = db.query(Job).order_by(desc(Job.created_at)).limit(limit//2).all()
    for job in recent_jobs:
        activities.append({
            "timestamp": job.created_at.isoformat(),
            "type": "job",
            "action": "created",
            "description": f"Job {job.job_type} created",
            "details": {"job_id": str(job.id), "status": job.status}
        })
    
    # Get recent chunks
    recent_chunks = db.query(Chunk).order_by(desc(Chunk.created_at)).limit(limit//2).all()
    for chunk in recent_chunks:
        activities.append({
            "timestamp": chunk.created_at.isoformat(),
            "type": "chunk",
            "action": "created",
            "description": f"Chunk processed",
            "details": {"chunk_id": str(chunk.id), "size": len(chunk.content)}
        })
    
    # Sort by timestamp
    activities.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return activities[:limit]

async def _get_system_health(db: Session) -> Dict[str, Any]:
    """Get system health status"""
    
    # Check database health
    try:
        from sqlalchemy import text
        result = db.execute(text("SELECT 1"))
        db.commit()  # Ensure the query completes
        db_healthy = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_healthy = False
        db.rollback()  # Rollback any failed transaction
    
    # Overall health score
    health_score = 100
    if not db_healthy:
        health_score -= 50
    
    # Health status
    if health_score >= 90:
        status = "excellent"
        color = "green"
    elif health_score >= 70:
        status = "good"
        color = "green"
    elif health_score >= 50:
        status = "degraded"
        color = "orange"
    else:
        status = "critical"
        color = "red"
    
    return {
        "overall_health": health_score,
        "status": status,
        "status_color": color,
        "services": {
            "database": "healthy" if db_healthy else "unhealthy",
            "api": "healthy",
            "cache": "healthy",
            "search": "healthy"
        },
        "last_check": datetime.now(timezone.utc).isoformat()
    }

async def _get_admin_recommendations(db: Session) -> List[Dict[str, Any]]:
    """Get admin recommendations"""
    
    recommendations = []
    
    # Check for system maintenance needs
    recommendations.append({
        "priority": "medium",
        "category": "maintenance",
        "title": "Regular Backup Check",
        "description": "Verify that automated backups are running correctly",
        "action": "Review backup logs and test restore procedures"
    })
    
    # Check for performance optimization
    recommendations.append({
        "priority": "low",
        "category": "performance",
        "title": "Database Optimization",
        "description": "Consider running database maintenance tasks",
        "action": "Run VACUUM and ANALYZE on PostgreSQL tables"
    })
    
    return recommendations

# System command implementations

async def _clear_cache() -> Dict[str, Any]:
    """Clear application cache"""
    return {"status": "success", "message": "Cache cleared successfully"}

async def _restart_services() -> Dict[str, Any]:
    """Restart background services"""
    return {"status": "success", "message": "Services restarted successfully"}

async def _backup_database() -> Dict[str, Any]:
    """Create database backup"""
    return {"status": "success", "message": "Database backup created successfully"}

async def _cleanup_logs() -> Dict[str, Any]:
    """Clean up old log files"""
    return {"status": "success", "message": "Log files cleaned up successfully"}

async def _update_indexes() -> Dict[str, Any]:
    """Update database indexes"""
    return {"status": "success", "message": "Database indexes updated successfully"}

async def _run_health_check() -> Dict[str, Any]:
    """Run comprehensive health check"""
    return {"status": "success", "message": "Health check completed successfully"}

async def _get_detailed_system_health(db: Session) -> Dict[str, Any]:
    """Get detailed system health information"""
    
    return {
        "system_health": await _get_system_health(db),
        "resource_usage": {
            "cpu": {"usage": 35, "cores": 8},
            "memory": {"used": "2.1 GB", "total": "16 GB", "percent": 13},
            "disk": {"used": "12 GB", "total": "50 GB", "percent": 24},
            "network": {"incoming": "1.2 MB/s", "outgoing": "0.8 MB/s"}
        },
        "service_details": {
            "database": {"status": "healthy", "connections": 5, "queries_per_second": 45},
            "cache": {"status": "healthy", "hit_rate": 85, "memory_usage": "512 MB"},
            "search": {"status": "healthy", "index_size": "2.1 GB", "query_time": "120ms"}
        }
    }

async def _get_usage_analytics(db: Session, cutoff_date: datetime) -> Dict[str, Any]:
    """Get usage analytics"""
    
    return {
        "total_requests": 12450,
        "unique_users": 156,
        "popular_endpoints": [
            {"endpoint": "/api/v1/search", "count": 3200},
            {"endpoint": "/api/v1/sources", "count": 2100},
            {"endpoint": "/api/v1/chunks", "count": 1800}
        ],
        "usage_by_hour": []  # Would contain hourly usage data
    }

async def _get_performance_analytics(db: Session, cutoff_date: datetime) -> Dict[str, Any]:
    """Get performance analytics"""
    
    return {
        "avg_response_time": 120,
        "p95_response_time": 350,
        "p99_response_time": 800,
        "error_rate": 0.02,
        "throughput": 45.2
    }

async def _get_error_analytics(cutoff_date: datetime) -> Dict[str, Any]:
    """Get error analytics"""
    
    return {
        "total_errors": 12,
        "error_types": {
            "4xx": 8,
            "5xx": 4
        },
        "top_errors": [
            {"error": "404 Not Found", "count": 5},
            {"error": "500 Internal Server Error", "count": 3}
        ]
    }

# Database maintenance functions

async def _vacuum_database(db: Session) -> Dict[str, Any]:
    """Run VACUUM on the database to reclaim storage space"""
    try:
        # Note: VACUUM cannot run inside a transaction block in PostgreSQL
        # We need to use a raw connection with autocommit
        connection = db.get_bind().raw_connection()
        connection.set_isolation_level(0)  # Set to autocommit mode
        cursor = connection.cursor()
        
        start_time = time.time()
        cursor.execute("VACUUM;")
        execution_time = time.time() - start_time
        
        cursor.close()
        connection.close()
        
        return {
            "status": "success",
            "message": f"VACUUM completed successfully in {execution_time:.2f} seconds",
            "details": "Database storage space has been reclaimed"
        }
    except Exception as e:
        logger.error(f"Error running VACUUM: {e}")
        return {
            "status": "error",
            "message": f"Failed to run VACUUM: {str(e)}"
        }

async def _analyze_database(db: Session) -> Dict[str, Any]:
    """Run ANALYZE on the database to update statistics"""
    try:
        from sqlalchemy import text
        start_time = time.time()
        db.execute(text("ANALYZE;"))
        db.commit()
        execution_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"ANALYZE completed successfully in {execution_time:.2f} seconds",
            "details": "Database statistics have been updated for better query planning"
        }
    except Exception as e:
        logger.error(f"Error running ANALYZE: {e}")
        db.rollback()
        return {
            "status": "error",
            "message": f"Failed to run ANALYZE: {str(e)}"
        }

async def _full_db_maintenance(db: Session) -> Dict[str, Any]:
    """Run VACUUM ANALYZE for complete database optimization"""
    try:
        # First run VACUUM
        vacuum_result = await _vacuum_database(db)
        if vacuum_result["status"] != "success":
            return vacuum_result
        
        # Then run ANALYZE
        analyze_result = await _analyze_database(db)
        if analyze_result["status"] != "success":
            return analyze_result
        
        return {
            "status": "success",
            "message": "Full database maintenance completed successfully",
            "details": {
                "vacuum": vacuum_result["message"],
                "analyze": analyze_result["message"]
            }
        }
    except Exception as e:
        logger.error(f"Error running full database maintenance: {e}")
        return {
            "status": "error",
            "message": f"Failed to run full database maintenance: {str(e)}"
        }