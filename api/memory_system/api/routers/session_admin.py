"""Session administration endpoints"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ....models import get_db
from ...services.session_cleanup import session_cleanup_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/cleanup/manual")
async def manual_session_cleanup():
    """
    Manually trigger a session cleanup cycle.
    
    This endpoint allows administrators to immediately run session cleanup
    instead of waiting for the scheduled background task.
    """
    try:
        result = await session_cleanup_service.manual_cleanup()
        
        if result['success']:
            logger.info("Manual session cleanup completed successfully")
            return {
                "message": "Session cleanup completed successfully",
                "result": result
            }
        else:
            logger.error(f"Manual session cleanup failed: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Cleanup failed: {result.get('error')}"
            )
            
    except Exception as e:
        logger.error(f"Manual cleanup endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to run manual cleanup"
        )


@router.get("/cleanup/stats")
async def get_cleanup_stats():
    """
    Get session cleanup statistics.
    
    Returns information about the last cleanup cycle and overall statistics.
    """
    try:
        stats = await session_cleanup_service.get_cleanup_stats()
        
        return {
            "cleanup_stats": stats,
            "service_status": {
                "running": session_cleanup_service._running,
                "cleanup_interval_hours": session_cleanup_service.cleanup_interval_hours,
                "stale_session_hours": session_cleanup_service.stale_session_hours,
                "old_session_days": session_cleanup_service.old_session_days
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get cleanup stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve cleanup statistics"
        )


@router.get("/cleanup/health")
async def cleanup_service_health():
    """Health check for session cleanup service"""
    try:
        is_running = session_cleanup_service._running
        stats = await session_cleanup_service.get_cleanup_stats()
        
        return {
            "status": "healthy" if is_running else "stopped",
            "service_running": is_running,
            "last_cleanup": stats.get('last_cleanup'),
            "total_cleanups": stats.get('total_cleanups', 0)
        }
        
    except Exception as e:
        logger.error(f"Cleanup health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/cleanup/restart")
async def restart_cleanup_service():
    """
    Restart the session cleanup background service.
    
    This endpoint allows administrators to restart the cleanup service
    if it has stopped or is having issues.
    """
    try:
        # Stop the service if running
        if session_cleanup_service._running:
            await session_cleanup_service.stop()
        
        # Start the service
        await session_cleanup_service.start()
        
        logger.info("Session cleanup service restarted successfully")
        return {
            "message": "Session cleanup service restarted successfully",
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Failed to restart cleanup service: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to restart cleanup service"
        )