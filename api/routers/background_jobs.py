"""
API Router for Background Jobs Management
Provides endpoints to monitor and control background jobs
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging

from ..services.background_jobs import get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["background-jobs"])


@router.get("/status")
async def get_jobs_status() -> Dict[str, Any]:
    """Get status of all background jobs"""
    try:
        job_manager = await get_job_manager()
        return job_manager.get_job_status()
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/{job_id}")
async def trigger_job(job_id: str) -> Dict[str, Any]:
    """Manually trigger a specific job"""
    try:
        job_manager = await get_job_manager()
        
        # Get the job
        job = job_manager.scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Run the job
        job.func()
        
        return {
            "success": True,
            "message": f"Job {job_id} triggered successfully",
            "job_name": job.name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pause/{job_id}")
async def pause_job(job_id: str) -> Dict[str, Any]:
    """Pause a specific job"""
    try:
        job_manager = await get_job_manager()
        job_manager.scheduler.pause_job(job_id)
        
        return {
            "success": True,
            "message": f"Job {job_id} paused"
        }
    except Exception as e:
        logger.error(f"Error pausing job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume/{job_id}")
async def resume_job(job_id: str) -> Dict[str, Any]:
    """Resume a paused job"""
    try:
        job_manager = await get_job_manager()
        job_manager.scheduler.resume_job(job_id)
        
        return {
            "success": True,
            "message": f"Job {job_id} resumed"
        }
    except Exception as e:
        logger.error(f"Error resuming job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def jobs_health_check() -> Dict[str, Any]:
    """Check if background jobs system is healthy"""
    try:
        job_manager = await get_job_manager()
        status = job_manager.get_job_status()
        
        return {
            "status": "healthy" if status['scheduler_running'] else "unhealthy",
            "service": "background-jobs",
            "scheduler_running": status['scheduler_running'],
            "total_jobs": status['total_jobs']
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "background-jobs",
            "error": str(e)
        }