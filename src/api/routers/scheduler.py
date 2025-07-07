"""Scheduler management router"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Scheduler service client
SCHEDULER_URL = "http://scheduler:8080"  # Will add this endpoint to scheduler


@router.get("/status")
async def get_scheduler_status():
    """Get current scheduler status and job information"""
    try:
        async with httpx.AsyncClient() as client:
            # Try to connect to scheduler health endpoint
            try:
                response = await client.get(f"{SCHEDULER_URL}/health", timeout=5.0)
                if response.status_code == 200:
                    scheduler_data = response.json()
                else:
                    scheduler_data = {"status": "error", "message": "Scheduler not responding"}
            except Exception:
                scheduler_data = {"status": "offline", "message": "Scheduler service unavailable"}
        
        # Get job history from our database (if we track it)
        job_stats = {
            "total_scheduled_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "last_run": None,
            "next_run": scheduler_data.get("next_run")
        }
        
        return {
            "scheduler": scheduler_data,
            "statistics": job_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scheduler status")


@router.post("/refresh")
async def trigger_manual_refresh(source_id: Optional[str] = None):
    """Manually trigger source refresh"""
    try:
        async with httpx.AsyncClient() as client:
            payload = {"source_id": source_id} if source_id else {"all": True}
            
            response = await client.post(
                f"{SCHEDULER_URL}/manual-refresh",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "message": "Refresh triggered successfully",
                    "details": result
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to trigger refresh",
                    "error": response.text
                }
                
    except Exception as e:
        logger.error(f"Error triggering manual refresh: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger refresh")


@router.get("/jobs")
async def get_scheduled_jobs():
    """Get information about all scheduled jobs"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SCHEDULER_URL}/jobs", timeout=10.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "jobs": [],
                    "message": "Could not retrieve job information"
                }
                
    except Exception as e:
        logger.error(f"Error getting scheduled jobs: {e}")
        return {
            "jobs": [],
            "error": str(e)
        }


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a scheduled job"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SCHEDULER_URL}/jobs/{job_id}/pause",
                timeout=10.0
            )
            
            if response.status_code == 200:
                return {"success": True, "message": f"Job {job_id} paused"}
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to pause job: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Error pausing job {job_id}: {e}")
        raise HTTPException(status_code=503, detail="Scheduler service unavailable")


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SCHEDULER_URL}/jobs/{job_id}/resume",
                timeout=10.0
            )
            
            if response.status_code == 200:
                return {"success": True, "message": f"Job {job_id} resumed"}
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to resume job: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Error resuming job {job_id}: {e}")
        raise HTTPException(status_code=503, detail="Scheduler service unavailable")


@router.get("/history")
async def get_refresh_history(limit: int = 50):
    """Get history of scheduled refreshes"""
    try:
        # This would typically query a database table that tracks refresh history
        # For now, return a placeholder structure
        return {
            "history": [
                {
                    "id": "refresh_2025_07_07_02_00",
                    "timestamp": "2025-07-07T02:00:00Z",
                    "type": "scheduled",
                    "sources_processed": 0,
                    "sources_successful": 0,
                    "sources_failed": 0,
                    "duration_seconds": 28,
                    "status": "completed",
                    "error": "Temporary failure in name resolution"
                }
            ],
            "total": 1,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting refresh history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get refresh history")


@router.put("/config")
async def update_scheduler_config(config: Dict[str, Any]):
    """Update scheduler configuration"""
    try:
        # Validate config parameters
        valid_params = {
            "refresh_schedule", "refresh_batch_size", 
            "refresh_delay_seconds", "scheduler_enabled"
        }
        
        invalid_params = set(config.keys()) - valid_params
        if invalid_params:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid configuration parameters: {invalid_params}"
            )
        
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{SCHEDULER_URL}/config",
                json=config,
                timeout=10.0
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Configuration updated",
                    "config": response.json()
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to update config: {response.text}"
                )
                
    except httpx.RequestError as e:
        logger.error(f"Error updating scheduler config: {e}")
        raise HTTPException(status_code=503, detail="Scheduler service unavailable")