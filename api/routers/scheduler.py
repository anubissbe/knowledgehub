"""Scheduler router for managing automatic source refresh schedules"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from ..dependencies import get_source_service, get_job_service
from ..services.scheduler import SchedulerService
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class SchedulerConfig(BaseModel):
    """Scheduler configuration model"""
    enabled: bool = True
    default_interval: int = 86400  # 24 hours in seconds
    check_interval: int = 3600     # 1 hour in seconds


class SchedulerStatus(BaseModel):
    """Scheduler status response"""
    enabled: bool
    last_check: datetime
    next_check: datetime
    sources_due: int
    total_sources: int


@router.get("/config")
async def get_scheduler_config():
    """Get current scheduler configuration"""
    try:
        scheduler_service = SchedulerService()
        config = await scheduler_service.get_config()
        return config
    except Exception as e:
        logger.error(f"Error getting scheduler config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/config")
async def update_scheduler_config(config: SchedulerConfig):
    """Update scheduler configuration"""
    try:
        scheduler_service = SchedulerService()
        await scheduler_service.update_config(config.dict())
        return {"message": "Scheduler configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating scheduler config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status")
async def get_scheduler_status():
    """Get current scheduler status"""
    try:
        scheduler_service = SchedulerService()
        status = await scheduler_service.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/run")
async def run_scheduler():
    """Manually trigger scheduler run"""
    try:
        scheduler_service = SchedulerService()
        result = await scheduler_service.run_scheduler()
        return result
    except Exception as e:
        logger.error(f"Error running scheduler: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sources/due")
async def get_sources_due():
    """Get sources that are due for refresh"""
    try:
        scheduler_service = SchedulerService()
        sources = await scheduler_service.get_sources_due_for_refresh()
        return {
            "sources": sources,
            "total": len(sources)
        }
    except Exception as e:
        logger.error(f"Error getting sources due: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sources/{source_id}/refresh")
async def refresh_source_now(
    source_id: str,
    source_service=Depends(get_source_service),
    job_service=Depends(get_job_service)
):
    """Manually refresh a specific source"""
    try:
        from uuid import UUID
        source_uuid = UUID(source_id)
        
        # Get source
        source = await source_service.get_by_id(source_uuid)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Create refresh job
        job = await job_service.create_scraping_job(
            source_id=source_uuid,
            url=source.url
        )
        
        # Queue the job
        await job_service.queue_scraping_job(job.id, source_uuid, source.url)
        
        return {
            "message": f"Refresh job created for {source.name}",
            "job_id": str(job.id)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid source ID")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")