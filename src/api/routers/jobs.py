"""Jobs router"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional, Dict, Any
import logging
from pydantic import BaseModel

from ..dependencies import get_job_service, get_db

logger = logging.getLogger(__name__)

router = APIRouter()


class JobStatusUpdate(BaseModel):
    """Model for job status updates"""
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


@router.get("/")
async def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    source_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    job_service=Depends(get_job_service)
):
    """List all jobs with optional filtering"""
    try:
        # Convert source_id to UUID if provided
        source_uuid = None
        if source_id:
            try:
                source_uuid = UUID(source_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid source_id format")
        
        jobs = await job_service.list_jobs(
            skip=skip,
            limit=limit,
            source_id=source_uuid,
            status=status
        )
        
        total = len(jobs)  # For now, just return the count of returned jobs
        
        return {
            "jobs": [job.to_dict() if hasattr(job, 'to_dict') else job.__dict__ for job in jobs],
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{job_id}")
async def get_job_status(
    job_id: UUID,
    db: Session = Depends(get_db),
    job_service=Depends(get_job_service)
):
    """Get the current status of a background job"""
    try:
        job = await job_service.get_job(db, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: UUID,
    job_service=Depends(get_job_service)
):
    """Cancel a pending or running job"""
    try:
        success = await job_service.cancel_job(job_id)
        if not success:
            raise HTTPException(
                status_code=404, 
                detail="Job not found or cannot be cancelled"
            )
        
        return {
            "message": "Job cancelled successfully",
            "job_id": str(job_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: UUID,
    job_service=Depends(get_job_service)
):
    """Retry a failed job"""
    try:
        new_job = await job_service.retry_job(job_id)
        if not new_job:
            raise HTTPException(
                status_code=404, 
                detail="Job not found or cannot be retried"
            )
        
        return {
            "message": "Job retry initiated successfully",
            "original_job_id": str(job_id),
            "new_job_id": str(new_job.id),
            "new_job": new_job.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/{job_id}")
async def update_job_status(
    job_id: UUID,
    update: JobStatusUpdate,
    job_service=Depends(get_job_service)
):
    """Update job status (used by workers)"""
    try:
        job = await job_service.update_job_status(
            job_id=job_id,
            status=update.status,
            stats=update.stats,
            error=update.error
        )
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "message": "Job status updated successfully",
            "job_id": str(job_id),
            "status": update.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job {job_id} status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")