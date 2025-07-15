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
    """List all jobs with optional filtering - now based on Redis queue state"""
    try:
        import redis.asyncio as redis
        import json
        import os
        from datetime import datetime
        
        # Connect to Redis
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_client = redis.from_url(redis_url, decode_responses=True)
        
        jobs = []
        
        # Get active jobs from Redis locks
        active_keys = await redis_client.keys("source:active:*")
        for key in active_keys:
            source_id_from_key = key.replace("source:active:", "")
            ttl = await redis_client.ttl(key)
            
            # Get source name
            try:
                from sqlalchemy import create_engine, text
                import os
                database_url = os.getenv('DATABASE_URL', 'postgresql://khuser:nq3okTS7f4tyEHz2UZvLfft2C@postgres:5432/knowledgehub')
                engine = create_engine(database_url)
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT name FROM knowledge_sources WHERE id = :id"), {"id": source_id_from_key})
                    row = result.fetchone()
                    source_name = row[0] if row else "Unknown Source"
            except:
                source_name = "Unknown Source"
            
            jobs.append({
                "id": f"active-{source_id_from_key}",
                "source_id": source_id_from_key,
                "source_name": source_name,
                "status": "running",
                "job_type": "crawl",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "ttl_seconds": ttl
            })
        
        # Get queued jobs from Redis queues
        for queue_name in ["crawl_jobs:high", "crawl_jobs:normal", "crawl_jobs:low"]:
            queue_length = await redis_client.llen(queue_name)
            for i in range(min(queue_length, 20)):  # Limit to avoid performance issues
                job_str = await redis_client.lindex(queue_name, i)
                if job_str:
                    try:
                        job_data = json.loads(job_str)
                        job_source_id = job_data.get("source_id")
                        
                        # Get source name
                        try:
                            from sqlalchemy import create_engine, text
                            import os
                            database_url = os.getenv('DATABASE_URL', 'postgresql://khuser:nq3okTS7f4tyEHz2UZvLfft2C@postgres:5432/knowledgehub')
                            engine = create_engine(database_url)
                            with engine.connect() as conn:
                                result = conn.execute(text("SELECT name FROM knowledge_sources WHERE id = :id"), {"id": job_source_id})
                                row = result.fetchone()
                                source_name = row[0] if row else "Unknown Source"
                        except:
                            source_name = "Unknown Source"
                        
                        jobs.append({
                            "id": job_data.get("job_id", f"queued-{i}"),
                            "source_id": job_source_id,
                            "source_name": source_name,
                            "status": "pending",
                            "job_type": job_data.get("job_type", "crawl"),
                            "created_at": datetime.utcnow().isoformat(),
                            "updated_at": datetime.utcnow().isoformat(),
                            "queue": queue_name
                        })
                    except json.JSONDecodeError:
                        continue
        
        await redis_client.close()
        
        # Apply filters
        if source_id:
            jobs = [job for job in jobs if job["source_id"] == source_id]
        if status:
            jobs = [job for job in jobs if job["status"] == status]
        
        # Apply pagination
        total = len(jobs)
        jobs = jobs[skip:skip + limit]
        
        return {
            "jobs": jobs,
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