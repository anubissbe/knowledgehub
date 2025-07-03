"""Job management service - Fixed version"""

from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import json
import logging

from ..models import get_db
from ..models.job import ScrapingJob as Job
from ..schemas.job import JobCreate

logger = logging.getLogger(__name__)


class JobService:
    """Service for managing crawl and processing jobs"""
    
    def __init__(self):
        """Initialize service"""
        self._db = None
    
    @property
    def db(self) -> Session:
        """Get database session"""
        if self._db is None:
            # Get a new session
            self._db = next(get_db())
        return self._db
    
    def __del__(self):
        """Clean up database session"""
        if self._db:
            self._db.close()
    
    async def create_job(
        self,
        source_id: UUID,
        job_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Job:
        """Create a new job"""
        job = Job(
            id=uuid4(),
            source_id=source_id,
            job_type=job_type,
            status="pending",
            config=config or {},
            stats={},
            created_at=datetime.utcnow()
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        
        # Queue job for processing (if message queue available)
        from .message_queue import message_queue
        if message_queue and message_queue.client:
            await message_queue.publish(
                f"{job_type}_jobs",
                json.dumps({
                    "job_id": str(job.id),
                    "source_id": str(source_id),
                    "config": config
                })
            )
        
        return job
    
    async def create_scraping_job(self, source_id: UUID, url: str) -> Job:
        """Create a scraping job for a source"""
        return await self.create_job(
            source_id=source_id,
            job_type="scraping",
            config={"url": url}
        )
    
    async def queue_scraping_job(self, job_id: UUID, source_id: UUID, url: str):
        """Queue a scraping job for processing"""
        # This would normally queue the job to a message queue
        # For now, just log it
        logger.info(f"Queuing scraping job {job_id} for source {source_id} at {url}")
    
    async def list_jobs(
        self,
        skip: int = 0,
        limit: int = 100,
        source_id: Optional[UUID] = None,
        status: Optional[str] = None,
        job_type: Optional[str] = None
    ) -> List[Job]:
        """List jobs with optional filters"""
        query = self.db.query(Job)
        
        if source_id:
            query = query.filter(Job.source_id == source_id)
        if status:
            query = query.filter(Job.status == status)
        if job_type:
            query = query.filter(Job.job_type == job_type)
        
        return query.order_by(desc(Job.created_at)).offset(skip).limit(limit).all()
    
    async def get_job(self, job_id: UUID) -> Optional[Job]:
        """Get a specific job by ID"""
        return self.db.query(Job).filter(Job.id == job_id).first()
    
    async def update_job_status(
        self,
        job_id: UUID,
        status: str,
        stats: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Optional[Job]:
        """Update job status and stats"""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.status = status
        
        if status == "running" and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in ["completed", "failed"]:
            job.completed_at = datetime.utcnow()
        
        if stats:
            job.stats = stats
        
        if error:
            job.error = error
        
        self.db.commit()
        self.db.refresh(job)
        
        return job
    
    def get_active_jobs(self) -> List[Job]:
        """Get all active (running) jobs"""
        return self.db.query(Job).filter(
            Job.status.in_(["pending", "running"])
        ).all()
    
    def get_stuck_jobs(self, timeout_minutes: int = 30) -> List[Job]:
        """Get jobs that have been running for too long"""
        timeout = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        
        return self.db.query(Job).filter(
            Job.status == "running",
            Job.started_at < timeout
        ).all()
    
    async def retry_job(self, job_id: UUID) -> Optional[Job]:
        """Retry a failed job"""
        job = await self.get_job(job_id)
        if not job or job.status != "failed":
            return None
        
        # Create new job with same config
        new_job = await self.create_job(
            source_id=job.source_id,
            job_type=job.job_type,
            config={
                **job.config,
                "retry_of": str(job.id),
                "retry_count": job.config.get("retry_count", 0) + 1
            }
        )
        
        return new_job
    
    async def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a pending or running job"""
        job = await self.get_job(job_id)
        if not job or job.status not in ["pending", "running"]:
            return False
        
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        
        self.db.commit()
        
        return True