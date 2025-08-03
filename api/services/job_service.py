"""Job management service - Fixed version"""

from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import json
import logging
import asyncio

from ..models import get_db
from ..models.job import ScrapingJob as Job, JobType, JobStatus
from ..schemas.job import JobCreate

logger = logging.getLogger(__name__)


class JobService:
    """Service for managing crawl and processing jobs"""
    
    def __init__(self):
        """Initialize service"""
        self._db = None
        from ..config import settings
        self.redis_url = settings.REDIS_URL
    
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
        # Convert string to enum
        try:
            if job_type.lower() == 'scraping':
                job_type_enum = JobType.SCRAPING
            elif job_type.lower() == 'reindexing':
                job_type_enum = JobType.REINDEXING
            elif job_type.lower() == 'deletion':
                job_type_enum = JobType.DELETION
            else:
                job_type_enum = JobType.SCRAPING
        except Exception:
            job_type_enum = JobType.SCRAPING
        
        job = Job(
            id=uuid4(),
            source_id=source_id,
            job_type=job_type_enum,
            status=JobStatus.PENDING,
            config=config or {},
            stats={},
            created_at=datetime.utcnow()
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        
        # Queue job for processing (if message queue available)
        try:
            from .message_queue import message_queue
            if message_queue and getattr(message_queue, 'client', None):
                # Publish to stream for new consumers
                await message_queue.publish(
                    f"{job_type}_jobs",
                    json.dumps({
                        "job_id": str(job.id),
                        "source_id": str(source_id),
                        "config": config
                    })
                )
                
                # Also push to Redis list for compatibility with simple scraper
                if job_type == "scraping":
                    import redis.asyncio as redis
                    redis_client = redis.from_url(
                        message_queue.url,
                        encoding="utf-8",
                        decode_responses=True
                    )
                    
                    # Determine priority based on config
                    priority = (config or {}).get("priority", "normal")
                    queue_name = f"crawl_jobs:{priority}"
                    
                    # Push job data to list
                    config_dict = config or {}
                    job_data = {
                        "job_id": str(job.id),
                        "source_id": str(source_id),
                        "url": config_dict.get("url", ""),
                        "max_depth": config_dict.get("max_depth", 2),
                        "max_pages": config_dict.get("max_pages", 50)
                    }
                    
                    await redis_client.rpush(queue_name, json.dumps(job_data))
                    await redis_client.close()
                    logger.info(f"Pushed scraping job {job.id} to {queue_name}")
                    
        except Exception as e:
            logger.warning(f"Failed to queue job to message queue: {e}")
        
        return job
    
    async def create_scraping_job(self, source_id: UUID, url: str) -> Job:
        """Create a scraping job for a source"""
        return await self.create_job(
            source_id=source_id,
            job_type="scraping",
            config={
                "url": str(url),
                "max_pages": 50,  # Default reasonable limit
                "max_depth": 2
            }
        )
    
    async def create_deletion_job(self, source_id: UUID) -> Job:
        """Create a deletion job for a source"""
        return await self.create_job(
            source_id=source_id,
            job_type="deletion",
            config={"operation": "delete_source"}
        )
    
    async def queue_scraping_job(self, job_id: UUID, source_id: UUID, url: str):
        """Queue a scraping job for processing"""
        import redis.asyncio as redis
        import json
        
        # Connect to Redis
        r = redis.from_url(self.redis_url)
        
        # Create job data
        job_data = {
            "job_id": str(job_id),
            "source_id": str(source_id),
            "url": str(url),  # Add the URL field
            "job_type": "initial_crawl"
        }
        
        # Add to normal priority queue
        await r.rpush("crawl_jobs:normal", json.dumps(job_data))
        
        logger.info(f"Queued scraping job {job_id} for source {source_id} at {str(url)}")
        
        # Close connection
        await r.aclose()
    
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
            # Convert string to enum
            try:
                if status.lower() == 'pending':
                    status_enum = JobStatus.PENDING
                elif status.lower() == 'running':
                    status_enum = JobStatus.RUNNING
                elif status.lower() == 'completed':
                    status_enum = JobStatus.COMPLETED
                elif status.lower() == 'failed':
                    status_enum = JobStatus.FAILED
                elif status.lower() == 'cancelled':
                    status_enum = JobStatus.CANCELLED
                else:
                    status_enum = None
                
                if status_enum:
                    query = query.filter(Job.status == status_enum)
            except Exception:
                pass
                
        if job_type:
            # Convert string to enum
            try:
                if job_type.lower() == 'scraping':
                    job_type_enum = JobType.SCRAPING
                elif job_type.lower() == 'reindexing':
                    job_type_enum = JobType.REINDEXING
                elif job_type.lower() == 'deletion':
                    job_type_enum = JobType.DELETION
                else:
                    job_type_enum = None
                    
                if job_type_enum:
                    query = query.filter(Job.job_type == job_type_enum)
            except Exception:
                pass
        
        return query.order_by(desc(Job.created_at)).offset(skip).limit(limit).all()
    
    async def get_job(self, db: Session, job_id: UUID) -> Optional[Job]:
        """Get a specific job by ID"""
        return db.query(Job).filter(Job.id == job_id).first()
    
    async def update_job_status(
        self,
        job_id: UUID,
        status: str,
        stats: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Optional[Job]:
        """Update job status and stats"""
        job = await self.get_job(self.db, job_id)
        if not job:
            return None
        
        # Convert string to enum
        try:
            if status.lower() == 'pending':
                status_enum = JobStatus.PENDING
            elif status.lower() == 'running':
                status_enum = JobStatus.RUNNING
            elif status.lower() == 'completed':
                status_enum = JobStatus.COMPLETED
            elif status.lower() == 'failed':
                status_enum = JobStatus.FAILED
            elif status.lower() == 'cancelled':
                status_enum = JobStatus.CANCELLED
            else:
                status_enum = JobStatus.PENDING
        except Exception:
            status_enum = JobStatus.PENDING
        
        job.status = status_enum
        
        if status.lower() == "running" and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status.lower() in ["completed", "failed", "cancelled"]:
            job.completed_at = datetime.utcnow()
            
            # Update source status when job completes
            if status.lower() == "completed" and job.source_id:
                from ..models.knowledge_source import KnowledgeSource, SourceStatus
                source = self.db.query(KnowledgeSource).filter(
                    KnowledgeSource.id == job.source_id
                ).first()
                
                if source and source.status == SourceStatus.PENDING:
                    source.status = SourceStatus.COMPLETED
                    source.last_scraped_at = datetime.utcnow()
                    logger.info(f"Updated source {source.id} status to COMPLETED")
        
        if stats:
            job.stats = stats
        
        if error:
            job.error = error
        
        self.db.commit()
        self.db.refresh(job)
        
        # Broadcast WebSocket notifications
        try:
            from ..routers.websocket import broadcast_to_all
            
            # Determine the notification type based on status
            notification_type = None
            if status.lower() == "completed":
                notification_type = "job_completed"
            elif status.lower() == "failed":
                notification_type = "job_failed"
            elif status.lower() == "cancelled":
                notification_type = "job_cancelled"
            
            if notification_type:
                # Send notification to all connected clients
                asyncio.create_task(broadcast_to_all({
                    "type": notification_type,
                    "job_id": str(job_id),
                    "status": status.lower(),
                    "stats": stats
                }))
                
                # Also send stats update notification
                asyncio.create_task(broadcast_to_all({
                    "type": "stats_updated",
                    "source_id": str(job.source_id) if job.source_id else None
                }))
                
        except Exception as e:
            logger.error(f"Failed to broadcast WebSocket notification: {e}")
        
        return job
    
    def get_active_jobs(self) -> List[Job]:
        """Get all active (running) jobs"""
        return self.db.query(Job).filter(
            Job.status.in_([JobStatus.PENDING, JobStatus.RUNNING])
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
        job = await self.get_job(self.db, job_id)
        if not job or job.status != "failed":
            return None
        
        # Create new job with same config
        new_job = await self.create_job(
            source_id=job.source_id,
            job_type=job.job_type,
            config={
                **(job.config or {}),
                "retry_of": str(job.id),
                "retry_count": (job.config or {}).get("retry_count", 0) + 1
            }
        )
        
        return new_job
    
    async def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a pending or running job"""
        
        # First check if this is a Redis-managed job (active or queued)
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Check if there's an active job for this source
            active_key = f"source:active:{job_id}"
            
            if await redis_client.exists(active_key):
                # Delete the active job lock to cancel it
                await redis_client.delete(active_key)
                
                # Send WebSocket notification for job cancellation
                try:
                    from ..routers.websocket import broadcast_to_all
                    
                    # Send job cancelled notification
                    asyncio.create_task(broadcast_to_all({
                        "type": "job_cancelled",
                        "job_id": str(job_id),
                        "status": "cancelled",
                        "source_id": str(job_id)
                    }))
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket notification: {e}")
                
                await redis_client.close()
                return True
            
            # Check if the job is in the Redis stream
            try:
                # Get all entries from the crawl_jobs stream
                stream_entries = await redis_client.xrange("crawl_jobs", "-", "+")
                job_found = False
                entry_id_to_delete = None
                
                for entry_id, fields in stream_entries:
                    if "message" in fields:
                        try:
                            job_data = json.loads(fields["message"])
                            # Check if this is the job we want to cancel
                            if job_data.get("source_id") == str(job_id):
                                job_found = True
                                entry_id_to_delete = entry_id
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                if job_found and entry_id_to_delete:
                    # Remove the specific entry from the stream
                    await redis_client.xdel("crawl_jobs", entry_id_to_delete)
                    logger.info(f"Removed job {job_id} from crawl_jobs stream")
                    
                    # Send WebSocket notification for job cancellation
                    try:
                        from ..routers.websocket import broadcast_to_all
                        
                        # Send job cancelled notification
                        asyncio.create_task(broadcast_to_all({
                            "type": "job_cancelled",
                            "job_id": str(job_id),
                            "status": "cancelled",
                            "source_id": str(job_id)
                        }))
                    except Exception as e:
                        logger.warning(f"Failed to send WebSocket notification: {e}")
                    
                    await redis_client.close()
                    return True
                    
            except Exception as e:
                logger.error(f"Error checking Redis stream: {e}")
            
            # Also check legacy list queues for backwards compatibility
            queues = ["crawl_jobs:high", "crawl_jobs:normal", "crawl_jobs:low"]
            for queue_name in queues:
                try:
                    queue_length = await redis_client.llen(queue_name)
                    jobs_to_keep = []
                    found_job = False
                    
                    # Get all jobs from the queue
                    for i in range(queue_length):
                        job_str = await redis_client.lindex(queue_name, i)
                        if job_str:
                            try:
                                job_data = json.loads(job_str)
                                if job_data.get("job_id") == str(job_id) or job_data.get("source_id") == str(job_id):
                                    found_job = True
                                    # Don't add this job to jobs_to_keep (effectively removing it)
                                    continue
                            except (json.JSONDecodeError, KeyError):
                                pass
                            jobs_to_keep.append(job_str)
                    
                    if found_job:
                        # Clear the queue and re-add jobs except the cancelled one
                        await redis_client.delete(queue_name)
                        if jobs_to_keep:
                            await redis_client.lpush(queue_name, *jobs_to_keep)
                        
                        # Send WebSocket notification for job cancellation
                        try:
                            from ..routers.websocket import broadcast_to_all
                            
                            # Send job cancelled notification
                            asyncio.create_task(broadcast_to_all({
                                "type": "job_cancelled",
                                "job_id": str(job_id),
                                "status": "cancelled",
                                "source_id": str(job_id)
                            }))
                        except Exception as e:
                            logger.warning(f"Failed to send WebSocket notification: {e}")
                        
                        await redis_client.close()
                        return True
                except Exception as e:
                    logger.error(f"Error checking queue {queue_name}: {e}")
                    continue
            
            await redis_client.close()
        except Exception as e:
            logger.error(f"Error checking Redis for job: {e}")
        
        # Fall back to database job lookup
        job = await self.get_job(self.db, job_id)
        if not job or job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
            return False
        
        # Store original status before updating
        original_status = job.status
        
        # Update job status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        if job.error is None:
            job.error = "Job cancelled by user"
        
        self.db.commit()
        self.db.refresh(job)
        
        # Send WebSocket notification for job cancellation
        try:
            from ..routers.websocket import broadcast_to_all
            
            # Send job cancelled notification
            asyncio.create_task(broadcast_to_all({
                "type": "job_cancelled",
                "job_id": str(job_id),
                "status": "cancelled",
                "source_id": str(job.source_id) if job.source_id else None
            }))
            
            # Also send stats update notification
            asyncio.create_task(broadcast_to_all({
                "type": "stats_updated",
                "source_id": str(job.source_id) if job.source_id else None
            }))
            
            logger.info(f"Sent WebSocket notification for cancelled job {job_id}")
        except Exception as e:
            logger.error(f"Failed to send WebSocket notification: {e}")
        
        # Remove from Redis streams and queues if was pending
        if original_status == JobStatus.PENDING:
            try:
                import redis.asyncio as redis
                redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                
                # Remove from stream
                try:
                    stream_entries = await redis_client.xrange("crawl_jobs", "-", "+")
                    for entry_id, fields in stream_entries:
                        if "message" in fields:
                            try:
                                job_data = json.loads(fields["message"])
                                if job_data.get("source_id") == str(job.source_id):
                                    await redis_client.xdel("crawl_jobs", entry_id)
                                    logger.info(f"Removed job {job_id} from crawl_jobs stream")
                                    break
                            except (json.JSONDecodeError, KeyError):
                                continue
                except Exception as e:
                    logger.error(f"Error removing job from stream: {e}")
                
                # Remove from all priority queues (legacy support)
                job_data = json.dumps({
                    "job_id": str(job_id),
                    "source_id": str(job.source_id),
                    "config": job.config or {}
                })
                
                for priority in ["high", "normal", "low"]:
                    queue_name = f"crawl_jobs:{priority}"
                    await redis_client.lrem(queue_name, 0, job_data)
                
                await redis_client.aclose()
                logger.info(f"Removed job {job_id} from Redis queues")
            except Exception as e:
                logger.error(f"Error removing job from Redis: {e}")
        
        # Add cancellation flag in Redis for running jobs
        if original_status == JobStatus.RUNNING:
            try:
                import redis.asyncio as redis
                redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                
                # Set a cancellation flag that the scraper can check
                await redis_client.setex(
                    f"job:cancelled:{job_id}",
                    300,  # Expire after 5 minutes
                    "1"
                )
                
                await redis_client.aclose()
                logger.info(f"Set cancellation flag for running job {job_id}")
            except Exception as e:
                logger.error(f"Error setting cancellation flag: {e}")
        
        return True