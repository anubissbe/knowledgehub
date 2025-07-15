"""Scheduler service for managing automatic source refresh schedules"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from ..config import settings
from ..models import get_db
from ..models.knowledge_source import KnowledgeSource, SourceStatus
from ..services.cache import redis_client
from ..services.job_service import JobService
from ..services.source_service import SourceService

logger = logging.getLogger(__name__)


class SchedulerService:
    """Service for managing automatic source refresh schedules"""
    
    def __init__(self):
        self.redis_key = "scheduler:config"
        self.status_key = "scheduler:status"
        self.job_service = JobService()
        self.source_service = SourceService()
        
    async def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        try:
            if redis_client and redis_client.client:
                config_str = await redis_client.get(self.redis_key)
                if config_str:
                    return json.loads(config_str)
        except Exception as e:
            logger.warning(f"Failed to get scheduler config from Redis: {e}")
        
        # Return default config
        return {
            "enabled": True,
            "default_interval": 86400,  # 24 hours
            "check_interval": 3600,     # 1 hour
        }
    
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update scheduler configuration"""
        try:
            if redis_client and redis_client.client:
                await redis_client.set(self.redis_key, json.dumps(config))
                logger.info(f"Updated scheduler config: {config}")
        except Exception as e:
            logger.error(f"Failed to update scheduler config: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        try:
            if redis_client and redis_client.client:
                status_str = await redis_client.get(self.status_key)
                if status_str:
                    status = json.loads(status_str)
                    
                    # Add current stats
                    sources_due = await self.get_sources_due_for_refresh()
                    total_sources = await self.source_service.count_sources()
                    
                    status.update({
                        "sources_due": len(sources_due),
                        "total_sources": total_sources,
                    })
                    
                    return status
        except Exception as e:
            logger.warning(f"Failed to get scheduler status from Redis: {e}")
        
        # Return default status
        config = await self.get_config()
        now = datetime.utcnow()
        
        return {
            "enabled": config.get("enabled", True),
            "last_check": now.isoformat(),
            "next_check": (now + timedelta(seconds=config.get("check_interval", 3600))).isoformat(),
            "sources_due": 0,
            "total_sources": 0,
        }
    
    async def update_status(self, status: Dict[str, Any]) -> None:
        """Update scheduler status"""
        try:
            if redis_client and redis_client.client:
                await redis_client.set(self.status_key, json.dumps(status))
        except Exception as e:
            logger.error(f"Failed to update scheduler status: {e}")
    
    async def get_sources_due_for_refresh(self) -> List[Dict[str, Any]]:
        """Get sources that are due for refresh based on their refresh intervals"""
        try:
            db = next(get_db())
            
            # Get all sources that have refresh intervals set
            sources = db.query(KnowledgeSource).filter(
                KnowledgeSource.config.op('->>')('refresh_interval').astext.cast(db.Integer) > 0
            ).all()
            
            due_sources = []
            now = datetime.utcnow()
            
            for source in sources:
                config = source.config or {}
                refresh_interval = config.get('refresh_interval', 86400)
                
                # Skip if refresh interval is 0 (never refresh)
                if refresh_interval <= 0:
                    continue
                
                # Check if source is due for refresh
                if source.last_scraped_at is None:
                    # Never scraped, due for refresh
                    due_sources.append({
                        "id": str(source.id),
                        "name": source.name,
                        "url": source.url,
                        "last_scraped_at": None,
                        "refresh_interval": refresh_interval,
                        "overdue_by": 0
                    })
                else:
                    next_refresh = source.last_scraped_at + timedelta(seconds=refresh_interval)
                    if now >= next_refresh:
                        overdue_by = int((now - next_refresh).total_seconds())
                        due_sources.append({
                            "id": str(source.id),
                            "name": source.name,
                            "url": source.url,
                            "last_scraped_at": source.last_scraped_at.isoformat(),
                            "refresh_interval": refresh_interval,
                            "overdue_by": overdue_by
                        })
            
            db.close()
            return due_sources
            
        except Exception as e:
            logger.error(f"Error getting sources due for refresh: {e}")
            return []
    
    async def run_scheduler(self) -> Dict[str, Any]:
        """Run the scheduler to refresh sources that are due"""
        try:
            config = await self.get_config()
            
            if not config.get("enabled", True):
                return {
                    "message": "Scheduler is disabled",
                    "jobs_created": 0,
                    "sources_processed": 0
                }
            
            # Get sources due for refresh
            due_sources = await self.get_sources_due_for_refresh()
            
            jobs_created = 0
            sources_processed = 0
            errors = []
            
            for source_info in due_sources:
                try:
                    source_id = UUID(source_info["id"])
                    
                    # Check if source is already being processed
                    if redis_client and redis_client.client:
                        active_key = f"source:active:{source_id}"
                        if await redis_client.exists(active_key):
                            logger.info(f"Source {source_info['name']} is already being processed, skipping")
                            continue
                    
                    # Create scraping job
                    job = await self.job_service.create_scraping_job(
                        source_id=source_id,
                        url=source_info["url"]
                    )
                    
                    # Queue the job
                    await self.job_service.queue_scraping_job(
                        job.id, 
                        source_id, 
                        source_info["url"]
                    )
                    
                    jobs_created += 1
                    sources_processed += 1
                    
                    logger.info(f"Created refresh job for source {source_info['name']} (job: {job.id})")
                    
                except Exception as e:
                    error_msg = f"Failed to create job for source {source_info.get('name', 'Unknown')}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # Update scheduler status
            now = datetime.utcnow()
            check_interval = config.get("check_interval", 3600)
            
            status = {
                "enabled": True,
                "last_check": now.isoformat(),
                "next_check": (now + timedelta(seconds=check_interval)).isoformat(),
                "last_run_jobs": jobs_created,
                "last_run_sources": sources_processed,
                "last_run_errors": len(errors)
            }
            
            await self.update_status(status)
            
            result = {
                "message": f"Scheduler run completed",
                "jobs_created": jobs_created,
                "sources_processed": sources_processed,
                "sources_due": len(due_sources),
                "errors": errors
            }
            
            logger.info(f"Scheduler run completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error running scheduler: {e}")
            raise
    
    async def start_background_scheduler(self) -> None:
        """Start the background scheduler task"""
        logger.info("Starting background scheduler...")
        
        async def scheduler_loop():
            while True:
                try:
                    config = await self.get_config()
                    
                    if config.get("enabled", True):
                        await self.run_scheduler()
                    
                    check_interval = config.get("check_interval", 3600)
                    await asyncio.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    # Sleep for a shorter time on error to retry sooner
                    await asyncio.sleep(300)  # 5 minutes
        
        # Run scheduler in background
        asyncio.create_task(scheduler_loop())
        logger.info("Background scheduler started")
    
    async def stop_scheduler(self) -> None:
        """Stop the scheduler"""
        try:
            config = await self.get_config()
            config["enabled"] = False
            await self.update_config(config)
            
            logger.info("Scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            raise