"""Scheduler service for automated source updates"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:3000")
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
REFRESH_SCHEDULE = os.getenv("REFRESH_SCHEDULE", "0 2 * * 0")  # Default: Sunday 2 AM
REFRESH_BATCH_SIZE = int(os.getenv("REFRESH_BATCH_SIZE", "2"))  # How many sources to refresh concurrently
REFRESH_DELAY_SECONDS = int(os.getenv("REFRESH_DELAY_SECONDS", "300"))  # 5 minutes between batches


class SourceScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.api_client = httpx.AsyncClient(
            base_url=API_BASE_URL,
            timeout=30.0
        )
        
    async def get_sources(self) -> List[Dict[str, Any]]:
        """Fetch all sources from the API"""
        try:
            response = await self.api_client.get("/api/v1/sources/")
            response.raise_for_status()
            data = response.json()
            return data.get("sources", [])
        except Exception as e:
            logger.error(f"Error fetching sources: {e}")
            return []
    
    async def refresh_source(self, source: Dict[str, Any]) -> bool:
        """Trigger a refresh for a single source"""
        source_id = source["id"]
        source_name = source["name"]
        
        try:
            logger.info(f"Refreshing source: {source_name} (ID: {source_id})")
            
            # Check if source needs refresh (based on last_scraped_at)
            last_scraped = source.get("last_scraped_at")
            if last_scraped:
                last_scraped_dt = datetime.fromisoformat(last_scraped.replace("Z", "+00:00"))
                days_since_update = (datetime.utcnow() - last_scraped_dt.replace(tzinfo=None)).days
                logger.info(f"Source {source_name} last updated {days_since_update} days ago")
            
            # Trigger refresh
            response = await self.api_client.post(f"/api/v1/sources/{source_id}/refresh")
            response.raise_for_status()
            
            job_data = response.json()
            logger.info(f"Refresh job created for {source_name}: {job_data}")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing source {source_name}: {e}")
            return False
    
    async def refresh_all_sources(self):
        """Refresh all sources with batching to avoid overload"""
        logger.info("Starting scheduled refresh of all sources")
        
        try:
            sources = await self.get_sources()
            if not sources:
                logger.warning("No sources found to refresh")
                return
            
            logger.info(f"Found {len(sources)} sources to refresh")
            
            # Filter sources by status and refresh settings
            active_sources = [
                s for s in sources 
                if s.get("status") == "completed" and 
                s.get("config", {}).get("refresh_interval", 0) > 0
            ]
            
            logger.info(f"Refreshing {len(active_sources)} active sources")
            
            # Process in batches
            for i in range(0, len(active_sources), REFRESH_BATCH_SIZE):
                batch = active_sources[i:i + REFRESH_BATCH_SIZE]
                logger.info(f"Processing batch {i//REFRESH_BATCH_SIZE + 1} of {(len(active_sources) + REFRESH_BATCH_SIZE - 1)//REFRESH_BATCH_SIZE}")
                
                # Refresh sources in parallel within batch
                tasks = [self.refresh_source(source) for source in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_count = sum(1 for r in results if r is True)
                logger.info(f"Batch complete: {success_count}/{len(batch)} sources refreshed successfully")
                
                # Wait between batches to avoid overwhelming the system
                if i + REFRESH_BATCH_SIZE < len(active_sources):
                    logger.info(f"Waiting {REFRESH_DELAY_SECONDS} seconds before next batch...")
                    await asyncio.sleep(REFRESH_DELAY_SECONDS)
            
            logger.info("Scheduled refresh completed")
            
        except Exception as e:
            logger.error(f"Error during scheduled refresh: {e}")
    
    async def manual_refresh(self, source_id: str = None):
        """Manually trigger refresh for a specific source or all sources"""
        if source_id:
            sources = await self.get_sources()
            source = next((s for s in sources if s["id"] == source_id), None)
            if source:
                success = await self.refresh_source(source)
                return {"success": success, "source": source["name"]}
            else:
                return {"success": False, "error": "Source not found"}
        else:
            await self.refresh_all_sources()
            return {"success": True, "message": "All sources refresh initiated"}
    
    def start(self):
        """Start the scheduler"""
        if not SCHEDULER_ENABLED:
            logger.info("Scheduler is disabled via SCHEDULER_ENABLED=false")
            return
        
        # Add the weekly refresh job
        self.scheduler.add_job(
            self.refresh_all_sources,
            CronTrigger.from_crontab(REFRESH_SCHEDULE),
            id="weekly_refresh",
            name="Weekly source refresh",
            replace_existing=True
        )
        
        logger.info(f"Scheduled weekly refresh with cron: {REFRESH_SCHEDULE}")
        
        # Log next run time
        job = self.scheduler.get_job("weekly_refresh")
        if job and hasattr(job, 'next_run_time'):
            logger.info(f"Next refresh scheduled for: {job.next_run_time}")
        
        self.scheduler.start()
        logger.info("Scheduler started successfully")
    
    async def stop(self):
        """Stop the scheduler and cleanup"""
        self.scheduler.shutdown()
        await self.api_client.aclose()
        logger.info("Scheduler stopped")


# Health check endpoint (optional)
async def health_check():
    """Simple health check for monitoring"""
    return {
        "status": "healthy",
        "scheduler_enabled": SCHEDULER_ENABLED,
        "refresh_schedule": REFRESH_SCHEDULE,
        "timestamp": datetime.utcnow().isoformat()
    }


async def main():
    """Main entry point"""
    scheduler = SourceScheduler()
    
    # Start scheduler
    scheduler.start()
    
    # Optional: Run an immediate refresh on startup (useful for testing)
    if os.getenv("REFRESH_ON_STARTUP", "false").lower() == "true":
        logger.info("Running immediate refresh on startup...")
        await scheduler.refresh_all_sources()
    
    try:
        # Keep the service running
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Optional: Log scheduler status
            if scheduler.scheduler.running:
                jobs = scheduler.scheduler.get_jobs()
                for job in jobs:
                    logger.debug(f"Job {job.id}: Next run at {job.next_run_time}")
                    
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())