"""Scraper worker main entry point"""

import asyncio
import os
import json
import signal
import sys
from typing import Optional
from datetime import datetime

try:
    from .incremental_crawler import IncrementalWebCrawler as WebCrawler
except ImportError:
    try:
        from .crawler import WebCrawler
    except ImportError:
        # Fallback to simple crawler if Playwright crawler fails
        from .simple_crawler import SimpleCrawler as WebCrawler
from .parsers import ContentParserFactory
from ..shared.config import Config
from ..shared.logging import setup_logging

# Setup logging
logger = setup_logging("scraper")


class ScraperWorker:
    """Main scraper worker that processes crawl jobs"""
    
    def __init__(self):
        self.config = Config()
        # Initialize crawler with API info if it's the incremental crawler
        try:
            api_key = os.getenv("API_KEY", "dev-api-key-123")
            self.crawler = WebCrawler(api_url=self.config.API_URL, api_key=api_key)
        except TypeError:
            # Regular crawler doesn't accept these params
            self.crawler = WebCrawler()
        self.parser_factory = ContentParserFactory()
        self.running = True
        
        # Redis connection for job queue
        import redis.asyncio as redis
        self.redis = redis.from_url(
            self.config.REDIS_URL,
            decode_responses=True
        )
        
        # API client for updating job status
        import httpx
        
        # Get API key from environment or use dev key
        api_key = os.getenv("API_KEY", "dev-api-key-123")
        
        self.api_client = httpx.AsyncClient(
            base_url=self.config.API_URL,
            timeout=30.0,
            headers={
                "X-API-Key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
    
    async def start(self):
        """Start the scraper worker"""
        logger.info("Starting scraper worker...")
        logger.info(f"API URL: {self.config.API_URL}")
        logger.info(f"Redis URL: {self.config.REDIS_URL}")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start crawler
        await self.crawler.start()
        
        # Main processing loop
        while self.running:
            try:
                # Get job from queue (blocking with timeout)
                job_data = await self._get_next_job()
                
                if job_data:
                    await self._process_job(job_data)
                else:
                    # No job available, wait a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        # Cleanup
        await self.cleanup()
    
    async def _get_next_job(self) -> Optional[dict]:
        """Get next job from Redis queue"""
        try:
            # Use BLPOP for blocking pop with timeout
            result = await self.redis.blpop(
                ["crawl_jobs:high", "crawl_jobs:normal", "crawl_jobs:low"],
                timeout=5
            )
            
            if result:
                queue, job_str = result
                return json.loads(job_str)
                
        except Exception as e:
            logger.error(f"Error getting job from queue: {e}")
        
        return None
    
    async def _process_job(self, job_data: dict):
        """Process a single crawl job"""
        job_id = job_data.get("job_id")
        source_id = job_data.get("source_id")
        
        logger.info(f"Processing job {job_id} for source {source_id}")
        
        try:
            # Check if job was cancelled before starting
            if await self._is_job_cancelled(job_id):
                logger.info(f"Job {job_id} was cancelled, skipping")
                return
            
            # Update job status to running
            await self._update_job_status(job_id, "running")
            
            # Get source details from API
            source = await self._get_source(source_id)
            if not source:
                raise ValueError(f"Source {source_id} not found")
            
            # Determine job type and process accordingly
            job_type = job_data.get("job_type", "initial_crawl")
            
            if job_type in ["initial_crawl", "refresh"]:
                result = await self._crawl_source(source, job_data)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            
            # Update job status based on result
            if result.get("cancelled"):
                await self._update_job_status(
                    job_id,
                    "cancelled",
                    result=result,
                    error="Job cancelled by user"
                )
            else:
                await self._update_job_status(
                    job_id,
                    "completed",
                    result=result
                )
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            await self._update_job_status(
                job_id,
                "failed",
                error=str(e)
            )
    
    async def _crawl_source(self, source: dict, job_data: dict) -> dict:
        """Crawl a source and extract content"""
        url = source["url"]
        job_id = job_data.get("job_id")
        # Get config from either crawl_config or config field
        crawl_config = source.get("crawl_config") or source.get("config", {})
        
        # Initialize crawl results
        results = {
            "pages_crawled": 0,
            "chunks_created": 0,
            "errors": [],
            "start_time": datetime.utcnow().isoformat(),
            "pages": [],
            "cancelled": False
        }
        
        # Prepare crawl parameters
        crawl_params = {
            "max_depth": crawl_config.get("max_depth", 2),
            "max_pages": crawl_config.get("max_pages", 100),
            "follow_patterns": crawl_config.get("follow_patterns", []),
            "exclude_patterns": crawl_config.get("exclude_patterns", []),
            "crawl_delay": crawl_config.get("crawl_delay", 1)
        }
        
        # Add source_id for incremental crawling if available
        if hasattr(self.crawler, 'crawl') and 'source_id' in self.crawler.crawl.__code__.co_varnames:
            crawl_params["source_id"] = source.get("id")
            crawl_params["force_refresh"] = crawl_config.get("force_refresh", False)
        
        # Start crawling
        async for page_data in self.crawler.crawl(url, **crawl_params):
            # Check for cancellation periodically
            if await self._is_job_cancelled(job_id):
                logger.info(f"Job {job_id} cancelled during crawling")
                results["cancelled"] = True
                results["errors"].append({"error": "Job cancelled by user"})
                break
            
            try:
                # Parse content based on type
                parser = self.parser_factory.get_parser(page_data["content_type"])
                chunks = await parser.parse(
                    content=page_data["content"],
                    url=page_data["url"],
                    metadata=page_data.get("metadata", {})
                )
                
                # Send chunks to RAG processor
                for chunk in chunks:
                    # Add content hash and other metadata from page
                    chunk_with_metadata = {
                        **chunk,
                        "content_hash": page_data.get("content_hash"),
                        "is_update": page_data.get("is_update", False)
                    }
                    await self._queue_chunk_for_processing({
                        "source_id": source["id"],
                        "job_id": job_data["job_id"],
                        "chunk": chunk_with_metadata
                    })
                
                results["pages_crawled"] += 1
                results["chunks_created"] += len(chunks)
                results["pages"].append({
                    "url": page_data["url"],
                    "title": page_data.get("title"),
                    "chunks": len(chunks)
                })
                
            except Exception as e:
                logger.error(f"Error processing page {page_data['url']}: {e}")
                results["errors"].append({
                    "url": page_data["url"],
                    "error": str(e)
                })
        
        results["end_time"] = datetime.utcnow().isoformat()
        return results
    
    async def _queue_chunk_for_processing(self, chunk_data: dict):
        """Queue chunk for RAG processing"""
        await self.redis.rpush(
            "rag_processing:normal",
            json.dumps(chunk_data)
        )
    
    async def _get_source(self, source_id: str) -> Optional[dict]:
        """Get source details from API"""
        try:
            response = await self.api_client.get(f"/api/v1/sources/{source_id}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching source {source_id}: {e}")
        
        return None
    
    async def _update_job_status(
        self,
        job_id: str,
        status: str,
        result: Optional[dict] = None,
        error: Optional[str] = None
    ):
        """Update job status via API"""
        logger.info(f"Updating job {job_id} status to: {status}")
        try:
            data = {"status": status}
            if result:
                data["result"] = result
            if error:
                data["error"] = error
            
            response = await self.api_client.patch(
                f"/api/v1/jobs/{job_id}",
                json=data
            )
            logger.info(f"Job status update response: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Job status update failed: {response.text}")
        except Exception as e:
            logger.error(f"Error updating job {job_id} status: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def _is_job_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled"""
        try:
            result = await self.redis.get(f"job:cancelled:{job_id}")
            return result == "1"
        except Exception as e:
            logger.error(f"Error checking job cancellation: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up scraper worker...")
        
        try:
            await self.crawler.stop()
            await self.redis.close()
            await self.api_client.aclose()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main entry point"""
    worker = ScraperWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())