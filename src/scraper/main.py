"""Scraper worker main entry point"""

import asyncio
import os
import json
import signal
import sys
from typing import Optional
from datetime import datetime

# Use the regular crawler for now
from .crawler import WebCrawler
from .parsers import ContentParserFactory
from ..shared.config import Config
from ..shared.logging import setup_logging
from aiohttp import web

# Setup logging
logger = setup_logging("scraper")


class ScraperWorker:
    """Main scraper worker that processes crawl jobs"""
    
    def __init__(self):
        self.config = Config()
        # Initialize crawler
        self.crawler = None
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
        
        # HTTP server for health checks
        self.app = web.Application()
        self.setup_routes()
        self.http_server = None
        self.health_check_port = int(os.environ.get('HEALTH_CHECK_PORT', '3014'))
    
    def setup_routes(self):
        """Setup HTTP routes for health checks"""
        self.app.router.add_get('/health', self.health_check)
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        try:
            # Check all dependencies
            dependencies = {}
            
            # Check Redis
            if self.redis:
                try:
                    await self.redis.ping()
                    dependencies['redis'] = 'healthy'
                except Exception as e:
                    dependencies['redis'] = f'unhealthy: {str(e)}'
            else:
                dependencies['redis'] = 'not_initialized'
            
            # Check API client
            if self.api_client:
                try:
                    # Just check if client is created, don't make actual request
                    dependencies['api'] = 'healthy'
                except Exception as e:
                    dependencies['api'] = f'unhealthy: {str(e)}'
            else:
                dependencies['api'] = 'not_initialized'
            
            # Check crawler
            if self.crawler:
                dependencies['crawler'] = 'healthy'
            else:
                dependencies['crawler'] = 'not_initialized'
            
            # Check processing status
            processing_status = {
                'running': self.running
            }
            
            # Determine overall status
            overall_status = 'healthy'
            if any('unhealthy' in str(v) for v in dependencies.values()):
                overall_status = 'unhealthy'
            elif any(v == 'not_initialized' for v in dependencies.values()):
                overall_status = 'degraded'
            
            return web.json_response({
                'status': overall_status,
                'service': 'scraper',
                'dependencies': dependencies,
                'processing': processing_status,
                'timestamp': datetime.utcnow().isoformat()
            }, status=200 if overall_status != 'unhealthy' else 503)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return web.json_response({
                'status': 'unhealthy',
                'service': 'scraper',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=503)
    
    async def start(self):
        """Start the scraper worker"""
        logger.info("Starting scraper worker...")
        logger.info(f"API URL: {self.config.API_URL}")
        logger.info(f"Redis URL: {self.config.REDIS_URL}")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # For now, use the regular crawler which is more stable
        self.crawler = WebCrawler()
        logger.info("Using regular WebCrawler")
        # Start regular crawler
        await self.crawler.start()
        
        # Start HTTP server for health checks
        logger.info(f"Starting health check server on port {self.health_check_port}")
        runner = web.AppRunner(self.app)
        await runner.setup()
        self.http_server = web.TCPSite(runner, '0.0.0.0', self.health_check_port)
        await self.http_server.start()
        logger.info(f"Health check server started on port {self.health_check_port}")
        
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
        """Get next job from Redis queue, avoiding sources already being processed"""
        try:
            # Check for jobs avoiding source conflicts
            for queue_name in ["crawl_jobs:high", "crawl_jobs:normal", "crawl_jobs:low"]:
                queue_length = await self.redis.llen(queue_name)
                if queue_length > 0:
                    # Check each job in queue to find one with source not being processed
                    for i in range(queue_length):
                        job_str = await self.redis.lindex(queue_name, i)
                        if job_str:
                            job_data = json.loads(job_str)
                            source_id = job_data.get("source_id")
                            
                            # Check if this source is already being processed
                            active_key = f"source:active:{source_id}"
                            is_active = await self.redis.get(active_key)
                            
                            if not is_active:
                                # Remove this specific job from queue and return it
                                await self.redis.lrem(queue_name, 1, job_str)
                                # Mark source as active for 1 hour
                                await self.redis.setex(active_key, 3600, "1")
                                logger.info(f"Selected job for source {source_id} (not currently active)")
                                return job_data
                            else:
                                logger.debug(f"Skipping job for source {source_id} (already active)")
            
            # If no suitable job found, wait a bit
            await asyncio.sleep(2)
            
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
        finally:
            # Always cleanup the active source flag
            active_key = f"source:active:{source_id}"
            await self.redis.delete(active_key)
            logger.info(f"Released source lock for {source_id}")
    
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
        
        # Process pages function
        async def process_pages(crawler):
            async for page_data in crawler.crawl(url, **crawl_params):
                # Check for cancellation periodically
                if await self._is_job_cancelled(job_id):
                    logger.info(f"Job {job_id} cancelled during crawling")
                    results["cancelled"] = True
                    results["errors"].append({"error": "Job cancelled by user"})
                    break
                
                try:
                    # Skip pages with errors (e.g., 404, network errors)
                    if "error" in page_data:
                        logger.warning(f"Skipping page {page_data['url']} due to error: {page_data['error']}")
                        results["errors"].append({
                            "url": page_data["url"],
                            "error": page_data["error"]
                        })
                        continue
                    
                    # Skip pages without content_type (shouldn't happen, but be defensive)
                    if "content_type" not in page_data:
                        logger.warning(f"Skipping page {page_data['url']} - missing content_type")
                        results["errors"].append({
                            "url": page_data["url"],
                            "error": "Missing content_type"
                        })
                        continue
                    
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
        
        # Use the regular crawler
        await process_pages(self.crawler)
        
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
            # Stop HTTP server
            if self.http_server:
                await self.http_server.stop()
                logger.info("Health check server stopped")
            
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