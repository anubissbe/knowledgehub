"""Scraper worker main entry point"""

import asyncio
import os
import json
import signal
import sys
from typing import Optional
from datetime import datetime

from .crawler import WebCrawler
from .parsers import ContentParserFactory
from ..shared.config import Config
from ..shared.logging import setup_logging

# Setup logging
logger = setup_logging("scraper")


class ScraperWorker:
    """Main scraper worker that processes crawl jobs"""
    
    def __init__(self):
        self.config = Config()
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
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Main processing loop
        while self.running:
            try:
                # Get job from queue
                job_data = await self._get_next_job()
                
                if job_data:
                    await self._process_job(job_data)
                else:
                    # No job available, wait
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        # Cleanup
        await self.cleanup()
    
    async def _get_next_job(self) -> Optional[dict]:
        """Get next job from Redis queue"""
        try:
            # Use BLPOP for blocking pop
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
        """Process a scraping job"""
        job_id = job_data.get("job_id")
        source_id = job_data.get("source_id")
        
        logger.info(f"Processing job {job_id} for source {source_id}")
        
        try:
            # Update job status to running
            await self._update_job_status(job_id, "running")
            
            # Get source details
            source = await self._get_source(source_id)
            if not source:
                raise Exception(f"Source {source_id} not found")
            
            # Crawl and extract content
            result = await self._crawl_source(source, job_data)
            
            # Update job as completed
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
        
        # Fix: Handle None crawl_config properly
        crawl_config = source.get("crawl_config")
        if crawl_config is None:
            # Use default config
            crawl_config = {}
        
        # Initialize crawl results
        results = {
            "pages_crawled": 0,
            "chunks_created": 0,
            "errors": [],
            "start_time": datetime.utcnow().isoformat(),
            "pages": []
        }
        
        # Start crawling
        async for page_data in self.crawler.crawl(
            url,
            max_depth=crawl_config.get("max_depth", 2),
            max_pages=crawl_config.get("max_pages", 100),
            follow_patterns=crawl_config.get("follow_patterns", []),
            exclude_patterns=crawl_config.get("exclude_patterns", [])
        ):
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
                    await self._queue_chunk_for_processing({
                        "source_id": source["id"],
                        "job_id": job_data["job_id"],
                        "chunk": chunk
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
        try:
            data = {"status": status}
            if result:
                data["result"] = result
            if error:
                data["error"] = error
            
            response = await self.api_client.put(
                f"/api/v1/jobs/{job_id}",
                json=data
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to update job status: {response.text}")
                
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up scraper worker...")
        
        try:
            await self.redis.aclose()
            await self.api_client.aclose()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main entry point"""
    worker = ScraperWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())