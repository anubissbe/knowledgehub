"""Incremental web crawler with delta detection"""

import asyncio
import hashlib
from typing import AsyncIterator, Dict, Optional, Set, Any, List
from datetime import datetime
from urllib.parse import urljoin, urlparse
import httpx
import logging

from .crawler import WebCrawler

logger = logging.getLogger(__name__)


class IncrementalWebCrawler(WebCrawler):
    """Enhanced web crawler with incremental and delta crawling capabilities"""
    
    def __init__(self, api_url: str, api_key: str):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.api_client = httpx.AsyncClient(
            base_url=api_url,
            headers={"X-API-Key": api_key},
            timeout=30.0
        )
        self.existing_docs_cache: Dict[str, Dict[str, Any]] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        await super().__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.api_client.aclose()
        await super().__aexit__(exc_type, exc_val, exc_tb)
        
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
        
    async def _load_existing_documents(self, source_id: str):
        """Load existing documents for a source into cache"""
        try:
            logger.info(f"Loading existing documents for source {source_id}")
            
            # Get all documents for this source
            response = await self.api_client.get(
                "/api/v1/documents/",
                params={"source_id": source_id, "limit": 10000}
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                
                # Cache document info by URL
                for doc in documents:
                    url = doc.get("url")
                    if url:
                        self.existing_docs_cache[url] = {
                            "id": doc.get("id"),
                            "content_hash": doc.get("content_hash"),
                            "updated_at": doc.get("updated_at"),
                            "metadata": doc.get("metadata", {})
                        }
                        
                logger.info(f"Loaded {len(self.existing_docs_cache)} existing documents")
            else:
                logger.warning(f"Failed to load existing documents: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")
            
    async def crawl(
        self,
        start_url: str,
        max_depth: int = 2,
        max_pages: int = 100,
        follow_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        crawl_delay: float = 1.0,
        source_id: Optional[str] = None,
        force_refresh: bool = False
    ) -> AsyncIterator[Dict]:
        """
        Crawl website with incremental and delta detection
        
        Args:
            start_url: Starting URL
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl
            follow_patterns: URL patterns to follow
            exclude_patterns: URL patterns to exclude
            crawl_delay: Delay between requests in seconds
            source_id: Source ID for incremental crawling
            force_refresh: Force refresh all pages regardless of changes
            
        Yields:
            Page data with content and metadata
        """
        # Load existing documents if source_id provided
        if source_id and not force_refresh:
            await self._load_existing_documents(source_id)
            
        visited: Set[str] = set()
        queue = [(start_url, 0)]
        pages_processed = 0
        new_pages_found = 0
        updated_pages = 0
        
        while queue and pages_processed < max_pages:
            url, depth = queue.pop(0)
            
            if url in visited or depth > max_depth:
                continue
                
            visited.add(url)
            
            # Check if we've already processed this URL
            existing_doc = self.existing_docs_cache.get(url)
            
            # For existing pages, use shorter delay for checking
            if existing_doc and not force_refresh:
                await asyncio.sleep(crawl_delay * 0.5)  # Shorter delay for checks
            else:
                await asyncio.sleep(crawl_delay)  # Full delay for new content
            
            try:
                # Navigate to page
                await self.page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Get page content
                content = await self.page.content()
                title = await self.page.title()
                
                # Calculate content hash
                content_hash = self._calculate_content_hash(content)
                
                # Determine if this is new or updated content
                is_new = existing_doc is None
                is_updated = False
                
                if existing_doc and not force_refresh:
                    # Check if content has changed
                    if existing_doc.get("content_hash") != content_hash:
                        is_updated = True
                        updated_pages += 1
                        logger.info(f"Content changed for {url}")
                    else:
                        # Content unchanged, but still find new links
                        logger.debug(f"Content unchanged for {url}, checking for new links")
                        
                if is_new:
                    new_pages_found += 1
                    logger.info(f"New page found: {url}")
                
                # Always yield the page data (for finding new links)
                # But mark whether it needs processing
                should_process = is_new or is_updated or force_refresh
                
                if should_process:
                    yield {
                        "url": url,
                        "content": content,
                        "title": title,
                        "content_hash": content_hash,
                        "is_new": is_new,
                        "is_update": is_updated,
                        "metadata": {
                            "url": url,
                            "title": title,
                            "crawled_at": datetime.utcnow().isoformat(),
                            "depth": depth,
                            "content_hash": content_hash
                        }
                    }
                
                pages_processed += 1
                
                # Extract links even from unchanged pages to find new content
                links = await self.page.evaluate("""
                    () => {
                        return Array.from(document.querySelectorAll('a[href]'))
                            .map(a => a.href)
                            .filter(href => href.startsWith('http'));
                    }
                """)
                
                # Process links
                for link in links:
                    absolute_url = urljoin(url, link)
                    
                    # Apply filters
                    if not self._should_follow_url(
                        absolute_url, start_url, follow_patterns, exclude_patterns
                    ):
                        continue
                        
                    if absolute_url not in visited:
                        queue.append((absolute_url, depth + 1))
                        
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                yield {
                    "url": url,
                    "error": str(e),
                    "metadata": {"url": url, "error": str(e)}
                }
                
        # Log summary
        logger.info(
            f"Incremental crawl complete: {pages_processed} pages checked, "
            f"{new_pages_found} new, {updated_pages} updated"
        )