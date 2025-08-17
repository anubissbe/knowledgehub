"""
Firecrawl Web Ingestion Service
Integrates with Firecrawl API for intelligent web scraping and content ingestion
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import re
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field, HttpUrl

from .hybrid_rag_service import HybridRAGService, get_hybrid_rag_service
from .cache import RedisCache
from .real_ai_intelligence import RealAIIntelligence
from ..config import settings

logger = logging.getLogger(__name__)


class CrawlMode(Enum):
    """Crawl modes for different use cases"""
    SINGLE_PAGE = "single_page"
    SITEMAP = "sitemap"
    RECURSIVE = "recursive"
    SELECTIVE = "selective"


class ContentFormat(Enum):
    """Output formats for crawled content"""
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass
class FirecrawlConfig:
    """Configuration for Firecrawl service"""
    api_url: str = "https://api.firecrawl.dev"
    api_key: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0
    max_concurrent_crawls: int = 5
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    default_format: ContentFormat = ContentFormat.MARKDOWN


class CrawlRequest(BaseModel):
    """Request for web crawling"""
    url: HttpUrl = Field(..., description="URL to crawl")
    mode: CrawlMode = Field(CrawlMode.SINGLE_PAGE, description="Crawl mode")
    max_pages: int = Field(10, ge=1, le=100, description="Maximum pages to crawl")
    max_depth: int = Field(2, ge=1, le=5, description="Maximum crawl depth")
    include_patterns: List[str] = Field(default_factory=list, description="URL patterns to include")
    exclude_patterns: List[str] = Field(default_factory=list, description="URL patterns to exclude")
    content_format: ContentFormat = Field(ContentFormat.MARKDOWN, description="Output format")
    extract_metadata: bool = Field(True, description="Extract page metadata")
    follow_robots: bool = Field(True, description="Follow robots.txt")
    rate_limit_delay: float = Field(1.0, description="Delay between requests")


class ProcessingOptions(BaseModel):
    """Options for content processing"""
    remove_ads: bool = Field(True, description="Remove advertisements")
    remove_navigation: bool = Field(True, description="Remove navigation elements")
    extract_main_content: bool = Field(True, description="Extract main content only")
    preserve_formatting: bool = Field(True, description="Preserve text formatting")
    extract_images: bool = Field(False, description="Extract image URLs")
    extract_links: bool = Field(True, description="Extract internal/external links")
    min_content_length: int = Field(100, description="Minimum content length")


class CrawlResult(BaseModel):
    """Result from web crawling"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    links: List[str] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    content_type: str = "text/html"
    content_length: int = 0
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time: float = 0.0


class FirecrawlIngestionService:
    """
    Service for intelligent web content ingestion using Firecrawl
    
    Features:
    - Multi-mode web crawling (single page, sitemap, recursive)
    - Content extraction and cleaning
    - Rate limiting and respectful crawling
    - Integration with hybrid RAG system
    - Duplicate detection and deduplication
    - Content quality filtering
    """
    
    def __init__(self, config: Optional[FirecrawlConfig] = None):
        self.config = config or FirecrawlConfig(
            api_key=getattr(settings, 'FIRECRAWL_API_KEY', None)
        )
        self.logger = logger
        self.cache = RedisCache(settings.REDIS_URL)
        self.ai_intelligence = RealAIIntelligence()
        
        # HTTP client for Firecrawl API
        self.client: Optional[httpx.AsyncClient] = None
        
        # RAG service for content ingestion
        self.hybrid_rag: Optional[HybridRAGService] = None
        
        # Crawl tracking
        self.active_crawls: Dict[str, Dict[str, Any]] = {}
        self.crawl_semaphore = asyncio.Semaphore(self.config.max_concurrent_crawls)
        
        # Performance tracking
        self.performance_stats = {
            "crawls_completed": 0,
            "pages_crawled": 0,
            "content_ingested": 0,
            "avg_crawl_time": 0.0,
            "success_rate": 0.0,
            "duplicate_rate": 0.0
        }
    
    async def initialize(self):
        """Initialize Firecrawl ingestion service"""
        try:
            # Initialize cache
            await self.cache.initialize()
            
            # Create HTTP client
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self.client = httpx.AsyncClient(
                base_url=self.config.api_url,
                headers=headers,
                timeout=self.config.timeout
            )
            
            # Initialize hybrid RAG service
            self.hybrid_rag = await get_hybrid_rag_service()
            
            # Test connection
            await self._health_check()
            
            logger.info("Firecrawl ingestion service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firecrawl service: {e}")
            # Continue without Firecrawl if initialization fails
            self.client = None
    
    async def crawl_and_ingest(
        self,
        request: CrawlRequest,
        processing_options: Optional[ProcessingOptions] = None,
        user_id: str = "system",
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Crawl web content and ingest into RAG system
        
        Args:
            request: Crawl configuration
            processing_options: Content processing options
            user_id: User identifier
            project_id: Optional project identifier
            
        Returns:
            Crawl and ingestion results
        """
        crawl_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            async with self.crawl_semaphore:
                # Track active crawl
                self.active_crawls[crawl_id] = {
                    "url": str(request.url),
                    "mode": request.mode.value,
                    "started_at": datetime.utcnow().isoformat(),
                    "status": "running",
                    "pages_found": 0,
                    "pages_processed": 0
                }
                
                # Execute crawl based on mode
                if request.mode == CrawlMode.SINGLE_PAGE:
                    results = await self._crawl_single_page(request, processing_options)
                elif request.mode == CrawlMode.SITEMAP:
                    results = await self._crawl_sitemap(request, processing_options)
                elif request.mode == CrawlMode.RECURSIVE:
                    results = await self._crawl_recursive(request, processing_options)
                elif request.mode == CrawlMode.SELECTIVE:
                    results = await self._crawl_selective(request, processing_options)
                else:
                    raise ValueError(f"Unknown crawl mode: {request.mode}")
                
                # Process and ingest results
                ingestion_results = await self._ingest_crawl_results(
                    results, user_id, project_id, request.url
                )
                
                execution_time = time.time() - start_time
                
                # Update tracking
                self.active_crawls[crawl_id]["status"] = "completed"
                self.active_crawls[crawl_id]["completed_at"] = datetime.utcnow().isoformat()
                self.active_crawls[crawl_id]["execution_time"] = execution_time
                
                # Track performance
                await self._track_crawl_performance(
                    request.mode, execution_time, len(results), True
                )
                
                return {
                    "crawl_id": crawl_id,
                    "success": True,
                    "url": str(request.url),
                    "mode": request.mode.value,
                    "pages_crawled": len(results),
                    "content_ingested": ingestion_results["documents_ingested"],
                    "execution_time": execution_time,
                    "ingestion_results": ingestion_results,
                    "crawl_results": results[:5]  # Return first 5 for preview
                }
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Crawl and ingestion failed: {e}")
            
            # Update tracking
            if crawl_id in self.active_crawls:
                self.active_crawls[crawl_id]["status"] = "failed"
                self.active_crawls[crawl_id]["error"] = str(e)
            
            # Track failure
            await self._track_crawl_performance(
                request.mode, execution_time, 0, False
            )
            
            return {
                "crawl_id": crawl_id,
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
        
        finally:
            # Clean up tracking after delay
            asyncio.create_task(self._cleanup_crawl_tracking(crawl_id, 300))
    
    async def _crawl_single_page(
        self,
        request: CrawlRequest,
        processing_options: Optional[ProcessingOptions]
    ) -> List[CrawlResult]:
        """Crawl a single page"""
        if not self.client:
            raise Exception("Firecrawl client not initialized")
        
        # Check cache first
        cache_key = f"firecrawl_page:{str(request.url)}"
        if self.config.enable_caching:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return [CrawlResult(**cached_result)]
        
        # Prepare crawl request
        crawl_data = {
            "url": str(request.url),
            "formats": [request.content_format.value],
            "includeTags": ["title", "meta", "h1", "h2", "h3", "p", "article"],
            "excludeTags": ["script", "style", "nav", "footer", "ads"] if processing_options and processing_options.remove_ads else [],
            "onlyMainContent": processing_options.extract_main_content if processing_options else True
        }
        
        response = await self._make_firecrawl_request(
            "POST",
            "/v1/scrape",
            json=crawl_data
        )
        
        if not response["success"]:
            raise Exception(f"Firecrawl API error: {response['error']}")
        
        data = response["data"]
        result = CrawlResult(
            url=str(request.url),
            title=data.get("metadata", {}).get("title", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            links=data.get("links", []),
            content_type=data.get("metadata", {}).get("contentType", "text/html"),
            content_length=len(data.get("content", "")),
            processing_time=response.get("response_time", 0.0)
        )
        
        # Cache result
        if self.config.enable_caching:
            await self.cache.set(cache_key, result.dict(), ttl=self.config.cache_ttl)
        
        return [result]
    
    async def _crawl_sitemap(
        self,
        request: CrawlRequest,
        processing_options: Optional[ProcessingOptions]
    ) -> List[CrawlResult]:
        """Crawl using sitemap"""
        if not self.client:
            raise Exception("Firecrawl client not initialized")
        
        # First, try to find sitemap
        sitemap_urls = await self._discover_sitemaps(request.url)
        
        if not sitemap_urls:
            logger.warning(f"No sitemap found for {request.url}, falling back to recursive crawl")
            return await self._crawl_recursive(request, processing_options)
        
        # Crawl pages from sitemap
        crawl_data = {
            "url": str(request.url),
            "formats": [request.content_format.value],
            "crawlerOptions": {
                "includes": request.include_patterns,
                "excludes": request.exclude_patterns,
                "maxDepth": request.max_depth,
                "limit": request.max_pages,
                "useSitemap": True
            }
        }
        
        response = await self._make_firecrawl_request(
            "POST",
            "/v1/crawl",
            json=crawl_data
        )
        
        if not response["success"]:
            raise Exception(f"Firecrawl crawl error: {response['error']}")
        
        # Process results
        results = []
        for page_data in response["data"].get("pages", []):
            result = CrawlResult(
                url=page_data.get("url", ""),
                title=page_data.get("metadata", {}).get("title", ""),
                content=page_data.get("content", ""),
                metadata=page_data.get("metadata", {}),
                links=page_data.get("links", []),
                content_length=len(page_data.get("content", ""))
            )
            results.append(result)
        
        return results
    
    async def _crawl_recursive(
        self,
        request: CrawlRequest,
        processing_options: Optional[ProcessingOptions]
    ) -> List[CrawlResult]:
        """Crawl recursively following links"""
        if not self.client:
            raise Exception("Firecrawl client not initialized")
        
        crawl_data = {
            "url": str(request.url),
            "formats": [request.content_format.value],
            "crawlerOptions": {
                "includes": request.include_patterns,
                "excludes": request.exclude_patterns,
                "maxDepth": request.max_depth,
                "limit": request.max_pages,
                "allowBackwardCrawling": False,
                "allowExternalContentLinks": False
            }
        }
        
        response = await self._make_firecrawl_request(
            "POST",
            "/v1/crawl",
            json=crawl_data
        )
        
        if not response["success"]:
            raise Exception(f"Firecrawl crawl error: {response['error']}")
        
        # Process results
        results = []
        for page_data in response["data"].get("pages", []):
            # Apply content filtering
            content = page_data.get("content", "")
            if processing_options and len(content) < processing_options.min_content_length:
                continue
            
            result = CrawlResult(
                url=page_data.get("url", ""),
                title=page_data.get("metadata", {}).get("title", ""),
                content=content,
                metadata=page_data.get("metadata", {}),
                links=page_data.get("links", []),
                content_length=len(content)
            )
            results.append(result)
        
        return results
    
    async def _crawl_selective(
        self,
        request: CrawlRequest,
        processing_options: Optional[ProcessingOptions]
    ) -> List[CrawlResult]:
        """Crawl only pages matching specific patterns"""
        # Start with recursive crawl
        all_results = await self._crawl_recursive(request, processing_options)
        
        # Filter results based on patterns
        filtered_results = []
        
        for result in all_results:
            # Check if URL matches include patterns
            if request.include_patterns:
                if not any(pattern in result.url for pattern in request.include_patterns):
                    continue
            
            # Check if URL matches exclude patterns
            if request.exclude_patterns:
                if any(pattern in result.url for pattern in request.exclude_patterns):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    async def _discover_sitemaps(self, base_url: HttpUrl) -> List[str]:
        """Discover sitemap URLs for a domain"""
        sitemaps = []
        domain = f"{base_url.scheme}://{base_url.host}"
        
        # Common sitemap locations
        common_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap.txt",
            "/robots.txt"
        ]
        
        for path in common_paths:
            try:
                url = urljoin(domain, path)
                response = await self.client.get(url)
                
                if response.status_code == 200:
                    if path == "/robots.txt":
                        # Parse robots.txt for sitemap references
                        for line in response.text.split("\n"):
                            if line.strip().lower().startswith("sitemap:"):
                                sitemap_url = line.split(":", 1)[1].strip()
                                sitemaps.append(sitemap_url)
                    else:
                        sitemaps.append(url)
                        
            except Exception as e:
                logger.debug(f"Failed to check {path}: {e}")
        
        return sitemaps
    
    async def _ingest_crawl_results(
        self,
        results: List[CrawlResult],
        user_id: str,
        project_id: Optional[str],
        source_url: HttpUrl
    ) -> Dict[str, Any]:
        """Ingest crawl results into hybrid RAG system"""
        if not self.hybrid_rag:
            return {
                "documents_ingested": 0,
                "error": "Hybrid RAG service not available"
            }
        
        ingested_count = 0
        duplicates_found = 0
        errors = []
        
        for result in results:
            try:
                # Check for duplicates
                if await self._is_duplicate_content(result.content):
                    duplicates_found += 1
                    continue
                
                # Prepare metadata
                metadata = {
                    "source": "firecrawl",
                    "source_url": str(source_url),
                    "page_url": result.url,
                    "title": result.title,
                    "crawled_at": result.crawled_at.isoformat(),
                    "content_type": result.content_type,
                    "content_length": result.content_length,
                    "user_id": user_id,
                    **result.metadata
                }
                
                if project_id:
                    metadata["project_id"] = project_id
                
                # Ingest into hybrid RAG
                ingest_result = await self.hybrid_rag.ingest_document(
                    content=result.content,
                    metadata=metadata,
                    doc_id=f"firecrawl_{hash(result.url)}"
                )
                
                if ingest_result["success"]:
                    ingested_count += 1
                else:
                    errors.append(f"Failed to ingest {result.url}: {ingest_result.get('error')}")
                    
            except Exception as e:
                errors.append(f"Error processing {result.url}: {str(e)}")
        
        # Update performance stats
        self.performance_stats["content_ingested"] += ingested_count
        self.performance_stats["duplicate_rate"] = duplicates_found / len(results) if results else 0
        
        return {
            "documents_ingested": ingested_count,
            "duplicates_skipped": duplicates_found,
            "errors": errors,
            "total_processed": len(results),
            "success_rate": ingested_count / len(results) if results else 0
        }
    
    async def _is_duplicate_content(self, content: str) -> bool:
        """Check if content is duplicate based on hash"""
        content_hash = str(hash(content))
        cache_key = f"content_hash:{content_hash}"
        
        exists = await self.cache.get(cache_key)
        if exists:
            return True
        
        # Cache hash to detect future duplicates
        await self.cache.set(cache_key, True, ttl=86400)  # 24 hours
        return False
    
    async def _make_firecrawl_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Firecrawl API with retry logic"""
        if not self.client:
            return {"success": False, "error": "Firecrawl client not initialized"}
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.request(
                    method=method,
                    url=endpoint,
                    json=json,
                    params=params
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "data": data,
                        "response_time": response_time
                    }
                else:
                    error_msg = f"Firecrawl API error {response.status_code}: {response.text}"
                    if attempt == self.config.max_retries - 1:
                        return {"success": False, "error": error_msg}
                    
                    # Retry on server errors
                    if response.status_code >= 500:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        return {"success": False, "error": error_msg}
                        
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return {"success": False, "error": str(e)}
                
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return {"success": False, "error": "Max retries exceeded"}
    
    async def _health_check(self) -> bool:
        """Check if Firecrawl service is healthy"""
        try:
            if not self.client:
                return False
            
            # Simple health check - try to access API
            response = await self.client.get("/v1/status")
            return response.status_code in [200, 404]  # 404 is ok for status endpoint
            
        except Exception as e:
            logger.warning(f"Firecrawl health check failed: {e}")
            return False
    
    async def _track_crawl_performance(
        self,
        mode: CrawlMode,
        execution_time: float,
        pages_crawled: int,
        success: bool
    ):
        """Track crawl performance metrics"""
        try:
            if success:
                self.performance_stats["crawls_completed"] += 1
                self.performance_stats["pages_crawled"] += pages_crawled
                
                # Update average crawl time
                total_crawls = self.performance_stats["crawls_completed"]
                old_avg = self.performance_stats["avg_crawl_time"]
                self.performance_stats["avg_crawl_time"] = (
                    (old_avg * (total_crawls - 1) + execution_time) / total_crawls
                )
                
                # Update success rate
                total_attempts = total_crawls + sum(1 for crawl in self.active_crawls.values() if crawl.get("status") == "failed")
                self.performance_stats["success_rate"] = total_crawls / total_attempts if total_attempts > 0 else 1.0
            
            # Track in AI intelligence system
            await self.ai_intelligence.track_performance_metric(
                "firecrawl_crawl",
                execution_time=execution_time,
                success=success,
                metadata={
                    "mode": mode.value,
                    "pages_crawled": pages_crawled
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to track Firecrawl performance: {e}")
    
    async def _cleanup_crawl_tracking(self, crawl_id: str, delay: int):
        """Clean up crawl tracking after delay"""
        await asyncio.sleep(delay)
        self.active_crawls.pop(crawl_id, None)
    
    async def get_crawl_status(self, crawl_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active crawl"""
        return self.active_crawls.get(crawl_id)
    
    async def list_active_crawls(self) -> List[Dict[str, Any]]:
        """List all active crawls"""
        return list(self.active_crawls.values())
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        return {
            **self.performance_stats,
            "active_crawls": len(self.active_crawls),
            "config": {
                "max_concurrent_crawls": self.config.max_concurrent_crawls,
                "enable_caching": self.config.enable_caching,
                "timeout": self.config.timeout
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        health = {
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {}
        }
        
        if not self.client:
            health["status"] = "disconnected"
            health["details"]["client"] = "not_initialized"
        else:
            is_healthy = await self._health_check()
            health["status"] = "healthy" if is_healthy else "unhealthy"
            health["details"]["api_connection"] = "ok" if is_healthy else "failed"
        
        health["details"]["performance"] = self.performance_stats
        health["details"]["active_crawls"] = len(self.active_crawls)
        
        return health
    
    async def close(self):
        """Close service connections"""
        if self.client:
            await self.client.aclose()
            self.client = None


# Global instance
_firecrawl_service: Optional[FirecrawlIngestionService] = None


async def get_firecrawl_service() -> FirecrawlIngestionService:
    """Get singleton Firecrawl service instance"""
    global _firecrawl_service
    
    if _firecrawl_service is None:
        _firecrawl_service = FirecrawlIngestionService()
        await _firecrawl_service.initialize()
    
    return _firecrawl_service