"""Web crawler using Playwright for JavaScript-heavy sites"""

import asyncio
from typing import AsyncIterator, Dict, Any, List, Optional, Set
from urllib.parse import urlparse, urljoin, urldefrag
import re
from datetime import datetime

from playwright.async_api import async_playwright, Browser, Page
import httpx

from ..shared.logging import setup_logging

logger = setup_logging("crawler")


class WebCrawler:
    """Asynchronous web crawler with JavaScript support"""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
        self.visited_urls: Set[str] = set()
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={
                "User-Agent": "KnowledgeHub/1.0 (compatible; bot)"
            }
        )
    
    async def start(self):
        """Start the crawler (initialize browser)"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox"]
        )
        logger.info("Crawler started with Playwright browser")
    
    async def stop(self):
        """Stop the crawler and cleanup resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        await self.http_client.aclose()
        logger.info("Crawler stopped")
    
    async def crawl(
        self,
        start_url: str,
        max_depth: int = 2,
        max_pages: int = 100,
        follow_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        crawl_delay: float = 0
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Crawl a website starting from the given URL
        
        Yields page data as it's crawled
        """
        follow_patterns = follow_patterns or []
        exclude_patterns = exclude_patterns or []
        
        # Validate start URL
        if not self._is_valid_url(start_url):
            logger.error(f"Invalid URL: {start_url}")
            yield {
                "url": start_url,
                "error": "Invalid URL format",
                "timestamp": datetime.utcnow().isoformat()
            }
            return
        
        # Reset visited URLs for new crawl
        self.visited_urls.clear()
        logger.info(f"Starting new crawl of {start_url} - cleared {len(self.visited_urls)} visited URLs")
        
        # Extract base domain for external URL detection
        base_domain = self._get_domain(start_url)
        
        # Queue of URLs to crawl with their depth
        queue = [(start_url, 0)]
        pages_crawled = 0
        
        while queue and pages_crawled < max_pages:
            if not queue:
                break
                
            url, depth = queue.pop(0)
            
            # Skip if already visited
            url_without_fragment = urldefrag(url)[0]
            if url_without_fragment in self.visited_urls:
                continue
            
            # Skip if exceeds max depth
            if depth > max_depth:
                continue
            
            # Check exclude patterns
            if self._matches_patterns(url, exclude_patterns):
                logger.debug(f"Skipping excluded URL: {url}")
                continue
            
            # Mark as visited
            self.visited_urls.add(url_without_fragment)
            
            try:
                # Crawl the page
                page_data = await self._crawl_page(url)
                
                if page_data:
                    pages_crawled += 1
                    yield page_data
                    
                    # Apply crawl delay to avoid rate limiting
                    if crawl_delay > 0:
                        logger.debug(f"Waiting {crawl_delay} seconds before next request...")
                        await asyncio.sleep(crawl_delay)
                    
                    # Extract and queue new URLs
                    if depth < max_depth:
                        for link in page_data.get("links", []):
                            # Skip external URLs unless explicitly allowed
                            if not self._is_same_domain(link, base_domain):
                                logger.debug(f"Skipping external URL: {link}")
                                continue
                                
                            # Apply follow patterns if specified
                            if follow_patterns and not self._matches_patterns(link, follow_patterns):
                                continue
                            
                            if link not in self.visited_urls:
                                queue.append((link, depth + 1))
                
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                yield {
                    "url": url,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    async def _crawl_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Crawl a single page"""
        logger.info(f"Crawling: {url}")
        
        # Determine if page needs JavaScript rendering
        if await self._needs_javascript(url):
            return await self._crawl_with_playwright(url)
        else:
            return await self._crawl_with_httpx(url)
    
    async def _needs_javascript(self, url: str) -> bool:
        """Determine if a page needs JavaScript rendering"""
        # Simple heuristic - can be improved
        js_indicators = [
            "react", "angular", "vue", "spa",
            "/app", "/dashboard", "#!",
            "stoplight.io"  # Stoplight uses React SPA
        ]
        
        url_lower = url.lower()
        return any(indicator in url_lower for indicator in js_indicators)
    
    async def _crawl_with_httpx(self, url: str) -> Optional[Dict[str, Any]]:
        """Crawl a page using httpx (for static content)"""
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            
            # Only process HTML and text content
            if "html" not in content_type and "text" not in content_type:
                logger.debug(f"Skipping non-HTML content: {content_type}")
                return None
            
            return {
                "url": url,
                "content": response.text,
                "content_type": content_type,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "links": self._extract_links(response.text, url),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "crawl_method": "httpx"
                }
            }
            
        except httpx.TimeoutException:
            logger.error(f"Timeout crawling {url}")
            return {
                "url": url,
                "error": "Request timeout",
                "timestamp": datetime.utcnow().isoformat()
            }
        except httpx.NetworkError as e:
            logger.error(f"Network error crawling {url}: {e}")
            return {
                "url": url,
                "error": f"Network error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error crawling {url}: {e.response.status_code}")
            return {
                "url": url,
                "error": f"HTTP {e.response.status_code}",
                "status_code": e.response.status_code,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Unexpected error crawling {url}: {e}")
            return {
                "url": url,
                "error": f"Unexpected error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _crawl_with_playwright(self, url: str) -> Optional[Dict[str, Any]]:
        """Crawl a page using Playwright (for JavaScript-heavy content)"""
        page = None
        try:
            # Create new page
            page = await self.browser.new_page()
            
            # Navigate to URL with longer timeout for JS-heavy sites
            response = await page.goto(
                url,
                wait_until="domcontentloaded",  # Faster than networkidle
                timeout=60000  # 60 seconds for complex sites
            )
            
            if not response:
                return None
            
            # Wait for content to load
            await page.wait_for_load_state("domcontentloaded")
            
            # Additional wait for React sites
            if "react.dev" in url:
                try:
                    # Wait for main content to appear
                    await page.wait_for_selector("main", timeout=10000)
                except:
                    logger.debug("Main content selector not found, continuing anyway")
            
            # Special handling for Stoplight documentation
            if "stoplight.io" in url:
                try:
                    # Wait for Stoplight's content container
                    await page.wait_for_selector("article, .sl-elements-article, [role='article']", timeout=15000)
                    # Give React time to render
                    await asyncio.sleep(2)
                except:
                    logger.debug("Stoplight content selector not found, continuing anyway")
            
            # Extract content
            content = await page.content()
            title = await page.title()
            
            # Extract all links
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => link.href);
                }
            """)
            
            # Extract metadata
            metadata = await page.evaluate("""
                () => {
                    const getMeta = (name) => {
                        const element = document.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
                        return element ? element.content : null;
                    };
                    
                    return {
                        description: getMeta('description') || getMeta('og:description'),
                        keywords: getMeta('keywords'),
                        author: getMeta('author'),
                        ogTitle: getMeta('og:title'),
                        ogType: getMeta('og:type'),
                        ogImage: getMeta('og:image')
                    };
                }
            """)
            
            return {
                "url": url,
                "content": content,
                "content_type": "text/html",
                "status_code": response.status,
                "title": title,
                "links": [self._normalize_url(link, url) for link in links],
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    **metadata,
                    "crawl_method": "playwright"
                }
            }
            
        except Exception as e:
            logger.error(f"Playwright error crawling {url}: {e}")
            return None
            
        finally:
            if page:
                await page.close()
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML content"""
        # Simple regex-based link extraction
        link_pattern = re.compile(r'href=[\'"]?([^\'" >]+)', re.IGNORECASE)
        links = []
        
        for match in link_pattern.finditer(html):
            link = match.group(1)
            normalized = self._normalize_url(link, base_url)
            if normalized:
                links.append(normalized)
        
        return links
    
    def _normalize_url(self, url: str, base_url: str) -> Optional[str]:
        """Normalize a URL relative to a base URL"""
        try:
            # Skip non-HTTP URLs
            if url.startswith(("javascript:", "mailto:", "tel:", "#")):
                return None
            
            # Make relative URLs absolute
            absolute_url = urljoin(base_url, url)
            
            # Parse and validate
            parsed = urlparse(absolute_url)
            if parsed.scheme not in ["http", "https"]:
                return None
            
            # Skip non-content resources (CSS, images, fonts, etc.)
            resource_extensions = {
                '.css', '.ico', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp',
                '.woff', '.woff2', '.ttf', '.eot', '.otf',
                '.mp3', '.mp4', '.avi', '.mov', '.wmv',
                '.zip', '.tar', '.gz', '.rar', '.7z',
                '.pdf'  # Skip PDFs for now as they need special handling
            }
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in resource_extensions):
                return None
            
            # Skip common resource paths
            if any(part in path_lower for part in ['/favicon', '/fonts/', '/images/', '/css/', '/js/', '/assets/']):
                return None
            
            # Remove fragment
            return urldefrag(absolute_url)[0]
            
        except Exception:
            return None
    
    def _matches_patterns(self, url: str, patterns: List[str]) -> bool:
        """Check if URL matches any of the patterns"""
        for pattern in patterns:
            if re.search(pattern, url):
                return True
        return False
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""
    
    def _is_same_domain(self, url: str, base_domain: str) -> bool:
        """Check if URL belongs to the same domain"""
        try:
            url_domain = self._get_domain(url)
            # Handle subdomains - check if domains share the same base
            if url_domain == base_domain:
                return True
            # Check if one is a subdomain of the other
            if url_domain.endswith('.' + base_domain) or base_domain.endswith('.' + url_domain):
                return True
            return False
        except Exception:
            return False