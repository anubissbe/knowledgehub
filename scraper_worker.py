#!/usr/bin/env python3
"""
KnowledgeHub Scraper Worker
Processes scraping jobs from Redis queue and populates the multi-dimensional databases
"""

import asyncio
import aiohttp
import redis
import json
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
from typing import Dict, List, Optional
import re
import time
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScraperWorker:
    def __init__(self, redis_url="redis://localhost:6381", api_base="http://localhost:3000"):
        self.redis_client = redis.from_url(redis_url)
        self.api_base = api_base
        self.session = None
        
    async def start(self):
        """Start the worker and process jobs"""
        self.session = aiohttp.ClientSession()
        logger.info("Scraper worker started, listening for jobs...")
        
        try:
            while True:
                # Check for jobs in priority order
                for queue in ['crawl_jobs:high', 'crawl_jobs:normal', 'crawl_jobs:low']:
                    job_data = self.redis_client.lpop(queue)
                    if job_data:
                        await self.process_job(json.loads(job_data))
                        break
                else:
                    # No jobs found, wait a bit
                    await asyncio.sleep(1)
        finally:
            await self.session.close()
    
    async def process_job(self, job: Dict):
        """Process a single scraping job"""
        logger.info(f"Processing job: {job['id']} for source: {job['source_name']}")
        
        try:
            # Update job status
            await self.update_job_status(job['id'], 'PROCESSING')
            
            # Get source configuration
            source = job['source']
            config = source.get('config', {})
            
            # Scrape the source
            if source['type'] == 'website':
                await self.scrape_website(source, job['id'])
            elif source['type'] == 'documentation':
                await self.scrape_documentation(source, job['id'])
            elif source['type'] == 'repository':
                await self.scrape_repository(source, job['id'])
            elif source['type'] == 'api':
                await self.scrape_api(source, job['id'])
            
            # Update job status to completed
            await self.update_job_status(job['id'], 'COMPLETED')
            logger.info(f"Job {job['id']} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing job {job['id']}: {str(e)}")
            await self.update_job_status(job['id'], 'FAILED', str(e))
    
    async def scrape_website(self, source: Dict, job_id: str):
        """Scrape a website source"""
        config = source.get('config', {})
        
        # Check if this is a JavaScript-heavy site that needs special handling
        js_required_domains = ['stoplight.io', 'swagger.io', 'redoc.ly']
        needs_js = any(domain in source['url'] for domain in js_required_domains)
        
        if needs_js:
            logger.info(f"Using JavaScript rendering for {source['url']}")
            await self.scrape_js_website(source, job_id)
            return
        
        max_depth = config.get('max_depth', 3)
        max_pages = config.get('max_pages', 500)
        crawl_delay = config.get('crawl_delay', 1.0)
        follow_patterns = config.get('follow_patterns', [])
        exclude_patterns = config.get('exclude_patterns', [])
        
        visited = set()
        to_visit = [(source['url'], 0)]  # (url, depth)
        pages_scraped = 0
        
        while to_visit and pages_scraped < max_pages:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
            
            # Check exclude patterns
            if any(re.match(pattern, url) for pattern in exclude_patterns):
                continue
            
            # Check follow patterns (if specified)
            if follow_patterns and not any(re.match(pattern, url) for pattern in follow_patterns):
                continue
            
            try:
                # Fetch the page
                async with self.session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {url}")
                        continue
                    
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' not in content_type:
                        continue
                    
                    html = await response.text()
                    visited.add(url)
                    pages_scraped += 1
                    
                    # Parse and extract content
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract text content
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    text = soup.get_text(separator=' ', strip=True)
                    title = soup.find('title').text if soup.find('title') else url
                    
                    # Extract metadata
                    metadata = {
                        'url': url,
                        'title': title,
                        'scraped_at': datetime.utcnow().isoformat(),
                        'depth': depth,
                        'job_id': job_id
                    }
                    
                    # Store the document
                    await self.store_document(source['id'], text, metadata)
                    
                    # Extract links for further crawling
                    if depth < max_depth:
                        for link in soup.find_all('a', href=True):
                            absolute_url = urljoin(url, link['href'])
                            parsed = urlparse(absolute_url)
                            
                            # Only follow links from the same domain
                            if parsed.netloc == urlparse(source['url']).netloc:
                                to_visit.append((absolute_url, depth + 1))
                    
                    # Respect crawl delay
                    await asyncio.sleep(crawl_delay)
                    
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                continue
        
        logger.info(f"Scraped {pages_scraped} pages from {source['name']}")
    
    async def scrape_js_website(self, source: Dict, job_id: str):
        """Scrape JavaScript-heavy websites using Playwright"""
        config = source.get('config', {})
        max_pages = config.get('max_pages', 500)  # Increase default for API docs
        crawl_delay = config.get('crawl_delay', 1.0)  # Reduce delay
        
        pages_scraped = 0
        visited_urls = set()
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            try:
                # Navigate to the main page
                logger.info(f"Loading JavaScript page: {source['url']}")
                await page.goto(source['url'], wait_until='domcontentloaded', timeout=15000)
                
                # Wait for content to load and sidebar to appear
                await page.wait_for_timeout(5000)
                
                # For Stoplight specifically, we need to find the navigation tree
                # First, let's scrape the current page
                visited_urls.add(source['url'])
                
                # First, we need to expand all navigation items to reveal API endpoints
                logger.info("Expanding Stoplight navigation to find all API endpoints...")
                
                # Click all expandable items in the sidebar to reveal nested endpoints
                await page.evaluate('''() => {
                    // Function to click all expandable elements
                    function expandAll() {
                        // Find all expandable buttons/chevrons in the sidebar
                        const expandables = document.querySelectorAll([
                            'button[aria-expanded="false"]',
                            '[class*="chevron"]:not([aria-expanded="true"])',
                            '[class*="toggle"]:not([aria-expanded="true"])',
                            '[class*="expand"]:not([aria-expanded="true"])',
                            'svg[class*="chevron"]',
                            'div[role="button"][aria-expanded="false"]'
                        ].join(', '));
                        
                        expandables.forEach(el => {
                            try { 
                                el.click();
                                // Some elements might be clickable parents
                                if (el.parentElement && el.parentElement.getAttribute('aria-expanded') === 'false') {
                                    el.parentElement.click();
                                }
                            } catch(e) {}
                        });
                        
                        return expandables.length;
                    }
                    
                    // Expand multiple times as some sections may be nested
                    let totalExpanded = 0;
                    for (let i = 0; i < 5; i++) {
                        const expanded = expandAll();
                        totalExpanded += expanded;
                        if (expanded === 0) break;
                    }
                    
                    return totalExpanded;
                }''')
                
                # Wait for navigation to fully expand
                await page.wait_for_timeout(3000)
                
                # Now extract all links including API endpoints
                all_links = await page.evaluate('''() => {
                    const links = [];
                    const seen = new Set();
                    
                    // Specifically look for API endpoint patterns in Stoplight
                    const apiPatterns = [
                        /\/(get|post|put|patch|delete|options|head)-/i,  // REST verbs in URL
                        /\/[a-z0-9-]+api[a-z0-9-]*$/i,  // API endpoints
                        /\/(v1|v2|v3|api)\//i,  // Version patterns
                        /\/endpoints?\//i,  // Endpoint paths
                        /\/operations?\//i,  // Operation paths
                    ];
                    
                    // Get all links from the navigation
                    const linkElements = document.querySelectorAll([
                        'nav a[href]',
                        '[role="navigation"] a[href]',
                        'aside a[href]',
                        '[class*="sidebar"] a[href]',
                        '[class*="navigation"] a[href]',
                        '[class*="menu"] a[href]',
                        '[class*="tree"] a[href]',
                        '[class*="nav"] a[href]',
                        'a[href*="/docs/"]'
                    ].join(', '));
                    
                    linkElements.forEach(a => {
                        const href = a.href;
                        const text = a.textContent.trim();
                        
                        // Skip if we've seen this URL
                        if (!href || seen.has(href)) return;
                        
                        // Skip non-content links
                        if (href.includes('#') && !href.includes('/docs/')) return;
                        if (href.includes('github.com') || href.includes('twitter.com')) return;
                        
                        // Include if it matches our patterns or is in docs
                        const isApiEndpoint = apiPatterns.some(pattern => pattern.test(href));
                        const isDocsPage = href.includes('/docs/');
                        
                        if (isApiEndpoint || isDocsPage) {
                            seen.add(href);
                            
                            // Try to determine the type of content
                            let contentType = 'page';
                            if (text.match(/^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)/i)) {
                                contentType = 'endpoint';
                            } else if (text.toLowerCase().includes('api')) {
                                contentType = 'api-section';
                            }
                            
                            // Get the parent section for context
                            const section = a.closest('[class*="group"], [class*="section"], li')
                                ?.querySelector('[class*="heading"], [class*="title"]')
                                ?.textContent?.trim() || 'General';
                            
                            links.push({
                                url: href,
                                text: text || 'Untitled',
                                contentType: contentType,
                                section: section
                            });
                        }
                    });
                    
                    // Also look for any OpenAPI/Swagger spec links
                    document.querySelectorAll('a[href*="openapi"], a[href*="swagger"], a[href*=".json"], a[href*=".yaml"]').forEach(a => {
                        if (!seen.has(a.href)) {
                            seen.add(a.href);
                            links.push({
                                url: a.href,
                                text: 'OpenAPI Specification',
                                contentType: 'openapi',
                                section: 'API Specification'
                            });
                        }
                    });
                    
                    return links;
                }''')
                
                logger.info(f"Found {len(all_links)} potential documentation links")
                
                # Process all found links
                for link_data in all_links:
                    if pages_scraped >= max_pages:
                        break
                    
                    link_url = link_data['url'] if isinstance(link_data, dict) else link_data
                    
                    # Skip if already visited
                    if link_url in visited_urls:
                        continue
                    
                    visited_urls.add(link_url)
                    
                    try:
                        logger.info(f"Scraping: {link_url} ({link_data.get('text', 'No title')})")
                        await page.goto(link_url, wait_until='domcontentloaded', timeout=15000)
                        await page.wait_for_timeout(2000)
                        
                        # Wait for content to stabilize (important for Stoplight)
                        await page.wait_for_timeout(1000)
                        
                        # For API endpoints, we need to extract structured information
                        is_api_endpoint = link_data.get('contentType') == 'endpoint' if isinstance(link_data, dict) else False
                        
                        if is_api_endpoint:
                            # Extract API endpoint specific information
                            api_info = await page.evaluate('''() => {
                                const info = {
                                    method: '',
                                    path: '',
                                    description: '',
                                    parameters: [],
                                    requestBody: '',
                                    responses: []
                                };
                                
                                // Look for HTTP method
                                const methodElement = document.querySelector('[class*="http-method"], [class*="method"], [data-method]');
                                if (methodElement) info.method = methodElement.textContent.trim();
                                
                                // Look for endpoint path
                                const pathElement = document.querySelector('[class*="http-path"], [class*="endpoint"], [class*="path"], code');
                                if (pathElement) info.path = pathElement.textContent.trim();
                                
                                // Look for description
                                const descElement = document.querySelector('[class*="description"], [class*="summary"], p');
                                if (descElement) info.description = descElement.textContent.trim();
                                
                                // Extract any code blocks (usually contain examples)
                                document.querySelectorAll('pre, code').forEach(code => {
                                    if (code.textContent.length > 50) {
                                        info.requestBody += code.textContent + '\\n\\n';
                                    }
                                });
                                
                                return info;
                            }''')
                        
                        # Get content
                        content = await page.content()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Extract text
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # For API endpoints, include structured information in the text
                        text = soup.get_text(separator=' ', strip=True)
                        
                        if is_api_endpoint and api_info.get('method'):
                            # Prepend structured API information
                            structured_text = f"API ENDPOINT: {api_info.get('method', '')} {api_info.get('path', '')}\n"
                            structured_text += f"Description: {api_info.get('description', '')}\n\n"
                            structured_text += text
                            text = structured_text
                        
                        title = await page.title()
                        
                        if len(text) > 100:  # Only store if meaningful content
                            metadata = {
                                'url': link_url,
                                'title': title,
                                'scraped_at': datetime.utcnow().isoformat(),
                                'job_id': job_id,
                                'render_method': 'playwright',
                                'content_type': link_data.get('contentType', 'page') if isinstance(link_data, dict) else 'page',
                                'section': link_data.get('section', 'General') if isinstance(link_data, dict) else 'General'
                            }
                            
                            # Add API-specific metadata if available
                            if is_api_endpoint and api_info.get('method'):
                                metadata['api_method'] = api_info.get('method')
                                metadata['api_path'] = api_info.get('path')
                            
                            await self.store_document(source['id'], text, metadata)
                            pages_scraped += 1
                            logger.info(f"Scraped page {pages_scraped}/{max_pages}: {title}")
                        
                        await asyncio.sleep(crawl_delay)
                        
                    except Exception as e:
                        logger.warning(f"Error scraping {link_url}: {str(e)}")
                        continue
                
                # Also scrape the main page if we haven't already
                if source['url'] not in visited_urls or pages_scraped == 0:
                    try:
                        await page.goto(source['url'], wait_until='domcontentloaded', timeout=15000)
                        await page.wait_for_timeout(2000)
                        
                        content = await page.content()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        text = soup.get_text(separator=' ', strip=True)
                        title = await page.title()
                        
                        if len(text) > 100:
                            metadata = {
                                'url': source['url'],
                                'title': title,
                                'scraped_at': datetime.utcnow().isoformat(),
                                'job_id': job_id,
                                'render_method': 'playwright'
                            }
                            
                            await self.store_document(source['id'], text, metadata)
                            pages_scraped += 1
                    except Exception as e:
                        logger.error(f"Error scraping main page: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error with Playwright scraping: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                await browser.close()
        
        logger.info(f"Scraped {pages_scraped} pages from {source['name']} using JavaScript rendering")
        
        # Update source stats to reflect scraped pages
        try:
            async with self.session.patch(
                f"{self.api_base}/api/sources/{source['id']}/stats",
                json={'documents': pages_scraped}
            ) as response:
                if response.status == 200:
                    logger.info(f"Updated source stats: {pages_scraped} documents")
        except Exception as e:
            logger.warning(f"Could not update source stats: {e}")
    
    async def scrape_documentation(self, source: Dict, job_id: str):
        """Scrape documentation sites (similar to website but with special handling)"""
        # Documentation sites often have special structure
        # This is a simplified version - you'd want to handle:
        # - API references
        # - Code examples extraction
        # - Version handling
        # - Better structure preservation
        await self.scrape_website(source, job_id)
    
    async def scrape_repository(self, source: Dict, job_id: str):
        """Scrape a code repository"""
        # This would integrate with GitHub/GitLab APIs
        # Extract:
        # - README files
        # - Documentation
        # - Code structure
        # - Issues/PRs for context
        logger.info(f"Repository scraping not yet implemented for {source['name']}")
    
    async def scrape_api(self, source: Dict, job_id: str):
        """Scrape an API endpoint"""
        # This would handle REST/GraphQL APIs
        # Extract structured data and convert to documents
        logger.info(f"API scraping not yet implemented for {source['name']}")
    
    async def store_document(self, source_id: str, content: str, metadata: Dict):
        """Store document and create chunks for vector search"""
        # Create document
        doc_data = {
            'source_id': source_id,
            'content': content,
            'metadata': metadata,
            'url': metadata['url'],
            'title': metadata.get('title', ''),
            'hash': hashlib.sha256(content.encode()).hexdigest()
        }
        
        # Store via API
        async with self.session.post(
            f"{self.api_base}/api/documents/",
            json=doc_data
        ) as response:
            if response.status == 200:
                doc = await response.json()
                logger.info(f"Stored document: {doc['id']}")
                
                # Create chunks for vector search
                chunks = self.create_chunks(content, chunk_size=1000, overlap=200)
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'document_id': doc['id'],
                        'content': chunk,
                        'position': i,
                        'metadata': {
                            'url': metadata['url'],
                            'title': metadata['title'],
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }
                    }
                    
                    # Store chunk
                    async with self.session.post(
                        f"{self.api_base}/api/chunks/",
                        json=chunk_data
                    ) as chunk_response:
                        if chunk_response.status == 200:
                            logger.debug(f"Stored chunk {i+1}/{len(chunks)}")
    
    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for vector search"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    async def update_job_status(self, job_id: str, status: str, error: Optional[str] = None):
        """Update job status via API"""
        data = {
            'status': status,
            'updated_at': datetime.utcnow().isoformat()
        }
        if error:
            data['error'] = error
        
        async with self.session.patch(
            f"{self.api_base}/api/jobs/{job_id}",
            json=data
        ) as response:
            if response.status == 200:
                logger.info(f"Updated job {job_id} status to {status}")

async def main():
    """Main entry point"""
    # Configuration from environment or config file
    REDIS_URL = "redis://192.168.1.25:6381"  # Adjust as needed
    API_BASE = "http://192.168.1.25:3000"
    
    worker = ScraperWorker(REDIS_URL, API_BASE)
    await worker.start()

if __name__ == "__main__":
    asyncio.run(main())