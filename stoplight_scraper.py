#!/usr/bin/env python3
"""
Specialized scraper for Stoplight API documentation sites.
This scraper is designed to handle the complex JavaScript navigation
and dynamic content loading that Stoplight uses.
"""

import asyncio
import json
import logging
from datetime import datetime
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import aiohttp
import hashlib
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoplightScraper:
    def __init__(self, api_base="http://localhost:3000"):
        self.api_base = api_base
        self.session = None
        
    async def scrape_stoplight_site(self, source_id: str, url: str):
        """Scrape a Stoplight documentation site comprehensively"""
        self.session = aiohttp.ClientSession()
        pages_scraped = 0
        visited_urls: Set[str] = set()
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,  # Run in headless mode
                    args=['--disable-blink-features=AutomationControlled']
                )
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = await context.new_page()
                
                # Enable console logging for debugging
                page.on("console", lambda msg: logger.debug(f"Browser console: {msg.text}"))
                
                logger.info(f"Loading Stoplight site: {url}")
                await page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Wait for the sidebar to load
                logger.info("Waiting for navigation sidebar to load...")
                try:
                    await page.wait_for_selector('[class*="sidebar"], [class*="navigation"], aside', timeout=10000)
                except:
                    logger.warning("Sidebar selector not found, continuing anyway")
                
                # Additional wait for dynamic content
                await page.wait_for_timeout(5000)
                
                # Try multiple strategies to find and expand navigation items
                logger.info("Attempting to expand navigation items...")
                
                # Strategy 1: Click all elements with aria-expanded="false"
                expanded_count = await page.evaluate('''() => {
                    let count = 0;
                    const expandables = document.querySelectorAll('[aria-expanded="false"]');
                    expandables.forEach(el => {
                        try {
                            el.click();
                            count++;
                        } catch (e) {}
                    });
                    return count;
                }''')
                logger.info(f"Expanded {expanded_count} items using aria-expanded")
                
                await page.wait_for_timeout(2000)
                
                # Strategy 2: Click all chevron/arrow icons
                expanded_count2 = await page.evaluate('''() => {
                    let count = 0;
                    // Look for various chevron/arrow patterns
                    const selectors = [
                        'svg[class*="chevron"]',
                        'svg[class*="arrow"]',
                        'svg[class*="caret"]',
                        '[class*="chevron"]',
                        '[class*="arrow-right"]',
                        '[class*="expand"]',
                        'button:has(svg)'
                    ];
                    
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => {
                            try {
                                // Check if parent or element has collapsed state
                                const parent = el.closest('[aria-expanded]');
                                if (parent && parent.getAttribute('aria-expanded') === 'false') {
                                    parent.click();
                                    count++;
                                } else if (el.tagName === 'BUTTON' || el.onclick) {
                                    el.click();
                                    count++;
                                }
                            } catch (e) {}
                        });
                    });
                    return count;
                }''')
                logger.info(f"Expanded {expanded_count2} additional items using chevron selectors")
                
                await page.wait_for_timeout(3000)
                
                # Now extract all links
                logger.info("Extracting all documentation links...")
                all_links = await page.evaluate('''() => {
                    const links = [];
                    const seen = new Set();
                    
                    // Get all links from various possible containers
                    const linkSelectors = [
                        'a[href*="/docs/"]',
                        'aside a[href]',
                        'nav a[href]',
                        '[class*="sidebar"] a[href]',
                        '[class*="navigation"] a[href]',
                        '[class*="menu"] a[href]',
                        '[class*="tree"] a[href]',
                        '[role="navigation"] a[href]',
                        'ul li a[href]'
                    ];
                    
                    linkSelectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(a => {
                            const href = a.href;
                            const text = a.textContent.trim();
                            
                            // Only include documentation links
                            if (href && 
                                !seen.has(href) && 
                                href.includes('/docs/') &&
                                !href.includes('#') &&
                                !href.includes('github.com') &&
                                !href.includes('twitter.com')) {
                                
                                seen.add(href);
                                
                                // Categorize the link
                                let category = 'General';
                                let linkType = 'page';
                                
                                // Check if it's an API endpoint
                                if (text.match(/^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/)) {
                                    linkType = 'endpoint';
                                    category = 'API Endpoints';
                                } else if (text.toLowerCase().includes('api')) {
                                    linkType = 'api-section';
                                    category = 'API Documentation';
                                }
                                
                                // Try to get the parent section
                                const section = a.closest('li')?.parentElement?.previousElementSibling?.textContent?.trim() ||
                                               a.closest('[class*="group"]')?.querySelector('[class*="heading"]')?.textContent?.trim() ||
                                               'General';
                                
                                links.push({
                                    url: href,
                                    text: text || 'Untitled',
                                    type: linkType,
                                    category: category,
                                    section: section
                                });
                            }
                        });
                    });
                    
                    return links;
                }''')
                
                logger.info(f"Found {len(all_links)} documentation links")
                
                # If we didn't find many links, try a different approach
                if len(all_links) < 10:
                    logger.info("Few links found, trying alternative extraction method...")
                    
                    # Take a screenshot for debugging
                    await page.screenshot(path='/tmp/stoplight_debug.png')
                    
                    # Try to get links from the page content itself
                    page_links = await page.evaluate('''() => {
                        const content = document.body.innerText;
                        const urls = [];
                        
                        // Look for patterns that might be API endpoints
                        const patterns = [
                            /\/api\/v\d+\/[\w-]+/g,
                            /\/[\w-]+\/[\w-]+\/[\w-]+/g
                        ];
                        
                        patterns.forEach(pattern => {
                            const matches = content.match(pattern);
                            if (matches) {
                                matches.forEach(match => {
                                    if (!urls.includes(match)) {
                                        urls.push(match);
                                    }
                                });
                            }
                        });
                        
                        return urls;
                    }''')
                    
                    logger.info(f"Found {len(page_links)} potential API paths in content")
                
                # Process the main page first
                logger.info("Scraping main page...")
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator=' ', strip=True)
                
                if len(text) > 100:
                    await self.store_document(source_id, text, {
                        'url': url,
                        'title': await page.title(),
                        'scraped_at': datetime.utcnow().isoformat(),
                        'type': 'main-page'
                    })
                    pages_scraped += 1
                    visited_urls.add(url)
                
                # Process all found links
                for link_data in all_links:
                    if link_data['url'] in visited_urls:
                        continue
                        
                    try:
                        logger.info(f"Scraping: {link_data['text']} ({link_data['type']})")
                        await page.goto(link_data['url'], wait_until='domcontentloaded', timeout=20000)
                        await page.wait_for_timeout(2000)
                        
                        # Extract content
                        content = await page.content()
                        soup = BeautifulSoup(content, 'html.parser')
                        for script in soup(["script", "style"]):
                            script.decompose()
                        text = soup.get_text(separator=' ', strip=True)
                        
                        # For API endpoints, try to extract structured information
                        if link_data['type'] == 'endpoint':
                            api_details = await self.extract_api_details(page)
                            if api_details:
                                text = f"API ENDPOINT: {api_details}\n\n{text}"
                        
                        if len(text) > 100:
                            await self.store_document(source_id, text, {
                                'url': link_data['url'],
                                'title': await page.title(),
                                'scraped_at': datetime.utcnow().isoformat(),
                                'type': link_data['type'],
                                'category': link_data['category'],
                                'section': link_data['section']
                            })
                            pages_scraped += 1
                            visited_urls.add(link_data['url'])
                            
                            # Respect rate limits
                            await asyncio.sleep(1)
                            
                    except Exception as e:
                        logger.error(f"Error scraping {link_data['url']}: {str(e)}")
                        continue
                
                logger.info(f"Completed scraping. Total pages: {pages_scraped}")
                await browser.close()
                
        finally:
            await self.session.close()
            
        return pages_scraped
    
    async def extract_api_details(self, page):
        """Extract API endpoint details from the page"""
        try:
            details = await page.evaluate('''() => {
                const info = [];
                
                // Look for HTTP method
                const method = document.querySelector('[class*="method"], [class*="http-method"]')?.textContent?.trim();
                if (method) info.push(`Method: ${method}`);
                
                // Look for path
                const path = document.querySelector('[class*="path"], [class*="endpoint"], code')?.textContent?.trim();
                if (path) info.push(`Path: ${path}`);
                
                // Look for description
                const desc = document.querySelector('[class*="description"], p')?.textContent?.trim()?.substring(0, 200);
                if (desc) info.push(`Description: ${desc}`);
                
                return info.join(' | ');
            }''')
            return details
        except:
            return None
    
    async def store_document(self, source_id: str, content: str, metadata: Dict):
        """Store document via API"""
        doc_data = {
            'source_id': source_id,
            'content': content,
            'metadata': metadata,
            'url': metadata['url'],
            'title': metadata.get('title', ''),
            'hash': hashlib.sha256(content.encode()).hexdigest()
        }
        
        async with self.session.post(
            f"{self.api_base}/api/documents/",
            json=doc_data
        ) as response:
            if response.status == 200:
                doc = await response.json()
                logger.info(f"Stored document: {doc.get('id', 'unknown')}")
                
                # Create chunks
                chunks = self.create_chunks(content)
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'document_id': doc['id'],
                        'content': chunk,
                        'position': i,
                        'chunk_type': 'text',
                        'metadata': {
                            'url': metadata['url'],
                            'title': metadata.get('title', ''),
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }
                    }
                    
                    async with self.session.post(
                        f"{self.api_base}/api/chunks/",
                        json=chunk_data
                    ) as chunk_response:
                        if chunk_response.status != 200:
                            logger.warning(f"Failed to store chunk {i}")
    
    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks

async def main():
    """Test the Stoplight scraper"""
    scraper = StoplightScraper(api_base="http://192.168.1.25:3000")
    
    # Checkmarx Stoplight API documentation
    source_id = "563f2058-42d1-48a3-9187-59f06dd64f39"
    url = "https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/3w7wczsazj6pg-introduction"
    
    pages = await scraper.scrape_stoplight_site(source_id, url)
    print(f"Scraped {pages} pages")

if __name__ == "__main__":
    asyncio.run(main())