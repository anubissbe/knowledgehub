#!/usr/bin/env python3
"""
Enhanced Stoplight.io scraper with better navigation handling
"""

import asyncio
from playwright.async_api import async_playwright
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_stoplight_scraping():
    """Test scraping Stoplight documentation with enhanced selectors"""
    
    url = "https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        
        try:
            logger.info(f"Loading page: {url}")
            await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for the main content to load
            logger.info("Waiting for content to load...")
            await page.wait_for_timeout(5000)
            
            # Take a screenshot for debugging
            await page.screenshot(path="stoplight_loaded.png")
            
            # Try different methods to find navigation links
            logger.info("Looking for navigation elements...")
            
            # Method 1: Look for Stoplight-specific navigation
            nav_links_v1 = await page.evaluate('''() => {
                const links = [];
                
                // Stoplight uses specific class patterns
                const selectors = [
                    'a[href*="/docs/"]',
                    '[class*="sl-"] a',
                    '[data-testid*="nav"] a',
                    '[class*="TableOfContents"] a',
                    '[class*="toc"] a',
                    '[class*="sidebar"] a',
                    '[class*="navigation"] a',
                    '.sl-flex a[href]'
                ];
                
                const seen = new Set();
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(a => {
                        const href = a.href;
                        if (href && !seen.has(href) && href.includes('/docs/')) {
                            seen.add(href);
                            links.push({
                                url: href,
                                text: a.textContent.trim(),
                                classes: a.className
                            });
                        }
                    });
                });
                
                return links;
            }''')
            
            logger.info(f"Method 1 found {len(nav_links_v1)} links")
            
            # Method 2: Look for expandable sections and click them
            logger.info("Expanding navigation sections...")
            expanded_count = await page.evaluate('''() => {
                let count = 0;
                
                // Find all expandable elements
                const expandables = document.querySelectorAll([
                    '[aria-expanded="false"]',
                    '[class*="chevron"]:not([aria-expanded="true"])',
                    '[class*="collapse"]:not(.show)',
                    'button[class*="expand"]',
                    'button[class*="toggle"]'
                ].join(', '));
                
                expandables.forEach(el => {
                    try {
                        el.click();
                        count++;
                    } catch (e) {}
                });
                
                return count;
            }''')
            
            logger.info(f"Expanded {expanded_count} sections")
            await page.wait_for_timeout(2000)
            
            # Method 3: After expansion, get all links again
            all_links = await page.evaluate('''() => {
                const links = [];
                const seen = new Set();
                
                // Get all links after expansion
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.href;
                    const text = a.textContent.trim();
                    
                    if (href && !seen.has(href)) {
                        // Filter for relevant documentation links
                        if (href.includes('/docs/') && 
                            !href.includes('#') && 
                            !href.includes('github.com') &&
                            !href.includes('twitter.com')) {
                            
                            seen.add(href);
                            
                            // Detect if it's an API endpoint
                            const isEndpoint = text.match(/^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)/i) ||
                                             href.match(/\/(get|post|put|patch|delete)-/i);
                            
                            links.push({
                                url: href,
                                text: text || 'Untitled',
                                type: isEndpoint ? 'endpoint' : 'page'
                            });
                        }
                    }
                });
                
                return links;
            }''')
            
            logger.info(f"Total links found: {len(all_links)}")
            
            # Use nav_links_v1 if all_links is empty
            final_links = all_links if all_links else nav_links_v1
            
            # Print first 10 links for debugging
            for i, link in enumerate(final_links[:10]):
                link_type = link.get('type', 'unknown')
                logger.info(f"  {i+1}. [{link_type}] {link['text']} -> {link['url']}")
            
            # Method 4: Check if there's a table of contents or sitemap
            toc_exists = await page.evaluate('''() => {
                const tocSelectors = [
                    '[class*="table-of-contents"]',
                    '[class*="toc"]',
                    '[class*="TableOfContents"]',
                    '[role="navigation"]',
                    'nav'
                ];
                
                for (const selector of tocSelectors) {
                    const el = document.querySelector(selector);
                    if (el && el.querySelectorAll('a').length > 0) {
                        return true;
                    }
                }
                return false;
            }''')
            
            logger.info(f"Table of contents found: {toc_exists}")
            
            # Take another screenshot after expansion
            await page.screenshot(path="stoplight_expanded.png")
            
            return final_links
            
        finally:
            await browser.close()

if __name__ == "__main__":
    links = asyncio.run(test_stoplight_scraping())
    logger.info(f"\nSummary: Found {len(links)} total links")