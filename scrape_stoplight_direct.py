#!/usr/bin/env python3
"""
Direct scraper for Stoplight API documentation
"""

import asyncio
from playwright.async_api import async_playwright
import sys
sys.path.insert(0, '.')
from scraper_db_storage import store_scraped_document
from api.models import get_db
from api.models.source import KnowledgeSource
from api.models.knowledge_source import SourceStatus
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def scrape_stoplight_api_docs(source_id: str = "a2ef8910-0b25-4138-abcb-428666ce691d"):
    """Scrape Checkmarx Stoplight API documentation"""
    
    db = next(get_db())
    source = db.query(KnowledgeSource).filter(KnowledgeSource.id == source_id).first()
    
    if not source:
        logger.error(f"Source {source_id} not found")
        return
    
    # Update source status
    source.status = SourceStatus.CRAWLING
    db.commit()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        try:
            logger.info(f"Loading Stoplight page: {source.url}")
            await page.goto(source.url, wait_until='domcontentloaded', timeout=30000)
            await page.wait_for_timeout(5000)
            
            # Expand navigation
            logger.info("Expanding navigation...")
            await page.evaluate('''() => {
                // Click all expandable elements
                const expandables = document.querySelectorAll('[aria-expanded="false"], button[class*="expand"], button[class*="toggle"]');
                expandables.forEach(el => {
                    try { el.click(); } catch(e) {}
                });
            }''')
            await page.wait_for_timeout(2000)
            
            # Get all documentation links
            all_links = await page.evaluate('''() => {
                const links = [];
                const seen = new Set();
                
                // Find all documentation links
                document.querySelectorAll('a[href*="/docs/"]').forEach(a => {
                    const href = a.href;
                    const text = a.textContent.trim();
                    
                    if (href && !seen.has(href) && !href.includes('#') && href.includes('checkmarx')) {
                        seen.add(href);
                        
                        // Detect API endpoints
                        const isEndpoint = text.match(/^(GET|POST|PUT|PATCH|DELETE)/i) || 
                                         href.includes('/get-') || href.includes('/post-') || 
                                         href.includes('/put-') || href.includes('/delete-');
                        
                        links.push({
                            url: href,
                            text: text || 'Untitled',
                            type: isEndpoint ? 'endpoint' : 'page'
                        });
                    }
                });
                
                return links;
            }''')
            
            logger.info(f"Found {len(all_links)} documentation links")
            
            # Scrape each page
            pages_scraped = 0
            for link in all_links:
                try:
                    logger.info(f"Scraping [{link['type']}]: {link['text']}")
                    await page.goto(link['url'], wait_until='domcontentloaded', timeout=15000)
                    await page.wait_for_timeout(2000)
                    
                    # Get page content
                    content = await page.evaluate('''() => {
                        // Remove navigation and footer
                        const nav = document.querySelector('nav, [role="navigation"], aside');
                        if (nav) nav.remove();
                        
                        const footer = document.querySelector('footer');
                        if (footer) footer.remove();
                        
                        // Get main content
                        const main = document.querySelector('main, [role="main"], article, .sl-elements-article');
                        return main ? main.innerText : document.body.innerText;
                    }''')
                    
                    # Get title
                    title = await page.title()
                    
                    # Store document
                    if content and len(content.strip()) > 50:
                        doc_id = store_scraped_document(
                            source_id=source_id,
                            url=link['url'],
                            title=title,
                            content=content,
                            metadata={
                                'type': link['type'],
                                'original_text': link['text']
                            }
                        )
                        logger.info(f"Stored document: {doc_id}")
                        pages_scraped += 1
                    
                except Exception as e:
                    logger.error(f"Error scraping {link['url']}: {e}")
                    continue
            
            logger.info(f"Scraped {pages_scraped} pages total")
            
            # Update source status
            source.status = SourceStatus.COMPLETED
            source.last_scraped_at = datetime.now(timezone.utc)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            source.status = SourceStatus.FAILED
            db.commit()
            
        finally:
            await browser.close()
            db.close()

if __name__ == "__main__":
    import sys
    from datetime import datetime, timezone
    
    source_id = sys.argv[1] if len(sys.argv) > 1 else "a2ef8910-0b25-4138-abcb-428666ce691d"
    asyncio.run(scrape_stoplight_api_docs(source_id))