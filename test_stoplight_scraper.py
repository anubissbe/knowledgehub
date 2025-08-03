import asyncio
from playwright.async_api import async_playwright
import json

async def test_stoplight_scraping():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Loading Stoplight page...")
        await page.goto('https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction', wait_until='networkidle')
        
        # Wait for content to load
        await page.wait_for_timeout(5000)
        
        print("\nTrying to expand navigation...")
        # Try different ways to expand the navigation
        
        # Method 1: Click on expandable items
        expand_buttons = await page.query_selector_all('[aria-expanded="false"]')
        print(f"Found {len(expand_buttons)} expandable items")
        
        for button in expand_buttons[:5]:  # Expand first 5
            try:
                await button.click()
                await page.wait_for_timeout(500)
            except:
                pass
        
        # Method 2: Look for navigation structure
        print("\nAnalyzing page structure...")
        structure = await page.evaluate('''() => {
            const info = {
                title: document.title,
                url: window.location.href,
                navElements: [],
                links: []
            };
            
            // Find navigation elements
            const navSelectors = [
                'nav', '[role="navigation"]', 'aside', '[class*="sidebar"]',
                '[class*="navigation"]', '[class*="menu"]', '[class*="tree"]',
                '[class*="sl-"]', '.sl-flex', '[data-testid*="nav"]',
                '[class*="TableOfContents"]', '[class*="toc"]'
            ];
            
            navSelectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    info.navElements.push({
                        selector: selector,
                        count: elements.length,
                        firstClass: elements[0].className
                    });
                }
            });
            
            // Get all links
            const allLinks = document.querySelectorAll('a[href]');
            const linkMap = new Map();
            
            allLinks.forEach(link => {
                const href = link.href;
                const text = link.textContent.trim();
                if (href && !href.includes('javascript:') && !href.includes('mailto:')) {
                    if (!linkMap.has(href)) {
                        linkMap.set(href, {
                            url: href,
                            text: text,
                            classes: link.className
                        });
                    }
                }
            });
            
            info.links = Array.from(linkMap.values())
                .filter(link => link.url.includes('checkmarx'))
                .slice(0, 50); // First 50 links
            
            return info;
        }''')
        
        print(f"\nPage Title: {structure['title']}")
        print(f"Current URL: {structure['url']}")
        print(f"\nNavigation Elements Found:")
        for nav in structure['navElements']:
            print(f"  - {nav['selector']}: {nav['count']} elements (class: {nav['firstClass']})")
        
        print(f"\nTotal Checkmarx Links Found: {len(structure['links'])}")
        print("\nSample Links:")
        for link in structure['links'][:10]:
            print(f"  - {link['text'][:50]}: {link['url']}")
        
        # Method 3: Look for specific Stoplight patterns
        print("\n\nLooking for API endpoints...")
        api_links = await page.evaluate('''() => {
            const links = [];
            
            // Look for HTTP method indicators
            const methodPatterns = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'];
            const allElements = document.querySelectorAll('*');
            
            allElements.forEach(el => {
                const text = el.textContent || '';
                methodPatterns.forEach(method => {
                    if (text.includes(method + ' /') || text.includes(method + ' ')) {
                        const parent = el.closest('a') || el.parentElement?.closest('a');
                        if (parent && parent.href) {
                            links.push({
                                method: method,
                                text: text.substring(0, 100),
                                url: parent.href
                            });
                        }
                    }
                });
            });
            
            return links.slice(0, 20);
        }''')
        
        print(f"\nAPI Endpoints Found: {len(api_links)}")
        for link in api_links[:10]:
            print(f"  - {link['method']}: {link['text'][:60]}...")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_stoplight_scraping())