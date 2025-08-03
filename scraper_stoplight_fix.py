"""
Enhanced Stoplight scraper that properly discovers all API documentation pages
"""
import asyncio
from playwright.async_api import async_playwright
import json
from urllib.parse import urljoin, urlparse
import time

async def scrape_stoplight_api_docs(base_url):
    """
    Scrape Stoplight documentation with improved discovery
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        discovered_urls = set()
        visited_urls = set()
        
        print(f"Loading initial page: {base_url}")
        
        try:
            # Load the main page
            await page.goto(base_url, wait_until='domcontentloaded', timeout=60000)
            await page.wait_for_timeout(5000)  # Let JS fully load
            
            # Method 1: Extract from the table of contents / navigation
            print("\nExtracting navigation structure...")
            nav_links = await page.evaluate('''() => {
                const links = new Map();
                const baseUrl = window.location.origin;
                
                // Find all links in navigation areas
                const selectors = [
                    // Generic navigation
                    'nav a', '[role="navigation"] a', 'aside a',
                    // Stoplight specific
                    '[class*="TableOfContents"] a',
                    '[class*="toc"] a',
                    '[class*="sidebar"] a',
                    '[class*="sl-"] a',
                    // Links with /docs/ in them
                    'a[href*="/docs/"]'
                ];
                
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(link => {
                        const href = link.getAttribute('href');
                        if (!href) return;
                        
                        // Convert to absolute URL
                        let fullUrl;
                        if (href.startsWith('http')) {
                            fullUrl = href;
                        } else if (href.startsWith('/')) {
                            fullUrl = baseUrl + href;
                        } else {
                            fullUrl = new URL(href, window.location.href).href;
                        }
                        
                        // Only include checkmarx docs
                        if (fullUrl.includes('checkmarx') && fullUrl.includes('/docs/')) {
                            const text = link.textContent.trim();
                            links.set(fullUrl, text);
                        }
                    });
                });
                
                return Array.from(links.entries()).map(([url, text]) => ({url, text}));
            }''')
            
            print(f"Found {len(nav_links)} navigation links")
            for link in nav_links:
                discovered_urls.add(link['url'])
            
            # Method 2: Look for API endpoint patterns in the current page
            print("\nLooking for API endpoints on current page...")
            api_endpoints = await page.evaluate('''() => {
                const endpoints = [];
                const baseUrl = window.location.origin;
                
                // Find all elements that look like API endpoints
                document.querySelectorAll('*').forEach(el => {
                    const text = el.textContent || '';
                    
                    // Match REST API patterns
                    const match = text.match(/^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\\s+(\\/[^\\s]*)/);
                    if (match) {
                        // Try to find the associated link
                        let linkEl = el.closest('a');
                        if (!linkEl) {
                            // Sometimes the link is a sibling or parent
                            linkEl = el.parentElement?.querySelector('a');
                        }
                        
                        if (linkEl && linkEl.href) {
                            endpoints.push({
                                method: match[1],
                                path: match[2],
                                url: linkEl.href,
                                text: text.substring(0, 100)
                            });
                        }
                    }
                });
                
                return endpoints;
            }''')
            
            print(f"Found {len(api_endpoints)} API endpoints on current page")
            for endpoint in api_endpoints:
                discovered_urls.add(endpoint['url'])
            
            # Method 3: Expand all collapsible sections
            print("\nExpanding collapsible sections...")
            expanded = await page.evaluate('''() => {
                let count = 0;
                
                // Click all expandable elements
                const expandSelectors = [
                    '[aria-expanded="false"]',
                    'button[aria-expanded="false"]',
                    'div[aria-expanded="false"]',
                    '[class*="collaps"][aria-expanded="false"]',
                    '[class*="expand"][aria-expanded="false"]'
                ];
                
                expandSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        try {
                            el.click();
                            count++;
                        } catch (e) {}
                    });
                });
                
                return count;
            }''')
            
            print(f"Expanded {expanded} sections")
            await page.wait_for_timeout(2000)
            
            # Re-extract links after expansion
            expanded_links = await page.evaluate('''() => {
                const links = new Set();
                document.querySelectorAll('a[href*="/docs/"]').forEach(a => {
                    if (a.href.includes('checkmarx')) {
                        links.add(a.href);
                    }
                });
                return Array.from(links);
            }''')
            
            for link in expanded_links:
                discovered_urls.add(link)
            
            print(f"\nTotal unique URLs discovered: {len(discovered_urls)}")
            
            # Method 4: Check for sitemap or API index
            print("\nChecking for API index or sitemap...")
            
            # Look for links to API reference, index, or overview pages
            index_patterns = [
                'api-reference', 'api-overview', 'api-index',
                'reference', 'endpoints', 'operations',
                'all-endpoints', 'complete-reference'
            ]
            
            potential_index_links = await page.evaluate('''(patterns) => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.href.toLowerCase();
                    const text = a.textContent.toLowerCase();
                    
                    patterns.forEach(pattern => {
                        if (href.includes(pattern) || text.includes(pattern)) {
                            links.push({url: a.href, text: a.textContent});
                        }
                    });
                });
                return links;
            }''', index_patterns)
            
            print(f"Found {len(potential_index_links)} potential index pages")
            
            # Visit index pages to discover more endpoints
            for index_link in potential_index_links[:3]:  # Visit top 3
                try:
                    print(f"\nVisiting index page: {index_link['text']}")
                    await page.goto(index_link['url'], wait_until='domcontentloaded', timeout=30000)
                    await page.wait_for_timeout(3000)
                    
                    # Extract links from index page
                    index_urls = await page.evaluate('''() => {
                        const urls = new Set();
                        document.querySelectorAll('a[href*="/docs/"]').forEach(a => {
                            if (a.href.includes('checkmarx')) {
                                urls.add(a.href);
                            }
                        });
                        return Array.from(urls);
                    }''')
                    
                    for url in index_urls:
                        discovered_urls.add(url)
                    
                    print(f"  Found {len(index_urls)} more URLs")
                    
                except Exception as e:
                    print(f"  Error visiting index page: {e}")
            
        except Exception as e:
            print(f"Error during scraping: {e}")
        
        finally:
            await browser.close()
        
        # Save results
        results = {
            'base_url': base_url,
            'total_discovered': len(discovered_urls),
            'urls': list(discovered_urls),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('stoplight_discovered_urls.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"DISCOVERY COMPLETE")
        print(f"Total unique URLs found: {len(discovered_urls)}")
        print(f"Results saved to: stoplight_discovered_urls.json")
        print(f"{'='*60}")
        
        # Show sample URLs
        print("\nSample discovered URLs:")
        for i, url in enumerate(list(discovered_urls)[:20]):
            print(f"{i+1}. {url}")
        
        return discovered_urls

async def main():
    base_url = "https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction"
    urls = await scrape_stoplight_api_docs(base_url)
    
    # Group URLs by type
    api_endpoints = []
    guide_pages = []
    
    for url in urls:
        if any(method in url.lower() for method in ['get-', 'post-', 'put-', 'delete-', 'patch-']):
            api_endpoints.append(url)
        else:
            guide_pages.append(url)
    
    print(f"\n\nURL Categories:")
    print(f"API Endpoints: {len(api_endpoints)}")
    print(f"Guide Pages: {len(guide_pages)}")
    
    if len(urls) < 20:
        print("\n⚠️  Warning: Found fewer than expected URLs.")
        print("The Stoplight site might be using advanced client-side rendering.")
        print("Consider using the Stoplight API or export functionality if available.")

if __name__ == "__main__":
    asyncio.run(main())