#!/usr/bin/env python3
"""
Enhanced script to find ALL URLs from Checkmarx Stoplight API documentation
"""

import asyncio
import json
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import time
import re

async def deep_navigation_scan(page):
    """Perform a deep scan of the navigation structure"""
    print("Performing deep navigation scan...")
    
    # Get all navigation structure details
    nav_structure = await page.evaluate("""
        () => {
            const structure = {
                sections: [],
                links: [],
                expandableItems: [],
                hiddenContent: []
            };
            
            // Find all section headers
            const sectionSelectors = [
                '.sl-heading',
                'h2', 'h3', 'h4',
                '[role="heading"]',
                '.sl-text-heading'
            ];
            
            sectionSelectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => {
                    structure.sections.push({
                        text: el.textContent.trim(),
                        parent: el.parentElement?.className
                    });
                });
            });
            
            // Find all expandable items
            const expandables = document.querySelectorAll(
                '[aria-expanded], [data-expanded], .expandable, .collapsible'
            );
            expandables.forEach(el => {
                structure.expandableItems.push({
                    expanded: el.getAttribute('aria-expanded') || el.getAttribute('data-expanded'),
                    text: el.textContent.substring(0, 50)
                });
            });
            
            // Check for version dropdowns or branch selectors
            const versionSelectors = document.querySelectorAll(
                'select, [role="combobox"], .version-selector, .branch-selector'
            );
            versionSelectors.forEach(el => {
                if (el.tagName === 'SELECT') {
                    structure.hiddenContent.push({
                        type: 'dropdown',
                        options: Array.from(el.options).map(o => o.text)
                    });
                }
            });
            
            return structure;
        }
    """)
    
    return nav_structure

async def search_for_api_patterns(page):
    """Search for API endpoint patterns in the page content"""
    print("Searching for API endpoint patterns...")
    
    api_patterns = await page.evaluate("""
        () => {
            const patterns = {
                restEndpoints: [],
                apiCategories: [],
                hiddenApis: []
            };
            
            // Look for REST endpoint patterns
            const endpointRegex = /\/(api|v\\d+|rest)\/[\\w-\/]+/gi;
            const textContent = document.body.innerText;
            const matches = textContent.match(endpointRegex) || [];
            patterns.restEndpoints = [...new Set(matches)];
            
            // Find API category mentions
            const categoryRegex = /(\\w+)\\s+(API|Service|Management|REST)/gi;
            const categoryMatches = textContent.match(categoryRegex) || [];
            patterns.apiCategories = [...new Set(categoryMatches)];
            
            // Check scripts for hidden API references
            const scripts = document.querySelectorAll('script');
            scripts.forEach(script => {
                if (script.textContent.includes('api') || script.textContent.includes('endpoint')) {
                    const apiRefs = script.textContent.match(/["']([^"']*api[^"']*)["']/gi) || [];
                    patterns.hiddenApis = patterns.hiddenApis.concat(apiRefs.slice(0, 5));
                }
            });
            
            return patterns;
        }
    """)
    
    return api_patterns

async def click_and_wait(page, selector, wait_time=500):
    """Click an element and wait for content to load"""
    try:
        element = await page.query_selector(selector)
        if element:
            await element.click()
            await asyncio.sleep(wait_time / 1000)
            return True
    except:
        pass
    return False

async def expand_everything_aggressively(page):
    """Aggressively try to expand all collapsible content"""
    print("Aggressively expanding all content...")
    
    # Multiple rounds of expansion with different strategies
    strategies = [
        # Strategy 1: Click anything that looks like an arrow or chevron
        """
        document.querySelectorAll('svg, [class*="chevron"], [class*="arrow"], [class*="expand"]').forEach(el => {
            try { el.click(); } catch(e) {}
            try { el.parentElement.click(); } catch(e) {}
        });
        """,
        
        # Strategy 2: Click parent elements of SVGs
        """
        document.querySelectorAll('button:has(svg), div:has(svg)').forEach(el => {
            if (el.querySelector('path')) {
                try { el.click(); } catch(e) {}
            }
        });
        """,
        
        # Strategy 3: Trigger keyboard events on navigation items
        """
        document.querySelectorAll('nav [tabindex], nav button, nav [role="button"]').forEach(el => {
            try {
                el.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
                el.dispatchEvent(new KeyboardEvent('keydown', {key: ' ', bubbles: true}));
            } catch(e) {}
        });
        """,
        
        # Strategy 4: Force aria-expanded to true
        """
        document.querySelectorAll('[aria-expanded="false"]').forEach(el => {
            el.setAttribute('aria-expanded', 'true');
            try { el.click(); } catch(e) {}
        });
        """
    ]
    
    for i, strategy in enumerate(strategies):
        print(f"  Applying expansion strategy {i+1}...")
        await page.evaluate(strategy)
        await asyncio.sleep(0.5)

async def extract_all_possible_urls(page):
    """Extract every possible URL from the page"""
    print("Extracting all possible URLs...")
    
    all_urls = await page.evaluate("""
        () => {
            const urls = new Set();
            
            // Get all href attributes
            document.querySelectorAll('[href]').forEach(el => {
                const href = el.getAttribute('href');
                if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
                    urls.add(new URL(href, window.location.origin).href);
                }
            });
            
            // Check data attributes that might contain URLs
            document.querySelectorAll('[data-href], [data-url], [data-link]').forEach(el => {
                ['data-href', 'data-url', 'data-link'].forEach(attr => {
                    const val = el.getAttribute(attr);
                    if (val && val.includes('/')) {
                        try {
                            urls.add(new URL(val, window.location.origin).href);
                        } catch(e) {}
                    }
                });
            });
            
            // Check onclick handlers
            document.querySelectorAll('[onclick]').forEach(el => {
                const onclick = el.getAttribute('onclick');
                const urlMatch = onclick.match(/["']([^"']*\\/docs\\/[^"']*)["']/);
                if (urlMatch) {
                    try {
                        urls.add(new URL(urlMatch[1], window.location.origin).href);
                    } catch(e) {}
                }
            });
            
            return Array.from(urls);
        }
    """)
    
    return all_urls

async def scrape_checkmarx_enhanced():
    """Enhanced scraping of Checkmarx documentation"""
    base_url = "https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction"
    
    async with async_playwright() as p:
        print("Launching browser with enhanced settings...")
        browser = await p.chromium.launch(
            headless=True,  # Run headless
            args=['--disable-blink-features=AutomationControlled']
        )
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        # Enable console logging
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        
        try:
            print(f"Navigating to {base_url}")
            await page.goto(base_url, wait_until='networkidle', timeout=60000)
            
            # Wait for navigation to fully load
            await page.wait_for_selector('nav, [role="navigation"]', timeout=10000)
            await asyncio.sleep(3)
            
            # Perform aggressive expansion
            await expand_everything_aggressively(page)
            
            # Try scrolling the entire page to trigger lazy loading
            print("Scrolling to trigger lazy loading...")
            await page.evaluate("""
                async () => {
                    // Scroll main content
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(r => setTimeout(r, 1000));
                    
                    // Scroll navigation if it has its own scroll
                    const nav = document.querySelector('nav, [role="navigation"], aside');
                    if (nav) {
                        nav.scrollTop = nav.scrollHeight;
                        await new Promise(r => setTimeout(r, 1000));
                    }
                }
            """)
            
            # Perform deep navigation scan
            nav_structure = await deep_navigation_scan(page)
            print(f"Found {len(nav_structure['sections'])} sections")
            print(f"Found {len(nav_structure['expandableItems'])} expandable items")
            
            # Search for API patterns
            api_patterns = await search_for_api_patterns(page)
            print(f"Found {len(api_patterns['restEndpoints'])} REST endpoint patterns")
            print(f"Found {len(api_patterns['apiCategories'])} API categories")
            
            # Extract all URLs
            all_urls = await extract_all_possible_urls(page)
            
            # Filter for Checkmarx documentation URLs
            checkmarx_docs = [
                url for url in all_urls 
                if 'checkmarx' in url and '/docs/' in url and 'stoplight.io' in url
            ]
            unique_docs = sorted(list(set(checkmarx_docs)))
            
            # Look for version or branch variations
            print("\nChecking for version/branch variations...")
            branch_pattern = re.compile(r'/branches/[^/]+/')
            branches_found = set()
            for url in unique_docs:
                match = branch_pattern.search(url)
                if match:
                    branches_found.add(match.group())
            
            # Save comprehensive results
            results = {
                'total_unique_urls': len(unique_docs),
                'urls': unique_docs,
                'navigation_structure': nav_structure,
                'api_patterns': api_patterns,
                'branches_found': list(branches_found),
                'all_urls_count': len(all_urls),
                'scrape_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('/opt/projects/knowledgehub/checkmarx_enhanced_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print detailed summary
            print("\n" + "="*60)
            print("ENHANCED SCAN RESULTS")
            print("="*60)
            print(f"Total unique documentation URLs: {len(unique_docs)}")
            print(f"Total URLs found on page: {len(all_urls)}")
            print(f"Branches/versions found: {branches_found}")
            print(f"Navigation sections: {len(nav_structure['sections'])}")
            print(f"API categories discovered: {len(api_patterns['apiCategories'])}")
            
            # Show some discovered API categories
            if api_patterns['apiCategories']:
                print("\nDiscovered API categories:")
                for cat in api_patterns['apiCategories'][:10]:
                    print(f"  - {cat}")
            
            # Analysis
            if len(unique_docs) > 50:
                print(f"\n✅ SUCCESS! Found {len(unique_docs)} URLs - {len(unique_docs) - 50} MORE than your current 50!")
                print("\nAdditional URLs discovered:")
                # Show URLs not in the original 50
                original_50_count = 50  # Assuming you have 50
                if len(unique_docs) > original_50_count:
                    for i, url in enumerate(unique_docs[original_50_count:], 1):
                        print(f"{i}. {url}")
            else:
                print(f"\n⚠️  Found {len(unique_docs)} URLs - same as or less than your current collection")
                print("\nPossible reasons:")
                print("- All pages are already discovered")
                print("- Some content requires authentication")
                print("- Dynamic loading not fully captured")
                print("- Version-specific content not accessible")
            
            print(f"\nFull results saved to: /opt/projects/knowledgehub/checkmarx_enhanced_results.json")
            
            # Wait before closing to observe the page
            print("\nBrowser will close in 5 seconds...")
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"Error during enhanced scraping: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_checkmarx_enhanced())