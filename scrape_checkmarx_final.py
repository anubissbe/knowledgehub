#!/usr/bin/env python3
"""
Final optimized script to extract all Checkmarx documentation URLs
"""

import asyncio
import json
from playwright.async_api import async_playwright
import time
from collections import defaultdict

async def wait_and_expand(page):
    """Wait for page load and expand navigation items"""
    print("Waiting for page to load and expanding navigation...")
    
    # Wait for navigation to be present
    try:
        await page.wait_for_selector('nav, [role="navigation"], aside', timeout=10000)
    except:
        print("Navigation not found with standard selectors")
    
    # Give the page time to initialize
    await asyncio.sleep(3)
    
    # Try multiple expansion methods
    expansion_methods = [
        # Method 1: Click all buttons with SVG icons
        """
        const buttons = document.querySelectorAll('button');
        let clicked = 0;
        buttons.forEach(btn => {
            if (btn.querySelector('svg') && !btn.querySelector('[aria-expanded="true"]')) {
                try {
                    btn.click();
                    clicked++;
                } catch(e) {}
            }
        });
        return clicked;
        """,
        
        # Method 2: Click elements with chevron/arrow classes
        """
        const elements = document.querySelectorAll('[class*="chevron"], [class*="arrow"], [class*="caret"]');
        let clicked = 0;
        elements.forEach(el => {
            try {
                if (el.tagName === 'BUTTON' || el.tagName === 'A') {
                    el.click();
                } else if (el.parentElement) {
                    el.parentElement.click();
                }
                clicked++;
            } catch(e) {}
        });
        return clicked;
        """,
        
        # Method 3: Force expand by changing attributes
        """
        const expandables = document.querySelectorAll('[aria-expanded="false"]');
        expandables.forEach(el => {
            el.setAttribute('aria-expanded', 'true');
            el.click();
        });
        return expandables.length;
        """
    ]
    
    total_expanded = 0
    for method in expansion_methods:
        try:
            expanded = await page.evaluate(method)
            total_expanded += expanded
            await asyncio.sleep(0.5)
        except:
            pass
    
    print(f"Expanded {total_expanded} elements")
    return total_expanded

async def extract_all_urls(page):
    """Extract all documentation URLs from the page"""
    print("Extracting all URLs...")
    
    urls_data = await page.evaluate("""
        () => {
            const data = {
                urls: new Set(),
                sections: [],
                stats: {
                    total_links: 0,
                    nav_links: 0,
                    hidden_links: 0
                }
            };
            
            // Get all links
            const allLinks = document.querySelectorAll('a[href]');
            data.stats.total_links = allLinks.length;
            
            allLinks.forEach(link => {
                const href = link.getAttribute('href');
                if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
                    try {
                        const absoluteUrl = new URL(href, window.location.origin).href;
                        data.urls.add(absoluteUrl);
                        
                        // Check if it's in navigation
                        const isInNav = link.closest('nav, [role="navigation"], aside') !== null;
                        if (isInNav) data.stats.nav_links++;
                        
                        // Check if it's hidden
                        const style = window.getComputedStyle(link);
                        if (style.display === 'none' || style.visibility === 'hidden') {
                            data.stats.hidden_links++;
                        }
                    } catch(e) {}
                }
            });
            
            // Get section information
            const sections = document.querySelectorAll('h1, h2, h3, [role="heading"]');
            sections.forEach(section => {
                data.sections.push(section.textContent.trim());
            });
            
            // Check for API endpoint mentions in text
            const bodyText = document.body.innerText;
            const apiMatches = bodyText.match(/\/api\/[\\w\\/-]+/gi) || [];
            apiMatches.forEach(api => {
                data.urls.add(window.location.origin + api);
            });
            
            return {
                urls: Array.from(data.urls),
                sections: data.sections,
                stats: data.stats
            };
        }
    """)
    
    return urls_data

async def navigate_through_sections(page, base_url):
    """Navigate through different sections to discover more pages"""
    print("Navigating through sections to discover more content...")
    
    discovered_urls = set()
    
    # Get all navigation links
    nav_links = await page.evaluate("""
        () => {
            const links = [];
            document.querySelectorAll('nav a[href], aside a[href], [role="navigation"] a[href]').forEach(link => {
                const href = link.getAttribute('href');
                if (href && href.includes('/docs/') && !href.startsWith('#')) {
                    links.push({
                        url: new URL(href, window.location.origin).href,
                        text: link.textContent.trim()
                    });
                }
            });
            return links;
        }
    """)
    
    print(f"Found {len(nav_links)} navigation links to explore")
    
    # Visit first few links to see if they reveal more content
    for i, link_data in enumerate(nav_links[:5]):  # Limit to first 5 to save time
        try:
            print(f"  Visiting: {link_data['text'][:50]}...")
            await page.goto(link_data['url'], wait_until='domcontentloaded', timeout=15000)
            await asyncio.sleep(1)
            
            # Extract URLs from this page
            sub_urls = await page.evaluate("""
                () => {
                    const urls = new Set();
                    document.querySelectorAll('a[href]').forEach(link => {
                        const href = link.getAttribute('href');
                        if (href && href.includes('/docs/') && !href.startsWith('#')) {
                            urls.add(new URL(href, window.location.origin).href);
                        }
                    });
                    return Array.from(urls);
                }
            """)
            
            discovered_urls.update(sub_urls)
            
        except Exception as e:
            print(f"  Error visiting link: {e}")
    
    # Go back to main page
    await page.goto(base_url, wait_until='domcontentloaded', timeout=30000)
    
    return list(discovered_urls)

async def scrape_checkmarx_comprehensive():
    """Comprehensive Checkmarx documentation scraping"""
    base_url = "https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction"
    
    async with async_playwright() as p:
        print("Launching browser...")
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        all_urls = set()
        
        try:
            # Initial page load
            print(f"Loading main page: {base_url}")
            await page.goto(base_url, wait_until='domcontentloaded', timeout=30000)
            
            # Expand navigation
            await wait_and_expand(page)
            
            # Extract URLs from main page
            main_page_data = await extract_all_urls(page)
            all_urls.update(main_page_data['urls'])
            
            print(f"\nMain page statistics:")
            print(f"  Total links: {main_page_data['stats']['total_links']}")
            print(f"  Navigation links: {main_page_data['stats']['nav_links']}")
            print(f"  Hidden links: {main_page_data['stats']['hidden_links']}")
            print(f"  Sections found: {len(main_page_data['sections'])}")
            
            # Navigate through sections
            discovered_urls = await navigate_through_sections(page, base_url)
            all_urls.update(discovered_urls)
            
            # Filter for Checkmarx documentation URLs
            checkmarx_docs = [
                url for url in all_urls
                if 'checkmarx' in url.lower() 
                and '/docs/' in url 
                and 'stoplight.io' in url
                and not url.endswith('.png')
                and not url.endswith('.jpg')
                and not url.endswith('.svg')
            ]
            
            # Remove duplicates and sort
            unique_docs = sorted(list(set(checkmarx_docs)))
            
            # Categorize URLs
            categories = defaultdict(list)
            for url in unique_docs:
                # Extract the last part of the URL as category hint
                parts = url.split('/')
                if len(parts) > 1:
                    last_part = parts[-1]
                    if '-api' in last_part:
                        categories['API Services'].append(url)
                    elif 'rest-api' in last_part:
                        categories['REST APIs'].append(url)
                    elif 'service' in last_part:
                        categories['Services'].append(url)
                    elif 'management' in last_part:
                        categories['Management'].append(url)
                    else:
                        categories['Other'].append(url)
            
            # Save results
            results = {
                'total_unique_urls': len(unique_docs),
                'urls': unique_docs,
                'categories': dict(categories),
                'main_page_sections': main_page_data['sections'][:20],  # First 20 sections
                'statistics': main_page_data['stats'],
                'scrape_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            output_file = '/opt/projects/knowledgehub/checkmarx_comprehensive_results.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print summary
            print("\n" + "="*60)
            print("COMPREHENSIVE SCAN RESULTS")
            print("="*60)
            print(f"Total unique documentation URLs: {len(unique_docs)}")
            print(f"\nURLs by category:")
            for category, urls in categories.items():
                print(f"  {category}: {len(urls)} URLs")
            
            print(f"\nSample URLs from each category:")
            for category, urls in categories.items():
                if urls:
                    print(f"\n{category}:")
                    for url in urls[:3]:
                        print(f"  - {url}")
            
            # Comparison with expected 50
            if len(unique_docs) > 50:
                print(f"\n✅ SUCCESS! Found {len(unique_docs)} URLs - {len(unique_docs) - 50} MORE than expected!")
            elif len(unique_docs) == 50:
                print(f"\n✓ Found exactly 50 URLs - matches your current collection")
            else:
                print(f"\n⚠️  Found {len(unique_docs)} URLs - less than expected 50")
            
            print(f"\nDetailed results saved to: {output_file}")
            
            # List all unique URL patterns
            print("\nUnique URL patterns found:")
            patterns = set()
            for url in unique_docs:
                # Extract pattern after /main/
                match = url.split('/main/')
                if len(match) > 1:
                    pattern = match[1].split('-')[0] if '-' in match[1] else match[1]
                    patterns.add(pattern[:8] + "...")
            
            for pattern in sorted(patterns)[:10]:
                print(f"  - {pattern}")
            
        except Exception as e:
            print(f"Error during comprehensive scraping: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_checkmarx_comprehensive())