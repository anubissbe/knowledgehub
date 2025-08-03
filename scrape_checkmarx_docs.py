#!/usr/bin/env python3
"""
Script to scrape all URLs from Checkmarx Stoplight API documentation
"""

import asyncio
import json
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
import time

async def expand_all_sections(page):
    """Expand all collapsible sections in the navigation"""
    print("Expanding all navigation sections...")
    
    # Wait for the page to load
    await page.wait_for_load_state('networkidle')
    await asyncio.sleep(2)
    
    # Try multiple selectors for expandable sections
    expand_selectors = [
        'button[aria-expanded="false"]',
        '[data-testid="nav-toggle"]',
        '.sl-elements-chevron-button',
        '.sl-flex button:has(svg)',
        'button:has(path[d*="M5"])',  # Chevron icons
        '[role="button"][aria-expanded="false"]'
    ]
    
    expanded_count = 0
    for selector in expand_selectors:
        try:
            buttons = await page.query_selector_all(selector)
            for button in buttons:
                try:
                    await button.click()
                    expanded_count += 1
                    await asyncio.sleep(0.1)  # Small delay between clicks
                except:
                    pass
        except:
            pass
    
    print(f"Expanded {expanded_count} sections")
    await asyncio.sleep(1)  # Wait for animations to complete

async def extract_navigation_urls(page):
    """Extract all URLs from the navigation sidebar"""
    print("Extracting URLs from navigation...")
    
    # Get all links from the navigation
    nav_links = await page.evaluate("""
        () => {
            const links = new Set();
            
            // Try multiple selectors for navigation links
            const selectors = [
                'nav a[href]',
                '.sl-elements-navigation a[href]',
                '[role="navigation"] a[href]',
                'aside a[href]',
                '.sl-flex a[href]',
                'a[href*="/docs/"]'
            ];
            
            selectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(link => {
                    const href = link.getAttribute('href');
                    if (href && !href.startsWith('#')) {
                        // Get absolute URL
                        const absoluteUrl = new URL(href, window.location.origin).href;
                        links.add(absoluteUrl);
                    }
                });
            });
            
            return Array.from(links);
        }
    """)
    
    return nav_links

async def check_for_hidden_sections(page):
    """Look for any hidden or additional documentation sections"""
    print("Checking for hidden sections...")
    
    hidden_info = await page.evaluate("""
        () => {
            const info = {
                hiddenElements: [],
                additionalSections: [],
                totalLinks: 0
            };
            
            // Check for hidden elements
            const hidden = document.querySelectorAll('[style*="display: none"], [hidden], .hidden, .collapsed');
            info.hiddenElements = Array.from(hidden).map(el => ({
                tag: el.tagName,
                class: el.className,
                text: el.textContent?.substring(0, 50)
            }));
            
            // Count all links on the page
            info.totalLinks = document.querySelectorAll('a[href]').length;
            
            // Look for dropdowns or menus
            const dropdowns = document.querySelectorAll('select, [role="menu"], .dropdown');
            info.additionalSections = Array.from(dropdowns).map(el => ({
                tag: el.tagName,
                class: el.className,
                options: el.tagName === 'SELECT' ? Array.from(el.options).map(o => o.text) : []
            }));
            
            return info;
        }
    """)
    
    return hidden_info

async def scrape_checkmarx_docs():
    """Main function to scrape Checkmarx documentation"""
    base_url = "https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction"
    
    async with async_playwright() as p:
        print("Launching browser...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        try:
            print(f"Navigating to {base_url}")
            await page.goto(base_url, wait_until='networkidle', timeout=30000)
            
            # Initial wait for page to fully load
            await asyncio.sleep(3)
            
            # Expand all sections multiple times to ensure we get everything
            for i in range(3):
                print(f"\nExpansion round {i+1}...")
                await expand_all_sections(page)
                await asyncio.sleep(1)
            
            # Extract URLs
            nav_urls = await extract_navigation_urls(page)
            print(f"\nFound {len(nav_urls)} navigation URLs")
            
            # Check for hidden sections
            hidden_info = await check_for_hidden_sections(page)
            print(f"Found {len(hidden_info['hiddenElements'])} hidden elements")
            print(f"Total links on page: {hidden_info['totalLinks']}")
            
            # Also try to get URLs by scrolling through the navigation
            print("\nScrolling through navigation to find more links...")
            nav_element = await page.query_selector('nav, [role="navigation"], aside')
            if nav_element:
                await nav_element.scroll_into_view_if_needed()
                # Scroll to bottom of navigation
                await page.evaluate("""
                    () => {
                        const nav = document.querySelector('nav, [role="navigation"], aside');
                        if (nav) {
                            nav.scrollTop = nav.scrollHeight;
                        }
                    }
                """)
                await asyncio.sleep(1)
            
            # Extract URLs again after scrolling
            nav_urls_after_scroll = await extract_navigation_urls(page)
            nav_urls.extend(nav_urls_after_scroll)
            
            # Remove duplicates and sort
            unique_urls = sorted(list(set(nav_urls)))
            
            # Filter for Checkmarx docs
            checkmarx_urls = [url for url in unique_urls if 'checkmarx' in url and '/docs/' in url]
            
            # Save results
            results = {
                'total_unique_urls': len(checkmarx_urls),
                'urls': checkmarx_urls,
                'hidden_sections': hidden_info,
                'scrape_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('/opt/projects/knowledgehub/checkmarx_urls_extracted.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print summary
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Total unique documentation URLs found: {len(checkmarx_urls)}")
            print(f"Hidden elements found: {len(hidden_info['hiddenElements'])}")
            print(f"Additional sections found: {len(hidden_info['additionalSections'])}")
            
            # Print first 10 URLs as sample
            print("\nSample URLs found:")
            for i, url in enumerate(checkmarx_urls[:10]):
                print(f"{i+1}. {url}")
            
            if len(checkmarx_urls) > 50:
                print(f"\n✅ Found {len(checkmarx_urls)} URLs - MORE than the 50 you currently have!")
            else:
                print(f"\n⚠️  Found {len(checkmarx_urls)} URLs - This might be less than or equal to your current 50")
            
            print(f"\nFull results saved to: /opt/projects/knowledgehub/checkmarx_urls_extracted.json")
            
            # Take a screenshot for reference
            await page.screenshot(path='/opt/projects/knowledgehub/checkmarx_docs_screenshot.png', full_page=False)
            print("Screenshot saved to: /opt/projects/knowledgehub/checkmarx_docs_screenshot.png")
            
        except Exception as e:
            print(f"Error during scraping: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_checkmarx_docs())