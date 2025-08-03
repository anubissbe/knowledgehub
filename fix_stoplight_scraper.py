import asyncio
from playwright.async_api import async_playwright
import json

async def scrape_stoplight_fully():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Use headful for debugging
        page = await browser.new_page()
        
        print("Loading Stoplight page...")
        try:
            await page.goto('https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction', 
                          wait_until='domcontentloaded', timeout=60000)
        except Exception as e:
            print(f"Initial load error: {e}")
        
        # Wait for the page to stabilize
        await page.wait_for_timeout(5000)
        
        print("\nExtracting all navigation links...")
        
        # Get all links using a more comprehensive approach
        all_links = await page.evaluate('''() => {
            const links = new Set();
            const baseUrl = 'https://checkmarx.stoplight.io';
            
            // Method 1: Get all anchor tags
            document.querySelectorAll('a[href]').forEach(a => {
                const href = a.href;
                if (href && href.startsWith(baseUrl) && href.includes('/docs/')) {
                    links.add(href);
                }
            });
            
            // Method 2: Look in the React/Vue app data
            // Stoplight often stores route data in the app state
            if (window.__STOPLIGHT_HYDRATION_DATA__) {
                console.log('Found Stoplight hydration data');
                try {
                    const data = window.__STOPLIGHT_HYDRATION_DATA__;
                    // Extract links from the data structure
                    JSON.stringify(data, (key, value) => {
                        if (typeof value === 'string' && value.includes('/docs/')) {
                            if (value.startsWith('/')) {
                                links.add(baseUrl + value);
                            } else if (value.startsWith('http')) {
                                links.add(value);
                            }
                        }
                        return value;
                    });
                } catch (e) {
                    console.error('Error processing hydration data:', e);
                }
            }
            
            // Method 3: Check for API endpoint patterns
            document.querySelectorAll('*').forEach(el => {
                const text = el.textContent || '';
                // Look for REST method patterns
                if (text.match(/^(GET|POST|PUT|DELETE|PATCH)\s+\//)) {
                    const parent = el.closest('a');
                    if (parent && parent.href) {
                        links.add(parent.href);
                    }
                }
            });
            
            return Array.from(links);
        }''')
        
        print(f"\nFound {len(all_links)} unique documentation links")
        
        # Filter and organize links
        api_endpoints = []
        guide_pages = []
        
        for link in all_links:
            if any(method in link for method in ['get-', 'post-', 'put-', 'delete-', 'patch-']):
                api_endpoints.append(link)
            else:
                guide_pages.append(link)
        
        print(f"\nAPI Endpoints: {len(api_endpoints)}")
        print(f"Guide Pages: {len(guide_pages)}")
        
        # Save the results
        results = {
            'total_links': len(all_links),
            'api_endpoints': api_endpoints[:20],  # First 20 as sample
            'guide_pages': guide_pages[:20],
            'all_links': all_links
        }
        
        with open('stoplight_links.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nSample API Endpoints:")
        for link in api_endpoints[:10]:
            print(f"  - {link}")
        
        print("\n\nNow let's try the table of contents approach...")
        
        # Try to interact with the navigation
        try:
            # Look for expandable sections
            expandables = await page.query_selector_all('[aria-expanded="false"]')
            print(f"\nFound {len(expandables)} expandable sections")
            
            # Expand them
            for i, exp in enumerate(expandables[:10]):
                try:
                    await exp.click()
                    await page.wait_for_timeout(200)
                    print(f"Expanded section {i+1}")
                except:
                    pass
            
            # Re-extract links after expansion
            await page.wait_for_timeout(2000)
            
            expanded_links = await page.evaluate('''() => {
                const links = new Set();
                document.querySelectorAll('a[href*="/docs/"]').forEach(a => {
                    if (a.href.includes('checkmarx')) {
                        links.add(a.href);
                    }
                });
                return Array.from(links);
            }''')
            
            print(f"\nAfter expansion: {len(expanded_links)} links")
            
        except Exception as e:
            print(f"Error during expansion: {e}")
        
        await browser.close()
        
        return all_links

if __name__ == "__main__":
    links = asyncio.run(scrape_stoplight_fully())
    print(f"\n\nTotal links found: {len(links)}")
    
    # Update the scraper configuration
    if len(links) > 20:
        print("\n✅ Success! Found many documentation pages.")
        print("The scraper needs to be updated to properly discover these links.")