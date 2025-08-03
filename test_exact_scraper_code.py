import asyncio
from playwright.async_api import async_playwright

async def test_exact_code():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Loading page...")
        await page.goto('https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction')
        await page.wait_for_timeout(5000)
        
        # This is the EXACT code from the scraper
        all_links = await page.evaluate('''() => {
            const links = [];
            const seen = new Set();
            
            // Simply get ALL links that contain /docs/
            document.querySelectorAll('a[href]').forEach(link => {
                const href = link.href;
                const text = link.textContent.trim();
                
                // Skip if already seen or not a docs link
                if (!href || seen.has(href) || !href.includes('/docs/')) return;
                
                // Only include checkmarx docs
                if (href.includes('checkmarx')) {
                    seen.add(href);
                    
                    // Determine content type
                    let contentType = 'page';
                    if (text.match(/^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)/i)) {
                        contentType = 'endpoint';
                    } else if (text.toLowerCase().includes('api') || href.includes('-api')) {
                        contentType = 'api-section';
                    }
                    
                    links.push({
                        url: href,
                        text: text || 'Untitled',
                        contentType: contentType,
                        section: 'API Documentation'
                    });
                }
            });
            
            console.log('Found ' + links.length + ' documentation links');
            return links;
        }''')
        
        print(f"\nFound {len(all_links)} links")
        
        # Let's also check what happens without the complex logic
        simple_test = await page.evaluate('''() => {
            const links = document.querySelectorAll('a[href]');
            let count = 0;
            links.forEach(link => {
                if (link.href.includes('/docs/') && link.href.includes('checkmarx')) {
                    count++;
                }
            });
            return count;
        }''')
        
        print(f"Simple count: {simple_test} links")
        
        # Test if there's an error in the evaluation
        test_error = await page.evaluate('''() => {
            try {
                const links = [];
                document.querySelectorAll('a[href]').forEach(link => {
                    links.push(link.href);
                });
                return {success: true, count: links.length, sample: links.slice(0, 5)};
            } catch (e) {
                return {success: false, error: e.toString()};
            }
        }''')
        
        print(f"\nError test: {test_error}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_exact_code())