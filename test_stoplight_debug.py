import asyncio
from playwright.async_api import async_playwright
import json

async def debug_stoplight():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Loading page...")
        await page.goto('https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction')
        await page.wait_for_timeout(10000)  # Wait 10 seconds
        
        # Take a screenshot
        await page.screenshot(path='stoplight_loaded.png')
        print("Screenshot saved to stoplight_loaded.png")
        
        # Try different methods to find links
        print("\n1. Checking page readiness...")
        ready_state = await page.evaluate('document.readyState')
        print(f"Document ready state: {ready_state}")
        
        print("\n2. Looking for navigation elements...")
        nav_count = await page.evaluate('''() => {
            const navs = document.querySelectorAll('nav, [role="navigation"], aside, [class*="sidebar"]');
            console.log('Navigation elements:', navs);
            return navs.length;
        }''')
        print(f"Found {nav_count} navigation elements")
        
        print("\n3. Checking all links on page...")
        all_links_count = await page.evaluate('''() => {
            const links = document.querySelectorAll('a[href]');
            console.log('Total links:', links.length);
            return links.length;
        }''')
        print(f"Total links on page: {all_links_count}")
        
        print("\n4. Looking for docs links...")
        docs_links = await page.evaluate('''() => {
            const links = [];
            document.querySelectorAll('a[href]').forEach(a => {
                if (a.href.includes('/docs/')) {
                    links.push({
                        href: a.href,
                        text: a.textContent.trim().substring(0, 50)
                    });
                }
            });
            return links;
        }''')
        print(f"Found {len(docs_links)} docs links")
        
        # Print first 10
        for i, link in enumerate(docs_links[:10]):
            print(f"  {i+1}. {link['text']} -> {link['href']}")
        
        print("\n5. Checking for React/Vue app...")
        has_react = await page.evaluate('typeof React !== "undefined"')
        has_vue = await page.evaluate('typeof Vue !== "undefined"')
        has_stoplight = await page.evaluate('typeof window.__STOPLIGHT_HYDRATION_DATA__ !== "undefined"')
        print(f"React: {has_react}, Vue: {has_vue}, Stoplight data: {has_stoplight}")
        
        # input("\nPress Enter to close browser...")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_stoplight())