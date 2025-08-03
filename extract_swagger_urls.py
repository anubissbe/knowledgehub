import asyncio
from playwright.async_api import async_playwright
import re

async def extract_swagger_urls():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print('📄 Loading Swagger documentation page...')
        url = 'https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/t4nqvzyp5zfg5-try-it-out-in-swagger'
        await page.goto(url, wait_until='domcontentloaded')
        await page.wait_for_timeout(5000)
        
        # Get the page text content
        content = await page.inner_text('body')
        
        print('\n🔍 Searching for Swagger URLs in page content...\n')
        
        # Look for URLs that might be Swagger endpoints
        # Common patterns: swagger-ui, api-docs, openapi
        url_pattern = r'https?://[^\s<>"]+(?:swagger|api-docs|openapi)[^\s<>"]*'
        urls = re.findall(url_pattern, content, re.IGNORECASE)
        
        # Also look for any Checkmarx API URLs
        checkmarx_pattern = r'https?://[^\s<>"]*checkmarx[^\s<>"]+/(?:api|v\d+|swagger)[^\s<>"]*'
        checkmarx_urls = re.findall(checkmarx_pattern, content, re.IGNORECASE)
        
        all_urls = list(set(urls + checkmarx_urls))
        
        if all_urls:
            print(f'✅ Found {len(all_urls)} potential API/Swagger URLs:')
            for url in all_urls:
                print(f'   - {url}')
        
        # Also extract any code blocks that might contain URLs
        print('\n📋 Checking for URLs in code blocks...')
        code_blocks = await page.query_selector_all('code, pre')
        
        code_urls = []
        for block in code_blocks:
            text = await block.inner_text()
            if 'http' in text or 'swagger' in text.lower():
                # Extract URLs from code blocks
                found_urls = re.findall(r'https?://[^\s]+', text)
                code_urls.extend(found_urls)
        
        code_urls = list(set(code_urls))
        if code_urls:
            print(f'\n🔗 Found {len(code_urls)} URLs in code blocks:')
            for url in code_urls:
                if 'checkmarx' in url or 'swagger' in url.lower():
                    print(f'   - {url}')
        
        # Check for any tables that might list endpoints
        print('\n📊 Checking for API endpoint tables...')
        tables = await page.query_selector_all('table')
        print(f'Found {len(tables)} tables on the page')
        
        await browser.close()
        
        return all_urls + code_urls

asyncio.run(extract_swagger_urls())