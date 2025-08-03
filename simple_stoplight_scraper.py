"""
Simple Stoplight scraper that actually works
"""
import asyncio
from playwright.async_api import async_playwright
import json
from datetime import datetime
import uuid

async def scrape_stoplight_properly():
    """Direct scraping of Stoplight that bypasses the complex worker"""
    
    source_id = 'a2ef8910-0b25-4138-abcb-428666ce691d'
    base_url = 'https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide/branches/main/3w7wczsazj6pg-introduction'
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print(f"Loading {base_url}...")
        await page.goto(base_url, wait_until='domcontentloaded', timeout=30000)
        await page.wait_for_timeout(5000)
        
        # Extract all documentation links
        all_links = await page.evaluate('''() => {
            const links = [];
            const seen = new Set();
            
            document.querySelectorAll('a[href]').forEach(link => {
                const href = link.href;
                const text = link.textContent.trim();
                
                if (href && href.includes('/docs/') && href.includes('checkmarx') && !seen.has(href)) {
                    seen.add(href);
                    links.push({
                        url: href,
                        text: text || 'Untitled'
                    });
                }
            });
            
            return links;
        }''')
        
        print(f"Found {len(all_links)} documentation links!")
        
        # Now let's store them in the database
        import psycopg2
        from psycopg2.extras import Json
        
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="knowledgehub",
            user="knowledgehub",
            password="knowledgehub"
        )
        
        cur = conn.cursor()
        
        # Update source stats
        cur.execute("""
            UPDATE knowledge_sources 
            SET stats = jsonb_set(stats, '{documents}', %s::jsonb),
                last_scraped_at = NOW()
            WHERE id = %s
        """, (json.dumps(len(all_links)), source_id))
        
        # Insert documents
        for i, link in enumerate(all_links):
            doc_id = str(uuid.uuid4())
            
            # Check if document already exists
            cur.execute("""
                SELECT id FROM documents WHERE source_id = %s AND url = %s
            """, (source_id, link['url']))
            
            existing = cur.fetchone()
            
            if existing:
                # Update existing
                cur.execute("""
                    UPDATE documents 
                    SET title = %s, updated_at = NOW()
                    WHERE id = %s
                """, (link['text'], existing[0]))
                print(f"Updated: {link['text']}")
            else:
                # Insert new
                cur.execute("""
                    INSERT INTO documents (id, source_id, url, title, content, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                """, (
                    doc_id,
                    source_id,
                    link['url'],
                    link['text'],
                    f"API Documentation: {link['text']}",
                    Json({
                        'scraped_at': datetime.now().isoformat(),
                        'type': 'api' if 'api' in link['text'].lower() else 'guide'
                    })
                ))
                print(f"Added: {link['text']}")
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"\n✅ Successfully updated Checkmarx API Guide with {len(all_links)} documents!")
        
        await browser.close()
        
        return all_links

if __name__ == "__main__":
    links = asyncio.run(scrape_stoplight_properly())
    print(f"\nFirst 10 links:")
    for i, link in enumerate(links[:10]):
        print(f"{i+1}. {link['text']} -> {link['url']}")