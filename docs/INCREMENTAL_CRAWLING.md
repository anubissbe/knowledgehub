# Incremental & Delta Crawling Guide

KnowledgeHub now supports intelligent incremental crawling that dramatically reduces update times by only processing new or changed content.

## Overview

The incremental crawling system provides:
- **Change Detection**: SHA-256 content hashing to detect modified pages
- **New Page Discovery**: Continues crawling to find pages beyond what's indexed
- **Performance**: 95%+ faster updates compared to full re-crawls
- **Automatic**: No configuration needed - works out of the box

## How It Works

### 1. Initial Crawl
When a source is first added, the system performs a full crawl:
- Visits all pages up to the configured limits
- Stores content hash for each page
- Creates document records with metadata

### 2. Incremental Updates
On subsequent crawls (manual or scheduled):
- Loads existing document hashes into memory
- Checks each URL against the cache
- For existing pages: Compares content hash
- For new pages: Processes normally
- Continues crawling to discover new content

### 3. Delta Detection
The system detects three types of changes:
- **New Pages**: URLs not previously indexed
- **Updated Pages**: Content hash differs from stored hash
- **Unchanged Pages**: Skipped but links extracted for discovery

## Performance Examples

### GitHub Documentation
- **Initial Crawl**: 1,838 pages in ~25 minutes
- **Incremental Update**: Same pages checked in ~30 seconds
- **With Changes**: Only modified pages re-processed

### React Documentation
- **Initial Crawl**: 68 pages in ~2 minutes  
- **Incremental Update**: Checked in ~3 seconds

## Technical Implementation

### Content Hashing
```python
def _calculate_content_hash(self, content: str) -> str:
    """Calculate SHA-256 hash of content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
```

### Document Caching
```python
async def _load_existing_documents(self, source_id: str):
    """Load existing documents for a source into cache"""
    response = await self.api_client.get("/api/v1/documents/", 
                                       params={"source_id": source_id})
    for doc in documents:
        self.existing_docs_cache[url] = {
            "id": doc.get("id"),
            "content_hash": doc.get("content_hash"),
            "updated_at": doc.get("updated_at")
        }
```

### Crawl Logic
The crawler:
1. Loads existing documents into memory cache
2. For each URL encountered:
   - Checks if URL exists in cache
   - If exists: Compares content hash
   - If new or changed: Yields for processing
   - Always extracts links for discovery
3. Continues until max_pages limit reached

## API Endpoints

### Documents API
New endpoints for document management:

```bash
# List documents for a source
GET /api/v1/documents/?source_id={source_id}

# Get specific document
GET /api/v1/documents/{document_id}

# Get document chunks
GET /api/v1/documents/{document_id}/chunks
```

## Configuration

### Crawl Configuration
No special configuration needed! The system automatically:
- Detects when to use incremental crawling
- Manages content hashes
- Optimizes crawl delays

### Force Refresh
To bypass incremental logic and re-crawl everything:
```python
crawl_config = {
    "force_refresh": True,
    # ... other config
}
```

## Best Practices

### 1. Regular Updates
- Schedule weekly updates to catch changes
- The scheduler automatically uses incremental crawling

### 2. Page Limits
- Set appropriate max_pages limits
- System will find new pages up to this limit

### 3. Rate Limiting
- Incremental checks use shorter delays (50% of normal)
- Helps avoid rate limiting on large sites

## Monitoring

### Crawl Statistics
Each job reports:
- Pages checked vs pages processed
- New pages found
- Updated pages detected
- Total time saved

### Logs
```
INFO: Loading existing documents for source 123e4567-e89b-12d3-a456-426614174000
INFO: Loaded 1838 existing documents
INFO: Content unchanged for https://example.com/page1, checking for new links
INFO: New page found: https://example.com/new-page
INFO: Content changed for https://example.com/updated-page
INFO: Incremental crawl complete: 1850 pages checked, 12 new, 3 updated
```

## Troubleshooting

### All Pages Re-crawled
**Cause**: Content hashes not stored properly
**Solution**: Check document records have content_hash field

### Slow Performance
**Cause**: Too many API calls to load documents
**Solution**: Ensure document limit is sufficient (default: 10,000)

### Missing New Pages
**Cause**: max_pages limit reached
**Solution**: Increase max_pages in source configuration

## Future Enhancements

Planned improvements:
- Partial content hashing for large pages
- Configurable hash algorithms
- Bloom filters for very large sites
- Differential content storage