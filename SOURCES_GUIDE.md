# 📚 KnowledgeHub Source Management Guide

## Overview

KnowledgeHub supports ingesting knowledge from multiple sources into its multi-dimensional database system (PostgreSQL, Weaviate, Neo4j, TimescaleDB). This guide explains how to add and manage sources.

## 🔧 Source Types

KnowledgeHub supports these source types:
- **website** - General websites and web pages
- **documentation** - Technical documentation sites
- **repository** - Code repositories (GitHub, GitLab, etc.)
- **api** - REST/GraphQL APIs
- **wiki** - Wiki-style knowledge bases

## 📍 How to Add Sources

### 1. Via Claude Helper Commands

```bash
# First, load the helpers
source /opt/projects/knowledgehub/claude_code_helpers.sh

# Add a documentation source
claude-add-source "https://docs.python.org/3/" "documentation" "Python Docs"

# Add a website
claude-add-source "https://example.com" "website" "Example Site"

# List all sources
claude-list-sources

# Refresh a source (re-scrape)
claude-refresh-source src_123456
```

### 2. Via Web UI

Navigate to http://192.168.1.25:3100/sources and use the UI to:
- Add new sources with the "Add Source" button
- Configure crawling parameters
- Monitor scraping progress
- View indexed documents

### 3. Via API

```bash
# Add a source
curl -X POST http://192.168.1.25:3000/api/sources/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "FastAPI Documentation",
    "url": "https://fastapi.tiangolo.com/",
    "type": "documentation",
    "config": {
      "max_depth": 3,
      "max_pages": 1000,
      "crawl_delay": 1.0,
      "follow_patterns": [".*\\.html$", ".*\\.md$"],
      "exclude_patterns": [".*/_static/.*", ".*/assets/.*"]
    }
  }'

# List sources
curl http://192.168.1.25:3000/api/sources/

# Get source details
curl http://192.168.1.25:3000/api/sources/src_123456

# Refresh source
curl -X POST http://192.168.1.25:3000/api/sources/src_123456/refresh

# Delete source
curl -X DELETE http://192.168.1.25:3000/api/sources/src_123456
```

### 4. Via Python Script

```bash
python3 /opt/projects/knowledgehub/manage_sources.py add "https://react.dev" "documentation" "React Docs"
python3 /opt/projects/knowledgehub/manage_sources.py list
python3 /opt/projects/knowledgehub/manage_sources.py refresh src_123456
```

## ⚙️ Configuration Options

When adding sources, you can configure:

```json
{
  "config": {
    // Crawling depth
    "max_depth": 3,              // How deep to follow links
    "max_pages": 500,            // Maximum pages to crawl
    
    // Rate limiting
    "crawl_delay": 1.0,          // Seconds between requests
    
    // URL filtering
    "follow_patterns": [         // Regex patterns to include
      ".*\\.html$",
      ".*\\.md$"
    ],
    "exclude_patterns": [        // Regex patterns to exclude
      ".*/_static/.*",
      ".*/assets/.*"
    ],
    
    // Authentication (if needed)
    "authentication": {
      "type": "basic",
      "username": "user",
      "password": "pass"
    },
    
    // Custom headers
    "custom_headers": {
      "User-Agent": "KnowledgeHub/1.0"
    },
    
    // Auto-refresh
    "refresh_interval": 86400    // Seconds (24 hours)
  }
}
```

## 🗄️ Data Flow

1. **Source Added** → Status: PENDING
2. **Job Queued** → Redis queue for processing
3. **Scraper Worker** → Fetches and processes content
4. **Content Storage**:
   - Raw content → PostgreSQL (documents table)
   - Chunks → PostgreSQL (document_chunks table)
   - Embeddings → Weaviate (vector search)
   - Relationships → Neo4j (knowledge graph)
   - Time-series data → TimescaleDB
5. **Status Updated** → COMPLETED or ERROR

## 🚀 Running the Scraper Worker

The scraper worker processes queued jobs:

```bash
# Run the scraper worker
python3 /opt/projects/knowledgehub/scraper_worker.py

# Or run with custom settings
REDIS_URL="redis://192.168.1.25:6379" \
API_BASE="http://192.168.1.25:3000" \
python3 /opt/projects/knowledgehub/scraper_worker.py
```

## 📊 Source Status Types

- **PENDING** - Newly added, waiting to be scraped
- **QUEUED** - Job created and in queue
- **CRAWLING** - Actively scraping content
- **INDEXING** - Processing and storing content
- **COMPLETED** - Successfully scraped
- **ERROR** - Failed (check logs)
- **PAUSED** - Temporarily stopped

## 🎯 Best Practices

1. **Start Small**: Test with a small max_pages first
2. **Respect Rate Limits**: Use appropriate crawl_delay
3. **Filter Wisely**: Use patterns to avoid irrelevant content
4. **Monitor Progress**: Check status regularly
5. **Handle Errors**: Check logs if status is ERROR

## 🔍 Searching Indexed Content

Once sources are indexed, search through:

```bash
# Via Claude helpers
claude-search "python async programming"

# Via API
curl -X POST http://192.168.1.25:3000/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python async programming"}'

# Via Web UI
# Navigate to http://192.168.1.25:3100/search
```

## 🚨 Troubleshooting

1. **Source stuck in PENDING**: Check if scraper worker is running
2. **ERROR status**: Check logs at `/var/log/knowledgehub/scraper.log`
3. **No results**: Verify source was successfully crawled
4. **Slow crawling**: Adjust max_depth and max_pages

## 📝 Example Sources to Add

```bash
# Programming documentation
claude-add-source "https://docs.python.org/3/" "documentation" "Python 3 Docs"
claude-add-source "https://developer.mozilla.org/en-US/docs/Web/JavaScript" "documentation" "MDN JavaScript"
claude-add-source "https://react.dev/" "documentation" "React Documentation"
claude-add-source "https://fastapi.tiangolo.com/" "documentation" "FastAPI Docs"

# AI/ML resources
claude-add-source "https://huggingface.co/docs" "documentation" "Hugging Face Docs"
claude-add-source "https://pytorch.org/docs/stable/index.html" "documentation" "PyTorch Docs"

# Development tools
claude-add-source "https://docs.docker.com/" "documentation" "Docker Docs"
claude-add-source "https://kubernetes.io/docs/" "documentation" "Kubernetes Docs"
```

## 🔐 Security Notes

- Sources are scraped from your server's IP
- Authentication credentials are stored encrypted
- Respect robots.txt and rate limits
- Don't scrape sensitive/private content without permission

---

Remember: The more quality sources you add, the smarter your KnowledgeHub becomes!