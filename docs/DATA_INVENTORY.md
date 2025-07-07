# KnowledgeHub Data Inventory

## Overview
As of July 6, 2025, your KnowledgeHub system contains:
- **12 Knowledge Sources**
- **1,270 Documents** 
- **1,266 Document Chunks** (for vector search)

## Data Sources

### 1. Checkmarx Documentation (693 documents total)
- **Checkmarx One API Reference Guide**: 427 documents
  - URL: https://checkmarx.stoplight.io/docs/checkmarx-one-api-reference-guide
  - Crawled: July 5, 2025
  
- **Checkmarx SCA Documentation**: 78 documents
  - URL: https://docs.checkmarx.com/en/34965-68708-checkmarx-sca.html
  - Crawled: July 6, 2025
  
- **Checkmarx One**: 65 documents
  - URL: https://docs.checkmarx.com/en/34965-67042-checkmarx-one.html
  - Crawled: July 6, 2025
  
- **Checkmarx Documentation - Full**: 62 documents
  - URL: https://docs.checkmarx.com/en/34965-checkmarx-documentation.html
  - Crawled: July 6, 2025
  
- **Checkmarx SAST Documentation**: 61 documents
  - URL: https://docs.checkmarx.com/en/34965-68565-checkmarx-sast.html
  - Crawled: July 6, 2025

### 2. GitHub Documentation (497 documents total)
- **GitHub Docs - Actions**: 163 documents
  - URL: https://docs.github.com/en/actions
  - Crawled: July 5, 2025
  
- **GitHub Docs - Get-Started**: 144 documents
  - URL: https://docs.github.com/en/get-started
  - Crawled: July 5, 2025
  
- **GitHub Documentation**: 144 documents
  - URL: https://docs.github.com/en
  - Crawled: June 30, 2025
  
- **GitHub Docs - REST**: 46 documents
  - URL: https://docs.github.com/en/rest
  - Crawled: July 5, 2025

### 3. Other Documentation (80 documents total)
- **Prisma Cloud API**: 75 documents
  - URL: https://pan.dev/prisma-cloud/api/
  - Crawled: July 6, 2025
  
- **Python 3.13 Documentation**: 5 documents
  - URL: Unknown
  - Crawled: Date unknown

## Data Storage Locations

### 1. PostgreSQL Database (Port 5433)
- **Location**: `/opt/projects/knowledgehub/data/postgres`
- **Content**: Metadata for all documents, sources, and crawl statistics
- **Tables**: knowledge_sources, documents, document_chunks, scraping_jobs

### 2. Weaviate Vector Database (Port 8090)
- **Location**: `/opt/projects/knowledgehub/data/weaviate` (35MB)
- **Content**: Vector embeddings for semantic search
- **Collection**: Knowledge_chunks
- **Vectors**: 1,266 document chunks with embeddings

### 3. MinIO Object Storage (Ports 9010/9011)
- **Location**: `/opt/projects/knowledgehub/data/minio` (104KB)
- **Content**: Currently minimal usage - may store raw documents
- **Status**: Configured but appears to have minimal data

### 4. Redis Cache (Port 6381)
- **Location**: `/opt/projects/knowledgehub/data/redis` (28KB)
- **Content**: Job queues and caching
- **Purpose**: Processing queue for crawl jobs

## Storage Summary
- **Total Disk Usage**: ~35.1 MB (mostly Weaviate vectors)
- **Document Count**: 1,270
- **Chunk Count**: 1,266
- **Sources**: 12 active sources

## Recent Activity
Most recent documents crawled (July 6, 2025):
- Checkmarx release notes (versions 3.5 - 3.40)
- Checkmarx documentation pages
- API documentation updates

## Access Methods

### Query Documents
```bash
# List all sources with document counts
docker exec knowledgehub-postgres psql -U khuser -d knowledgehub -c \
  "SELECT name, url, (stats->>'documents')::int as docs FROM knowledge_sources ORDER BY docs DESC;"

# Search documents by title
docker exec knowledgehub-postgres psql -U khuser -d knowledgehub -c \
  "SELECT title, url FROM documents WHERE title ILIKE '%api%' LIMIT 10;"
```

### Search Vectors
```bash
# Search using the API
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "checkmarx api", "limit": 10}'
```

### Web Interface
Access the KnowledgeHub UI at: http://192.168.1.24:3010

## Notes
- All sources show "completed" status with zero errors
- The system is actively maintaining documentation from security tools (Checkmarx, Prisma Cloud) and development platforms (GitHub)
- Vector embeddings are successfully generated for all documents
- The data is relatively compact (~35MB) despite having over 1,200 documents, indicating efficient storage