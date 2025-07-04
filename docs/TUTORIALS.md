# KnowledgeHub Tutorials

Step-by-step tutorials to help you master KnowledgeHub quickly and effectively.

## Table of Contents

- [Tutorial 1: Your First Knowledge Source](#tutorial-1-your-first-knowledge-source)
- [Tutorial 2: Advanced Search Techniques](#tutorial-2-advanced-search-techniques)
- [Tutorial 3: Setting Up API Integration](#tutorial-3-setting-up-api-integration)
- [Tutorial 4: Monitoring and Optimization](#tutorial-4-monitoring-and-optimization)
- [Tutorial 5: Incremental Crawling Setup](#tutorial-5-incremental-crawling-setup)
- [Tutorial 6: Bulk Source Management](#tutorial-6-bulk-source-management)
- [Tutorial 7: Custom Embedding Workflows](#tutorial-7-custom-embedding-workflows)
- [Tutorial 8: Production Deployment](#tutorial-8-production-deployment)

---

## Tutorial 1: Your First Knowledge Source

**Goal**: Learn how to add and configure your first knowledge source in KnowledgeHub.

**Time**: 15 minutes

**Prerequisites**: KnowledgeHub running locally

### Step 1: Access the Dashboard

1. Open your browser and navigate to `http://localhost:3101`
2. You should see the KnowledgeHub dashboard
3. Take note of the system health indicators - all should be green

### Step 2: Navigate to Sources

1. Click on **📚 Sources** in the main navigation
2. You'll see an empty list (this is your first source!)
3. Click the **"+ Add Source"** button

### Step 3: Configure Your First Source

Let's add the React documentation as our first source:

```
Name: React Documentation
Description: Official React documentation and guides
Base URL: https://react.dev
Source Type: web
```

**Advanced Configuration**:
```json
{
  "max_depth": 3,
  "max_pages": 200,
  "crawl_delay": 1.0,
  "follow_patterns": ["**/learn/**", "**/reference/**"],
  "exclude_patterns": ["**/blog/**", "**/community/**"]
}
```

### Step 4: Start the Crawl

1. Click **"Create Source"**
2. You'll be redirected to the Sources page
3. Notice the new source with status "Active"
4. A crawl job will automatically start

### Step 5: Monitor Progress

1. Click on **⚙️ Jobs** in the navigation
2. You'll see a new crawl job in "Running" status
3. Watch the real-time progress updates
4. The job should complete in 2-3 minutes for React docs

### Step 6: Verify Content Processing

1. Wait for the crawl job to complete (status: "Completed")
2. You should see RAG processing jobs start automatically
3. These jobs convert crawled content into searchable chunks
4. Wait for all jobs to complete

### Step 7: Test Your First Search

1. Navigate to **🔍 Search**
2. Try searching for: `"useState hook"`
3. You should see relevant results from the React documentation
4. Click on a result to see the full content

**Expected Results**:
- Crawl job processes ~68 pages
- RAG processing creates ~500-800 chunks
- Search returns relevant React documentation
- Total setup time: ~5 minutes

### Troubleshooting

**If the crawl fails**:
- Check that https://react.dev is accessible
- Verify your internet connection
- Look at job error messages for specific issues

**If search returns no results**:
- Ensure all RAG processing jobs completed
- Check that Weaviate service is running: `curl http://localhost:8090/v1/meta`

---

## Tutorial 2: Advanced Search Techniques

**Goal**: Master different search types and techniques for finding the right information.

**Time**: 20 minutes

**Prerequisites**: At least one source with content (Tutorial 1 completed)

### Step 1: Understanding Search Types

Navigate to **🔍 Search** and let's explore the three search types:

#### Semantic Search
Best for: Conceptual questions, finding similar ideas

**Example Query**: `"What are the benefits of using hooks?"`

**Try these queries**:
- `"How to handle form input in React"`
- `"Best practices for component architecture"`
- `"When should I use useEffect"`

#### Keyword Search
Best for: Finding specific terms, API references

**Example Query**: `useState`

**Try these queries**:
- `useEffect dependency array`
- `React.memo`
- `createContext`

#### Hybrid Search (Recommended)
Best for: Most situations, combines both approaches

**Example Query**: `"How to optimize React performance"`

### Step 2: Advanced Query Techniques

#### Using Quotes for Exact Phrases
```
Search: "React Strict Mode"
Result: Finds exact mentions of React Strict Mode
```

#### Excluding Terms
```
Search: hooks -class
Result: Information about hooks, excluding class-based content
```

#### Required Terms
```
Search: +React +performance +optimization
Result: Content that must contain all three terms
```

#### Wildcards
```
Search: use*
Result: Matches useState, useEffect, useContext, etc.
```

### Step 3: Using Search Filters

#### Filter by Source
1. Select specific sources from the dropdown
2. Useful when you know which documentation set contains your answer

#### Filter by Date
1. Use date ranges to find recently updated content
2. Helpful for finding latest API changes

#### Filter by Content Type
1. Focus on specific types of content
2. Useful in larger knowledge bases

### Step 4: Interpreting Results

Each result shows:
- **Relevance Score** (0.0-1.0): Higher is better
- **Source**: Which knowledge base
- **Snippet**: Key content preview
- **URL**: Direct link to original

**Good relevance scores**:
- 0.8-1.0: Excellent match
- 0.6-0.8: Good match
- 0.4-0.6: Moderate match
- Below 0.4: Consider rephrasing query

### Step 5: Search Strategy Workshop

Let's practice with these scenarios:

#### Scenario 1: Learning a New Concept
**Goal**: Understand React Context
**Strategy**: Start broad, then narrow down

```
Search 1: "What is React Context"
Search 2: "React Context vs Redux"
Search 3: "useContext hook examples"
```

#### Scenario 2: Solving a Specific Problem
**Goal**: Fix a performance issue
**Strategy**: Be specific about the problem

```
Search 1: "React component re-rendering too often"
Search 2: "useMemo vs useCallback performance"
Search 3: "React DevTools profiler"
```

#### Scenario 3: API Reference Lookup
**Goal**: Find specific API documentation
**Strategy**: Use exact API names

```
Search 1: useEffect
Search 2: "useEffect cleanup function"
Search 3: useEffect dependency array
```

### Step 6: Advanced Techniques

#### Multi-step Research
1. Start with a broad conceptual search
2. Use results to identify specific topics
3. Drill down with targeted searches
4. Cross-reference between sources

#### Building Search Queries
1. **Too many results**: Add more specific terms
2. **No results**: Remove terms or try synonyms
3. **Wrong results**: Use exclude operators (-)
4. **Missing context**: Try question format

---

## Tutorial 3: Setting Up API Integration

**Goal**: Learn to interact with KnowledgeHub programmatically using the REST API.

**Time**: 30 minutes

**Prerequisites**: Basic familiarity with REST APIs and curl/HTTP clients

### Step 1: API Authentication

First, get your API key from the web interface:

1. Navigate to **⚙️ Settings** (if available) or use the default development key
2. For this tutorial, we'll use direct API access (no auth required in development)

### Step 2: Explore the API Documentation

1. Open `http://localhost:3000/docs` in your browser
2. This shows the interactive OpenAPI documentation
3. Explore the available endpoints

### Step 3: Basic API Operations

#### Health Check
```bash
curl -X GET http://localhost:3000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "timestamp": 1640995200.123,
  "services": {
    "api": "operational",
    "database": "operational",
    "redis": "operational",
    "weaviate": "operational"
  }
}
```

#### List Sources
```bash
curl -X GET http://localhost:3000/api/v1/sources
```

**Expected Response**:
```json
{
  "sources": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "name": "React Documentation",
      "base_url": "https://react.dev",
      "status": "active",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 1
}
```

### Step 4: Create a Source via API

```bash
curl -X POST http://localhost:3000/api/v1/sources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Vue.js Documentation",
    "description": "Official Vue.js documentation",
    "base_url": "https://vuejs.org/guide/",
    "source_type": "web",
    "config": {
      "max_depth": 3,
      "max_pages": 500,
      "crawl_delay": 1.0
    }
  }'
```

**Expected Response**:
```json
{
  "id": "456e7890-e89b-12d3-a456-426614174001",
  "name": "Vue.js Documentation",
  "status": "active",
  "job_id": "789e0123-e89b-12d3-a456-426614174002"
}
```

### Step 5: Monitor Job Progress

```bash
# Get job details
curl -X GET http://localhost:3000/api/v1/jobs/789e0123-e89b-12d3-a456-426614174002
```

**Response includes**:
```json
{
  "id": "789e0123-e89b-12d3-a456-426614174002",
  "status": "running",
  "progress": 45,
  "total_pages": 0,
  "processed_pages": 23,
  "created_at": "2024-01-01T10:00:00Z",
  "started_at": "2024-01-01T10:00:05Z"
}
```

### Step 6: Perform Searches

#### Basic Search
```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "component composition",
    "search_type": "hybrid",
    "limit": 5
  }'
```

#### Advanced Search with Filters
```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "reactive data",
    "search_type": "semantic",
    "limit": 10,
    "filters": {
      "source_ids": ["456e7890-e89b-12d3-a456-426614174001"]
    }
  }'
```

### Step 7: Working with Documents

#### List Documents for a Source
```bash
curl -X GET "http://localhost:3000/api/v1/documents/?source_id=456e7890-e89b-12d3-a456-426614174001&limit=10"
```

#### Get Document Details
```bash
curl -X GET http://localhost:3000/api/v1/documents/doc-id-here
```

#### Get Document Chunks
```bash
curl -X GET http://localhost:3000/api/v1/documents/doc-id-here/chunks
```

### Step 8: Python Integration Example

Create a simple Python client:

```python
import requests
import json

class KnowledgeHubClient:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_source(self, name, base_url, **config):
        data = {
            "name": name,
            "base_url": base_url,
            "source_type": "web",
            "config": config
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/sources",
            json=data
        )
        return response.json()
    
    def search(self, query, search_type="hybrid", limit=10):
        data = {
            "query": query,
            "search_type": search_type,
            "limit": limit
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/search",
            json=data
        )
        return response.json()
    
    def get_job_status(self, job_id):
        response = self.session.get(
            f"{self.base_url}/api/v1/jobs/{job_id}"
        )
        return response.json()

# Usage example
client = KnowledgeHubClient()

# Create a source
result = client.create_source(
    name="Python Documentation",
    base_url="https://docs.python.org/3/",
    max_depth=3,
    max_pages=1000
)
print(f"Created source: {result['id']}")

# Search for content
results = client.search("list comprehensions")
for result in results["results"]:
    print(f"Score: {result['score']:.3f} - {result['title']}")
```

### Step 9: JavaScript/Node.js Integration

```javascript
class KnowledgeHubClient {
    constructor(baseUrl = 'http://localhost:3000') {
        this.baseUrl = baseUrl;
    }
    
    async createSource(name, baseUrl, config = {}) {
        const response = await fetch(`${this.baseUrl}/api/v1/sources`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name,
                base_url: baseUrl,
                source_type: 'web',
                config
            })
        });
        return response.json();
    }
    
    async search(query, searchType = 'hybrid', limit = 10) {
        const response = await fetch(`${this.baseUrl}/api/v1/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                search_type: searchType,
                limit
            })
        });
        return response.json();
    }
}

// Usage
const client = new KnowledgeHubClient();

// Create source
const source = await client.createSource(
    'MDN Documentation',
    'https://developer.mozilla.org/en-US/docs/',
    { max_depth: 3, max_pages: 2000 }
);

// Search
const results = await client.search('async await');
console.log(results);
```

---

## Tutorial 4: Monitoring and Optimization

**Goal**: Learn to monitor system performance and optimize KnowledgeHub for your use case.

**Time**: 25 minutes

**Prerequisites**: KnowledgeHub running with some sources configured

### Step 1: System Health Monitoring

#### Dashboard Overview
1. Navigate to the main dashboard
2. Review the system health indicators:
   - **API Status**: Should be "Operational"
   - **Database**: Connection status and query performance
   - **Redis**: Cache hit rate and memory usage
   - **Weaviate**: Vector database status and index size

#### Detailed Health Checks
```bash
# Overall system health
curl http://localhost:3000/health

# Individual service checks
curl http://localhost:3000/health/database
curl http://localhost:3000/health/redis
curl http://localhost:3000/health/weaviate
```

### Step 2: Performance Metrics

#### Search Performance
Monitor these key metrics:
- **Response Time**: Should be under 500ms for most queries
- **Cache Hit Rate**: Aim for 60%+ for repeated searches
- **Result Quality**: Relevance scores above 0.6

#### Crawling Performance
Track these indicators:
- **Pages per Minute**: Varies by site, typically 10-50
- **Error Rate**: Should be under 5%
- **Incremental Efficiency**: 95%+ pages skipped on updates

#### Resource Usage
```bash
# Check container resource usage
docker stats

# Check database performance
docker exec knowledgehub-postgres pg_stat_activity

# Check Redis memory usage
docker exec knowledgehub-redis redis-cli info memory
```

### Step 3: Job Queue Monitoring

#### Queue Depths
```bash
# Check pending jobs
docker exec knowledgehub-redis redis-cli llen crawl_jobs:pending
docker exec knowledgehub-redis redis-cli llen rag_jobs:pending

# Check processing jobs
docker exec knowledgehub-redis redis-cli scard crawl_jobs:processing
```

#### Queue Management
If queues are backing up:
1. **Scale workers**: Add more scraper/rag-processor containers
2. **Adjust batch sizes**: Reduce concurrent operations
3. **Optimize configuration**: Reduce crawl depth or page limits

### Step 4: Database Optimization

#### Index Performance
```sql
-- Connect to PostgreSQL
docker exec -it knowledgehub-postgres psql -U khuser knowledgehub

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE tablename IN ('knowledge_sources', 'documents', 'document_chunks');

-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

#### Database Maintenance
```sql
-- Analyze tables for better query planning
ANALYZE knowledge_sources;
ANALYZE documents;
ANALYZE document_chunks;

-- Vacuum to reclaim space
VACUUM ANALYZE;
```

### Step 5: Cache Optimization

#### Redis Cache Analysis
```bash
# Check cache hit rates
docker exec knowledgehub-redis redis-cli info stats | grep hit

# Check memory usage patterns
docker exec knowledgehub-redis redis-cli info memory

# Check key patterns
docker exec knowledgehub-redis redis-cli --scan --pattern "search:*" | wc -l
```

#### Cache Tuning
Update configuration based on usage:
```yaml
# In docker-compose.yml or environment
SEARCH_CACHE_TTL=7200      # Increase for stable content
SOURCE_CACHE_TTL=1800      # Increase for stable sources
JOB_CACHE_TTL=30           # Decrease for faster updates
```

### Step 6: Weaviate Optimization

#### Vector Database Health
```bash
# Check Weaviate status
curl http://localhost:8090/v1/meta

# Check collection statistics
curl http://localhost:8090/v1/objects | jq '.objects | length'

# Check memory usage
curl http://localhost:8090/v1/nodes
```

#### Performance Tuning
```json
{
  "vectorIndexConfig": {
    "distance": "cosine",
    "ef": 200,           // Increase for better recall
    "efConstruction": 128, // Increase for better indexing
    "maxConnections": 16   // Increase for faster search
  }
}
```

### Step 7: Scaling Strategies

#### Horizontal Scaling
Scale specific services based on bottlenecks:

```bash
# Scale scraper workers
docker compose up -d --scale scraper=3

# Scale RAG processors
docker compose up -d --scale rag-processor=2

# Scale API instances (with load balancer)
docker compose up -d --scale api=2
```

#### Resource Allocation
Adjust container resources:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Step 8: Performance Benchmarking

#### Search Performance Test
```bash
# Install Apache Bench
sudo apt install apache2-utils

# Create test search query
cat > search_query.json << 'EOF'
{
  "query": "installation guide",
  "search_type": "hybrid",
  "limit": 10
}
EOF

# Benchmark search performance
ab -n 100 -c 5 -T 'application/json' -p search_query.json \
   http://localhost:3000/api/v1/search
```

#### Crawling Performance Test
```bash
# Time a full crawl
time curl -X POST http://localhost:3000/api/v1/sources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Performance",
    "base_url": "https://httpbin.org/",
    "config": {"max_pages": 10}
  }'
```

### Step 9: Alerting Setup

#### Basic Monitoring Script
```bash
#!/bin/bash
# monitor.sh - Basic health monitoring

check_service() {
    local service=$1
    local url=$2
    
    if curl -f -s "$url" > /dev/null; then
        echo "✅ $service is healthy"
    else
        echo "❌ $service is unhealthy"
        # Add alerting logic here (email, webhook, etc.)
    fi
}

check_service "API" "http://localhost:3000/health"
check_service "Weaviate" "http://localhost:8090/v1/meta"

# Check queue depths
pending_jobs=$(docker exec knowledgehub-redis redis-cli llen crawl_jobs:pending)
if [ "$pending_jobs" -gt 100 ]; then
    echo "⚠️  High queue depth: $pending_jobs pending jobs"
fi
```

#### Integration with Monitoring Tools
For production, consider integrating with:
- **Prometheus + Grafana**: Metrics collection and visualization
- **ELK Stack**: Log aggregation and analysis
- **PagerDuty/Slack**: Alerting and incident management

---

## Tutorial 5: Incremental Crawling Setup

**Goal**: Understand and optimize incremental crawling for maximum performance benefits.

**Time**: 20 minutes

**Prerequisites**: At least one source configured and crawled

### Step 1: Understanding Incremental Crawling

Incremental crawling provides 95%+ speed improvements by:
- **Content Hashing**: SHA-256 fingerprinting of each page
- **Change Detection**: Comparing current vs stored hashes
- **Selective Processing**: Only processing new/changed content
- **Link Discovery**: Still following links to find new pages

### Step 2: Verify Incremental Setup

#### Check Existing Content Hashes
```bash
# Connect to database
docker exec -it knowledgehub-postgres psql -U khuser knowledgehub

# Check if content hashes exist
SELECT url, content_hash, updated_at 
FROM documents 
WHERE content_hash IS NOT NULL 
LIMIT 5;
```

If content_hash column is empty, the initial crawl didn't store hashes.

#### Force Hash Generation
```sql
-- For existing documents without hashes, trigger a re-crawl
UPDATE knowledge_sources 
SET last_crawl_at = NULL 
WHERE name = 'Your Source Name';
```

### Step 3: Test Incremental Performance

#### Baseline Test - Full Refresh
```bash
# Start with a known source
curl -X POST http://localhost:3000/api/v1/sources/SOURCE_ID/refresh \
  -H "Content-Type: application/json" \
  -d '{"force_refresh": true}'

# Note the job ID and time the completion
# Example result: 1,838 pages in 25 minutes
```

#### Incremental Test
```bash
# Wait 1-2 minutes, then refresh again (without force_refresh)
curl -X POST http://localhost:3000/api/v1/sources/SOURCE_ID/refresh

# Time this completion
# Example result: 1,838 pages checked in 30 seconds (95%+ faster)
```

### Step 4: Monitor Incremental Performance

#### Job Performance Metrics
When reviewing completed jobs, look for:
```json
{
  "result": {
    "pages_checked": 1838,
    "pages_processed": 12,
    "pages_unchanged": 1826,
    "pages_new": 8,
    "pages_updated": 4,
    "efficiency": 99.3,
    "time_saved_seconds": 1485
  }
}
```

#### Understanding the Numbers
- **pages_checked**: Total pages visited
- **pages_processed**: New/changed pages that went through full processing
- **efficiency**: Percentage of pages skipped (higher is better)
- **time_saved**: Estimated time saved vs full crawl

### Step 5: Optimizing for Incremental Crawling

#### Source Configuration
```json
{
  "max_depth": 4,
  "max_pages": 5000,
  "crawl_delay": 0.5,          // Reduced delay for checking unchanged pages
  "incremental_delay": 0.2,    // Even faster for hash checks
  "follow_patterns": ["**"],
  "exclude_patterns": [
    "**/search/**",             // Skip dynamic content
    "**/login/**",              // Skip auth pages
    "**/?page=*",               // Skip pagination
    "**/?sort=*"                // Skip sorting views
  ]
}
```

#### Exclude Dynamic Content
Pages that change frequently but aren't useful:
- Search result pages
- User-generated content sections
- Advertisement areas
- Timestamp-dependent pages

### Step 6: Content Hash Strategy

#### Understanding Hashing
The system calculates SHA-256 hashes of:
- Main content text (excluding navigation, ads, etc.)
- Normalized whitespace
- Cleaned HTML structure

#### Custom Hash Configuration
For sites with dynamic elements:
```json
{
  "content_extraction": {
    "exclude_selectors": [
      ".advertisement",
      ".user-comments", 
      ".last-updated",
      ".random-links"
    ],
    "normalize_whitespace": true,
    "ignore_case": false
  }
}
```

### Step 7: Scheduling Incremental Updates

#### Automated Scheduling
```json
{
  "scheduler": {
    "enabled": true,
    "refresh_schedule": "0 2 * * 0",  // Sunday 2 AM
    "incremental_by_default": true,
    "force_refresh_interval": 30      // Full refresh every 30 days
  }
}
```

#### Manual Scheduling
```bash
# Set up a cron job for regular incremental updates
0 */6 * * * curl -X POST http://localhost:3000/api/v1/sources/refresh-all
```

### Step 8: Troubleshooting Incremental Issues

#### All Pages Being Re-processed
**Symptoms**: Incremental crawl takes as long as full crawl

**Diagnosis**:
```sql
SELECT COUNT(*) as total_docs,
       COUNT(content_hash) as docs_with_hash,
       COUNT(*) - COUNT(content_hash) as missing_hashes
FROM documents;
```

**Solutions**:
1. Ensure content_hash column exists
2. Verify scraper is calling hash calculation
3. Check for database transaction issues

#### Missing New Content
**Symptoms**: New pages not being discovered

**Diagnosis**:
```bash
# Check if max_pages limit is being reached
curl http://localhost:3000/api/v1/jobs/JOB_ID
```

**Solutions**:
1. Increase max_pages limit
2. Review exclude patterns
3. Check follow patterns

#### False Change Detection
**Symptoms**: Pages marked as changed when they haven't

**Common Causes**:
- Dynamic timestamps in content
- Rotating advertisements
- Session-specific content
- Random elements

**Solutions**:
```json
{
  "content_cleaning": {
    "remove_patterns": [
      "Last updated: \\d{4}-\\d{2}-\\d{2}",
      "Session ID: [a-f0-9]+",
      "Advertisement \\d+"
    ]
  }
}
```

### Step 9: Advanced Incremental Features

#### Conditional Updates
Only update if significant changes:
```json
{
  "incremental_config": {
    "min_change_threshold": 0.1,  // 10% content change required
    "ignore_minor_changes": true,
    "change_detection_method": "fuzzy"
  }
}
```

#### Batch Hash Updates
For better performance:
```json
{
  "performance": {
    "batch_hash_updates": true,
    "batch_size": 100,
    "parallel_hash_calculation": 4
  }
}
```

---

## Tutorial 6: Bulk Source Management

**Goal**: Learn to efficiently manage multiple knowledge sources at scale.

**Time**: 30 minutes

**Prerequisites**: Basic familiarity with KnowledgeHub API

### Step 1: Planning Your Knowledge Architecture

Before adding multiple sources, plan your knowledge taxonomy:

#### Documentation Categories
```
├── Product Documentation
│   ├── API References
│   ├── User Guides
│   └── Developer Guides
├── Internal Knowledge
│   ├── Runbooks
│   ├── Architecture Docs
│   └── Process Documentation
└── External Resources
    ├── Technology Documentation
    ├── Best Practices
    └── Industry Standards
```

#### Source Naming Convention
Use consistent naming patterns:
- `{Category} - {Product/Service} - {Type}`
- Examples:
  - `API - GitHub - Reference`
  - `Guide - Docker - User Manual`
  - `Internal - Platform - Architecture`

### Step 2: Create Source Templates

#### High-Volume Documentation Site
```json
{
  "name": "Template - Large Documentation",
  "config": {
    "max_depth": 5,
    "max_pages": 5000,
    "crawl_delay": 1.0,
    "follow_patterns": [
      "**/docs/**",
      "**/guide/**", 
      "**/tutorial/**",
      "**/reference/**"
    ],
    "exclude_patterns": [
      "**/admin/**",
      "**/private/**",
      "**/search/**",
      "**/login/**",
      "**/*.pdf",
      "**/*.zip"
    ]
  }
}
```

#### API Documentation
```json
{
  "name": "Template - API Documentation",
  "config": {
    "max_depth": 3,
    "max_pages": 1000,
    "crawl_delay": 0.5,
    "follow_patterns": [
      "**/api/**",
      "**/reference/**",
      "**/endpoints/**"
    ],
    "exclude_patterns": [
      "**/v1/**",
      "**/deprecated/**",
      "**/beta/**"
    ]
  }
}
```

#### Blog/Article Sites
```json
{
  "name": "Template - Blog Content",
  "config": {
    "max_depth": 2,
    "max_pages": 500,
    "crawl_delay": 1.5,
    "follow_patterns": [
      "**/blog/**",
      "**/articles/**",
      "**/posts/**"
    ],
    "exclude_patterns": [
      "**/comments/**",
      "**/author/**",
      "**/tag/**",
      "**/category/**"
    ]
  }
}
```

### Step 3: Bulk Source Creation Script

#### Python Bulk Creation Script
```python
import requests
import json
import time
from typing import List, Dict

class BulkSourceManager:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_sources_from_config(self, sources_config: List[Dict]):
        results = []
        
        for source_config in sources_config:
            try:
                print(f"Creating source: {source_config['name']}")
                
                response = self.session.post(
                    f"{self.base_url}/api/v1/sources",
                    json=source_config
                )
                
                if response.status_code == 201:
                    result = response.json()
                    results.append({
                        'name': source_config['name'],
                        'id': result['id'],
                        'job_id': result.get('job_id'),
                        'status': 'created'
                    })
                    print(f"✅ Created: {result['id']}")
                else:
                    print(f"❌ Failed: {response.status_code} - {response.text}")
                    results.append({
                        'name': source_config['name'],
                        'status': 'failed',
                        'error': response.text
                    })
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error creating {source_config['name']}: {e}")
                results.append({
                    'name': source_config['name'],
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def monitor_crawl_jobs(self, job_ids: List[str]):
        """Monitor multiple crawl jobs until completion"""
        pending_jobs = set(job_ids)
        
        while pending_jobs:
            completed_in_cycle = set()
            
            for job_id in pending_jobs:
                try:
                    response = self.session.get(
                        f"{self.base_url}/api/v1/jobs/{job_id}"
                    )
                    
                    if response.status_code == 200:
                        job = response.json()
                        status = job['status']
                        
                        if status in ['completed', 'failed', 'cancelled']:
                            print(f"Job {job_id}: {status}")
                            if status == 'completed':
                                pages = job.get('processed_pages', 0)
                                print(f"  Processed {pages} pages")
                            completed_in_cycle.add(job_id)
                        else:
                            progress = job.get('progress', 0)
                            print(f"Job {job_id}: {status} ({progress}%)")
                
                except Exception as e:
                    print(f"Error checking job {job_id}: {e}")
            
            pending_jobs -= completed_in_cycle
            
            if pending_jobs:
                time.sleep(30)  # Check every 30 seconds

# Usage example
sources_config = [
    {
        "name": "React Documentation",
        "description": "Official React documentation",
        "base_url": "https://react.dev",
        "source_type": "web",
        "config": {
            "max_depth": 3,
            "max_pages": 200,
            "crawl_delay": 1.0
        }
    },
    {
        "name": "Vue.js Documentation", 
        "description": "Official Vue.js documentation",
        "base_url": "https://vuejs.org/guide/",
        "source_type": "web",
        "config": {
            "max_depth": 3,
            "max_pages": 300,
            "crawl_delay": 1.0
        }
    },
    {
        "name": "Angular Documentation",
        "description": "Official Angular documentation", 
        "base_url": "https://angular.io/docs",
        "source_type": "web",
        "config": {
            "max_depth": 4,
            "max_pages": 500,
            "crawl_delay": 1.0
        }
    }
]

manager = BulkSourceManager()
results = manager.create_sources_from_config(sources_config)

# Extract job IDs for monitoring
job_ids = [r['job_id'] for r in results if r.get('job_id')]
if job_ids:
    print(f"\nMonitoring {len(job_ids)} crawl jobs...")
    manager.monitor_crawl_jobs(job_ids)
```

### Step 4: Configuration Management

#### Environment-Based Configurations
```python
# config/development.json
{
  "default_config": {
    "max_depth": 2,
    "max_pages": 100,
    "crawl_delay": 0.5
  },
  "templates": {
    "small_site": {"max_pages": 50},
    "large_site": {"max_pages": 2000},
    "api_docs": {"max_depth": 3, "follow_patterns": ["**/api/**"]}
  }
}

# config/production.json  
{
  "default_config": {
    "max_depth": 4,
    "max_pages": 5000,
    "crawl_delay": 1.0
  }
}
```

#### CSV-Based Source Definition
```csv
name,base_url,template,max_pages,description
React Docs,https://react.dev,large_site,2000,Official React documentation
Vue Docs,https://vuejs.org/guide/,large_site,1500,Official Vue.js documentation
Angular Docs,https://angular.io/docs,large_site,3000,Official Angular documentation
Fastify API,https://fastify.dev/docs/,api_docs,500,Fastify web framework API
Express API,https://expressjs.com/,api_docs,300,Express.js web framework
```

### Step 5: Batch Operations

#### Refresh All Sources
```python
def refresh_all_sources(incremental=True):
    """Refresh all active sources"""
    
    # Get all sources
    response = requests.get(f"{base_url}/api/v1/sources")
    sources = response.json()['sources']
    
    job_ids = []
    
    for source in sources:
        if source['status'] == 'active':
            print(f"Refreshing: {source['name']}")
            
            refresh_data = {"force_refresh": not incremental}
            response = requests.post(
                f"{base_url}/api/v1/sources/{source['id']}/refresh",
                json=refresh_data
            )
            
            if response.status_code == 200:
                job_id = response.json().get('job_id')
                if job_id:
                    job_ids.append(job_id)
            
            time.sleep(1)  # Rate limiting
    
    return job_ids

# Refresh all sources incrementally
job_ids = refresh_all_sources(incremental=True)
```

#### Bulk Configuration Updates
```python
def update_source_configs(updates: Dict[str, Dict]):
    """Update configuration for multiple sources"""
    
    for source_id, new_config in updates.items():
        try:
            response = requests.patch(
                f"{base_url}/api/v1/sources/{source_id}",
                json={"config": new_config}
            )
            
            if response.status_code == 200:
                print(f"✅ Updated source {source_id}")
            else:
                print(f"❌ Failed to update {source_id}: {response.text}")
        
        except Exception as e:
            print(f"❌ Error updating {source_id}: {e}")

# Example: Increase crawl delay for all sources
updates = {
    "source-id-1": {"crawl_delay": 2.0},
    "source-id-2": {"crawl_delay": 2.0}, 
    "source-id-3": {"crawl_delay": 2.0}
}

update_source_configs(updates)
```

### Step 6: Source Health Monitoring

#### Health Check Script
```python
def check_source_health():
    """Check health of all sources"""
    
    response = requests.get(f"{base_url}/api/v1/sources")
    sources = response.json()['sources']
    
    health_report = {
        'healthy': [],
        'needs_attention': [],
        'failed': []
    }
    
    for source in sources:
        source_id = source['id']
        name = source['name']
        
        # Check last crawl
        last_crawl = source.get('last_crawl_at')
        if not last_crawl:
            health_report['needs_attention'].append({
                'name': name,
                'issue': 'Never crawled'
            })
            continue
        
        # Check recent jobs
        jobs_response = requests.get(
            f"{base_url}/api/v1/jobs?source_id={source_id}&limit=5"
        )
        
        if jobs_response.status_code == 200:
            jobs = jobs_response.json()['jobs']
            
            if jobs:
                latest_job = jobs[0]
                if latest_job['status'] == 'failed':
                    health_report['failed'].append({
                        'name': name,
                        'issue': latest_job.get('error_message', 'Unknown error')
                    })
                elif latest_job['status'] == 'completed':
                    health_report['healthy'].append(name)
                else:
                    health_report['needs_attention'].append({
                        'name': name,
                        'issue': f"Job status: {latest_job['status']}"
                    })
    
    return health_report

# Run health check
health = check_source_health()
print(f"Healthy sources: {len(health['healthy'])}")
print(f"Need attention: {len(health['needs_attention'])}")
print(f"Failed sources: {len(health['failed'])}")

for item in health['failed']:
    print(f"❌ {item['name']}: {item['issue']}")
```

### Step 7: Performance Optimization for Multiple Sources

#### Staggered Crawling
```python
def staggered_refresh(source_ids: List[str], batch_size=3, delay_between_batches=300):
    """Refresh sources in batches to avoid overwhelming the system"""
    
    for i in range(0, len(source_ids), batch_size):
        batch = source_ids[i:i + batch_size]
        
        print(f"Starting batch {i//batch_size + 1}: {len(batch)} sources")
        
        job_ids = []
        for source_id in batch:
            response = requests.post(
                f"{base_url}/api/v1/sources/{source_id}/refresh"
            )
            
            if response.status_code == 200:
                job_id = response.json().get('job_id')
                if job_id:
                    job_ids.append(job_id)
        
        # Wait for batch to complete before starting next
        if job_ids:
            monitor_crawl_jobs(job_ids)
        
        # Delay between batches
        if i + batch_size < len(source_ids):
            print(f"Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
```

#### Resource Usage Monitoring
```python
def monitor_resource_usage():
    """Monitor system resources during bulk operations"""
    
    import psutil
    
    # Check memory usage
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    
    # Check disk usage
    disk = psutil.disk_usage('/')
    print(f"Disk usage: {disk.percent}%")
    
    # Check queue depths
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        crawl_queue = r.llen('crawl_jobs:pending')
        rag_queue = r.llen('rag_jobs:pending')
        
        print(f"Crawl queue: {crawl_queue} jobs")
        print(f"RAG queue: {rag_queue} jobs")
        
    except Exception as e:
        print(f"Could not check queues: {e}")

# Monitor during bulk operations
monitor_resource_usage()
```

This tutorial provides comprehensive guidance for managing multiple knowledge sources efficiently at scale. The scripts and techniques shown can be adapted for specific organizational needs and deployment environments.

---

## Tutorial 7: Custom Embedding Workflows

**Goal**: Learn to customize and optimize the embedding generation process for your specific use case.

**Time**: 35 minutes

**Prerequisites**: Understanding of vector embeddings and some Python experience

### Step 1: Understanding the Embedding Pipeline

The default embedding workflow:
```
Text Chunks → Preprocessing → Model (all-MiniLM-L6-v2) → 384D Vectors → Weaviate Storage
```

#### Default Model Characteristics
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Max Length**: 256 tokens
- **Language**: English (optimized)
- **Use Case**: General purpose text similarity

### Step 2: Choosing Alternative Models

#### For Code Documentation
```python
# Better for code and technical content
models = [
    "microsoft/codebert-base",           # Code-specific embeddings
    "sentence-transformers/all-mpnet-base-v2",  # Higher quality, slower
    "jinaai/jina-embeddings-v2-base-en"  # Latest high-performance model
]
```

#### For Multilingual Content
```python
# Support for multiple languages
models = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased",
    "intfloat/multilingual-e5-large"
]
```

#### For Domain-Specific Content
```python
# Specialized models
models = [
    "allenai/scibert_scivocab_uncased",  # Scientific papers
    "emilyalsentzer/Bio_ClinicalBERT",   # Medical content
    "nlpaueb/legal-bert-base-uncased"    # Legal documents
]
```

### Step 3: Custom Embeddings Service Setup

#### Create Custom Embeddings Service
```python
# custom_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch

app = FastAPI()

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    dimensions: int

class CustomEmbeddingsService:
    def __init__(self):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_name: str):
        """Load model if not already cached"""
        if model_name not in self.models:
            print(f"Loading model: {model_name}")
            self.models[model_name] = SentenceTransformer(
                model_name, 
                device=self.device
            )
        return self.models[model_name]
    
    def generate_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings for texts using specified model"""
        model = self.load_model(model_name)
        
        # Batch processing for efficiency
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings

service = CustomEmbeddingsService()

@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    try:
        embeddings = service.generate_embeddings(
            request.texts, 
            request.model_name
        )
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model_name=request.model_name,
            dimensions=embeddings.shape[1]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_available_models():
    """List available models"""
    return {
        "loaded_models": list(service.models.keys()),
        "available_models": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2", 
            "microsoft/codebert-base",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
```

#### Docker Configuration
```dockerfile
# Dockerfile.custom-embeddings
FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn sentence-transformers torch numpy

COPY custom_embeddings.py .

EXPOSE 8100

CMD ["python", "custom_embeddings.py"]
```

### Step 4: Model Comparison Framework

#### Embedding Quality Test
```python
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingEvaluator:
    def __init__(self, embeddings_url="http://localhost:8100"):
        self.url = embeddings_url
    
    def test_model_quality(self, model_name: str, test_queries: List[str]):
        """Test model quality using predefined queries"""
        
        # Test cases: (query, expected_similar, expected_different)
        test_cases = [
            (
                "How to install Docker",
                ["Docker installation guide", "Setting up Docker"],
                ["Python tutorials", "Database optimization"]
            ),
            (
                "React component lifecycle", 
                ["useEffect hook", "Component mounting"],
                ["CSS styling", "Database queries"]
            ),
            (
                "API authentication methods",
                ["JWT tokens", "OAuth implementation"], 
                ["UI design", "File compression"]
            )
        ]
        
        results = []
        
        for query, similar_docs, different_docs in test_cases:
            # Get embeddings for query and documents
            all_texts = [query] + similar_docs + different_docs
            
            response = requests.post(
                f"{self.url}/embeddings",
                json={
                    "texts": all_texts,
                    "model_name": model_name
                }
            )
            
            if response.status_code == 200:
                embeddings = np.array(response.json()["embeddings"])
                query_embedding = embeddings[0:1]
                
                # Calculate similarities
                similarities = cosine_similarity(query_embedding, embeddings[1:]).flatten()
                
                # Split similarities
                similar_sims = similarities[:len(similar_docs)]
                different_sims = similarities[len(similar_docs):]
                
                # Calculate metrics
                avg_similar = np.mean(similar_sims)
                avg_different = np.mean(different_sims)
                separation = avg_similar - avg_different
                
                results.append({
                    "query": query,
                    "similar_avg": avg_similar,
                    "different_avg": avg_different,
                    "separation": separation
                })
        
        return results
    
    def compare_models(self, models: List[str], test_queries: List[str]):
        """Compare multiple models"""
        comparison = {}
        
        for model in models:
            print(f"Testing model: {model}")
            results = self.test_model_quality(model, test_queries)
            
            # Calculate overall scores
            avg_separation = np.mean([r["separation"] for r in results])
            avg_similar = np.mean([r["similar_avg"] for r in results])
            
            comparison[model] = {
                "avg_separation": avg_separation,
                "avg_similarity": avg_similar,
                "detailed_results": results
            }
        
        return comparison

# Usage
evaluator = EmbeddingEvaluator()

models_to_test = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "microsoft/codebert-base"
]

test_queries = [
    "Docker installation", 
    "React components",
    "API security"
]

comparison = evaluator.compare_models(models_to_test, test_queries)

# Print results
for model, scores in comparison.items():
    print(f"\n{model}:")
    print(f"  Separation Score: {scores['avg_separation']:.3f}")
    print(f"  Similarity Score: {scores['avg_similarity']:.3f}")
```

### Step 5: Chunking Strategy Optimization

#### Intelligent Chunking
```python
class SmartChunker:
    def __init__(self, model_name: str, max_tokens: int = 256):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
    
    def semantic_chunking(self, text: str, min_chunk_size: int = 100) -> List[str]:
        """Split text based on semantic boundaries"""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph exceeds token limit
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            tokens = self.tokenizer.encode(test_chunk)
            
            if len(tokens) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                # Current chunk is full, start new one
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too long, split it
                if len(self.tokenizer.encode(paragraph)) > self.max_tokens:
                    sub_chunks = self.split_long_paragraph(paragraph)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a paragraph that's too long for the model"""
        sentences = paragraph.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            tokens = self.tokenizer.encode(test_chunk)
            
            if len(tokens) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# Usage
chunker = SmartChunker("sentence-transformers/all-MiniLM-L6-v2")
chunks = chunker.semantic_chunking(long_text)
```

### Step 6: Domain-Specific Fine-tuning

#### Fine-tune for Your Domain
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def create_training_data():
    """Create training examples for your domain"""
    
    # Format: (anchor, positive, negative)
    examples = [
        InputExample(
            texts=["Docker container", "Container deployment", "Virtual machine"]
        ),
        InputExample(
            texts=["React hooks", "useState example", "Angular directives"] 
        ),
        InputExample(
            texts=["API authentication", "JWT token", "UI styling"]
        )
        # Add more examples based on your domain
    ]
    
    return examples

def fine_tune_model(base_model: str, training_examples: List[InputExample]):
    """Fine-tune a model for your specific domain"""
    
    # Load base model
    model = SentenceTransformer(base_model)
    
    # Create data loader
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=16)
    
    # Define loss function
    train_loss = losses.TripletLoss(model)
    
    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=100,
        output_path="./fine-tuned-model"
    )
    
    return model

# Create training data
training_data = create_training_data()

# Fine-tune model
fine_tuned_model = fine_tune_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    training_data
)
```

### Step 7: Multi-Model Ensemble

#### Ensemble Embeddings
```python
class EnsembleEmbedder:
    def __init__(self, model_configs: List[dict]):
        self.models = []
        for config in model_configs:
            model = SentenceTransformer(config["name"])
            self.models.append({
                "model": model,
                "weight": config.get("weight", 1.0),
                "name": config["name"]
            })
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate ensemble embeddings"""
        all_embeddings = []
        
        for model_info in self.models:
            embeddings = model_info["model"].encode(texts)
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Apply weight
            embeddings = embeddings * model_info["weight"]
            all_embeddings.append(embeddings)
        
        # Average weighted embeddings
        ensemble_embeddings = np.mean(all_embeddings, axis=0)
        
        # Normalize final embeddings
        ensemble_embeddings = ensemble_embeddings / np.linalg.norm(
            ensemble_embeddings, axis=1, keepdims=True
        )
        
        return ensemble_embeddings

# Configure ensemble
model_configs = [
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "weight": 1.0},
    {"name": "microsoft/codebert-base", "weight": 1.5},  # Higher weight for code
    {"name": "sentence-transformers/all-mpnet-base-v2", "weight": 0.8}
]

ensemble = EnsembleEmbedder(model_configs)
embeddings = ensemble.encode(["Your text here"])
```

### Step 8: Integration with KnowledgeHub

#### Custom RAG Processor
```python
# custom_rag_processor.py
import asyncio
import json
from typing import List, Dict
import httpx

class CustomRAGProcessor:
    def __init__(
        self, 
        api_url: str,
        embeddings_url: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.api_url = api_url
        self.embeddings_url = embeddings_url
        self.model_name = model_name
        self.client = httpx.AsyncClient()
    
    async def process_document(self, document: Dict) -> List[Dict]:
        """Process a document with custom embedding logic"""
        
        # Smart chunking
        chunker = SmartChunker(self.model_name)
        chunks = chunker.semantic_chunking(document["content"])
        
        # Generate embeddings
        embedding_response = await self.client.post(
            f"{self.embeddings_url}/embeddings",
            json={
                "texts": chunks,
                "model_name": self.model_name
            }
        )
        
        embeddings = embedding_response.json()["embeddings"]
        
        # Create chunk records
        chunk_records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_record = {
                "document_id": document["id"],
                "chunk_index": i,
                "content": chunk,
                "embedding": embedding,
                "metadata": {
                    "model_name": self.model_name,
                    "chunk_type": "semantic",
                    "token_count": len(chunk.split())
                }
            }
            chunk_records.append(chunk_record)
        
        return chunk_records
    
    async def store_chunks(self, chunks: List[Dict]):
        """Store chunks in the system"""
        for chunk in chunks:
            await self.client.post(
                f"{self.api_url}/api/v1/chunks",
                json=chunk
            )

# Usage in worker
processor = CustomRAGProcessor(
    api_url="http://localhost:3000",
    embeddings_url="http://localhost:8100",
    model_name="microsoft/codebert-base"  # Use code-specific model
)
```

This tutorial provides comprehensive guidance for customizing the embedding workflow to match your specific domain and quality requirements. The techniques can be mixed and matched based on your needs and computational resources.

---

## Tutorial 8: Production Deployment

**Goal**: Deploy KnowledgeHub to production with high availability, security, and monitoring.

**Time**: 45 minutes

**Prerequisites**: Basic knowledge of Docker, cloud platforms, and system administration

### Step 1: Production Architecture Planning

#### Infrastructure Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Web Frontend  │
│   (nginx/ALB)   │────│   (3 replicas)  │────│   (2 replicas)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│   Workers Pool  │──────────────┘
                        │ Scraper(2) +    │
                        │ RAG(2) +        │
                        │ Scheduler(1)    │
                        └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │    Weaviate     │
│   (Primary +    │    │   (Cluster)     │    │   (Cluster)     │
│   Read Replica) │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### Scaling Strategy
- **API Gateway**: Horizontal scaling behind load balancer
- **Workers**: Auto-scaling based on queue depth
- **Databases**: Read replicas and clustering
- **Storage**: Distributed object storage (S3/MinIO cluster)

### Step 2: Docker Compose Production Setup

#### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - api
      - web-ui
    restart: unless-stopped
    networks:
      - knowledgehub-network

  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - WEAVIATE_URL=${WEAVIATE_URL}
      - S3_ENDPOINT_URL=${S3_ENDPOINT_URL}
      - WORKERS=4
      - LOG_LEVEL=INFO
      - METRICS_ENABLED=true
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - knowledgehub-network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  web-ui:
    build:
      context: ./src/web-ui
      dockerfile: ../../docker/web-ui.Dockerfile
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
    environment:
      - VITE_API_URL=${VITE_API_URL}
      - VITE_WS_URL=${VITE_WS_URL}
    networks:
      - knowledgehub-network
    restart: unless-stopped

  scraper:
    build:
      context: .
      dockerfile: docker/scraper.Dockerfile
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - API_URL=http://api:3000
      - CONCURRENCY=2
    depends_on:
      - api
      - redis
    networks:
      - knowledgehub-network
    restart: unless-stopped

  rag-processor:
    build:
      context: .
      dockerfile: docker/rag-processor.Dockerfile
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - EMBEDDINGS_SERVICE_URL=http://embeddings-service:8100
      - WEAVIATE_URL=${WEAVIATE_URL}
    depends_on:
      - api
      - embeddings-service
    networks:
      - knowledgehub-network
    restart: unless-stopped

  scheduler:
    build:
      context: .
      dockerfile: docker/scheduler.Dockerfile
    deploy:
      replicas: 1
      resources:
        limits:
          cpus: '1'
          memory: 2G
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - API_URL=http://api:3000
      - SCHEDULER_ENABLED=true
    depends_on:
      - api
    networks:
      - knowledgehub-network
    restart: unless-stopped

  embeddings-service:
    build:
      context: .
      dockerfile: docker/embeddings.Dockerfile
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    environment:
      - MODEL_CACHE_DIR=/app/models
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - embeddings_models:/app/models
    networks:
      - knowledgehub-network
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_SHARED_PRELOAD_LIBRARIES=pg_stat_statements
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - knowledgehub-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 4gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - knowledgehub-network
    restart: unless-stopped

  weaviate:
    image: semitechnologies/weaviate:1.22.4
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=text2vec-transformers
      - ENABLE_MODULES=text2vec-transformers
      - TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080
      - CLUSTER_HOSTNAME=node1
    volumes:
      - weaviate_data:/var/lib/weaviate
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    networks:
      - knowledgehub-network
    restart: unless-stopped

  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
    environment:
      - ENABLE_CUDA=1
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    networks:
      - knowledgehub-network
    restart: unless-stopped

  # Monitoring stack
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - knowledgehub-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - knowledgehub-network
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  weaviate_data:
    driver: local
  embeddings_models:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  knowledgehub-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Step 3: Nginx Configuration

#### Production Nginx Config
```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        least_conn;
        server api:3000 max_fails=3 fail_timeout=30s;
    }
    
    upstream websocket_backend {
        ip_hash;  # Sticky sessions for WebSocket
        server api:3000 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=search:10m rate=5r/s;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Main application
    server {
        listen 80;
        server_name knowledgehub.example.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name knowledgehub.example.com;

        ssl_certificate /etc/ssl/certs/knowledgehub.crt;
        ssl_certificate_key /etc/ssl/certs/knowledgehub.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-SHA384;
        ssl_prefer_server_ciphers on;

        # Frontend
        location / {
            proxy_pass http://web-ui:3101;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Caching for static assets
            location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
                expires 1y;
                add_header Cache-Control "public, immutable";
            }
        }

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Search endpoint with stricter rate limiting
        location /api/v1/search {
            limit_req zone=search burst=10 nodelay;
            
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket connections
        location /ws {
            proxy_pass http://websocket_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket timeouts
            proxy_read_timeout 86400;
        }

        # Health check
        location /health {
            proxy_pass http://api_backend;
            access_log off;
        }
    }
}
```

### Step 4: Environment Configuration

#### Production Environment Variables
```bash
# .env.production
# Database
DATABASE_URL=postgresql://khuser:${DATABASE_PASSWORD}@postgres:5432/knowledgehub
POSTGRES_DB=knowledgehub
POSTGRES_USER=khuser
POSTGRES_PASSWORD=${DATABASE_PASSWORD}

# Redis
REDIS_URL=redis://redis:6379/0

# Weaviate
WEAVIATE_URL=http://weaviate:8080

# Object Storage (S3/MinIO)
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY=${AWS_ACCESS_KEY_ID}
S3_SECRET_KEY=${AWS_SECRET_ACCESS_KEY}
S3_BUCKET=knowledgehub-prod
S3_REGION=us-east-1

# API Configuration
API_HOST=0.0.0.0
API_PORT=3000
DEBUG=false
SECRET_KEY=${SECRET_KEY}
WORKERS=4

# Frontend
VITE_API_URL=https://knowledgehub.example.com
VITE_WS_URL=wss://knowledgehub.example.com

# Security
API_RATE_LIMIT=1000
SEARCH_RATE_LIMIT=100
SESSION_TIMEOUT=3600
CORS_ORIGINS=https://knowledgehub.example.com

# Monitoring
LOG_LEVEL=INFO
JSON_LOGGING=true
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Grafana
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
```

### Step 5: SSL/TLS Certificate Setup

#### Let's Encrypt with Certbot
```bash
# Install Certbot
sudo apt install certbot

# Generate certificates
sudo certbot certonly --standalone \
  -d knowledgehub.example.com \
  -d api.knowledgehub.example.com

# Auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

#### Certificate Deployment Script
```bash
#!/bin/bash
# deploy-certs.sh

DOMAIN="knowledgehub.example.com"
CERT_PATH="/etc/letsencrypt/live/$DOMAIN"
NGINX_CERT_PATH="./ssl"

# Create nginx cert directory
mkdir -p $NGINX_CERT_PATH

# Copy certificates
sudo cp $CERT_PATH/fullchain.pem $NGINX_CERT_PATH/knowledgehub.crt
sudo cp $CERT_PATH/privkey.pem $NGINX_CERT_PATH/knowledgehub.key

# Set permissions
sudo chown root:root $NGINX_CERT_PATH/*
sudo chmod 644 $NGINX_CERT_PATH/knowledgehub.crt
sudo chmod 600 $NGINX_CERT_PATH/knowledgehub.key

echo "Certificates deployed successfully"
```

### Step 6: Monitoring Setup

#### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files: []

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

#### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "KnowledgeHub Production Metrics",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(api_requests_total{status_code=~\"5..\"}[5m]) / rate(api_requests_total[5m]) * 100",
            "legendFormat": "Error %"
          }
        ]
      },
      {
        "title": "Queue Depths",
        "type": "graph",
        "targets": [
          {
            "expr": "queue_size{queue_name=\"crawl_jobs:pending\"}",
            "legendFormat": "Crawl Queue"
          },
          {
            "expr": "queue_size{queue_name=\"rag_jobs:pending\"}",
            "legendFormat": "RAG Queue"
          }
        ]
      }
    ]
  }
}
```

### Step 7: Backup and Recovery

#### Automated Backup Script
```bash
#!/bin/bash
# backup.sh

set -e

BACKUP_DIR="/opt/backups/knowledgehub/$(date +%Y-%m-%d-%H%M%S)"
S3_BUCKET="knowledgehub-backups"
RETENTION_DAYS=30

echo "Starting backup to $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Database backup
echo "Backing up PostgreSQL..."
docker exec knowledgehub-postgres pg_dump -U khuser knowledgehub | gzip > "$BACKUP_DIR/database.sql.gz"

# Redis backup
echo "Backing up Redis..."
docker exec knowledgehub-redis redis-cli --rdb - | gzip > "$BACKUP_DIR/redis.rdb.gz"

# Weaviate backup (if supported)
echo "Backing up Weaviate..."
curl -X POST "http://localhost:8090/v1/backups" \
  -H "Content-Type: application/json" \
  -d '{"id": "backup-'$(date +%s)'"}' || echo "Weaviate backup failed"

# Configuration backup
echo "Backing up configuration..."
cp .env.production "$BACKUP_DIR/"
cp docker-compose.prod.yml "$BACKUP_DIR/"
cp -r nginx/ "$BACKUP_DIR/"

# Upload to S3
echo "Uploading to S3..."
aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/$(basename $BACKUP_DIR)/"

# Cleanup old local backups
echo "Cleaning up old backups..."
find /opt/backups/knowledgehub -name "20*" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} +

# Cleanup old S3 backups
aws s3 ls "s3://$S3_BUCKET/" | while read -r line; do
  backup_date=$(echo $line | awk '{print $2}' | cut -d'-' -f1-3)
  if [[ $(date -d "$backup_date" +%s) -lt $(date -d "$RETENTION_DAYS days ago" +%s) ]]; then
    backup_name=$(echo $line | awk '{print $2}')
    aws s3 rm "s3://$S3_BUCKET/$backup_name" --recursive
  fi
done

echo "Backup completed successfully"
```

#### Recovery Procedures
```bash
#!/bin/bash
# restore.sh

BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ]; then
  echo "Usage: $0 <backup_directory>"
  exit 1
fi

echo "Restoring from $BACKUP_DIR"

# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore database
echo "Restoring PostgreSQL..."
docker-compose -f docker-compose.prod.yml up -d postgres
sleep 10
gunzip -c "$BACKUP_DIR/database.sql.gz" | docker exec -i knowledgehub-postgres psql -U khuser knowledgehub

# Restore Redis
echo "Restoring Redis..."
docker-compose -f docker-compose.prod.yml up -d redis
sleep 5
gunzip -c "$BACKUP_DIR/redis.rdb.gz" | docker exec -i knowledgehub-redis redis-cli --pipe

# Restore configuration
echo "Restoring configuration..."
cp "$BACKUP_DIR/.env.production" .
cp "$BACKUP_DIR/docker-compose.prod.yml" .
cp -r "$BACKUP_DIR/nginx/" .

# Start all services
echo "Starting all services..."
docker-compose -f docker-compose.prod.yml up -d

echo "Restoration completed"
```

### Step 8: Deployment Automation

#### CI/CD with GitHub Actions
```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup SSH
      uses: webfactory/ssh-agent@v0.7.0
      with:
        ssh-private-key: ${{ secrets.DEPLOY_SSH_KEY }}
    
    - name: Deploy to production
      run: |
        ssh -o StrictHostKeyChecking=no ${{ secrets.DEPLOY_USER }}@${{ secrets.DEPLOY_HOST }} << 'EOF'
          cd /opt/knowledgehub
          
          # Backup current deployment
          docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U khuser knowledgehub > backup-pre-deploy.sql
          
          # Pull latest changes
          git pull origin main
          
          # Build and deploy
          docker-compose -f docker-compose.prod.yml build --no-cache
          docker-compose -f docker-compose.prod.yml up -d
          
          # Health check
          sleep 30
          curl -f http://localhost/health || exit 1
          
          echo "Deployment successful"
        EOF
```

#### Rolling Update Script
```bash
#!/bin/bash
# rolling-update.sh

SERVICES=("api" "web-ui" "scraper" "rag-processor")

for service in "${SERVICES[@]}"; do
  echo "Updating $service..."
  
  # Build new image
  docker-compose -f docker-compose.prod.yml build $service
  
  # Get current replicas
  replicas=$(docker service ls --filter name=${service} --format "{{.Replicas}}")
  
  if [ -n "$replicas" ]; then
    # Rolling update for swarm mode
    docker service update --image knowledgehub_${service}:latest knowledgehub_${service}
  else
    # Standard compose update
    docker-compose -f docker-compose.prod.yml up -d --no-deps $service
  fi
  
  # Health check
  sleep 10
  curl -f http://localhost/health || {
    echo "Health check failed for $service"
    exit 1
  }
  
  echo "$service updated successfully"
done

echo "Rolling update completed"
```

This comprehensive tutorial covers all aspects of deploying KnowledgeHub to production with enterprise-grade reliability, security, and monitoring. The configuration can be adapted for different cloud providers and deployment scenarios.