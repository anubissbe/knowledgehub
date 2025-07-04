# Tutorials

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

### Step 6: Test Your First Search

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
- Check that Weaviate service is running

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

### Step 3: Interpreting Results

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

### Step 4: Search Strategy Workshop

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

---

## Tutorial 3: Setting Up API Integration

**Goal**: Learn to interact with KnowledgeHub programmatically using the REST API.

**Time**: 30 minutes

**Prerequisites**: Basic familiarity with REST APIs and curl/HTTP clients

### Step 1: Explore the API Documentation

1. Open `http://localhost:3000/docs` in your browser
2. This shows the interactive OpenAPI documentation
3. Explore the available endpoints

### Step 2: Basic API Operations

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

### Step 3: Create a Source via API

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

### Step 4: Python Integration Example

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

### Step 5: JavaScript/Node.js Integration

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

### Step 3: Database Optimization

#### Check Slow Queries
```bash
# Connect to PostgreSQL
docker exec -it knowledgehub-postgres psql -U khuser knowledgehub

# Check slow queries
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

### Step 4: Cache Optimization

#### Redis Cache Analysis
```bash
# Check cache hit rates
docker exec knowledgehub-redis redis-cli info stats | grep hit

# Check memory usage patterns
docker exec knowledgehub-redis redis-cli info memory
```

### Step 5: Performance Benchmarking

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

---

## Tutorial 5: Incremental Crawling Setup

**Goal**: Understand and optimize incremental crawling for maximum performance benefits.

**Time**: 20 minutes

**Prerequisites**: At least one source configured and crawled

### Understanding Incremental Crawling

Incremental crawling provides 95%+ speed improvements by:
- **Content Hashing**: SHA-256 fingerprinting of each page
- **Change Detection**: Comparing current vs stored hashes
- **Selective Processing**: Only processing new/changed content
- **Link Discovery**: Still following links to find new pages

### Test Incremental Performance

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

### Monitor Incremental Performance

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

### Optimizing for Incremental Crawling

#### Source Configuration
```json
{
  "max_depth": 4,
  "max_pages": 5000,
  "crawl_delay": 0.5,
  "incremental_delay": 0.2,
  "follow_patterns": ["**"],
  "exclude_patterns": [
    "**/search/**",
    "**/login/**",
    "**/?page=*",
    "**/?sort=*"
  ]
}
```

---

## Tutorial 6: Bulk Source Management

**Goal**: Learn to efficiently manage multiple knowledge sources at scale.

**Time**: 30 minutes

**Prerequisites**: Basic familiarity with KnowledgeHub API

### Planning Your Knowledge Architecture

Before adding multiple sources, plan your knowledge taxonomy:

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

### Python Bulk Creation Script

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
                    print(f"❌ Failed: {response.status_code}")
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error creating {source_config['name']}: {e}")
        
        return results

# Usage example
sources_config = [
    {
        "name": "React Documentation",
        "base_url": "https://react.dev",
        "source_type": "web",
        "config": {
            "max_depth": 3,
            "max_pages": 200
        }
    },
    {
        "name": "Vue.js Documentation",
        "base_url": "https://vuejs.org/guide/",
        "source_type": "web",
        "config": {
            "max_depth": 3,
            "max_pages": 300
        }
    }
]

manager = BulkSourceManager()
results = manager.create_sources_from_config(sources_config)
```

### Source Health Monitoring

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
        # Check last crawl
        last_crawl = source.get('last_crawl_at')
        if not last_crawl:
            health_report['needs_attention'].append({
                'name': source['name'],
                'issue': 'Never crawled'
            })
            continue
        
        # Check recent jobs
        jobs_response = requests.get(
            f"{base_url}/api/v1/jobs?source_id={source['id']}&limit=5"
        )
        
        if jobs_response.status_code == 200:
            jobs = jobs_response.json()['jobs']
            
            if jobs and jobs[0]['status'] == 'failed':
                health_report['failed'].append({
                    'name': source['name'],
                    'issue': jobs[0].get('error_message', 'Unknown error')
                })
            else:
                health_report['healthy'].append(source['name'])
    
    return health_report
```

---

## Tutorial 7: Custom Embedding Workflows

**Goal**: Learn to customize and optimize the embedding generation process for your specific use case.

**Time**: 35 minutes

**Prerequisites**: Understanding of vector embeddings and some Python experience

### Understanding the Embedding Pipeline

The default embedding workflow:
```
Text Chunks → Preprocessing → Model (all-MiniLM-L6-v2) → 384D Vectors → Weaviate Storage
```

### Choosing Alternative Models

#### For Code Documentation
```python
# Better for code and technical content
models = [
    "microsoft/codebert-base",
    "sentence-transformers/all-mpnet-base-v2",
    "jinaai/jina-embeddings-v2-base-en"
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

### Custom Embeddings Service

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

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
    
    def generate_embeddings(self, texts: List[str], model_name: str):
        """Generate embeddings for texts"""
        model = self.load_model(model_name)
        
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings

service = CustomEmbeddingsService()

@app.post("/embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        embeddings = service.generate_embeddings(
            request.texts, 
            request.model_name
        )
        
        return {
            "embeddings": embeddings.tolist(),
            "model_name": request.model_name,
            "dimensions": embeddings.shape[1]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Intelligent Chunking

```python
class SmartChunker:
    def __init__(self, model_name: str, max_tokens: int = 256):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
    
    def semantic_chunking(self, text: str, min_chunk_size: int = 100):
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
                current_chunk = paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
```

---

## Tutorial 8: Production Deployment

**Goal**: Deploy KnowledgeHub to production with high availability, security, and monitoring.

**Time**: 45 minutes

**Prerequisites**: Basic knowledge of Docker, cloud platforms, and system administration

### Production Architecture

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

### Production Docker Compose

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
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - WORKERS=4
      - LOG_LEVEL=INFO
    restart: unless-stopped

  # Additional services...
```

### SSL/TLS Certificate Setup

```bash
# Install Certbot
sudo apt install certbot

# Generate certificates
sudo certbot certonly --standalone \
  -d knowledgehub.example.com

# Auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

### Monitoring Setup

#### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']
```

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/opt/backups/knowledgehub/$(date +%Y-%m-%d-%H%M%S)"
S3_BUCKET="knowledgehub-backups"

echo "Starting backup to $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Database backup
docker exec knowledgehub-postgres pg_dump -U khuser knowledgehub | gzip > "$BACKUP_DIR/database.sql.gz"

# Redis backup
docker exec knowledgehub-redis redis-cli --rdb - | gzip > "$BACKUP_DIR/redis.rdb.gz"

# Upload to S3
aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/$(basename $BACKUP_DIR)/"

echo "Backup completed successfully"
```

### CI/CD with GitHub Actions

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
    
    - name: Deploy to production
      run: |
        ssh ${{ secrets.DEPLOY_USER }}@${{ secrets.DEPLOY_HOST }} << 'EOF'
          cd /opt/knowledgehub
          
          # Pull latest changes
          git pull origin main
          
          # Build and deploy
          docker-compose -f docker-compose.prod.yml build
          docker-compose -f docker-compose.prod.yml up -d
          
          # Health check
          sleep 30
          curl -f http://localhost/health || exit 1
          
          echo "Deployment successful"
        EOF
```

---

## Next Steps

Congratulations on completing these tutorials! You now have a comprehensive understanding of KnowledgeHub. Here are some next steps:

1. **Explore Advanced Features**: Try custom models, fine-tuning, and enterprise integrations
2. **Optimize Performance**: Use the monitoring tools to identify and fix bottlenecks
3. **Scale Your Deployment**: Add more sources and expand to production
4. **Contribute**: Share your experiences and contribute to the project

For more information:
- [User Guide](User-Guide) - Complete feature reference
- [API Documentation](API-Documentation) - Full API reference
- [Architecture Overview](Architecture) - System design details
- [Troubleshooting Guide](Troubleshooting) - Common issues and solutions