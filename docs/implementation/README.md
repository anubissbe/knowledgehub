# AI Knowledge Hub - Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the AI Knowledge Hub system. We'll build the system incrementally, starting with core components and progressively adding features.

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git
- 8GB+ RAM
- 20GB+ free disk space

## Phase 1: Project Setup & Core Infrastructure

### 1.1 Initialize Project Structure

```bash
# Create project directory
mkdir ai-knowledge-hub && cd ai-knowledge-hub

# Initialize git repository
git init

# Create directory structure
mkdir -p src/{api,mcp-server,scraper,rag-processor,web-ui}
mkdir -p docs/{architecture,implementation,deployment,api}
mkdir -p scripts docker k8s tests
mkdir -p data/{postgres,redis,weaviate}

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements files
touch requirements.txt requirements-dev.txt
touch src/{api,mcp-server,scraper,rag-processor}/requirements.txt
```

### 1.2 Docker Development Environment

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # PostgreSQL for document metadata
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: knowledgehub
      POSTGRES_USER: khuser
      POSTGRES_PASSWORD: khpassword
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U khuser -d knowledgehub"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for caching and message queue
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - ./data/redis:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Weaviate vector database
  weaviate:
    image: semitechnologies/weaviate:latest
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
    volumes:
      - ./data/weaviate:/var/lib/weaviate
    ports:
      - "8080:8080"
    depends_on:
      - t2v-transformers

  # Transformer model for embeddings
  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
    environment:
      ENABLE_CUDA: '0'

  # MinIO for blob storage (S3-compatible)
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - ./data/minio:/data
    ports:
      - "9000:9000"
      - "9001:9001"
```

### 1.3 Environment Configuration

Create `.env`:

```env
# Application
APP_NAME=AI Knowledge Hub
APP_ENV=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=3000
API_WORKERS=4

# Database URLs
DATABASE_URL=postgresql://khuser:khpassword@localhost:5432/knowledgehub
REDIS_URL=redis://localhost:6379
WEAVIATE_URL=http://localhost:8080

# S3/MinIO Configuration
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=knowledge-hub

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Scraping Configuration
MAX_CONCURRENT_SCRAPERS=5
SCRAPER_TIMEOUT=30000
RATE_LIMIT_REQUESTS_PER_SECOND=1
USER_AGENT=Mozilla/5.0 (compatible; KnowledgeHubBot/1.0)

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24
```

## Phase 2: Database Setup & Models

### 2.1 PostgreSQL Schema

Create `src/api/database/schema.sql`:

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Knowledge sources table
CREATE TABLE knowledge_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL UNIQUE,
    status VARCHAR(50) DEFAULT 'pending',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_scraped_at TIMESTAMP,
    stats JSONB DEFAULT '{}'
);

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT,
    content_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, url)
);

-- Document chunks table (for tracking)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(50),
    content TEXT NOT NULL,
    embedding_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Scraping jobs table
CREATE TABLE scraping_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT,
    stats JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_sources_status ON knowledge_sources(status);
CREATE INDEX idx_documents_source ON documents(source_id);
CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_jobs_source ON scraping_jobs(source_id);
CREATE INDEX idx_jobs_status ON scraping_jobs(status);
```

### 2.2 SQLAlchemy Models

Create `src/api/models/base.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Create `src/api/models/knowledge_source.py`:

```python
from sqlalchemy import Column, String, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from .base import Base

class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    url = Column(Text, nullable=False, unique=True)
    status = Column(String(50), default="pending")
    config = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scraped_at = Column(DateTime, nullable=True)
    stats = Column(JSON, default={})
```

## Phase 3: MCP Server Implementation

### 3.1 MCP Protocol Server

Create `src/mcp-server/server.py`:

```python
import asyncio
import json
from typing import Dict, List, Any, Optional
from fastmcp import FastMCP
from datetime import datetime
import logging

from .tools import KnowledgeTools
from .resources import KnowledgeResources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeHubMCPServer:
    def __init__(self, knowledge_api_url: str):
        self.mcp = FastMCP("AI Knowledge Hub")
        self.tools = KnowledgeTools(knowledge_api_url)
        self.resources = KnowledgeResources(knowledge_api_url)
        self.setup_server()
    
    def setup_server(self):
        """Register all MCP tools and resources"""
        
        # Search tool
        @self.mcp.tool()
        async def search_knowledge(
            query: str,
            source_filter: Optional[str] = None,
            limit: int = 10,
            include_metadata: bool = True
        ) -> str:
            """
            Search the knowledge base using hybrid search.
            
            Args:
                query: Natural language search query
                source_filter: Optional source name to filter results
                limit: Maximum number of results (default: 10)
                include_metadata: Include document metadata in results
            
            Returns:
                JSON string with search results
            """
            try:
                results = await self.tools.search(
                    query=query,
                    source_filter=source_filter,
                    limit=limit,
                    include_metadata=include_metadata
                )
                return json.dumps(results, indent=2)
            except Exception as e:
                logger.error(f"Search error: {e}")
                return json.dumps({"error": str(e)})
        
        # Memory storage tool
        @self.mcp.tool()
        async def store_memory(
            content: str,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None
        ) -> str:
            """
            Store information in the knowledge base.
            
            Args:
                content: Text content to store
                tags: Optional list of tags for categorization
                metadata: Optional metadata dictionary
            
            Returns:
                Confirmation message with memory ID
            """
            try:
                memory_id = await self.tools.store_memory(
                    content=content,
                    tags=tags or [],
                    metadata=metadata or {}
                )
                return f"Memory stored successfully with ID: {memory_id}"
            except Exception as e:
                logger.error(f"Store error: {e}")
                return f"Error storing memory: {str(e)}"
        
        # Context retrieval tool
        @self.mcp.tool()
        async def get_relevant_context(
            query: str,
            max_tokens: int = 4000,
            recency_weight: float = 0.1
        ) -> str:
            """
            Retrieve relevant context for a query.
            
            Args:
                query: Context query
                max_tokens: Maximum tokens to return
                recency_weight: Weight for recent documents (0-1)
            
            Returns:
                Relevant context as formatted text
            """
            try:
                context = await self.tools.get_context(
                    query=query,
                    max_tokens=max_tokens,
                    recency_weight=recency_weight
                )
                return context
            except Exception as e:
                logger.error(f"Context error: {e}")
                return f"Error retrieving context: {str(e)}"
        
        # List sources tool
        @self.mcp.tool()
        async def list_knowledge_sources() -> str:
            """
            List all available knowledge sources.
            
            Returns:
                JSON string with source information
            """
            try:
                sources = await self.tools.list_sources()
                return json.dumps(sources, indent=2)
            except Exception as e:
                logger.error(f"List sources error: {e}")
                return json.dumps({"error": str(e)})
        
        # Register resources
        @self.mcp.resource("knowledge://sources")
        async def get_sources_resource() -> str:
            """Get detailed information about all knowledge sources"""
            return await self.resources.get_sources_details()
        
        @self.mcp.resource("knowledge://stats")
        async def get_stats_resource() -> str:
            """Get system statistics and metrics"""
            return await self.resources.get_system_stats()
        
        @self.mcp.resource("knowledge://health")
        async def get_health_resource() -> str:
            """Get system health status"""
            return await self.resources.get_health_status()
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Knowledge Hub MCP Server...")
        await self.mcp.run()

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    server = KnowledgeHubMCPServer(
        knowledge_api_url=os.getenv("API_URL", "http://localhost:3000")
    )
    
    asyncio.run(server.run())
```

### 3.2 MCP Tools Implementation

Create `src/mcp-server/tools.py`:

```python
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

class KnowledgeTools:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = None
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def search(
        self,
        query: str,
        source_filter: Optional[str] = None,
        limit: int = 10,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Execute hybrid search against the knowledge base"""
        await self._ensure_session()
        
        params = {
            "q": query,
            "limit": limit,
            "include_metadata": include_metadata
        }
        
        if source_filter:
            params["source"] = source_filter
        
        async with self.session.get(
            f"{self.api_url}/api/v1/search",
            params=params
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Format results for AI consumption
            formatted_results = []
            for result in data.get("results", []):
                formatted_results.append({
                    "content": result["content"],
                    "source": result["source_name"],
                    "url": result["url"],
                    "score": result["score"],
                    "metadata": result.get("metadata", {}) if include_metadata else None
                })
            
            return {
                "query": query,
                "count": len(formatted_results),
                "results": formatted_results
            }
    
    async def store_memory(
        self,
        content: str,
        tags: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """Store a memory item in the knowledge base"""
        await self._ensure_session()
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        payload = {
            "content": content,
            "content_hash": content_hash,
            "tags": tags,
            "metadata": {
                **metadata,
                "stored_at": datetime.utcnow().isoformat(),
                "stored_by": "mcp_client"
            }
        }
        
        async with self.session.post(
            f"{self.api_url}/api/v1/memories",
            json=payload
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["id"]
    
    async def get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        recency_weight: float = 0.1
    ) -> str:
        """Get relevant context optimized for AI consumption"""
        # First, perform a search
        search_results = await self.search(query, limit=20)
        
        # Sort by score and recency
        results = search_results["results"]
        for result in results:
            # Adjust score based on recency if metadata available
            if result.get("metadata", {}).get("updated_at"):
                # Implementation of recency scoring
                pass
        
        # Build context within token limit
        context_parts = [
            f"# Relevant Context for: {query}\n"
        ]
        
        current_tokens = 50  # Rough estimate for header
        
        for i, result in enumerate(results):
            # Rough token estimation (1 token ≈ 4 chars)
            result_text = f"\n## [{i+1}] {result['source']} - {result['url']}\n{result['content']}\n"
            result_tokens = len(result_text) // 4
            
            if current_tokens + result_tokens > max_tokens:
                break
            
            context_parts.append(result_text)
            current_tokens += result_tokens
        
        return "\n".join(context_parts)
    
    async def list_sources(self) -> List[Dict[str, Any]]:
        """List all knowledge sources"""
        await self._ensure_session()
        
        async with self.session.get(
            f"{self.api_url}/api/v1/sources"
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            return [{
                "name": source["name"],
                "url": source["url"],
                "status": source["status"],
                "documents": source.get("stats", {}).get("documents", 0),
                "last_updated": source.get("last_scraped_at", "Never")
            } for source in data]
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
```

## Phase 4: Web Scraping Engine

### 4.1 Scraping Worker Implementation

Create `src/scraper/worker.py`:

```python
import asyncio
from typing import Dict, List, Optional, Any
from playwright.async_api import async_playwright, Page, Browser
import hashlib
from bs4 import BeautifulSoup
import json
import logging
from urllib.parse import urljoin, urlparse
from datetime import datetime

from .content_extractor import StructuredContentExtractor
from .rate_limiter import TokenBucket

logger = logging.getLogger(__name__)

class ScrapingWorker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.browser: Optional[Browser] = None
        self.rate_limiter = TokenBucket(
            rate=config.get("rate_limit_rps", 1),
            capacity=config.get("rate_limit_burst", 10)
        )
        self.content_extractor = StructuredContentExtractor()
    
    async def start(self):
        """Initialize the browser instance"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox'
            ]
        )
    
    async def stop(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
    
    async def scrape_page(self, url: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape a single page and extract structured content"""
        await self.rate_limiter.acquire()
        
        page = await self.browser.new_page()
        try:
            # Configure page
            await self._configure_page(page, context)
            
            # Navigate to URL
            logger.info(f"Navigating to: {url}")
            response = await page.goto(
                url,
                wait_until="networkidle",
                timeout=self.config.get("timeout", 30000)
            )
            
            if not response or response.status >= 400:
                raise Exception(f"HTTP {response.status if response else 'No response'}")
            
            # Wait for dynamic content
            await self._wait_for_content(page)
            
            # Extract structured content
            html = await page.content()
            structured_content = await self._extract_structured_content(
                page, html, url
            )
            
            # Extract links for crawling
            links = await self._extract_links(page, url)
            
            return {
                "url": url,
                "status": "success",
                "content": structured_content,
                "links": links,
                "metadata": {
                    "title": await page.title(),
                    "scraped_at": datetime.utcnow().isoformat(),
                    "content_hash": hashlib.sha256(html.encode()).hexdigest()
                }
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "status": "error",
                "error": str(e)
            }
        finally:
            await page.close()
    
    async def _configure_page(self, page: Page, context: Dict[str, Any]):
        """Configure page settings and authentication"""
        # Set user agent
        await page.set_extra_http_headers({
            "User-Agent": self.config.get("user_agent", "KnowledgeHubBot/1.0")
        })
        
        # Handle authentication if needed
        if context.get("auth"):
            auth = context["auth"]
            if auth["type"] == "basic":
                await page.set_extra_http_headers({
                    "Authorization": f"Basic {auth['credentials']}"
                })
            elif auth["type"] == "form":
                # Navigate to login page and fill form
                await page.goto(auth["login_url"])
                await page.fill(auth["username_selector"], auth["username"])
                await page.fill(auth["password_selector"], auth["password"])
                await page.click(auth["submit_selector"])
                await page.wait_for_navigation()
    
    async def _wait_for_content(self, page: Page):
        """Wait for dynamic content to load"""
        # Try multiple strategies
        strategies = [
            # Wait for common content selectors
            ("article", 2000),
            (".content", 2000),
            ("#content", 2000),
            ("main", 2000),
            (".documentation", 2000),
            (".docs-content", 2000)
        ]
        
        for selector, timeout in strategies:
            try:
                await page.wait_for_selector(selector, timeout=timeout)
                break
            except:
                continue
        
        # Additional wait for JavaScript rendering
        await page.wait_for_load_state("networkidle", timeout=5000)
    
    async def _extract_structured_content(
        self, 
        page: Page, 
        html: str, 
        url: str
    ) -> Dict[str, Any]:
        """Extract content preserving structure"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove navigation, headers, footers
        for tag in soup.find_all(['nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # Find main content area
        content_area = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find(class_='content') or
            soup.find(class_='documentation') or
            soup.find('body')
        )
        
        # Extract structured sections
        sections = []
        current_section = {
            "heading": "Introduction",
            "level": 1,
            "content": [],
            "subsections": []
        }
        
        for element in content_area.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'code', 'table', 'ul', 'ol']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Start new section
                level = int(element.name[1])
                new_section = {
                    "heading": element.get_text(strip=True),
                    "level": level,
                    "content": [],
                    "subsections": []
                }
                
                # Find appropriate parent section
                if level == 1:
                    if current_section["content"]:
                        sections.append(current_section)
                    current_section = new_section
                else:
                    # Nested section logic
                    current_section["subsections"].append(new_section)
            
            elif element.name == 'pre' or (element.name == 'code' and element.parent.name != 'pre'):
                # Code block
                current_section["content"].append({
                    "type": "code",
                    "language": element.get('class', [''])[0].replace('language-', '') if element.get('class') else 'plain',
                    "content": element.get_text(strip=True)
                })
            
            elif element.name == 'table':
                # Table - convert to markdown
                table_md = self._table_to_markdown(element)
                current_section["content"].append({
                    "type": "table",
                    "content": table_md
                })
            
            elif element.name in ['ul', 'ol']:
                # List
                items = [li.get_text(strip=True) for li in element.find_all('li')]
                current_section["content"].append({
                    "type": "list",
                    "ordered": element.name == 'ol',
                    "items": items
                })
            
            else:
                # Regular paragraph
                text = element.get_text(strip=True)
                if text:
                    current_section["content"].append({
                        "type": "text",
                        "content": text
                    })
        
        # Don't forget the last section
        if current_section["content"]:
            sections.append(current_section)
        
        return {
            "url": url,
            "sections": sections,
            "raw_text": content_area.get_text(separator='\n', strip=True)
        }
    
    def _table_to_markdown(self, table) -> str:
        """Convert HTML table to Markdown format"""
        rows = table.find_all('tr')
        if not rows:
            return ""
        
        # Extract headers
        headers = []
        header_row = rows[0]
        for th in header_row.find_all(['th', 'td']):
            headers.append(th.get_text(strip=True))
        
        if not headers:
            return ""
        
        # Build markdown
        md_lines = ['| ' + ' | '.join(headers) + ' |']
        md_lines.append('|' + '---|' * len(headers))
        
        # Add data rows
        for row in rows[1:]:
            cells = []
            for td in row.find_all(['td', 'th']):
                cells.append(td.get_text(strip=True))
            if cells:
                md_lines.append('| ' + ' | '.join(cells) + ' |')
        
        return '\n'.join(md_lines)
    
    async def _extract_links(self, page: Page, base_url: str) -> List[str]:
        """Extract all links from the page"""
        links = await page.evaluate("""
            () => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    links.push(a.href);
                });
                return links;
            }
        """)
        
        # Filter and normalize links
        base_domain = urlparse(base_url).netloc
        filtered_links = []
        
        for link in links:
            try:
                parsed = urlparse(link)
                # Only include links from same domain
                if parsed.netloc == base_domain or not parsed.netloc:
                    normalized = urljoin(base_url, link)
                    # Skip anchors, downloads, etc.
                    if not any(normalized.endswith(ext) for ext in ['.pdf', '.zip', '.tar.gz', '#']):
                        filtered_links.append(normalized)
            except:
                continue
        
        return list(set(filtered_links))
```

### 4.2 Crawler Manager

Create `src/scraper/crawler_manager.py`:

```python
import asyncio
from typing import Dict, Set, List, Optional, Any
from urllib.parse import urlparse
import logging
from datetime import datetime
import json

from .worker import ScrapingWorker
from .url_frontier import URLFrontier

logger = logging.getLogger(__name__)

class CrawlerManager:
    def __init__(self, config: Dict[str, Any], message_queue):
        self.config = config
        self.message_queue = message_queue
        self.active_jobs: Dict[str, Dict] = {}
        self.workers: List[ScrapingWorker] = []
        self.url_frontier = URLFrontier()
    
    async def start(self):
        """Initialize crawler manager and workers"""
        # Create worker pool
        num_workers = self.config.get("max_concurrent_scrapers", 5)
        for i in range(num_workers):
            worker = ScrapingWorker(self.config)
            await worker.start()
            self.workers.append(worker)
        
        # Start job processor
        asyncio.create_task(self._process_jobs())
    
    async def _process_jobs(self):
        """Main job processing loop"""
        while True:
            try:
                # Get job from queue
                message = await self.message_queue.get("scraping_jobs")
                if message:
                    job = json.loads(message)
                    await self.process_job(job)
            except Exception as e:
                logger.error(f"Job processing error: {e}")
                await asyncio.sleep(1)
    
    async def process_job(self, job: Dict[str, Any]):
        """Process a scraping job"""
        job_id = job["job_id"]
        source_id = job["source_id"]
        start_url = job["url"]
        
        logger.info(f"Starting job {job_id} for {start_url}")
        
        # Initialize job state
        self.active_jobs[job_id] = {
            "source_id": source_id,
            "start_url": start_url,
            "status": "running",
            "pages_scraped": 0,
            "pages_failed": 0,
            "started_at": datetime.utcnow()
        }
        
        # Initialize URL frontier
        await self.url_frontier.add_url(job_id, start_url, depth=0)
        
        # Process URLs
        await self._crawl_site(job_id)
        
        # Complete job
        self.active_jobs[job_id]["status"] = "completed"
        self.active_jobs[job_id]["completed_at"] = datetime.utcnow()
        
        # Notify completion
        await self._notify_completion(job_id)
    
    async def _crawl_site(self, job_id: str):
        """Crawl entire site for a job"""
        job = self.active_jobs[job_id]
        max_pages = self.config.get("max_pages_per_site", 1000)
        max_depth = self.config.get("max_crawl_depth", 10)
        
        tasks = []
        semaphore = asyncio.Semaphore(len(self.workers))
        
        while True:
            # Get next URL from frontier
            url_info = await self.url_frontier.get_next_url(job_id)
            if not url_info:
                break
            
            if job["pages_scraped"] >= max_pages:
                logger.info(f"Reached max pages limit for job {job_id}")
                break
            
            if url_info["depth"] > max_depth:
                continue
            
            # Create scraping task
            task = asyncio.create_task(
                self._scrape_with_worker(
                    job_id, 
                    url_info["url"], 
                    url_info["depth"],
                    semaphore
                )
            )
            tasks.append(task)
            
            # Process in batches
            if len(tasks) >= len(self.workers):
                results = await asyncio.gather(*tasks, return_exceptions=True)
                tasks = []
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Scraping error: {result}")
                    elif result:
                        await self._process_scrape_result(job_id, result)
        
        # Wait for remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if not isinstance(result, Exception) and result:
                    await self._process_scrape_result(job_id, result)
    
    async def _scrape_with_worker(
        self, 
        job_id: str, 
        url: str, 
        depth: int,
        semaphore: asyncio.Semaphore
    ):
        """Scrape URL using available worker"""
        async with semaphore:
            # Get available worker (round-robin)
            worker = self.workers[hash(url) % len(self.workers)]
            
            # Scrape page
            result = await worker.scrape_page(url, {
                "job_id": job_id,
                "depth": depth
            })
            
            return result
    
    async def _process_scrape_result(self, job_id: str, result: Dict[str, Any]):
        """Process scraping result"""
        job = self.active_jobs[job_id]
        
        if result["status"] == "success":
            job["pages_scraped"] += 1
            
            # Add discovered links to frontier
            for link in result.get("links", []):
                await self.url_frontier.add_url(
                    job_id, 
                    link, 
                    depth=result.get("depth", 0) + 1
                )
            
            # Send content to RAG processor
            await self.message_queue.publish("rag_processing", json.dumps({
                "job_id": job_id,
                "source_id": job["source_id"],
                "document": result
            }))
            
            # Update progress
            if job["pages_scraped"] % 10 == 0:
                await self._notify_progress(job_id)
        else:
            job["pages_failed"] += 1
            logger.warning(f"Failed to scrape {result['url']}: {result.get('error')}")
    
    async def _notify_progress(self, job_id: str):
        """Send progress notification"""
        job = self.active_jobs[job_id]
        await self.message_queue.publish("notifications", json.dumps({
            "type": "job_progress",
            "job_id": job_id,
            "progress": {
                "pages_scraped": job["pages_scraped"],
                "pages_failed": job["pages_failed"],
                "status": job["status"]
            }
        }))
    
    async def _notify_completion(self, job_id: str):
        """Send completion notification"""
        job = self.active_jobs[job_id]
        duration = (job["completed_at"] - job["started_at"]).total_seconds()
        
        await self.message_queue.publish("notifications", json.dumps({
            "type": "job_completed",
            "job_id": job_id,
            "stats": {
                "pages_scraped": job["pages_scraped"],
                "pages_failed": job["pages_failed"],
                "duration_seconds": duration
            }
        }))
```

## Phase 5: RAG Processing Pipeline

### 5.1 Content Chunker

Create `src/rag-processor/chunker.py`:

```python
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from enum import Enum

class ChunkType(Enum):
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"

@dataclass
class Chunk:
    content: str
    chunk_type: ChunkType
    metadata: Dict[str, Any]
    position: int
    parent_heading: Optional[str] = None

class HybridChunker:
    def __init__(self, config: Dict[str, Any]):
        self.max_chunk_size = config.get("max_chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.min_chunk_size = config.get("min_chunk_size", 100)
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk document using hybrid strategy based on content type
        """
        chunks = []
        position = 0
        
        for section in document.get("sections", []):
            section_chunks = self._chunk_section(
                section, 
                position,
                parent_heading=section.get("heading")
            )
            chunks.extend(section_chunks)
            position += len(section_chunks)
        
        return chunks
    
    def _chunk_section(
        self, 
        section: Dict[str, Any], 
        start_position: int,
        parent_heading: Optional[str] = None
    ) -> List[Chunk]:
        """Chunk a document section"""
        chunks = []
        position = start_position
        
        # Add heading as a chunk
        if section.get("heading"):
            chunks.append(Chunk(
                content=section["heading"],
                chunk_type=ChunkType.HEADING,
                metadata={"level": section.get("level", 1)},
                position=position,
                parent_heading=parent_heading
            ))
            position += 1
        
        # Process content items
        current_text_buffer = []
        
        for item in section.get("content", []):
            item_type = item.get("type")
            
            if item_type == "code":
                # Flush text buffer first
                if current_text_buffer:
                    text_chunks = self._chunk_text(
                        "\n".join(current_text_buffer),
                        position,
                        parent_heading=section.get("heading")
                    )
                    chunks.extend(text_chunks)
                    position += len(text_chunks)
                    current_text_buffer = []
                
                # Add code as separate chunk
                chunks.append(Chunk(
                    content=item["content"],
                    chunk_type=ChunkType.CODE,
                    metadata={"language": item.get("language", "unknown")},
                    position=position,
                    parent_heading=section.get("heading")
                ))
                position += 1
            
            elif item_type == "table":
                # Flush text buffer
                if current_text_buffer:
                    text_chunks = self._chunk_text(
                        "\n".join(current_text_buffer),
                        position,
                        parent_heading=section.get("heading")
                    )
                    chunks.extend(text_chunks)
                    position += len(text_chunks)
                    current_text_buffer = []
                
                # Add table as chunk
                chunks.append(Chunk(
                    content=item["content"],
                    chunk_type=ChunkType.TABLE,
                    metadata={},
                    position=position,
                    parent_heading=section.get("heading")
                ))
                position += 1
            
            elif item_type == "list":
                # Add to text buffer with formatting
                list_text = "\n".join([
                    f"{'*' if not item.get('ordered') else f'{i+1}.'} {item}"
                    for i, item in enumerate(item.get("items", []))
                ])
                current_text_buffer.append(list_text)
            
            else:  # text
                current_text_buffer.append(item.get("content", ""))
        
        # Process remaining text buffer
        if current_text_buffer:
            text_chunks = self._chunk_text(
                "\n".join(current_text_buffer),
                position,
                parent_heading=section.get("heading")
            )
            chunks.extend(text_chunks)
        
        # Process subsections
        for subsection in section.get("subsections", []):
            subsection_chunks = self._chunk_section(
                subsection,
                position + len(chunks),
                parent_heading=section.get("heading")
            )
            chunks.extend(subsection_chunks)
        
        return chunks
    
    def _chunk_text(
        self, 
        text: str, 
        start_position: int,
        parent_heading: Optional[str] = None
    ) -> List[Chunk]:
        """Chunk text content using recursive splitting"""
        if len(text) <= self.max_chunk_size:
            return [Chunk(
                content=text,
                chunk_type=ChunkType.TEXT,
                metadata={},
                position=start_position,
                parent_heading=parent_heading
            )]
        
        chunks = []
        
        # Try to split on paragraph boundaries first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        position = start_position
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        chunk_type=ChunkType.TEXT,
                        metadata={},
                        position=position,
                        parent_heading=parent_heading
                    ))
                    position += 1
                
                # Handle oversized paragraph
                if len(para) > self.max_chunk_size:
                    # Split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sentence_chunk = ""
                    
                    for sentence in sentences:
                        if len(sentence_chunk) + len(sentence) <= self.max_chunk_size:
                            sentence_chunk += sentence + " "
                        else:
                            if sentence_chunk:
                                chunks.append(Chunk(
                                    content=sentence_chunk.strip(),
                                    chunk_type=ChunkType.TEXT,
                                    metadata={},
                                    position=position,
                                    parent_heading=parent_heading
                                ))
                                position += 1
                            sentence_chunk = sentence + " "
                    
                    if sentence_chunk:
                        current_chunk = sentence_chunk
                else:
                    current_chunk = para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                chunk_type=ChunkType.TEXT,
                metadata={},
                position=position,
                parent_heading=parent_heading
            ))
        
        return chunks
```

### 5.2 Embedding Service

Create `src/rag-processor/embedder.py`:

```python
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = config.get("batch_size", 32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def initialize(self):
        """Load the embedding model"""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Model loaded on {self.device}")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.model:
            self.initialize()
        
        # Run embedding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self._embed_sync,
            texts
        )
        
        return embeddings
    
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding function"""
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Generate embeddings
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if not self.model:
            self.initialize()
        return self.model.get_sentence_embedding_dimension()
```

### 5.3 RAG Processor Main

Create `src/rag-processor/processor.py`:

```python
import asyncio
import json
from typing import Dict, Any, List
import logging
from datetime import datetime
import hashlib

from .chunker import HybridChunker, Chunk
from .embedder import EmbeddingService
from ..api.services.vector_store import VectorStoreService
from ..api.services.document_store import DocumentStoreService

logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self, config: Dict[str, Any], message_queue):
        self.config = config
        self.message_queue = message_queue
        self.chunker = HybridChunker(config.get("chunking", {}))
        self.embedder = EmbeddingService(config.get("embedding", {}))
        self.vector_store = VectorStoreService(config.get("weaviate", {}))
        self.document_store = DocumentStoreService(config.get("postgres", {}))
    
    async def start(self):
        """Initialize processor and start processing loop"""
        # Initialize services
        self.embedder.initialize()
        await self.vector_store.initialize()
        
        # Start processing loop
        asyncio.create_task(self._process_documents())
    
    async def _process_documents(self):
        """Main document processing loop"""
        while True:
            try:
                # Get document from queue
                message = await self.message_queue.get("rag_processing")
                if message:
                    data = json.loads(message)
                    await self.process_document(data)
            except Exception as e:
                logger.error(f"Document processing error: {e}")
                await asyncio.sleep(1)
    
    async def process_document(self, data: Dict[str, Any]):
        """Process a single document through the RAG pipeline"""
        job_id = data["job_id"]
        source_id = data["source_id"]
        document = data["document"]
        
        logger.info(f"Processing document: {document['url']}")
        
        try:
            # Step 1: Store document metadata
            doc_id = await self.document_store.create_document({
                "source_id": source_id,
                "url": document["url"],
                "title": document["metadata"].get("title", ""),
                "content": document["content"].get("raw_text", ""),
                "content_hash": document["metadata"].get("content_hash"),
                "metadata": document["metadata"]
            })
            
            # Step 2: Chunk the document
            chunks = self.chunker.chunk_document(document["content"])
            logger.info(f"Created {len(chunks)} chunks for {document['url']}")
            
            # Step 3: Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedder.embed_texts(texts)
            
            # Step 4: Prepare vector store objects
            vector_objects = []
            chunk_records = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create unique ID for vector
                chunk_id = hashlib.sha256(
                    f"{doc_id}_{i}_{chunk.content[:50]}".encode()
                ).hexdigest()[:16]
                
                # Prepare vector object
                vector_obj = {
                    "id": chunk_id,
                    "vector": embedding,
                    "properties": {
                        "doc_id": str(doc_id),
                        "source_id": str(source_id),
                        "url": document["url"],
                        "chunk_index": i,
                        "chunk_type": chunk.chunk_type.value,
                        "content": chunk.content,
                        "parent_heading": chunk.parent_heading or "",
                        "position": chunk.position,
                        "metadata": json.dumps(chunk.metadata)
                    }
                }
                vector_objects.append(vector_obj)
                
                # Prepare chunk record for database
                chunk_records.append({
                    "document_id": doc_id,
                    "chunk_index": i,
                    "chunk_type": chunk.chunk_type.value,
                    "content": chunk.content,
                    "embedding_id": chunk_id,
                    "metadata": chunk.metadata
                })
            
            # Step 5: Batch insert to vector store
            await self.vector_store.batch_insert(vector_objects)
            
            # Step 6: Store chunk records in database
            await self.document_store.create_chunks(chunk_records)
            
            # Step 7: Update document status
            await self.document_store.update_document(doc_id, {
                "status": "indexed",
                "indexed_at": datetime.utcnow()
            })
            
            # Step 8: Send completion notification
            await self.message_queue.publish("notifications", json.dumps({
                "type": "document_indexed",
                "job_id": job_id,
                "document": {
                    "id": str(doc_id),
                    "url": document["url"],
                    "chunks": len(chunks)
                }
            }))
            
            logger.info(f"Successfully indexed document: {document['url']}")
            
        except Exception as e:
            logger.error(f"Error processing document {document['url']}: {e}")
            
            # Send error notification
            await self.message_queue.publish("notifications", json.dumps({
                "type": "document_error",
                "job_id": job_id,
                "document": {
                    "url": document["url"],
                    "error": str(e)
                }
            }))
```

## Phase 6: API Implementation

### 6.1 FastAPI Application

Create `src/api/main.py`:

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

from .routers import sources, search, jobs, websocket
from .services.startup import initialize_services
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI Knowledge Hub API...")
    await initialize_services()
    yield
    # Shutdown
    logger.info("Shutting down AI Knowledge Hub API...")

# Create FastAPI app
app = FastAPI(
    title="AI Knowledge Hub API",
    description="Intelligent documentation indexing and knowledge management",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
app.add_middleware(AuthMiddleware)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Time: {process_time:.3f}s"
    )
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Include routers
app.include_router(sources.router, prefix="/api/v1/sources", tags=["sources"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "AI Knowledge Hub API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "documentation": "/api/docs",
            "sources": "/api/v1/sources",
            "search": "/api/v1/search",
            "jobs": "/api/v1/jobs",
            "websocket": "/ws"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    # TODO: Add actual health checks for dependencies
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "database": "operational",
            "redis": "operational",
            "weaviate": "operational"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )
```

### 6.2 Source Management Router

Create `src/api/routers/sources.py`:

```python
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from ..models.schemas import (
    SourceCreate, SourceUpdate, SourceResponse, 
    SourceListResponse, JobResponse
)
from ..services.source_service import SourceService
from ..services.job_service import JobService
from ..dependencies import get_source_service, get_job_service

router = APIRouter()

@router.post("/", response_model=JobResponse, status_code=202)
async def create_source(
    source: SourceCreate,
    background_tasks: BackgroundTasks,
    source_service: SourceService = Depends(get_source_service),
    job_service: JobService = Depends(get_job_service)
):
    """
    Add a new knowledge source and start scraping.
    
    This endpoint immediately returns a job ID while the scraping
    happens asynchronously in the background.
    """
    # Check if source already exists
    existing = await source_service.get_by_url(source.url)
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Source with this URL already exists"
        )
    
    # Create source
    db_source = await source_service.create(source)
    
    # Create scraping job
    job = await job_service.create_scraping_job(
        source_id=db_source.id,
        url=source.url
    )
    
    # Queue scraping task
    background_tasks.add_task(
        job_service.queue_scraping_job,
        job.id,
        db_source.id,
        source.url
    )
    
    return JobResponse(
        job_id=str(job.id),
        message=f"Scraping job created for {source.name}",
        status="queued"
    )

@router.get("/", response_model=SourceListResponse)
async def list_sources(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    source_service: SourceService = Depends(get_source_service)
):
    """Get list of all knowledge sources"""
    sources = await source_service.list_sources(
        skip=skip,
        limit=limit,
        status=status
    )
    
    total = await source_service.count_sources(status=status)
    
    return SourceListResponse(
        sources=sources,
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: UUID,
    source_service: SourceService = Depends(get_source_service)
):
    """Get detailed information about a knowledge source"""
    source = await source_service.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    return source

@router.put("/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: UUID,
    update: SourceUpdate,
    source_service: SourceService = Depends(get_source_service)
):
    """Update knowledge source properties"""
    source = await source_service.update(source_id, update)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    return source

@router.delete("/{source_id}", status_code=202)
async def delete_source(
    source_id: UUID,
    background_tasks: BackgroundTasks,
    source_service: SourceService = Depends(get_source_service),
    job_service: JobService = Depends(get_job_service)
):
    """Delete a knowledge source and all its indexed content"""
    source = await source_service.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Create deletion job
    job = await job_service.create_deletion_job(source_id)
    
    # Queue deletion task
    background_tasks.add_task(
        source_service.delete_source_data,
        source_id
    )
    
    return JobResponse(
        job_id=str(job.id),
        message=f"Deletion job created for {source.name}",
        status="queued"
    )

@router.post("/{source_id}/rescrape", status_code=202)
async def rescrape_source(
    source_id: UUID,
    background_tasks: BackgroundTasks,
    source_service: SourceService = Depends(get_source_service),
    job_service: JobService = Depends(get_job_service)
):
    """Trigger a new scraping job for an existing source"""
    source = await source_service.get_by_id(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Create rescraping job
    job = await job_service.create_scraping_job(
        source_id=source_id,
        url=source.url,
        is_rescrape=True
    )
    
    # Queue scraping task
    background_tasks.add_task(
        job_service.queue_scraping_job,
        job.id,
        source_id,
        source.url
    )
    
    return JobResponse(
        job_id=str(job.id),
        message=f"Rescraping job created for {source.name}",
        status="queued"
    )
```

### 6.3 Search Router

Create `src/api/routers/search.py`:

```python
from fastapi import APIRouter, Query, Depends, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from ..services.search_service import SearchService
from ..dependencies import get_search_service

router = APIRouter()

class SearchQuery(BaseModel):
    query: str
    source_filter: Optional[str] = None
    limit: int = 10
    include_metadata: bool = True
    search_type: str = "hybrid"  # hybrid, vector, keyword

class SearchResult(BaseModel):
    content: str
    source_name: str
    url: str
    score: float
    chunk_type: str
    metadata: Optional[dict] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    search_time_ms: float

@router.post("/", response_model=SearchResponse)
async def search(
    search_query: SearchQuery,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Execute a search query against the knowledge base.
    
    Supports three search types:
    - hybrid: Combines semantic and keyword search (default)
    - vector: Pure semantic/similarity search
    - keyword: Traditional keyword-based search
    """
    import time
    start_time = time.time()
    
    # Execute search
    results = await search_service.search(
        query=search_query.query,
        source_filter=search_query.source_filter,
        limit=search_query.limit,
        search_type=search_query.search_type
    )
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append(SearchResult(
            content=result["content"],
            source_name=result["source_name"],
            url=result["url"],
            score=result["score"],
            chunk_type=result["chunk_type"],
            metadata=result.get("metadata") if search_query.include_metadata else None
        ))
    
    search_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return SearchResponse(
        query=search_query.query,
        results=formatted_results,
        total=len(formatted_results),
        search_time_ms=search_time
    )

@router.get("/suggest")
async def search_suggestions(
    q: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=20),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get search suggestions based on partial query.
    
    Returns suggested queries based on indexed content.
    """
    suggestions = await search_service.get_suggestions(q, limit)
    return {"suggestions": suggestions}
```

## Phase 7: Web UI Implementation

### 7.1 React Application Setup

Create `src/web-ui/package.json`:

```json
{
  "name": "knowledge-hub-ui",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@emotion/react": "^11.11.0",
    "@emotion/styled": "^11.11.0",
    "@mui/icons-material": "^5.13.0",
    "@mui/material": "^5.13.0",
    "@mui/x-data-grid": "^6.3.0",
    "@reduxjs/toolkit": "^1.9.5",
    "@tanstack/react-query": "^4.29.0",
    "axios": "^1.4.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-markdown": "^8.0.7",
    "react-redux": "^8.0.5",
    "react-router-dom": "^6.11.0",
    "react-syntax-highlighter": "^15.5.0",
    "socket.io-client": "^4.6.0",
    "typescript": "^5.0.0",
    "vite": "^4.3.0"
  },
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "type-check": "tsc --noEmit"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.0.0",
    "eslint": "^8.40.0",
    "prettier": "^2.8.0"
  }
}
```

### 7.2 Main App Component

Create `src/web-ui/src/App.tsx`:

```typescript
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Provider } from 'react-redux';

import { store } from './store';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Sources from './pages/Sources';
import Search from './pages/Search';
import KnowledgeGraph from './pages/KnowledgeGraph';
import Settings from './pages/Settings';
import { WebSocketProvider } from './contexts/WebSocketContext';

// Create theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

// Create query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

function App() {
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <WebSocketProvider>
            <Router>
              <Layout>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/sources" element={<Sources />} />
                  <Route path="/search" element={<Search />} />
                  <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
                  <Route path="/settings" element={<Settings />} />
                </Routes>
              </Layout>
            </Router>
          </WebSocketProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </Provider>
  );
}

export default App;
```

### 7.3 Sources Management Page

Create `src/web-ui/src/pages/Sources.tsx`:

```typescript
import React, { useState } from 'react';
import {
  Box,
  Button,
  Paper,
  Typography,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  LinearProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { DataGrid, GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';

import { api } from '../services/api';
import { useWebSocket } from '../contexts/WebSocketContext';
import { Source } from '../types';

export default function Sources() {
  const [open, setOpen] = useState(false);
  const [newSource, setNewSource] = useState({ name: '', url: '' });
  const queryClient = useQueryClient();
  const { jobProgress } = useWebSocket();

  // Fetch sources
  const { data, isLoading } = useQuery({
    queryKey: ['sources'],
    queryFn: () => api.getSources(),
  });

  // Add source mutation
  const addSourceMutation = useMutation({
    mutationFn: (source: { name: string; url: string }) =>
      api.createSource(source),
    onSuccess: () => {
      queryClient.invalidateQueries(['sources']);
      setOpen(false);
      setNewSource({ name: '', url: '' });
    },
  });

  // Delete source mutation
  const deleteSourceMutation = useMutation({
    mutationFn: (id: string) => api.deleteSource(id),
    onSuccess: () => {
      queryClient.invalidateQueries(['sources']);
    },
  });

  // Rescrape source mutation
  const rescrapeSourceMutation = useMutation({
    mutationFn: (id: string) => api.rescrapeSource(id),
    onSuccess: () => {
      queryClient.invalidateQueries(['sources']);
    },
  });

  const columns: GridColDef[] = [
    {
      field: 'name',
      headerName: 'Name',
      flex: 1,
      minWidth: 200,
    },
    {
      field: 'url',
      headerName: 'URL',
      flex: 2,
      minWidth: 300,
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 150,
      renderCell: (params: GridRenderCellParams) => {
        const status = params.value as string;
        const jobId = params.row.active_job_id;
        const progress = jobId ? jobProgress[jobId] : null;

        if (progress) {
          return (
            <Box sx={{ width: '100%' }}>
              <Typography variant="caption">
                {progress.pages_scraped} pages
              </Typography>
              <LinearProgress
                variant="determinate"
                value={progress.progress || 0}
              />
            </Box>
          );
        }

        const color =
          status === 'completed'
            ? 'success'
            : status === 'error'
            ? 'error'
            : status === 'indexing'
            ? 'warning'
            : 'default';

        return <Chip label={status} color={color} size="small" />;
      },
    },
    {
      field: 'stats',
      headerName: 'Documents',
      width: 120,
      valueGetter: (params) => params.row.stats?.documents || 0,
    },
    {
      field: 'last_scraped_at',
      headerName: 'Last Updated',
      width: 180,
      valueFormatter: (params) => {
        if (!params.value) return 'Never';
        return format(new Date(params.value), 'MMM d, yyyy HH:mm');
      },
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 150,
      sortable: false,
      renderCell: (params: GridRenderCellParams) => (
        <>
          <IconButton
            size="small"
            onClick={() => rescrapeSourceMutation.mutate(params.row.id)}
            disabled={params.row.status === 'indexing'}
          >
            <RefreshIcon />
          </IconButton>
          <IconButton
            size="small"
            color="error"
            onClick={() => deleteSourceMutation.mutate(params.row.id)}
            disabled={params.row.status === 'indexing'}
          >
            <DeleteIcon />
          </IconButton>
        </>
      ),
    },
  ];

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Typography variant="h4">Knowledge Sources</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpen(true)}
        >
          Add Source
        </Button>
      </Box>

      <Paper sx={{ height: 600 }}>
        <DataGrid
          rows={data?.sources || []}
          columns={columns}
          loading={isLoading}
          pageSize={10}
          rowsPerPageOptions={[10, 25, 50]}
          disableSelectionOnClick
        />
      </Paper>

      {/* Add Source Dialog */}
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Knowledge Source</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="normal"
            label="Name"
            fullWidth
            variant="outlined"
            value={newSource.name}
            onChange={(e) =>
              setNewSource({ ...newSource, name: e.target.value })
            }
          />
          <TextField
            margin="normal"
            label="Documentation URL"
            fullWidth
            variant="outlined"
            value={newSource.url}
            onChange={(e) =>
              setNewSource({ ...newSource, url: e.target.value })
            }
            helperText="Enter the root URL of the documentation site"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button
            onClick={() => addSourceMutation.mutate(newSource)}
            variant="contained"
            disabled={!newSource.name || !newSource.url}
          >
            Add Source
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
```

### 7.4 Search Interface

Create `src/web-ui/src/pages/Search.tsx`:

```typescript
import React, { useState } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  InputAdornment,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon,
  Description as DescriptionIcon,
  TableChart as TableIcon,
} from '@mui/icons-material';
import { useMutation } from '@tanstack/react-query';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { api } from '../services/api';
import { SearchResult } from '../types';

export default function Search() {
  const [query, setQuery] = useState('');
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());

  const searchMutation = useMutation({
    mutationFn: (searchQuery: string) =>
      api.search({
        query: searchQuery,
        limit: 20,
        search_type: 'hybrid',
      }),
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      searchMutation.mutate(query);
    }
  };

  const toggleExpanded = (index: number) => {
    const newExpanded = new Set(expandedResults);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedResults(newExpanded);
  };

  const getChunkIcon = (chunkType: string) => {
    switch (chunkType) {
      case 'code':
        return <CodeIcon />;
      case 'table':
        return <TableIcon />;
      default:
        return <DescriptionIcon />;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Search Knowledge Base
      </Typography>

      {/* Search Form */}
      <Paper component="form" onSubmit={handleSearch} sx={{ p: 2, mb: 3 }}>
        <TextField
          fullWidth
          placeholder="Ask a question or search for documentation..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
            endAdornment: searchMutation.isLoading && (
              <InputAdornment position="end">
                <CircularProgress size={20} />
              </InputAdornment>
            ),
          }}
        />
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Button
            type="submit"
            variant="contained"
            disabled={!query.trim() || searchMutation.isLoading}
          >
            Search
          </Button>
          <Typography variant="body2" color="text.secondary" sx={{ ml: 'auto' }}>
            {searchMutation.data &&
              `${searchMutation.data.total} results in ${searchMutation.data.search_time_ms.toFixed(
                0
              )}ms`}
          </Typography>
        </Box>
      </Paper>

      {/* Search Results */}
      {searchMutation.data && (
        <Box>
          {searchMutation.data.results.map((result: SearchResult, index: number) => (
            <Card key={index} sx={{ mb: 2 }}>
              <CardContent>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    mb: 1,
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getChunkIcon(result.chunk_type)}
                    <Typography variant="subtitle1" component="span">
                      {result.source_name}
                    </Typography>
                    <Chip
                      label={result.chunk_type}
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={`Score: ${result.score.toFixed(3)}`}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </Box>
                  <IconButton
                    size="small"
                    onClick={() => toggleExpanded(index)}
                  >
                    <ExpandMoreIcon
                      sx={{
                        transform: expandedResults.has(index)
                          ? 'rotate(180deg)'
                          : 'rotate(0deg)',
                        transition: 'transform 0.3s',
                      }}
                    />
                  </IconButton>
                </Box>

                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: 'block', mb: 1 }}
                >
                  {result.url}
                </Typography>

                {/* Content Preview */}
                <Box
                  sx={{
                    maxHeight: expandedResults.has(index) ? 'none' : '100px',
                    overflow: 'hidden',
                    position: 'relative',
                  }}
                >
                  {result.chunk_type === 'code' ? (
                    <SyntaxHighlighter
                      language={result.metadata?.language || 'text'}
                      style={vscDarkPlus}
                      customStyle={{
                        margin: 0,
                        fontSize: '0.875rem',
                      }}
                    >
                      {result.content}
                    </SyntaxHighlighter>
                  ) : (
                    <ReactMarkdown>{result.content}</ReactMarkdown>
                  )}
                  
                  {!expandedResults.has(index) && (
                    <Box
                      sx={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        height: '40px',
                        background:
                          'linear-gradient(to bottom, transparent, white)',
                      }}
                    />
                  )}
                </Box>

                {/* Metadata */}
                {expandedResults.has(index) && result.metadata && (
                  <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Metadata
                    </Typography>
                    <pre style={{ fontSize: '0.75rem', overflow: 'auto' }}>
                      {JSON.stringify(result.metadata, null, 2)}
                    </pre>
                  </Box>
                )}
              </CardContent>
            </Card>
          ))}
        </Box>
      )}

      {/* Empty State */}
      {!searchMutation.data && !searchMutation.isLoading && (
        <Box sx={{ textAlign: 'center', mt: 8 }}>
          <SearchIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            Enter a search query to explore the knowledge base
          </Typography>
        </Box>
      )}
    </Box>
  );
}
```

## Conclusion

This implementation guide provides a comprehensive foundation for building the AI Knowledge Hub. The system combines:

1. **MCP Protocol Server** for AI agent integration
2. **Advanced Web Scraping** with Playwright for JavaScript-heavy sites
3. **Hybrid Search** combining semantic and keyword matching
4. **Modern Web UI** with React and Material-UI
5. **Scalable Architecture** using microservices and event-driven design

### Next Steps

1. **Testing**: Implement comprehensive test suites
2. **Monitoring**: Set up Prometheus and Grafana
3. **Documentation**: Complete API documentation
4. **Security**: Implement authentication and rate limiting
5. **Optimization**: Performance tuning and caching strategies

The modular design allows you to start with basic functionality and progressively add features while maintaining a clean, maintainable codebase.