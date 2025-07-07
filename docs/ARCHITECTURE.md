# KnowledgeHub Architecture Documentation

## Table of Contents
- [System Overview](#system-overview)
- [Microservices Architecture](#microservices-architecture)
- [Service Interactions](#service-interactions)
- [Data Flow](#data-flow)
- [Database Design](#database-design)
- [API Gateway Pattern](#api-gateway-pattern)
- [Background Processing](#background-processing)
- [Real-time Communication](#real-time-communication)
- [Storage Architecture](#storage-architecture)
- [Security Architecture](#security-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Scalability Considerations](#scalability-considerations)

## System Overview

KnowledgeHub is a modern, cloud-native knowledge management system built using a microservices architecture. It provides intelligent document processing, vector-based search, and AI-powered content understanding capabilities.

### High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[React Web UI<br/>Port: 5173 (dev)]
    end
    
    subgraph "API Gateway Layer"
        API[FastAPI Gateway<br/>Port: 3000]
        WS[WebSocket Server]
        MCP[MCP Server<br/>Port: 3008]
    end
    
    subgraph "Application Services"
        SCHED[Scheduler Service<br/>Automated Tasks]
        SCRAPER[Scraper Worker<br/>Web Crawling]
        RAG[RAG Processor<br/>Content Processing]
    end
    
    subgraph "AI/ML Services"
        AI[AI Service<br/>Port: 8002]
        TRANS[Transformers Service<br/>sentence-transformers]
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL<br/>Metadata)]
        REDIS[(Redis<br/>Cache & Queues)]
        WEAVIATE[(Weaviate<br/>Vector DB)]
        MINIO[(MinIO<br/>Object Storage)]
    end
    
    UI --> API
    API --> WS
    API --> PG
    API --> REDIS
    API --> WEAVIATE
    
    SCHED --> API
    SCRAPER --> API
    RAG --> API
    RAG --> AI
    
    WEAVIATE --> TRANS
    AI --> TRANS
    
    API --> MCP
    
    style UI fill:#e1f5fe
    style API fill:#f3e5f5
    style PG fill:#e8f5e8
    style WEAVIATE fill:#fff3e0
    style REDIS fill:#ffebee
```

## Microservices Architecture

### Service Catalog

| Service | Technology | Port | Purpose | Dependencies |
|---------|------------|------|---------|--------------|
| **API Gateway** | FastAPI | 3000 | Central API hub, request routing | PostgreSQL, Redis, Weaviate |
| **Web UI** | React + TypeScript | 5173 (dev) | User interface | API Gateway |
| **MCP Server** | WebSocket + JSON-RPC | 3008 | Model Context Protocol | API Gateway |
| **Scraper Worker** | Playwright + HTTPX | - | Web content extraction | API Gateway, Redis |
| **RAG Processor** | Python + Transformers | - | Content chunking & embeddings | API Gateway, AI Service |
| **Scheduler** | APScheduler | - | Automated task execution | API Gateway |
| **AI Service** | FastAPI + sentence-transformers | 8002 | AI analysis and vector generation | PostgreSQL |
| **Transformers Service** | Weaviate Plugin | - | Model inference | - |

### Service Responsibilities

#### 1. API Gateway (`api`)
**Primary Responsibilities:**
- Central request routing and validation
- Authentication and authorization 
- Rate limiting and throttling
- Business logic orchestration
- Database transaction management

**Key Features:**
- OpenAPI/Swagger documentation
- Pydantic request/response validation
- SQLAlchemy ORM integration
- Redis caching layer
- WebSocket support for real-time updates

**Endpoints:**
- `/api/v1/sources` - Knowledge source management
- `/api/v1/search` - Intelligent search capabilities
- `/api/v1/jobs` - Background job management
- `/api/v1/memories` - Conversation memory system
- `/api/v1/chunks` - Document chunk operations
- `/health` - System health monitoring

#### 2. Web UI (`web-ui`)
**Primary Responsibilities:**
- User interface rendering
- State management
- API integration
- Real-time updates display

**Technology Stack:**
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development
- **UI Library**: Material-UI components
- **State Management**: React Query for server state
- **Styling**: CSS-in-JS with emotion

**Pages & Components:**
- Dashboard - System overview and metrics
- Sources - Knowledge source management
- Search - Advanced search interface  
- Jobs - Background task monitoring
- Memory - Conversation context management
- Settings - System configuration

#### 3. Scraper Worker (`scraper`)
**Primary Responsibilities:**
- Web content crawling and extraction
- Incremental crawling with delta detection
- Content parsing and cleanup
- Queue-based job processing

**Key Features:**
- **Playwright Integration**: JavaScript-enabled crawling
- **Incremental Crawling**: SHA-256 content hashing for change detection
- **Smart Parsing**: Content extraction with noise removal
- **Configurable Limits**: Depth, page count, and pattern filtering
- **Error Handling**: Graceful failure recovery

**Crawling Pipeline:**
1. Receive job from Redis queue
2. Initialize Playwright browser session
3. Load existing document cache (for incremental)
4. Crawl pages with configurable depth/limits
5. Extract and clean content
6. Calculate content hashes
7. Queue content for RAG processing
8. Update job status and metrics

#### 4. RAG Processor (`rag-processor`)
**Primary Responsibilities:**
- Intelligent text chunking
- Embedding generation coordination
- Vector storage management
- Content enrichment

**Processing Pipeline:**
1. Receive content from scraper
2. Apply smart chunking algorithms
3. Generate embeddings via embeddings service
4. Store vectors in Weaviate
5. Update metadata in PostgreSQL
6. Notify completion via WebSocket

**Chunking Strategies:**
- **Semantic Chunking**: Preserve meaning boundaries
- **Token-aware Splitting**: Respect model token limits
- **Overlap Management**: Maintain context continuity
- **Metadata Preservation**: Track source and location

#### 5. Scheduler Service (`scheduler`)
**Primary Responsibilities:**
- Automated job scheduling
- Weekly source refreshes
- System maintenance tasks
- Performance optimization

**Scheduling Features:**
- **Cron-based Triggers**: Weekly refresh cycles
- **Intelligent Batching**: Process multiple sources efficiently
- **Load Balancing**: Distribute work across time
- **Error Recovery**: Retry failed operations

### Service Communication Patterns

#### 1. Synchronous Communication
- **HTTP REST**: API Gateway ↔ Frontend
- **gRPC**: Internal service calls (when needed)
- **WebSocket**: Real-time updates

#### 2. Asynchronous Communication  
- **Redis Queues**: Job distribution and processing
- **Event Publishing**: Status updates and notifications
- **Message Passing**: Inter-service coordination

#### 3. Data Access Patterns
- **API Gateway**: Direct database access via SQLAlchemy
- **Workers**: Queue-based job processing
- **Caching**: Redis for frequently accessed data
- **Vector Operations**: Direct Weaviate integration

## Service Interactions

### Content Ingestion Flow

```mermaid
sequenceDiagram
    participant UI as Web UI
    participant API as API Gateway
    participant Q as Redis Queue
    participant S as Scraper
    participant R as RAG Processor
    participant E as Embeddings
    participant W as Weaviate
    participant DB as PostgreSQL
    
    UI->>API: Create source
    API->>DB: Store source metadata
    API->>Q: Queue crawl job
    API->>UI: Return job ID
    
    S->>Q: Poll for jobs
    S->>S: Crawl website
    S->>API: Send extracted content
    API->>Q: Queue for RAG processing
    
    R->>Q: Poll for content
    R->>R: Chunk content
    R->>E: Generate embeddings
    E->>R: Return vectors
    R->>W: Store vectors
    R->>DB: Store metadata
    R->>API: Update job status
    API->>UI: WebSocket update
```

### Search Process Flow

```mermaid
sequenceDiagram
    participant UI as Web UI
    participant API as API Gateway
    participant E as Embeddings
    participant W as Weaviate
    participant DB as PostgreSQL
    participant C as Redis Cache
    
    UI->>API: Search query
    API->>C: Check cache
    alt Cache miss
        API->>E: Generate query embedding
        E->>API: Return vector
        API->>W: Vector similarity search
        W->>API: Return similar chunks
        API->>DB: Get metadata
        DB->>API: Return enriched results
        API->>C: Cache results
    end
    API->>UI: Return search results
```

### Job Management Flow

```mermaid
sequenceDiagram
    participant API as API Gateway  
    participant Q as Redis Queue
    participant W as Worker
    participant WS as WebSocket
    participant UI as Web UI
    
    API->>Q: Queue job
    API->>WS: Job created event
    WS->>UI: Real-time update
    
    W->>Q: Poll for jobs
    W->>W: Process job
    W->>API: Update progress
    API->>WS: Progress update
    WS->>UI: Real-time progress
    
    W->>API: Job completion
    API->>WS: Completion event
    WS->>UI: Final status
```

## Data Flow

### Primary Data Flows

#### 1. Content Ingestion Pipeline
```
Source URL → Scraper → Content Extraction → RAG Processor → 
Chunking → Embedding Generation → Vector Storage + Metadata Storage
```

#### 2. Search Pipeline  
```
User Query → Embedding Generation → Vector Search → 
Metadata Enrichment → Result Ranking → Response Formatting
```

#### 3. Job Processing Pipeline
```
Job Creation → Queue Distribution → Worker Processing → 
Progress Updates → Status Notification → Completion Handling
```

### Data Transformation Points

1. **Content Extraction**: Raw HTML → Clean text
2. **Chunking**: Long documents → Semantic chunks
3. **Embedding**: Text chunks → Vector representations
4. **Search**: Query text → Similarity scores → Ranked results
5. **Metadata**: Source info → Searchable attributes

## Database Design

### PostgreSQL Schema

#### Core Tables

**knowledge_sources**
```sql
CREATE TABLE knowledge_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    base_url VARCHAR(500) NOT NULL,
    source_type VARCHAR(50) NOT NULL DEFAULT 'web',
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    config JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_crawl_at TIMESTAMP WITH TIME ZONE
);
```

**documents**
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    url VARCHAR(1000) NOT NULL,
    title VARCHAR(500),
    content TEXT,
    content_hash VARCHAR(64), -- SHA-256 for incremental crawling
    metadata JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_id, url)
);
```

**document_chunks**
```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(50) NOT NULL DEFAULT 'content',
    content TEXT NOT NULL,
    embedding_id UUID, -- Weaviate vector ID
    summary TEXT,
    keywords TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**scraping_jobs**
```sql
CREATE TABLE scraping_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL, -- 'crawl', 'delete', 'refresh'
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    total_pages INTEGER DEFAULT 0,
    processed_pages INTEGER DEFAULT 0,
    error_message TEXT,
    result JSONB,
    config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);
```

#### Indexes for Performance

```sql
-- Search optimization
CREATE INDEX idx_documents_source_url ON documents(source_id, url);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_embedding_id ON document_chunks(embedding_id);

-- Job management
CREATE INDEX idx_jobs_status ON scraping_jobs(status);
CREATE INDEX idx_jobs_source_status ON scraping_jobs(source_id, status);
CREATE INDEX idx_jobs_created_at ON scraping_jobs(created_at DESC);

-- Source management  
CREATE INDEX idx_sources_status ON knowledge_sources(status);
CREATE INDEX idx_sources_updated_at ON knowledge_sources(updated_at DESC);
```

### Weaviate Schema

**Knowledge_chunks Collection**
```json
{
  "class": "Knowledge_chunks",
  "vectorizer": "text2vec-transformers",
  "moduleConfig": {
    "text2vec-transformers": {
      "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
  },
  "properties": [
    {
      "name": "content",
      "dataType": ["text"],
      "description": "The actual text content of the chunk"
    },
    {
      "name": "source_id", 
      "dataType": ["string"],
      "description": "Reference to knowledge source"
    },
    {
      "name": "document_id",
      "dataType": ["string"], 
      "description": "Reference to source document"
    },
    {
      "name": "chunk_index",
      "dataType": ["int"],
      "description": "Position within document"
    },
    {
      "name": "chunk_type",
      "dataType": ["string"],
      "description": "Type of content (content, heading, code)"
    },
    {
      "name": "url",
      "dataType": ["string"],
      "description": "Source URL"
    },
    {
      "name": "title",
      "dataType": ["string"],
      "description": "Document title"
    }
  ]
}
```

### Redis Data Structures

#### Job Queues
```
crawl_jobs:pending     # List of pending crawl jobs
crawl_jobs:processing  # Set of currently processing jobs
rag_jobs:pending       # List of pending RAG processing jobs
rag_jobs:processing    # Set of currently processing RAG jobs
```

#### Caching
```
search:cache:{hash}         # Search result cache (TTL: 1 hour)
source:{id}:metadata        # Source metadata cache (TTL: 15 minutes) 
job:{id}:status            # Job status cache (TTL: 1 minute)
system:health              # System health cache (TTL: 30 seconds)
```

#### Session Management
```
session:{id}               # User session data
websocket:connections      # Active WebSocket connections
rate_limit:{ip}:{endpoint} # Rate limiting counters
```

## API Gateway Pattern

### Request Lifecycle

1. **Request Reception**: FastAPI receives HTTP request
2. **Authentication**: Validate API key and permissions
3. **Rate Limiting**: Check request limits per IP/user
4. **Input Validation**: Pydantic model validation
5. **Business Logic**: Route to appropriate service handler
6. **Database Operations**: SQLAlchemy ORM transactions
7. **Response Formatting**: Standardized JSON response
8. **Logging**: Request/response audit trail

### Middleware Stack

```python
# Middleware execution order (LIFO)
app.add_middleware(CORSMiddleware)           # Cross-origin requests
app.add_middleware(RateLimitMiddleware)     # Rate limiting  
app.add_middleware(AuthenticationMiddleware) # API key validation
app.add_middleware(LoggingMiddleware)       # Request/response logging
app.add_middleware(ErrorHandlingMiddleware) # Global error handling
```

### Error Handling Strategy

```python
# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "details": str(exc) if DEBUG else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

## Background Processing

### Queue-Based Architecture

#### Job Distribution Pattern
```
API Gateway → Redis Queue → Worker Pool → Result Processing → Status Update
```

#### Worker Scaling Strategy
- **Horizontal Scaling**: Multiple worker containers
- **Load Balancing**: Redis queue distribution
- **Graceful Shutdown**: Complete current jobs before stopping
- **Health Monitoring**: Worker heartbeat system

### Job Types & Processing

#### 1. Crawl Jobs
```json
{
  "job_type": "crawl",
  "source_id": "uuid",
  "config": {
    "max_depth": 3,
    "max_pages": 1000,
    "force_refresh": false
  },
  "priority": "normal"
}
```

#### 2. RAG Processing Jobs  
```json
{
  "job_type": "rag_process",
  "document_id": "uuid", 
  "content": "Document content...",
  "chunk_config": {
    "chunk_size": 1000,
    "overlap": 200
  }
}
```

#### 3. Cleanup Jobs
```json
{
  "job_type": "cleanup",
  "source_id": "uuid",
  "operation": "delete_source"
}
```

### Job State Management

```mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Running: Worker picks up
    Running --> Completed: Success
    Running --> Failed: Error
    Running --> Cancelled: User cancellation
    Failed --> Pending: Retry
    Cancelled --> [*]
    Completed --> [*]
    Failed --> [*]
```

## Real-time Communication

### WebSocket Implementation

#### Connection Management
```python
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)
```

#### Message Types
```json
// Job progress update
{
  "type": "job_update",
  "data": {
    "job_id": "uuid",
    "status": "running",
    "progress": 45,
    "message": "Processing page 45 of 100"
  }
}

// System alert
{
  "type": "system_alert", 
  "data": {
    "level": "warning",
    "message": "High memory usage detected",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}

// Search progress (for long searches)
{
  "type": "search_progress",
  "data": {
    "query_id": "uuid",
    "stage": "embedding_generation",
    "progress": 30
  }
}
```

### Event-Driven Updates

#### Publisher Pattern
```python
class EventPublisher:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def publish_job_update(self, job_id: str, status: str, progress: int):
        message = {
            "type": "job_update",
            "job_id": job_id,
            "status": status, 
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.redis.publish("job_updates", json.dumps(message))
```

## Storage Architecture

### Multi-Modal Storage Strategy

#### 1. Relational Data (PostgreSQL)
- **Purpose**: Structured metadata, relationships, transactions
- **Use Cases**: Source management, job tracking, user data
- **Benefits**: ACID compliance, complex queries, referential integrity

#### 2. Vector Data (Weaviate)
- **Purpose**: Semantic search, similarity matching
- **Use Cases**: Content search, recommendation engine
- **Benefits**: Neural search, automatic vectorization, GraphQL API

#### 3. Cache & Queues (Redis)
- **Purpose**: Fast access, temporary storage, message passing
- **Use Cases**: Session data, job queues, search cache
- **Benefits**: Sub-millisecond latency, pub/sub, persistence options

#### 4. Object Storage (MinIO)
- **Purpose**: Large files, backups, static assets
- **Use Cases**: Document storage, model artifacts, exports
- **Benefits**: S3 compatibility, distributed storage, lifecycle management

### Data Consistency Strategy

#### 1. Strong Consistency
- **PostgreSQL Transactions**: ACID guarantees for critical operations
- **Weaviate Operations**: Vector consistency within collection
- **Redis Atomic Operations**: Single-command consistency

#### 2. Eventual Consistency
- **Cross-Service Updates**: Background synchronization jobs
- **Cache Invalidation**: TTL-based and event-driven expiration
- **Search Index Updates**: Periodic refresh of vector embeddings

#### 3. Conflict Resolution
- **Optimistic Locking**: Version-based conflict detection
- **Last-Write-Wins**: Timestamp-based resolution for non-critical data
- **Manual Resolution**: User intervention for complex conflicts

## Security Architecture

### Authentication & Authorization

#### API Key Management
```python
class APIKeyAuth:
    def __init__(self, db: Session):
        self.db = db
    
    async def authenticate(self, api_key: str) -> Optional[User]:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        db_key = self.db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True
        ).first()
        return db_key.user if db_key else None
```

#### Permission Model
- **Admin**: Full system access
- **User**: Read/write to own resources
- **Read-Only**: Search and view only
- **Service**: Internal service communication

### Network Security

#### Container Isolation
- **Internal Networks**: Services communicate via Docker networks
- **Port Exposure**: Only necessary ports exposed to host
- **TLS Termination**: HTTPS handled by reverse proxy

#### Rate Limiting
```python
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    endpoint = request.url.path
    
    key = f"rate_limit:{client_ip}:{endpoint}"
    current = await redis.incr(key)
    
    if current == 1:
        await redis.expire(key, 60)  # 1 minute window
    
    if current > get_rate_limit(endpoint):
        raise HTTPException(429, "Rate limit exceeded")
    
    return await call_next(request)
```

### Data Security

#### Encryption at Rest
- **Database**: PostgreSQL encryption (configurable)
- **Object Storage**: MinIO server-side encryption
- **Secrets**: Docker secrets for sensitive configuration

#### Encryption in Transit
- **HTTPS**: TLS 1.3 for all external communication
- **Internal**: Optional service-to-service TLS
- **WebSocket**: WSS for real-time communication

### Input Validation & Sanitization

#### Pydantic Models
```python
class SourceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    base_url: HttpUrl
    description: Optional[str] = Field(None, max_length=1000)
    config: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('config')
    def validate_config(cls, v):
        # Custom validation logic
        return v
```

#### SQL Injection Prevention
- **Parameterized Queries**: SQLAlchemy ORM protection
- **Input Escaping**: Automatic parameter binding
- **Query Validation**: Type checking and bounds verification

## Deployment Architecture

### Container Orchestration

#### Docker Compose Configuration
```yaml
version: '3.8'
services:
  api:
    build: 
      context: .
      dockerfile: docker/api.Dockerfile
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - knowledgehub-network
    restart: unless-stopped
```

#### Health Checks
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Production Deployment Options

#### 1. Docker Compose (Single Server)
- **Use Case**: Development, small deployments
- **Benefits**: Simple setup, resource efficient
- **Limitations**: Single point of failure, limited scaling

#### 2. Kubernetes (Recommended)
- **Use Case**: Production, high availability
- **Benefits**: Auto-scaling, service discovery, rolling updates
- **Components**: Deployments, Services, ConfigMaps, Secrets

#### 3. Cloud Deployment
- **AWS**: ECS/EKS with RDS, ElastiCache, and S3
- **GCP**: GKE with Cloud SQL, Memorystore, and Cloud Storage
- **Azure**: AKS with Azure Database, Redis Cache, and Blob Storage

### Monitoring & Observability

#### Health Monitoring
```python
@router.get("/health")
async def health_check():
    checks = {
        "api": "operational",
        "database": await check_database(),
        "redis": await check_redis(), 
        "weaviate": await check_weaviate()
    }
    
    status = "healthy" if all(v == "operational" for v in checks.values()) else "degraded"
    
    return {
        "status": status,
        "timestamp": time.time(),
        "services": checks
    }
```

#### Metrics Collection
- **Application Metrics**: Request rates, response times, error rates
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Sources added, searches performed, jobs completed

#### Logging Strategy
```python
import structlog

logger = structlog.get_logger(__name__)

# Structured logging
logger.info(
    "job_completed",
    job_id=job_id,
    source_id=source_id,
    duration_seconds=duration,
    pages_processed=page_count
)
```

## Scalability Considerations

### Horizontal Scaling

#### Stateless Services
- **API Gateway**: Multiple instances behind load balancer
- **Workers**: Scale based on queue depth
- **Web UI**: CDN distribution of static assets

#### Database Scaling
- **PostgreSQL**: Read replicas for query distribution
- **Weaviate**: Multi-node cluster for large datasets
- **Redis**: Cluster mode for high throughput

### Performance Optimization

#### Caching Strategy
- **Application Cache**: Frequently accessed data in Redis
- **Database Cache**: Query result caching
- **CDN**: Static asset distribution

#### Query Optimization
- **Database Indexes**: Strategic index placement
- **Vector Search**: Approximate nearest neighbor algorithms
- **Batch Operations**: Reduce round-trip overhead

### Resource Management

#### Memory Management
- **Connection Pooling**: Database connection limits
- **Worker Limits**: Concurrent job processing bounds
- **Cache Eviction**: LRU policies for memory usage

#### CPU Optimization
- **Async Processing**: Non-blocking I/O operations
- **Background Jobs**: CPU-intensive work in separate workers
- **Load Balancing**: Distribute work across available cores

This comprehensive architecture documentation provides the foundation for understanding, deploying, and scaling the KnowledgeHub system. Regular updates to this document should accompany major system changes or optimizations.