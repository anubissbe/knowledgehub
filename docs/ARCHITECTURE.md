# KnowledgeHub Architecture Documentation

## System Overview

KnowledgeHub is a microservices-based intelligent documentation management system built with modern technologies for scalability, performance, and reliability.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KnowledgeHub System                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   React Web UI │    │   API Gateway   │    │   Scheduler     │         │
│  │  (TypeScript)   │◄──►│   (FastAPI)     │◄──►│ (APScheduler)   │         │
│  │   Port: 5173    │    │   Port: 3000    │    │   Background    │         │
│  └─────────────────┘    └─────────┬───────┘    └─────────────────┘         │
│                                   │                                         │
│                    ┌──────────────┼──────────────┐                         │
│                    │              │              │                         │
│                    ▼              ▼              ▼                         │
│         ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│         │   Scraper       │ │  RAG Processor  │ │   Embeddings    │        │
│         │   (Playwright)  │ │   (Chunking)    │ │  (GPU Server)   │        │
│         │   Background    │ │   Background    │ │   Port: 8100    │        │
│         └─────────────────┘ └─────────────────┘ └─────────────────┘        │
│                    │              │              │                         │
│                    └──────────────┼──────────────┘                         │
│                                   │                                         │
│         ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│         │   PostgreSQL    │ │     Redis       │ │    Weaviate     │        │
│         │   (Metadata)    │ │   (Queues)      │ │   (Vectors)     │        │
│         │   Port: 5432    │ │   Port: 6379    │ │   Port: 8080    │        │
│         └─────────────────┘ └─────────────────┘ └─────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Web UI (React Frontend)
- **Technology**: React 18, TypeScript, Vite
- **Port**: 5173
- **Purpose**: User interface for search, source management, and monitoring
- **Features**:
  - Real-time job progress updates
  - Source configuration and management
  - Search interface with filters
  - Job monitoring and cancellation
  - Memory management interface

### 2. API Gateway (FastAPI Backend)
- **Technology**: FastAPI, Python 3.11, SQLAlchemy
- **Port**: 3000
- **Purpose**: Central API hub for all operations
- **Features**:
  - RESTful API endpoints
  - WebSocket for real-time updates
  - Authentication and rate limiting
  - Request validation and error handling
  - Database ORM integration

### 3. Scheduler Service
- **Technology**: APScheduler, Python
- **Purpose**: Automated refresh and maintenance tasks
- **Features**:
  - Weekly source refresh scheduling
  - Delta detection for efficient updates
  - Batch processing to avoid overload
  - Configurable cron schedules
  - Error handling and retry logic

### 4. Scraper Service
- **Technology**: Playwright, HTTPX, Python
- **Purpose**: Web crawling and content extraction
- **Features**:
  - JavaScript-aware crawling with Playwright
  - Configurable crawl patterns and delays
  - Rate limiting and robots.txt compliance
  - Content parsing and extraction
  - Queue-based job processing

### 5. RAG Processor
- **Technology**: Python, Sentence Transformers
- **Purpose**: Content processing and embedding generation
- **Features**:
  - Smart content chunking with overlap
  - Batch processing for efficiency
  - Integration with embeddings service
  - Vector storage in Weaviate
  - Metadata management

### 6. Embeddings Service
- **Technology**: Sentence Transformers, FastAPI
- **Port**: 8100
- **Purpose**: Text embedding generation
- **Features**:
  - GPU acceleration with Tesla V100
  - Batch processing support
  - Model: all-MiniLM-L6-v2 (384 dimensions)
  - RESTful API interface
  - CPU fallback capability

### 7. PostgreSQL Database
- **Technology**: PostgreSQL 15
- **Port**: 5432
- **Purpose**: Primary data storage
- **Schema**:
  - Knowledge sources metadata
  - Document and chunk information
  - Job tracking and status
  - Memory items for conversation context
  - User management and API keys

### 8. Redis Cache/Queue
- **Technology**: Redis 7
- **Port**: 6379
- **Purpose**: Message queues and caching
- **Features**:
  - Job queue management
  - Session storage
  - Rate limiting counters
  - Caching for performance
  - Real-time communication

### 9. Weaviate Vector Database
- **Technology**: Weaviate
- **Port**: 8080
- **Purpose**: Vector search and storage
- **Features**:
  - High-performance vector similarity search
  - Schema-less configuration
  - GraphQL and REST APIs
  - Automatic indexing
  - Hybrid search capabilities

## Data Flow

### 1. Content Ingestion Flow
```
Source Addition → Job Creation → Queue → Scraper → Content Extraction → 
RAG Processor → Chunking → Embedding Generation → Vector Storage → 
Metadata Storage → Job Completion
```

### 2. Search Flow
```
User Query → API Gateway → Query Embedding → Weaviate Search → 
Result Ranking → Metadata Enrichment → Response Formation → UI Display
```

### 3. Scheduling Flow
```
Cron Trigger → Scheduler → Source Analysis → Delta Detection → 
Job Creation → Processing Pipeline → Status Updates
```

## Service Communication

### Synchronous Communication
- **Web UI ↔ API Gateway**: HTTP/HTTPS REST API
- **API Gateway ↔ Databases**: Direct database connections
- **Embeddings Service**: HTTP REST API calls

### Asynchronous Communication
- **Job Queues**: Redis-based message queues
- **Real-time Updates**: WebSocket connections
- **Background Processing**: Event-driven architecture

### Data Storage Patterns
- **PostgreSQL**: ACID transactions for metadata
- **Redis**: Fast access for queues and cache
- **Weaviate**: Optimized for vector operations

## Scalability Considerations

### Horizontal Scaling
- **Scraper Services**: Multiple instances can process different queues
- **RAG Processors**: Can be scaled based on processing load
- **API Gateway**: Stateless design allows multiple instances
- **Embeddings Service**: GPU resources can be distributed

### Vertical Scaling
- **Database**: Increased memory and CPU for larger datasets
- **Redis**: More memory for larger queues and cache
- **GPU Memory**: Larger batches for embedding generation

### Performance Optimizations
- **Batch Processing**: Groups operations for efficiency
- **Connection Pooling**: Reuses database connections
- **Caching**: Redis for frequently accessed data
- **Async Processing**: Non-blocking operations

## Security Architecture

### Authentication & Authorization
- **API Keys**: Service-to-service authentication
- **Rate Limiting**: Protection against abuse
- **Input Validation**: Request sanitization
- **CORS**: Cross-origin request control

### Data Security
- **Encryption**: Database connections use TLS
- **Secrets Management**: Environment variables
- **Network Isolation**: Docker network segmentation
- **Access Control**: Principle of least privilege

## Monitoring & Observability

### Health Checks
- **Service Health**: HTTP endpoints for status
- **Database Connectivity**: Connection pooling status
- **Queue Health**: Redis queue depth monitoring
- **Resource Usage**: CPU, memory, and GPU utilization

### Logging
- **Structured Logging**: JSON format for parsing
- **Log Levels**: Configurable verbosity
- **Error Tracking**: Exception capture and reporting
- **Performance Metrics**: Response times and throughput

### Metrics Collection
- **Custom Metrics**: Application-specific measurements
- **System Metrics**: Infrastructure monitoring
- **Business Metrics**: User activity and content growth

## Deployment Architecture

### Container Strategy
- **Docker Compose**: Local development and testing
- **Multi-stage Builds**: Optimized image sizes
- **Health Checks**: Container health monitoring
- **Resource Limits**: Memory and CPU constraints

### Environment Management
- **Environment Variables**: Configuration management
- **Secrets**: Secure credential handling
- **Multi-environment**: Development, staging, production
- **Configuration Validation**: Startup checks

### Backup & Recovery
- **Database Backups**: Automated PostgreSQL dumps
- **Vector Backups**: Weaviate data export
- **Configuration Backups**: Docker compose and env files
- **Recovery Procedures**: Documented restoration steps

## Future Architecture Considerations

### Microservices Evolution
- **Service Mesh**: Enhanced service communication
- **Circuit Breakers**: Fault tolerance patterns
- **Distributed Tracing**: Cross-service monitoring
- **API Gateway**: Enhanced routing and security

### Scalability Enhancements
- **Kubernetes**: Container orchestration
- **Load Balancing**: Traffic distribution
- **Auto-scaling**: Dynamic resource allocation
- **Caching Layers**: Multi-level caching strategy

### Technology Upgrades
- **Database Sharding**: Horizontal partitioning
- **Message Brokers**: Apache Kafka for high-volume
- **Search Engines**: Elasticsearch integration
- **ML Pipeline**: MLOps for model management