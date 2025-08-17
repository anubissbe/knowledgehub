# KnowledgeHub Integrated Services Deployment Guide

This guide covers the deployment and management of the complete KnowledgeHub infrastructure with all integrated AI services.

## 🏗️ Architecture Overview

### Service Stack
```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Web UI (3100)  │  Main API (3000)  │  Nginx Proxy (8080) │
├─────────────────────────────────────────────────────────────┤
│                      AI Services Layer                      │
├─────────────────────────────────────────────────────────────┤
│ Zep Memory │ Firecrawl │ Graphiti │ Phoenix │ AI Service    │
│   (8100)   │  (3002)   │  (8080)  │ (6006)  │   (8002)     │
├─────────────────────────────────────────────────────────────┤
│                    Data Storage Layer                       │
├─────────────────────────────────────────────────────────────┤
│ PostgreSQL │ TimescaleDB │ Redis │ Weaviate │ Qdrant │ Neo4j│
│   (5433)   │    (5434)   │(6381) │  (8090)  │ (6333) │(7474)│
├─────────────────────────────────────────────────────────────┤
│                  Monitoring & Observability                 │
├─────────────────────────────────────────────────────────────┤
│  Prometheus (9090)  │  Grafana (3001)  │  MinIO (9010)     │
└─────────────────────────────────────────────────────────────┘
```

### New Services Added

1. **Zep Memory System (Port 8100)**
   - Conversational memory and context management
   - PostgreSQL backend with pgvector
   - Integrates with main API for memory persistence

2. **Firecrawl Service (Port 3002)**
   - Intelligent web scraping and content ingestion
   - Playwright-powered browser automation
   - Rate-limited and optimized for large-scale scraping

3. **Graphiti GraphRAG (Port 8080)**
   - Graph-based Retrieval Augmented Generation
   - Neo4j integration for knowledge graphs
   - Advanced semantic understanding

4. **Phoenix Observability (Port 6006)**
   - AI model performance monitoring
   - Real-time metrics and tracing
   - ML model observability dashboard

5. **Enhanced Vector Storage**
   - Qdrant as high-performance alternative to Weaviate
   - Optimized for large-scale vector operations
   - Better performance characteristics

## 🚀 Deployment Process

### Prerequisites

- Docker 20.10.0+
- Docker Compose 1.29.0+
- 10GB+ available disk space
- 8GB+ RAM recommended
- Linux/macOS environment

### Quick Start

1. **Clone and Setup**:
```bash
cd /opt/projects/knowledgehub
cp .env.example .env
# Edit .env with your configuration
```

2. **Deploy Services**:
```bash
# Automated deployment with health checks
./deploy-integrated-services.sh

# Or manual deployment
docker-compose up -d
```

3. **Validate Deployment**:
```bash
# Comprehensive validation
./scripts/validate-integration.sh

# Quick health check
./scripts/health-check.sh
```

### Manual Deployment Steps

1. **Stage 1 - Core Infrastructure**:
```bash
docker-compose up -d postgres redis timescale minio
# Wait for services to be healthy
./scripts/health-check.sh
```

2. **Stage 2 - Vector and Graph Databases**:
```bash
docker-compose up -d weaviate qdrant neo4j
# Wait for initialization
sleep 60
```

3. **Stage 3 - AI Memory System**:
```bash
docker-compose up -d zep-postgres
# Wait for database initialization
docker-compose up -d zep
```

4. **Stage 4 - AI Services**:
```bash
docker-compose up -d phoenix playwright-service firecrawl graphiti
```

5. **Stage 5 - Application Layer**:
```bash
docker-compose up -d ai-service api webui nginx
```

6. **Stage 6 - Monitoring**:
```bash
docker-compose up -d prometheus grafana
```

## 🔧 Configuration Management

### Environment Variables

Key configuration options in `.env`:

```bash
# Database Configuration
DATABASE_PASSWORD=knowledgehub123
NEO4J_PASSWORD=knowledgehub123
TIMESCALE_PASSWORD=knowledgehub123
ZEP_DB_PASSWORD=zep123

# AI API Keys (Optional)
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
FIRECRAWL_API_KEY=your-key-here

# Resource Limits
REDIS_MEMORY_LIMIT=512mb
POSTGRES_SHARED_BUFFERS=256MB
NEO4J_HEAP_SIZE=1G
```

### Service-Specific Configuration

#### Zep Memory System
- Configuration: `zep-config.yaml`
- Database: Dedicated PostgreSQL with pgvector
- Memory window: 20 messages
- Automatic summarization enabled

#### Firecrawl Service
- Browser automation via Playwright
- Rate limiting: 5 requests/second
- Timeout: 300 seconds for complex pages
- Redis queue for job management

#### Graphiti GraphRAG
- Neo4j backend for graph storage
- Redis for caching and queues
- Supports OpenAI and Anthropic models
- Automatic knowledge graph construction

#### Phoenix Observability
- gRPC endpoint on port 4317
- Web UI on port 6006
- Automatic trace collection
- Model performance analytics

## 🔍 Monitoring and Observability

### Service Health Monitoring

```bash
# Continuous monitoring
./scripts/health-check.sh monitor

# Single health check
./scripts/health-check.sh check

# Resource usage
./scripts/health-check.sh resources
```

### Prometheus Metrics

All services expose metrics on `/metrics` endpoints:
- API performance and error rates
- Database connection pools
- Vector database operations
- Memory system usage
- AI service response times

### Grafana Dashboards

Access Grafana at `http://localhost:3001` (admin/admin):
- **KnowledgeHub Overview**: System-wide metrics
- **AI Performance**: AI service metrics
- **Database Performance**: All database metrics
- **Security Dashboard**: Security events and alerts

### Phoenix Observability

Access Phoenix at `http://localhost:6006`:
- Model performance traces
- Embedding drift detection
- Response time analysis
- Error pattern recognition

## 🛡️ Security Configuration

### Network Security

- Isolated Docker network with custom subnet
- Rate limiting on all public endpoints
- Security headers via Nginx
- Basic authentication for admin endpoints

### Data Security

- Database encryption at rest
- Redis AUTH protection
- MinIO access control
- Network isolation between services

### API Security

```nginx
# Rate limiting configuration
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;

# Security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
```

## 🔄 Service Integration Patterns

### Memory System Integration

```python
# API integration with Zep
from api.services.zep_memory import ZepMemoryService

zep = ZepMemoryService()
session = await zep.create_session(user_id="claude")
await zep.add_memory(session_id, message="User query", metadata={})
```

### Vector Database Integration

```python
# Dual vector database support
from api.services.vector_store import VectorStore

vector_store = VectorStore()
# Automatically selects best available DB (Qdrant > Weaviate)
results = await vector_store.similarity_search(query_vector, k=10)
```

### Graph RAG Integration

```python
# Graphiti integration for knowledge graphs
from api.services.graphrag_service import GraphRAGService

graph_rag = GraphRAGService()
enhanced_context = await graph_rag.enhance_context(
    query="What is machine learning?",
    context=existing_context
)
```

## 📊 Performance Optimization

### Resource Allocation

Recommended resource allocation:

```yaml
services:
  postgres:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
  
  redis:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
  
  weaviate:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### Performance Tuning

1. **Database Optimization**:
   - PostgreSQL: Tune `shared_buffers`, `work_mem`
   - Redis: Configure memory policy
   - Neo4j: Optimize heap and page cache

2. **Vector Database Tuning**:
   - Qdrant: Optimize segment size and indexing
   - Weaviate: Configure memory allocation

3. **AI Service Optimization**:
   - Enable GPU acceleration where available
   - Configure model caching
   - Optimize batch sizes

## 🚨 Troubleshooting

### Common Issues

1. **Service Won't Start**:
```bash
# Check logs
docker-compose logs <service-name>

# Check resource usage
docker stats

# Restart service
docker-compose restart <service-name>
```

2. **Database Connection Issues**:
```bash
# Test PostgreSQL connection
PGPASSWORD=knowledgehub123 psql -h localhost -p 5433 -U knowledgehub -d knowledgehub

# Test Redis connection
redis-cli -h localhost -p 6381 ping
```

3. **Memory Issues**:
```bash
# Check memory usage
free -h
docker system df

# Clean up resources
docker system prune
```

### Service-Specific Troubleshooting

#### Zep Memory System
- Check PostgreSQL connectivity
- Verify pgvector extension installation
- Review configuration in `zep-config.yaml`

#### Firecrawl Service
- Ensure Playwright service is running
- Check Redis connectivity
- Verify rate limiting settings

#### Vector Databases
- Check index status and health
- Monitor memory usage
- Verify data import completion

## 📈 Scaling Considerations

### Horizontal Scaling

1. **Database Scaling**:
   - PostgreSQL: Read replicas
   - Redis: Cluster mode
   - Vector DBs: Sharding

2. **Service Scaling**:
```bash
# Scale specific services
docker-compose up -d --scale api=3
docker-compose up -d --scale ai-service=2
```

3. **Load Balancing**:
   - Nginx upstream configuration
   - Health check integration
   - Failover handling

### Vertical Scaling

Adjust resource limits in docker-compose.yml:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 🔄 Backup and Recovery

### Automated Backup

```bash
# Full system backup
./backup-knowledgehub.sh

# Service-specific backups
docker-compose exec postgres pg_dump -U knowledgehub knowledgehub > postgres_backup.sql
docker-compose exec redis redis-cli BGSAVE
```

### Recovery Procedures

```bash
# Restore from backup
./restore-knowledgehub.sh backup-20240816

# Service-specific restore
docker-compose exec postgres psql -U knowledgehub -d knowledgehub < postgres_backup.sql
```

## 📚 API Documentation

### Service Endpoints

| Service | Endpoint | Purpose |
|---------|----------|---------|
| Main API | `http://localhost:3000/docs` | FastAPI documentation |
| Zep Memory | `http://localhost:8100/docs` | Memory system API |
| Graphiti | `http://localhost:8080/docs` | GraphRAG API |
| Phoenix | `http://localhost:6006` | Observability UI |
| Qdrant | `http://localhost:6333/docs` | Vector database API |

### Integration Examples

See the `examples/` directory for:
- Memory system integration
- Vector search implementation
- Graph RAG usage
- Observability setup

## 🎯 Next Steps

After successful deployment:

1. **Configure API Keys**: Add real API keys for external services
2. **Setup SSL**: Configure SSL certificates for production
3. **Scale Services**: Adjust resource allocation based on usage
4. **Monitor Performance**: Set up alerting and monitoring
5. **Backup Strategy**: Implement regular backup procedures

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review service logs
- Run validation scripts
- Open an issue with deployment logs

---

**Status**: ✅ Ready for production deployment  
**Last Updated**: August 16, 2024  
**Version**: 2.0.0 with integrated AI services