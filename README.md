# KnowledgeHub - Intelligent Documentation Management System

A comprehensive AI-powered knowledge management system that crawls, indexes, and provides intelligent search across multiple documentation sources using RAG (Retrieval-Augmented Generation) technology with GPU acceleration and automated scheduling.

## 🚀 Key Features

- **Multi-Source Crawling**: Automatically crawl and index documentation from websites, GitHub repos, and other sources
- **RAG-Powered Search**: Semantic search using vector embeddings and large language models
- **GPU Acceleration**: Hardware-accelerated embeddings using NVIDIA Tesla V100s
- **Automated Scheduling**: Weekly delta updates to keep content current
- **Memory System**: Advanced conversation memory for context retention
- **Job Management**: Complete lifecycle with real-time cancellation support
- **Real-time UI**: React-based interface with live progress tracking
- **Production Ready**: No mock data - fully functional with real embeddings

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Web UI │    │   API Gateway   │    │   Scheduler     │
│  (TypeScript)   │    │   (FastAPI)     │    │   (Weekly)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Scraper       │    │  RAG Processor  │    │   Embeddings    │
│   (Playwright)  │    │   (Chunking)    │    │  (GPU Server)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │    Weaviate     │
│   (Metadata)    │    │   (Queues)      │    │   (Vectors)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Current System Status

### Successfully Indexed Content
- **GitHub Documentation**: 1,835 documents/chunks
- **React.dev Documentation**: 58 documents/chunks  
- **FastAPI Documentation**: 53 documents/chunks
- **Python Tutorial**: 25 documents/chunks
- **Total**: 1,971 documents → 27,404 vector chunks

### System Health
- ✅ All services operational
- ✅ GPU embeddings active (Tesla V100)
- ✅ Weekly scheduler running
- ✅ Job cancellation system working
- ✅ Memory system functional
- ✅ All processing queues empty (work complete)

## 🛠️ Technology Stack

### Backend Services
- **FastAPI**: High-performance async API framework
- **PostgreSQL**: Primary database with enum types for job status
- **Redis**: Message queues and job coordination
- **Weaviate**: Vector database for semantic search
- **Playwright**: JavaScript-aware web scraping

### AI/ML Components
- **Sentence Transformers**: Text embeddings (all-MiniLM-L6-v2, 384 dimensions)
- **CUDA/GPU Support**: Tesla V100 acceleration
- **Smart Chunking**: Context-aware text segmentation
- **Vector Search**: Hybrid search with semantic ranking

### Frontend
- **React 18**: Modern UI framework with TypeScript
- **Vite**: Fast build tool and dev server
- **Material-UI**: Professional component library
- **TanStack Query**: Advanced data fetching and caching
- **Real-time Updates**: WebSocket-based live progress

### Infrastructure
- **Docker Compose**: Container orchestration
- **APScheduler**: Automated weekly scheduling
- **Job Queues**: Redis-based async processing
- **Health Monitoring**: Comprehensive system monitoring

## 📦 Installation & Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for acceleration)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/anubissbe/knowledgehub.git
cd knowledgehub
```

2. **Set up environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Start the system**:
```bash
# Standard deployment
docker compose up -d

# With GPU acceleration (recommended)
docker compose -f docker-compose.gpu.yml up -d
```

4. **Access the application**:
- **Web UI**: http://localhost:5173
- **API**: http://localhost:3000
- **API Docs**: http://localhost:3000/docs

### Verify Installation

```bash
# Check all services are running
docker compose ps

# Check system health
curl http://localhost:3000/health

# View processing logs
docker compose logs -f rag-processor
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://khuser:khpassword@postgres:5432/knowledgehub` |
| `REDIS_URL` | Redis connection | `redis://redis:6379/0` |
| `WEAVIATE_URL` | Weaviate vector DB | `http://weaviate:8080` |
| `EMBEDDINGS_URL` | Embeddings service | `http://embeddings:8100` |
| `USE_GPU` | Enable GPU acceleration | `true` |
| `SCHEDULER_ENABLED` | Enable auto-refresh | `true` |
| `REFRESH_SCHEDULE` | Cron for weekly refresh | `0 2 * * 0` |
| `REFRESH_BATCH_SIZE` | Sources per batch | `2` |
| `REFRESH_DELAY_SECONDS` | Delay between batches | `300` |

### Scheduler Configuration

The system includes automated weekly refresh:
```bash
# Default: Every Sunday at 2 AM
REFRESH_SCHEDULE="0 2 * * 0"

# Custom schedules
REFRESH_SCHEDULE="0 1 * * 1"  # Monday 1 AM  
REFRESH_SCHEDULE="0 6 * * *"  # Daily 6 AM
```

## 📚 Usage Guide

### Adding Knowledge Sources

#### Via Web Interface
1. Navigate to the Sources page
2. Click "Add Source"
3. Enter URL and configuration:
   - **URL**: Documentation website
   - **Type**: website/documentation
   - **Max Depth**: How deep to crawl
   - **Max Pages**: Page limit
   - **Refresh Interval**: Days between updates

#### Via API
```bash
curl -X POST "http://localhost:3000/api/v1/sources/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Example Documentation",
    "url": "https://example.com/docs",
    "type": "documentation",
    "refresh_interval": 7,
    "config": {
      "max_depth": 3,
      "max_pages": 500,
      "crawl_delay": 2
    }
  }'
```

### Intelligent Search

#### Web Interface
- Use the search bar with natural language queries
- Apply filters by source, date, or content type
- View highlighted results with relevance scores

#### API Search
```bash
curl -X POST "http://localhost:3000/api/v1/search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to handle async functions in React hooks?",
    "limit": 10,
    "filters": {
      "source_name": "React.dev Documentation"
    }
  }'
```

### Memory Management

The system includes advanced memory capabilities:

```bash
# Add memory
curl -X POST "http://localhost:3000/api/v1/memories/" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers concise code examples",
    "memory_type": "preference",
    "source": "conversation"
  }'

# Search memories  
curl -X GET "http://localhost:3000/api/v1/memories/search?query=preferences&limit=5"

# Get memory statistics
curl -X GET "http://localhost:3000/api/v1/memories/stats"
```

### Job Management

#### Monitor Jobs
```bash
# List all jobs
curl "http://localhost:3000/api/v1/jobs/"

# Filter by status
curl "http://localhost:3000/api/v1/jobs/?status=running"

# Get specific job
curl "http://localhost:3000/api/v1/jobs/{job_id}"
```

#### Cancel Jobs
```bash
# Cancel a running job
curl -X POST "http://localhost:3000/api/v1/jobs/{job_id}/cancel"

# Retry a failed job
curl -X POST "http://localhost:3000/api/v1/jobs/{job_id}/retry"
```

#### Job Cancellation Features
- **Pending Jobs**: Removed from Redis queue before processing
- **Running Jobs**: Gracefully stopped during crawling
- **Real-time Updates**: Immediate status updates in UI
- **Queue Management**: Automatic cleanup of cancelled work

## 🔄 Advanced Features

### Automated Scheduling

The scheduler automatically refreshes sources:
- **Delta Detection**: Only processes changed content
- **Batch Processing**: Handles multiple sources efficiently  
- **Configurable Timing**: Custom cron schedules
- **Error Handling**: Robust retry mechanisms

#### Manual Refresh
```bash
# Refresh specific source
curl -X POST "http://localhost:3000/api/v1/sources/{source_id}/refresh"

# Trigger scheduler manually
curl -X POST "http://localhost:3000/api/v1/scheduler/refresh"
```

### GPU Acceleration

When available, the system uses GPU acceleration:
- **Tesla V100 GPUs**: 2x GPUs for parallel processing
- **Embedding Generation**: 10x faster than CPU
- **Batch Processing**: Optimized for GPU memory
- **Automatic Fallback**: CPU backup if GPU unavailable

### Vector Search Technology

- **Semantic Search**: Understanding context and meaning
- **Hybrid Ranking**: Combines keyword and vector scores
- **Relevance Tuning**: Configurable similarity thresholds
- **Multi-language**: Supports various documentation languages

## 🏃 Development Guide

### Project Structure
```
knowledgehub/
├── src/
│   ├── api/                    # FastAPI backend
│   │   ├── routers/           # API endpoints
│   │   ├── services/          # Business logic  
│   │   ├── models/            # Database models
│   │   └── schemas/           # Pydantic schemas
│   ├── web-ui/                # React frontend
│   │   ├── src/components/    # UI components
│   │   ├── src/services/      # API integration
│   │   └── src/types/         # TypeScript types
│   ├── scraper/               # Web crawling
│   │   ├── crawler.py         # Playwright engine
│   │   └── parsers.py         # Content extraction
│   ├── rag_processor/         # AI processing
│   │   ├── main.py            # Queue processor
│   │   ├── chunker.py         # Smart chunking
│   │   └── embeddings_remote.py
│   ├── scheduler/             # Automated tasks
│   │   ├── main.py            # APScheduler service
│   │   └── Dockerfile         # Container config
│   └── shared/                # Common utilities
├── docker-compose.yml         # Main orchestration
├── docker-compose.gpu.yml     # GPU variant
└── docs/                      # Documentation
```

### Running Tests

```bash
# API functionality tests
python test_all_functionality.py
python test_basic_functionality.py  
python test_services.py

# Code quality checks
python test_code_quality.py
python test_imports.py
python test_structure.py

# System integration tests
./test_complete_system.sh
./test_in_docker.sh

# Memory system validation
cd /opt/projects/memory-system
python validate_system.py
```

### Development Commands

```bash
# Frontend development
cd src/web-ui
npm install
npm run dev              # Start dev server
npm run build           # Production build
npm run lint            # ESLint check

# Backend development  
cd src/api
python -m pytest       # Run tests
black --check .         # Code formatting
flake8 .               # Style guide

# Database operations
docker compose exec postgres psql -U khuser -d knowledgehub

# Queue monitoring
docker compose exec redis redis-cli
> LLEN rag_processing:normal
> LLEN crawl_jobs:normal
```

## 📊 Monitoring & Health

### System Monitoring

```bash
# Check all containers
docker compose ps

# Service health checks
curl http://localhost:3000/health
curl http://localhost:8100/health

# Resource usage
docker stats

# Log monitoring
docker compose logs -f api
docker compose logs -f scraper
docker compose logs -f rag-processor
docker compose logs -f scheduler
```

### Performance Metrics

The system tracks:
- **Processing Speed**: Pages/minute crawled
- **Queue Depth**: Pending jobs in Redis
- **Memory Usage**: Container resource consumption  
- **GPU Utilization**: Embedding generation efficiency
- **Search Performance**: Query response times

### Database Statistics

```sql
-- Connect to database
docker compose exec postgres psql -U khuser -d knowledgehub

-- Check content statistics
SELECT 
  ks.name,
  ks.status,
  COUNT(d.id) as documents,
  COUNT(dc.id) as chunks
FROM knowledge_sources ks
LEFT JOIN documents d ON ks.id = d.source_id  
LEFT JOIN document_chunks dc ON d.id = dc.document_id
GROUP BY ks.id, ks.name, ks.status;

-- Job statistics
SELECT status, COUNT(*) FROM scraping_jobs GROUP BY status;

-- Memory system stats
SELECT COUNT(*) as total_memories, 
       AVG(access_count) as avg_access_count
FROM memory_items;
```

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. Sources Stuck in "Pending"
```bash
# Check scraper logs
docker compose logs scraper

# Verify queue has jobs
docker compose exec redis redis-cli LLEN crawl_jobs:normal

# Restart scraper if needed
docker compose restart scraper
```

#### 2. Search Returns No Results
```bash
# Check Weaviate is running
curl http://localhost:8080/v1/meta

# Verify embeddings were generated
docker compose logs rag-processor | grep "Successfully processed"

# Check vector database content
curl http://localhost:8080/v1/objects | jq '.objects | length'
```

#### 3. High Memory Usage
```bash
# Reduce batch sizes
export REFRESH_BATCH_SIZE=1
export RAG_BATCH_SIZE=25

# Lower concurrent workers
export MAX_CONCURRENT_SCRAPERS=2

# Restart with new settings
docker compose restart
```

#### 4. GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Verify GPU configuration
docker compose logs embeddings | grep "CUDA"

# Use CPU fallback
export USE_GPU=false
docker compose restart embeddings
```

#### 5. Scheduler Not Running
```bash
# Check scheduler logs
docker compose logs scheduler

# Verify schedule configuration
curl http://localhost:3000/api/v1/scheduler/status

# Manual trigger
curl -X POST http://localhost:3000/api/v1/scheduler/refresh
```

### Performance Optimization

#### For High-Volume Processing
```bash
# Increase worker counts
export MAX_CONCURRENT_SCRAPERS=10
export RAG_BATCH_SIZE=100

# Optimize memory settings
export POSTGRES_SHARED_BUFFERS=512MB
export REDIS_MAXMEMORY=2GB

# Use GPU acceleration
export USE_GPU=true
export EMBEDDING_BATCH_SIZE=128
```

#### For Resource-Constrained Environments
```bash
# Reduce resource usage
export MAX_CONCURRENT_SCRAPERS=2
export RAG_BATCH_SIZE=25
export REFRESH_BATCH_SIZE=1

# Increase delays
export CRAWL_DELAY=3
export REFRESH_DELAY_SECONDS=600
```

## 🔒 Security & Production

### Security Features
- **API Key Authentication**: Configurable API keys
- **Rate Limiting**: Request throttling protection
- **Input Validation**: Comprehensive request validation
- **CORS Configuration**: Secure cross-origin requests
- **SQL Injection Prevention**: Parameterized queries

### Production Deployment
```bash
# Use production compose file
docker compose -f docker-compose.prod.yml up -d

# Set production environment
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=INFO

# Configure secure secrets
export DATABASE_PASSWORD=$(openssl rand -base64 32)
export API_SECRET_KEY=$(openssl rand -base64 32)
```

### Backup & Recovery
```bash
# Database backup
docker compose exec postgres pg_dump -U khuser knowledgehub > backup.sql

# Vector database backup
curl http://localhost:8080/v1/objects > weaviate_backup.json

# Redis backup
docker compose exec redis redis-cli BGSAVE
```

## 🤝 Contributing

### Getting Started
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test thoroughly
4. Commit changes: `git commit -m 'Add amazing feature'`  
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for all frontend code
- Add tests for new features
- Update documentation for changes
- Ensure Docker builds successfully

### Testing Requirements
- All tests must pass: `python test_all_functionality.py`
- Code quality checks: `python test_code_quality.py`
- Integration tests: `./test_complete_system.sh`
- Frontend tests: `npm test` (when available)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Weaviate**: Excellent vector database platform
- **Sentence Transformers**: High-quality embedding models
- **FastAPI**: Outstanding async web framework  
- **React**: Powerful frontend library
- **PostgreSQL**: Reliable database system
- **Docker**: Containerization platform

## 📞 Support

### Getting Help
- **Issues**: Create an issue on GitHub
- **Documentation**: Check the `/docs` folder
- **Examples**: See usage examples above
- **Community**: Join discussions in GitHub Discussions

### Reporting Bugs
When reporting issues, please include:
- System information (OS, Docker version)
- Error logs from relevant services
- Steps to reproduce the issue
- Expected vs actual behavior

---

**KnowledgeHub** - Intelligent Documentation Management  
Built with ❤️ for the AI-powered future of knowledge management