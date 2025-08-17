# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🧠 Project Overview

**KnowledgeHub** is an enterprise AI-enhanced development platform that provides persistent memory, advanced knowledge systems, and intelligent automation for AI coding assistants. It's designed as a universal backend that works with ANY AI coding tool (Claude Code, VSCode Copilot, Cursor, etc.) via REST APIs, WebSocket, and MCP (Model Context Protocol) integration.

### Core Value Proposition
- **Universal Backend**: Works with ANY AI coding assistant, not tool-specific
- **Persistent Memory**: Session continuity and context across all your tools
- **Learning System**: Continuously improves from usage patterns and mistakes
- **Enterprise Ready**: Multi-tenant, GDPR compliant, 99.9% uptime SLA
- **Production Tested**: 90%+ test coverage, comprehensive monitoring

## 🏗️ Architecture Overview

### Microservices Stack
```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Coding Tools                              │
│        (Claude Code, Cursor, Copilot, Codeium, etc)           │
└──────────────────────────┬──────────────────────────────────────┘
                          │ REST API / WebSocket / MCP
┌──────────────────────────┴──────────────────────────────────────┐
│                    KnowledgeHub API Gateway                     │
│                      FastAPI (Port 3000)                       │
├─────────────────────────────────────────────────────────────────┤
│ 8 AI Intelligence Systems + Enterprise Features                │
│ • Session Continuity     • Mistake Learning    • Decision AI   │
│ • Code Evolution        • Performance Intel    • Workflow AI   │
│ • GraphRAG + Neo4j      • LlamaIndex RAG      • Real-time AI  │
└────┬────────────┬────────────┬────────────┬────────────────────┘
     │            │            │            │
┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐
│PostgreSQL│ │TimescaleDB│ │  Neo4j  │ │ Weaviate│
│(Primary) │ │(Analytics)│ │ (Graph) │ │(Vectors)│
│   +      │ │     +     │ │    +    │ │    +    │
│  Redis   │ │   MinIO   │ │Prometheus│ │ Grafana │
└─────────┘ └─────────────┘ └─────────┘ └─────────┘
```

### Database Architecture
| Database | Purpose | Performance | Scale |
|----------|---------|-------------|--------|
| **PostgreSQL** | Primary data, sessions, metadata | 10K+ req/sec | Multi-TB |
| **TimescaleDB** | Time-series analytics, metrics | 1M+ points/sec | Hypertables |
| **Neo4j** | Knowledge graphs, relationships | Sub-100ms traversal | 100M+ nodes |
| **Weaviate** | Vector embeddings, semantic search | 1K+ vectors/sec | Billions |
| **Redis** | Cache, sessions, real-time data | 100K+ ops/sec | Memory-optimized |
| **MinIO** | Object storage, files, backups | S3-compatible | Petabyte+ |

### Core AI Intelligence Systems
1. **Session Continuity**: Auto-restore sessions with full context across tools
2. **Mistake Learning**: ML-powered error pattern recognition and prevention
3. **Decision Reasoning**: Track technical decisions with alternatives and outcomes
4. **Proactive Assistant**: Predict next tasks based on development patterns
5. **Code Evolution**: Track code changes with context and reasoning
6. **Performance Intelligence**: Monitor and optimize development workflows
7. **Workflow Automation**: Pattern-based automation and template generation
8. **Advanced Analytics**: Productivity metrics and project health insights

## 🚀 Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Docker & Docker Compose
- 8GB+ RAM recommended
- 10GB+ free disk space

### Quick Start
```bash
# Clone and start all services
git clone <repository>
cd knowledgehub

# Start all services (PostgreSQL, Redis, Weaviate, Neo4j, TimescaleDB, MinIO, API, Frontend)
docker-compose up -d

# Verify all services are healthy
docker-compose ps --filter "status=running"

# Check API health
curl http://localhost:3000/health

# Access web interface
open http://localhost:3100
```

### Service Endpoints
- **Main API**: http://localhost:3000 (FastAPI with OpenAPI docs at /docs)
- **Web UI**: http://localhost:3100 (React frontend)
- **PostgreSQL**: localhost:5433 (knowledgehub/knowledgehub123)
- **TimescaleDB**: localhost:5434 (analytics)
- **Redis**: localhost:6381
- **Weaviate**: localhost:8090
- **Neo4j**: localhost:7474 (web) / 7687 (bolt)
- **MinIO**: localhost:9010 (API) / 9011 (console)

## 💻 Common Development Commands

### Backend Development (Python/FastAPI)
```bash
# Install dependencies
pip install -r requirements.txt

# Run API locally (development)
cd api && python -m uvicorn main:app --reload --host 0.0.0.0 --port 3000

# Code quality checks
black . --line-length 88                    # Code formatting
flake8 . --max-line-length=88                # Linting
mypy . --ignore-missing-imports              # Type checking

# Testing
pytest                                       # Run all tests
pytest -m unit                              # Run unit tests only
pytest -m integration                       # Run integration tests
pytest -m e2e                              # Run end-to-end tests
pytest --cov=api --cov-report=html         # Generate coverage report

# Specific test categories
pytest -m "api"                             # API endpoint tests
pytest -m "database"                        # Database tests
pytest -m "ai"                             # AI service tests
pytest -m "memory"                         # Memory system tests
pytest -m "rag"                            # RAG system tests
pytest -m "performance"                    # Performance tests

# Database migrations
python -m alembic upgrade head              # Apply migrations
python -m alembic revision --autogenerate -m "Description"  # Create migration
```

### Frontend Development (React/TypeScript)
```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev                                 # Start dev server (http://localhost:5173)

# Build and quality checks
npm run build                              # Production build
npm run lint                               # ESLint checking
npm run preview                            # Preview production build

# Testing
npm test                                   # Run tests
npm run test:e2e                          # End-to-end tests with Playwright
```

### Database Management
```bash
# PostgreSQL (Primary Database)
docker exec -it knowledgehub-postgres-1 psql -U knowledgehub -d knowledgehub

# TimescaleDB (Analytics)
docker exec -it knowledgehub-timescale-1 psql -U knowledgehub -d knowledgehub_analytics

# Redis (Cache)
docker exec -it knowledgehub-redis-1 redis-cli

# Neo4j (Knowledge Graph)
# Access via web interface: http://localhost:7474
# Or use cypher-shell:
docker exec -it knowledgehub-neo4j-1 cypher-shell -u neo4j -p knowledgehub123

# MinIO (Object Storage)
# Access via web console: http://localhost:9011 (minioadmin/minioadmin)
```

### Docker Management
```bash
# View service status
docker-compose ps

# View logs
docker-compose logs -f api                 # API logs
docker-compose logs -f webui               # Frontend logs
docker-compose logs -f postgres            # Database logs

# Restart specific services
docker-compose restart api
docker-compose restart webui

# Stop all services
docker-compose down

# Full reset (removes all data)
docker-compose down -v
docker-compose up -d
```

### Monitoring and Troubleshooting
```bash
# Health checks
curl http://localhost:3000/health          # API health
curl http://localhost:3000/api/v1/admin/status  # Detailed status

# Performance metrics
curl http://localhost:3000/metrics         # Prometheus metrics

# Database connection tests
python check_database_connection.py        # Test all DB connections

# Comprehensive functionality test
python comprehensive_functionality_test.py # Test all AI features

# API endpoint testing
python test_api_endpoints.py               # Test all endpoints
```

## 🧪 Testing Strategy

### Test Organization
- **Unit Tests** (`tests/unit/`): Individual component testing
- **Integration Tests** (`tests/integration/`): Multi-component interactions
- **E2E Tests** (`tests/e2e/`): Full user workflow simulation
- **Performance Tests** (`tests/performance/`): Load and benchmark testing
- **RAG Tests** (`tests/rag/`): RAG system-specific testing

### Running Tests by Category
```bash
# Core system tests
pytest -m "unit or integration"            # Core functionality
pytest -m "api and not slow"               # Fast API tests
pytest -m "database"                       # Database layer tests

# AI system tests
pytest -m "ai"                            # AI service tests
pytest -m "memory"                        # Memory system tests
pytest -m "rag"                           # RAG system tests
pytest -m "graphrag"                      # GraphRAG with Neo4j
pytest -m "llamaindex"                    # LlamaIndex integration

# Performance and load tests
pytest -m "performance"                   # Performance benchmarks
pytest -m "load"                          # Load testing
pytest -m "slow"                          # Long-running tests

# Integration tests
pytest -m "e2e"                           # End-to-end workflows
pytest -m "websocket"                     # WebSocket functionality
pytest -m "mcp"                           # MCP server integration
```

### Test Coverage Requirements
- **Minimum Coverage**: 90% (enforced by pytest-cov)
- **Critical Paths**: 95%+ coverage required
- **New Features**: Must include comprehensive tests
- **AI Systems**: Require both unit and integration tests

## 🔧 Configuration Management

### Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql://knowledgehub:password@localhost:5433/knowledgehub
TIMESCALE_URL=postgresql://knowledgehub:password@localhost:5434/knowledgehub_analytics
REDIS_URL=redis://localhost:6381/0
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=knowledgehub123
WEAVIATE_URL=http://localhost:8090

# AI Service Configuration
AI_SERVICE_URL=http://localhost:8002
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Security Configuration
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_fernet_key_here
DISABLE_AUTH=true  # For development only

# Performance Configuration
REDIS_CACHE_TTL=3600
WEAVIATE_BATCH_SIZE=100
MAX_CONCURRENT_REQUESTS=100

# Feature Toggles
RAG_ADVANCED_ENABLED=true
GRAPHRAG_ENABLED=true
LLAMAINDEX_ENABLED=true
TIME_SERIES_ENABLED=true
```

### Configuration Files
- **Docker Compose**: `docker-compose.yml` (main services)
- **API Config**: `api/config.py` (application settings)
- **Frontend Config**: `frontend/vite.config.ts` (build configuration)
- **Database Migrations**: `migrations/` (SQL schema changes)
- **Pytest Config**: `pytest.ini` (test configuration)

## 🔍 Key File Locations

### Backend Structure
```
api/
├── main.py                    # FastAPI application entry point
├── config.py                  # Configuration management
├── database.py               # Database connection and ORM
├── routers/                  # API endpoint routers (40+ modules)
│   ├── claude_auto.py        # Session Continuity API
│   ├── mistake_learning.py   # Mistake Learning API
│   ├── decision_reasoning.py # Decision Reasoning API
│   ├── proactive.py          # Proactive Assistant API
│   ├── code_evolution.py     # Code Evolution API
│   ├── performance_metrics.py # Performance Intelligence API
│   ├── graphrag.py           # GraphRAG with Neo4j
│   ├── llamaindex_rag.py     # LlamaIndex RAG
│   └── ...
├── models/                   # SQLAlchemy ORM models
├── services/                 # Business logic services
├── middleware/               # FastAPI middleware
├── memory_system/           # Memory management system
└── shared/                  # Shared utilities and configs
```

### Frontend Structure
```
frontend/
├── src/
│   ├── main.tsx             # React application entry point
│   ├── components/          # Reusable UI components
│   ├── pages/              # Page components
│   │   ├── Dashboard.tsx    # Main dashboard
│   │   ├── AiIntelligence.tsx # AI Intelligence features
│   │   ├── MemorySystem.tsx # Memory management UI
│   │   └── ...
│   ├── services/           # API client services
│   ├── hooks/              # Custom React hooks
│   └── utils/              # Utility functions
├── package.json            # NPM dependencies and scripts
├── vite.config.ts          # Vite build configuration
└── tsconfig.json          # TypeScript configuration
```

### Database Schemas
```
migrations/
├── 001_initial_schema.sql       # Core tables (memories, sessions, users)
├── 002_learning_system_tables.sql # AI learning system tables
└── 003_timescale_analytics.sql  # TimescaleDB analytics schema
```

## 🤖 AI Integration Patterns

### Claude Code Integration
The system provides seamless integration with Claude Code via shell helpers:
```bash
# Source the helper functions
source ./claude_code_helpers.sh

# Initialize session (automatic context restoration)
claude-init

# Track errors and solutions
claude-error "TypeError" "Added null check" "solution details" true

# Record technical decisions
claude-decide "Use PostgreSQL" "Better ACID support" "MongoDB,Redis" "auth system" 0.9

# Get AI-powered suggestions
claude-suggest "implementing user authentication"

# Performance tracking
claude-track-performance "npm test" 1.5 true
```

### MCP Server Integration
For direct integration with Claude Desktop:
```bash
# MCP server configuration
cd mcp_server
python server.py  # Starts MCP server with 12 AI-enhanced tools

# Available MCP tools:
# - init_session, create_memory, search_memory
# - track_mistake, record_decision, get_suggestions
# - analytics, performance_metrics, session_status
```

### Universal API Integration
For any AI tool via REST API:
```python
import requests

# Example integration for any AI tool
api_base = "http://localhost:3000"

# Start session
session = requests.post(f"{api_base}/api/claude-auto/session/init", 
                       json={"user_id": "developer", "project_path": "/my/project"})

# Track mistake and solution
requests.post(f"{api_base}/api/mistake-learning/mistakes", 
             json={"error_type": "ImportError", "solution": "Fixed import path"})

# Get proactive suggestions
suggestions = requests.get(f"{api_base}/api/proactive/suggestions/developer")
```

## 🚨 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker status
docker-compose ps

# View specific service logs
docker-compose logs api
docker-compose logs postgres

# Restart problematic service
docker-compose restart api
```

**Database connection issues:**
```bash
# Test database connections
python check_database_connection.py

# Reset databases (WARNING: destroys data)
docker-compose down -v
docker-compose up -d
```

**Port conflicts:**
```bash
# Check what's using ports
lsof -i :3000  # API port
lsof -i :3100  # Frontend port
lsof -i :5433  # PostgreSQL port

# Kill conflicting processes
sudo kill -9 <PID>
```

**Memory/Performance issues:**
```bash
# Check Docker resource usage
docker stats

# Monitor API performance
curl http://localhost:3000/metrics

# Check database performance
docker exec -it knowledgehub-postgres-1 pg_stat_activity
```

### Performance Monitoring
- **API Metrics**: Available at `/metrics` endpoint (Prometheus format)
- **Database Health**: Use `check_database_connection.py` script
- **Comprehensive Testing**: Run `comprehensive_functionality_test.py`
- **Log Analysis**: All services log to stdout, viewable via `docker-compose logs`

### Recovery Procedures
- **Database Recovery**: Automated backups in `backups/` directory
- **Service Recovery**: Health checks with automatic restart
- **Data Migration**: Migration scripts in `migrations/` directory
- **Configuration Reset**: Default configs in repository root

## 📚 Additional Resources

### Documentation
- **API Documentation**: http://localhost:3000/docs (Swagger UI)
- **Architecture Docs**: See `README.md` for comprehensive system overview
- **Test Reports**: Generated in `htmlcov/` after running tests with coverage

### Development Tools
- **Database Admin**: Use tools like pgAdmin for PostgreSQL, Neo4j Browser for graph data
- **API Testing**: Swagger UI at `/docs` or tools like Postman/Insomnia
- **Monitoring**: Grafana dashboards (when monitoring stack is enabled)
- **Log Analysis**: Docker logs or external log aggregation tools

This AI-enhanced development platform is designed to make AI coding assistants smarter and more context-aware. The comprehensive test suite, monitoring, and enterprise features ensure production readiness while maintaining developer productivity.