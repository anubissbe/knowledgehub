# Installation Guide

This comprehensive guide covers all installation methods for KnowledgeHub, from quick Docker deployment to development setup and production installation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Installation (Docker)](#quick-installation-docker)
- [Development Installation](#development-installation)
- [Production Installation](#production-installation)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 20GB free space
- **OS**: Linux, macOS, or Windows 10+

#### Recommended for Production
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD
- **Network**: 1Gbps connection

### Software Dependencies

#### Required
- **Docker**: 24.0+ with Docker Compose
- **Git**: For cloning the repository

#### Optional (for development)
- **Python**: 3.11+
- **Node.js**: 18+ with npm
- **PostgreSQL**: 16+ (if not using Docker)
- **Redis**: 7+ (if not using Docker)

## Quick Installation (Docker)

The fastest way to get KnowledgeHub running is with Docker Compose.

### 1. Clone Repository

```bash
git clone https://github.com/anubissbe/knowledgehub.git
cd knowledgehub
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional for basic setup)
nano .env
```

Basic `.env` configuration:
```bash
# Database URLs (Docker defaults)
DATABASE_URL=postgresql://khuser:khpassword@postgres:5432/knowledgehub
REDIS_URL=redis://redis:6379/0
WEAVIATE_URL=http://weaviate-lite:8080

# MinIO Configuration
S3_ENDPOINT_URL=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=knowledgehub

# API Configuration
DEBUG=false
API_HOST=0.0.0.0
API_PORT=3000

# Frontend Configuration
VITE_API_URL=http://localhost:3000
VITE_WS_URL=ws://localhost:3000
```

### 3. Start Services

```bash
# Start all services in background
docker compose up -d

# Watch logs (optional)
docker compose logs -f
```

### 4. Wait for Initialization

Services take 2-3 minutes to fully initialize:

```bash
# Check service health
curl http://localhost:3000/health

# Expected response:
# {"status":"healthy","timestamp":...,"services":{"api":"operational",...}}
```

### 5. Access Application

- **Web UI**: http://localhost:3101
- **API Documentation**: http://localhost:3000/docs
- **MinIO Console**: http://localhost:9011 (admin/minioadmin)
- **Weaviate Console**: http://localhost:8090

### 6. First Time Setup

1. **Add a Knowledge Source**:
   - Go to Sources page in web UI
   - Click "Add Source"
   - Enter URL (e.g., `https://docs.github.com`)
   - Configure crawling parameters
   - Click "Create Source"

2. **Monitor Crawling**:
   - Go to Jobs page
   - Watch real-time progress
   - Check for completion

3. **Test Search**:
   - Go to Search page
   - Enter a query
   - Verify results appear

## Development Installation

For active development and contribution to KnowledgeHub.

### 1. System Dependencies

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-pip python3.11-venv nodejs npm postgresql-16 redis-server git curl

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 node postgresql@16 redis git docker

# Start services
brew services start postgresql@16
brew services start redis
```

#### Windows
```powershell
# Install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install dependencies
choco install python311 nodejs postgresql16 redis git docker-desktop
```

### 2. Clone and Setup

```bash
# Clone repository
git clone https://github.com/anubissbe/knowledgehub.git
cd knowledgehub

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Database Setup

```bash
# Create PostgreSQL database
createdb knowledgehub

# Apply schema
psql knowledgehub < src/api/database/schema.sql

# Create test database
createdb knowledgehub_test
psql knowledgehub_test < src/api/database/schema.sql
```

### 4. Environment Configuration

Create development environment file with local settings:

```bash
# Copy development environment
cp .env.example .env.dev
```

### 5. Start Core Services

Start supporting services with Docker:

```bash
# Start only infrastructure services
docker compose up -d postgres redis weaviate-lite minio
```

### 6. Frontend Setup

```bash
cd src/web-ui

# Install Node.js dependencies
npm install

# Start development server
npm run dev

# The frontend will be available at http://localhost:5173
```

### 7. Backend Services

Open separate terminals for each service:

#### Terminal 1: API Gateway
```bash
source venv/bin/activate
cd src/api
export $(cat ../../.env.dev | xargs)
uvicorn main:app --reload --port 3000
```

#### Terminal 2: Scraper Worker
```bash
source venv/bin/activate
cd src/scraper
export $(cat ../../.env.dev | xargs)
python main.py
```

#### Terminal 3: RAG Processor
```bash
source venv/bin/activate
cd src/rag_processor
export $(cat ../../.env.dev | xargs)
python main.py
```

### 8. Development Verification

```bash
# Test API
curl http://localhost:3000/health

# Test frontend
curl http://localhost:5173

# Run tests
python -m pytest tests/
cd src/web-ui && npm test
```

## Production Installation

For production deployments with high availability and performance.

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Configure firewall
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw --force enable
```

### 2. SSL/TLS Setup

```bash
# Install Certbot
sudo apt install -y certbot

# Get SSL certificates
sudo certbot certonly --standalone -d api.knowledgehub.example.com
sudo certbot certonly --standalone -d knowledgehub.example.com
```

### 3. Production Environment

Create production configuration with enhanced security and performance settings:

```bash
# Clone repository
git clone https://github.com/anubissbe/knowledgehub.git
cd knowledgehub

# Create production environment
cp .env.example .env.prod
```

Edit `.env.prod` with production values:
- External database connections
- Production S3/object storage
- Secure API keys
- SSL/TLS endpoints

### 4. Start Production Services

```bash
# Start with production configuration
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check service status
docker compose ps

# Monitor logs
docker compose logs -f api
```

### 5. Setup Monitoring

```bash
# Install monitoring stack
docker compose -f monitoring/docker-compose.monitoring.yml up -d

# Access monitoring:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### 6. Backup Configuration

Create automated backup script:

```bash
#!/bin/bash
# Daily backup script
BACKUP_DIR="/opt/backups/knowledgehub/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Database backup
docker exec knowledgehub-postgres pg_dump -U khuser knowledgehub | gzip > "$BACKUP_DIR/database.sql.gz"

# Configuration backup
cp -r /opt/knowledgehub/.env.prod "$BACKUP_DIR/"

# Clean old backups (keep 30 days)
find /opt/backups/knowledgehub -name "20*" -type d -mtime +30 -exec rm -rf {} +
```

## Configuration

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `REDIS_URL` | Redis connection string | - | Yes |
| `WEAVIATE_URL` | Weaviate API endpoint | - | Yes |
| `S3_ENDPOINT_URL` | S3/MinIO endpoint | - | Yes |
| `S3_ACCESS_KEY` | S3/MinIO access key | - | Yes |
| `S3_SECRET_KEY` | S3/MinIO secret key | - | Yes |
| `S3_BUCKET` | S3/MinIO bucket name | knowledgehub | No |
| `DEBUG` | Enable debug mode | false | No |
| `API_HOST` | API bind address | 0.0.0.0 | No |
| `API_PORT` | API port | 3000 | No |
| `LOG_LEVEL` | Logging level | INFO | No |

## Verification

### Health Checks

```bash
# Overall system health
curl http://localhost:3000/health

# Individual service health
curl http://localhost:3000/health/database
curl http://localhost:3000/health/redis
curl http://localhost:3000/health/weaviate
```

### Functional Tests

```bash
# Test source creation
curl -X POST http://localhost:3000/api/v1/sources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Documentation",
    "base_url": "https://docs.example.com",
    "source_type": "web"
  }'

# Test search functionality
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "installation guide",
    "search_type": "hybrid",
    "limit": 5
  }'
```

## Troubleshooting

### Common Issues

#### Services Won't Start

```bash
# Check Docker status
docker ps -a

# Check service logs
docker compose logs service-name

# Check disk space
df -h

# Check port conflicts
netstat -tulpn | grep :3000
```

#### Database Connection Issues

```bash
# Test database connectivity
docker exec knowledgehub-postgres pg_isready -U khuser

# Check database logs
docker compose logs postgres
```

#### Search Not Working

```bash
# Check Weaviate status
curl http://localhost:8090/v1/meta

# Check embeddings service
curl http://localhost:8100/health
```

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](Troubleshooting)
2. Search [GitHub Issues](https://github.com/anubissbe/knowledgehub/issues)
3. Create a new issue with error details

## Next Steps

After successful installation:

1. Read the [User Guide](User-Guide) to learn how to use KnowledgeHub
2. Review the [Configuration Guide](Configuration) for optimization
3. Set up [Monitoring](Monitoring) for production deployments
4. Configure [Security](Security) settings for your environment