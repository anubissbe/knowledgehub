# KnowledgeHub Installation Guide

This comprehensive guide covers all installation methods for KnowledgeHub, from quick Docker deployment to development setup and production installation.

## Table of Contents

- [Quick Installation (Docker)](#quick-installation-docker)
- [Development Installation](#development-installation)
- [Production Installation](#production-installation)
- [Prerequisites](#prerequisites)
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

```bash
# Copy development environment
cp .env.example .env.dev

# Edit for local development
nano .env.dev
```

Development `.env.dev`:
```bash
# Local database URLs
DATABASE_URL=postgresql://username:password@localhost:5432/knowledgehub
REDIS_URL=redis://localhost:6379/0

# For Weaviate, use Docker
WEAVIATE_URL=http://localhost:8090

# Local MinIO (or use Docker)
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Development settings
DEBUG=true
API_HOST=127.0.0.1
API_PORT=3000

# Frontend development
VITE_API_URL=http://localhost:3000
VITE_WS_URL=ws://localhost:3000
```

### 5. Start Core Services

Start supporting services with Docker:

```bash
# Start only infrastructure services
docker compose up -d postgres redis weaviate-lite minio
```

Or start manually (advanced):

```bash
# Start PostgreSQL
sudo systemctl start postgresql  # Linux
brew services start postgresql@16  # macOS

# Start Redis
sudo systemctl start redis-server  # Linux
brew services start redis  # macOS

# Start Weaviate (Docker recommended)
docker run -d \
  --name weaviate \
  -p 8090:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=text2vec-transformers \
  -e ENABLE_MODULES=text2vec-transformers \
  -e TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080 \
  semitechnologies/weaviate:1.22.4
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

#### Terminal 4: Scheduler (Optional)
```bash
source venv/bin/activate
cd src/scheduler
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

#### Hardware Requirements
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 500GB+ SSD
- **Network**: 10Gbps recommended

#### Software Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Configure firewall
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 3000  # API (if direct access needed)
sudo ufw --force enable
```

### 2. Production Environment

```bash
# Clone repository
git clone https://github.com/anubissbe/knowledgehub.git
cd knowledgehub

# Create production environment
cp .env.example .env.prod
```

Production `.env.prod`:
```bash
# Production database (external recommended)
DATABASE_URL=postgresql://khuser:secure_password@db.example.com:5432/knowledgehub
REDIS_URL=redis://redis.example.com:6379/0

# Production Weaviate (cluster recommended)
WEAVIATE_URL=http://weaviate.example.com:8080

# Production object storage (S3 recommended)
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY=AKIA...
S3_SECRET_KEY=your-secret-key
S3_BUCKET=knowledgehub-prod

# Production API settings
DEBUG=false
API_HOST=0.0.0.0
API_PORT=3000

# Production frontend
VITE_API_URL=https://api.knowledgehub.example.com
VITE_WS_URL=wss://api.knowledgehub.example.com

# Security settings
SECRET_KEY=your-very-long-secret-key
API_RATE_LIMIT=1000
MAX_UPLOAD_SIZE=100MB

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
```

### 3. SSL/TLS Setup

```bash
# Install Certbot
sudo apt install -y certbot

# Get SSL certificates
sudo certbot certonly --standalone -d api.knowledgehub.example.com
sudo certbot certonly --standalone -d knowledgehub.example.com

# Create nginx configuration
sudo tee /etc/nginx/sites-available/knowledgehub << 'EOF'
server {
    listen 80;
    server_name knowledgehub.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name knowledgehub.example.com;
    
    ssl_certificate /etc/letsencrypt/live/knowledgehub.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/knowledgehub.example.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:3101;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/ {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable configuration
sudo ln -s /etc/nginx/sites-available/knowledgehub /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Production Docker Compose

Create production docker-compose override:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    environment:
      - WORKERS=4
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  web-ui:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G

  scraper:
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G

  rag-processor:
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G

  postgres:
    restart: unless-stopped
    environment:
      - POSTGRES_SHARED_PRELOAD_LIBRARIES=pg_stat_statements
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  redis:
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
```

### 5. Start Production Services

```bash
# Start with production configuration
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check service status
docker compose ps

# Monitor logs
docker compose logs -f api
```

### 6. Production Monitoring

```bash
# Install monitoring stack
docker compose -f monitoring/docker-compose.monitoring.yml up -d

# Setup log rotation
sudo tee /etc/logrotate.d/knowledgehub << 'EOF'
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    size=1M
    missingok
    delaycompress
    copytruncate
}
EOF
```

### 7. Backup Configuration

```bash
# Create backup script
sudo tee /usr/local/bin/knowledgehub-backup.sh << 'EOF'
#!/bin/bash
set -e

BACKUP_DIR="/opt/backups/knowledgehub/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Database backup
docker exec knowledgehub-postgres pg_dump -U khuser knowledgehub | gzip > "$BACKUP_DIR/database.sql.gz"

# Redis backup
docker exec knowledgehub-redis redis-cli --rdb - | gzip > "$BACKUP_DIR/redis.rdb.gz"

# Configuration backup
cp -r /opt/knowledgehub/.env.prod "$BACKUP_DIR/"
cp -r /opt/knowledgehub/docker-compose*.yml "$BACKUP_DIR/"

# Clean old backups (keep 30 days)
find /opt/backups/knowledgehub -name "20*" -type d -mtime +30 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
EOF

sudo chmod +x /usr/local/bin/knowledgehub-backup.sh

# Setup daily backup cron
sudo crontab -e
# Add: 0 2 * * * /usr/local/bin/knowledgehub-backup.sh
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
| `WORKERS` | API worker processes | 1 | No |

### Service Configuration

Each service can be configured via environment variables or configuration files. See [Configuration Guide](CONFIGURATION.md) for detailed options.

## Verification

### Health Checks

```bash
# Overall system health
curl http://localhost:3000/health

# Individual service health
curl http://localhost:3000/health/database
curl http://localhost:3000/health/redis
curl http://localhost:3000/health/weaviate

# Expected response
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "services": {
    "api": "operational",
    "database": "operational",
    "redis": "operational",
    "weaviate": "operational"
  }
}
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

### Performance Tests

```bash
# Install Apache Bench
sudo apt install apache2-utils

# Test API performance
ab -n 1000 -c 10 http://localhost:3000/health

# Test search performance
ab -n 100 -c 5 -T 'application/json' -p search_query.json http://localhost:3000/api/v1/search
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

# Check memory usage
free -h

# Check port conflicts
netstat -tulpn | grep :3000
```

#### Database Connection Issues

```bash
# Test database connectivity
docker exec knowledgehub-postgres pg_isready -U khuser

# Check database logs
docker compose logs postgres

# Reset database
docker compose down
docker volume rm knowledgehub_postgres_data
docker compose up -d postgres
```

#### Search Not Working

```bash
# Check Weaviate status
curl http://localhost:8090/v1/meta

# Check embeddings service
curl http://localhost:8100/health

# Verify vector data
curl http://localhost:8090/v1/objects | jq '.objects | length'
```

#### Performance Issues

```bash
# Check resource usage
docker stats

# Check queue depths
docker exec knowledgehub-redis redis-cli llen crawl_jobs:pending

# Monitor API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:3000/health
```

### Recovery Procedures

#### Database Recovery

```bash
# Restore from backup
gunzip -c /opt/backups/knowledgehub/YYYY-MM-DD/database.sql.gz | \
docker exec -i knowledgehub-postgres psql -U khuser knowledgehub
```

#### Service Recovery

```bash
# Restart individual service
docker compose restart service-name

# Full system restart
docker compose down
docker compose up -d

# Reset to clean state
docker compose down -v
docker compose up -d
```

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Search [GitHub Issues](https://github.com/anubissbe/knowledgehub/issues)
3. Create a new issue with:
   - Operating system and version
   - Docker version
   - Error messages and logs
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read the [User Guide](USER_GUIDE.md) to learn how to use KnowledgeHub
2. Review the [Configuration Guide](CONFIGURATION.md) for optimization
3. Set up [Monitoring](MONITORING.md) for production deployments
4. Configure [Security](SECURITY.md) settings for your environment