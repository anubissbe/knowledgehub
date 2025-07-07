# KnowledgeHub Deployment Pipeline

## Overview

This document describes the deployment pipeline for KnowledgeHub, supporting both local development and production deployment.

## Deployment Methods

### 1. Local Deployment (Development/Production on Local Machine)

#### Quick Start
```bash
cd /opt/projects/knowledgehub
./knowledgehub.sh start
```

#### Using the deployment script directly:
```bash
cd /opt/projects/knowledgehub
./deploy/deploy-local.sh
```

This script:
- Checks Docker and disk space
- Creates necessary directories
- Backs up existing data
- Builds all Docker images
- Starts services in correct order
- Runs health checks
- Shows access URLs

#### Control Commands
```bash
./knowledgehub.sh start    # Start all services
./knowledgehub.sh stop     # Stop all services
./knowledgehub.sh restart  # Restart services
./knowledgehub.sh status   # Check status and health
./knowledgehub.sh logs     # View logs
./knowledgehub.sh update   # Update and redeploy
./knowledgehub.sh backup   # Create backup
./knowledgehub.sh clean    # Clean up everything
```

### 2. Full Deployment Script

For a more comprehensive deployment with image building and transfer:

```bash
cd /opt/projects/knowledgehub
./deploy/deploy.sh
```

This script:
- Checks SSH connectivity
- Builds all Docker images locally
- Exports images as tar.gz files
- Transfers images to Synology
- Loads images on Synology
- Restarts all services
- Runs comprehensive health checks

### 3. GitHub Actions CI/CD (Future)

The `.github/workflows/deploy.yml` file provides automated deployment on push to main branch:

1. **Test Stage**: Runs all Python and JavaScript tests
2. **Build Stage**: Builds Docker images
3. **Deploy Stage**: Deploys to Synology NAS
4. **Notify Stage**: Sends deployment status

To enable:
1. Add `SYNOLOGY_SSH_KEY` secret to GitHub repository
2. Ensure Synology NAS is accessible from GitHub Actions
3. Push to main branch to trigger deployment

## Pre-requisites

### Local Machine
- Docker and Docker Compose installed
- SSH access to Synology NAS
- rsync installed

### Synology NAS
- Docker package installed
- SSH enabled (port 2222)
- User with Docker permissions
- Directory: `/volume1/docker/knowledgehub`

## Configuration

### Environment Variables
Create `.env` file with:
```bash
# Database
DATABASE_URL=postgresql://khuser:khpassword@postgres:5432/knowledgehub

# Redis
REDIS_URL=redis://redis:6379/0

# Weaviate
WEAVIATE_URL=http://weaviate:8080

# MinIO
S3_ENDPOINT_URL=http://minio:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# API Keys
API_KEY=your-api-key-here
```

### SSH Configuration
Add to `~/.ssh/config`:
```
Host synology
    HostName 192.168.1.24
    Port 2222
    User Bert
    IdentityFile ~/.ssh/id_rsa
```

## Deployment Process

### 1. Pre-deployment
```bash
# Run tests
pytest
npm test

# Check current status
./scripts/check-health.sh

# Create backup
./scripts/backup.sh
```

### 2. Deploy
```bash
# Simple deployment
./deploy/deploy-simple.sh

# Or full deployment
./deploy/deploy.sh
```

### 3. Post-deployment
```bash
# Verify deployment
./scripts/check-health.sh

# Check logs
ssh synology "cd /volume1/docker/knowledgehub && docker compose logs --tail=50"

# Monitor services
ssh synology "cd /volume1/docker/knowledgehub && docker compose ps"
```

## Rollback Procedure

If deployment fails:

1. **Restore from backup**:
```bash
./scripts/restore.sh
```

2. **Or manually rollback**:
```bash
ssh synology "cd /volume1/docker/knowledgehub && docker compose down"
ssh synology "cd /volume1/docker/knowledgehub && git checkout HEAD~1"
ssh synology "cd /volume1/docker/knowledgehub && docker compose up -d"
```

## Health Checks

The deployment includes automated health checks for:
- PostgreSQL (5433)
- Redis (6381)
- Weaviate (8090)
- MinIO (9010)
- API Gateway (3000)
- MCP Server (3008)
- Scraper (3014)
- Frontend (3100)

## Monitoring

After deployment, monitor services at:
- **Application**: http://192.168.1.24:3100
- **Grafana**: http://192.168.1.24:3030
- **Prometheus**: http://192.168.1.24:9090

## Troubleshooting

### SSH Connection Issues
```bash
# Test SSH
ssh -p 2222 Bert@192.168.1.24 "echo OK"

# Check SSH key
ssh-keygen -t rsa -b 4096 -C "knowledgehub-deploy"
```

### Docker Issues
```bash
# Check Docker on Synology
ssh synology "docker version"

# Clean up old containers
ssh synology "docker system prune -f"
```

### Service Won't Start
```bash
# Check logs
ssh synology "cd /volume1/docker/knowledgehub && docker compose logs <service>"

# Restart individual service
ssh synology "cd /volume1/docker/knowledgehub && docker compose restart <service>"
```

## Security Considerations

1. **SSH Keys**: Use key-based authentication only
2. **Secrets**: Never commit secrets to repository
3. **Network**: Ensure Synology firewall allows required ports
4. **Backups**: Always backup before deployment

## Future Improvements

1. **Blue-Green Deployment**: Zero-downtime deployments
2. **Container Registry**: Use Synology's built-in registry
3. **Kubernetes**: Migrate to k3s for better orchestration
4. **Monitoring**: Add deployment metrics to Grafana