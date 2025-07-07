# KnowledgeHub Deployment Guide

## Overview

This guide covers the deployment of KnowledgeHub's complete infrastructure including the core application, monitoring, backup systems, and AI services.

## Prerequisites

- Docker and Docker Compose
- 8GB+ RAM (16GB recommended)
- 50GB+ storage space
- GPU support (optional, for AI acceleration)

## Core Services Deployment

### 1. Main Application Stack

```bash
cd /opt/projects/knowledgehub
docker compose up -d
```

**Services deployed:**
- API Gateway (port 3000)
- PostgreSQL (port 5433)
- Redis (port 6381)
- Weaviate (port 8090)
- MinIO (port 9010, 9011)
- RAG Processor (port 3013)
- Scraper Workers (ports 3014, 3015)
- Scheduler (background)

### 2. Monitoring Stack

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

**Services deployed:**
- Prometheus (port 9090)
- Grafana (port 3030) - admin/admin123
- AlertManager (port 9093)
- Node Exporter (port 9100)
- cAdvisor (port 8080)

### 3. AI Services

```bash
docker compose -f docker-compose.ai.yml up -d
```

**Services deployed:**
- AI Service (port 8001)

### 4. Logging Stack (Optional)

```bash
./scripts/deploy-logging.sh
```

**Services deployed:**
- Loki (log storage)
- Promtail (log collection)
- Grafana integration

## Service Health Checks

### Quick Health Check

```bash
# API Gateway
curl http://localhost:3000/health

# AI Service
curl http://localhost:8001/health

# RAG Processor
curl http://localhost:3013/health

# Scraper
curl http://localhost:3014/health
```

### Comprehensive Health Check

```bash
python3 test_health_checks.py
```

## Backup and Disaster Recovery

### Setup Automated Backups

1. **Test manual backup:**
```bash
./scripts/backup.sh
```

2. **Setup automated backups (cron):**
```bash
# Add to crontab
0 2 * * * /opt/projects/knowledgehub/scripts/backup-cron.sh
```

3. **Test restore process:**
```bash
# List available backups
ls backups/manifest_*.json

# Restore from backup
./scripts/restore.sh 20250705_143000
```

## Monitoring and Alerting

### Access Dashboards

- **Grafana**: http://localhost:3030 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

### Key Metrics to Monitor

1. **Application Metrics:**
   - API response times
   - Document processing rate
   - Chunk creation rate
   - Search query performance

2. **System Metrics:**
   - CPU usage
   - Memory usage
   - Disk space
   - Network I/O

3. **Database Metrics:**
   - PostgreSQL connections
   - Query performance
   - Redis memory usage
   - Weaviate vector operations

## Production Optimization

### Performance Tuning

1. **Database Optimization:**
```sql
-- Optimize PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '256MB';
SELECT pg_reload_conf();
```

2. **Redis Optimization:**
```bash
# Increase Redis memory limit
echo "maxmemory 4gb" >> config/redis.conf
echo "maxmemory-policy allkeys-lru" >> config/redis.conf
```

3. **GPU Acceleration:**
```bash
# Verify GPU availability
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Scaling Considerations

1. **Horizontal Scaling:**
   - Deploy multiple scraper workers
   - Load balance API requests
   - Shard Weaviate collections

2. **Vertical Scaling:**
   - Increase container memory limits
   - Optimize batch sizes
   - Tune connection pools

## Security Hardening

### 1. Network Security

```bash
# Configure firewall rules
ufw allow 3000/tcp  # API Gateway
ufw allow 3030/tcp  # Grafana
ufw allow 8001/tcp  # AI Service
```

### 2. Authentication

```bash
# Change default passwords
export GRAFANA_PASSWORD="your-secure-password"
export POSTGRES_PASSWORD="your-secure-password"
```

### 3. SSL/TLS Setup

```nginx
# Nginx reverse proxy with SSL
server {
    listen 443 ssl;
    server_name knowledgehub.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

1. **Services not starting:**
```bash
# Check container logs
docker logs knowledgehub-api
docker logs knowledgehub-rag
docker logs knowledgehub-scraper

# Check resource usage
docker stats
```

2. **Database connection issues:**
```bash
# Test PostgreSQL connection
docker exec -it knowledgehub-postgres psql -U postgres -d knowledgehub

# Check Redis connection
docker exec -it knowledgehub-redis redis-cli ping
```

3. **RAG processing issues:**
```bash
# Check RAG processor logs
docker logs knowledgehub-rag --tail=50

# Monitor processing queue
redis-cli llen rag_processing:normal
```

### Performance Issues

1. **Slow API responses:**
   - Check database query performance
   - Monitor Redis cache hit rates
   - Verify Weaviate vector search performance

2. **High memory usage:**
   - Tune embedding model batch sizes
   - Optimize chunk processing
   - Configure garbage collection

### Recovery Procedures

1. **Service Recovery:**
```bash
# Restart individual service
docker compose restart api

# Full stack restart
docker compose down && docker compose up -d
```

2. **Data Recovery:**
```bash
# Restore from backup
./scripts/restore.sh <backup_id>

# Verify data integrity
curl http://localhost:3000/api/v1/stats
```

## API Usage Examples

### AI Service Examples

```bash
# Threat analysis
curl -X POST http://localhost:8001/api/ai/analyze-threats \
  -H "Content-Type: application/json" \
  -d '{"content": "SELECT * FROM users WHERE id = 1 OR 1=1"}'

# Content similarity
curl -X POST http://localhost:8001/api/ai/content-similarity \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication security", "limit": 5}'

# Risk scoring
curl -X POST http://localhost:8001/api/ai/risk-scoring \
  -H "Content-Type: application/json" \
  -d '{
    "components": ["web_frontend", "api_gateway", "database"],
    "data_flows": [{"description": "user data to database"}],
    "environment": "production"
  }'
```

## Maintenance

### Regular Tasks

1. **Weekly:**
   - Review backup logs
   - Check disk space usage
   - Monitor error rates

2. **Monthly:**
   - Update container images
   - Review security logs
   - Optimize database performance

3. **Quarterly:**
   - Security audit
   - Performance review
   - Disaster recovery testing

### Updates and Upgrades

```bash
# Update containers
docker compose pull
docker compose up -d

# Update monitoring stack
docker compose -f docker-compose.monitoring.yml pull
docker compose -f docker-compose.monitoring.yml up -d
```

## Support and Documentation

- **Health Endpoints**: All services provide `/health` endpoints
- **Logs**: Centralized logging via Loki/Grafana
- **Metrics**: Prometheus metrics for all services
- **Backups**: Automated with 30-day retention
- **Documentation**: Auto-generated API docs at `/docs`

For additional support, check service logs and monitoring dashboards.