# KnowledgeHub Administrator Guide

## Table of Contents
1. [System Administration](#system-administration)
2. [User Management](#user-management)
3. [Source Management](#source-management)
4. [Performance Monitoring](#performance-monitoring)
5. [Backup and Recovery](#backup-and-recovery)
6. [Security Configuration](#security-configuration)
7. [Troubleshooting](#troubleshooting)

## System Administration

### Service Management
```bash
# Start all services
./knowledgehub.sh start

# Stop all services
./knowledgehub.sh stop

# Restart services
./knowledgehub.sh restart

# Check service status
./knowledgehub.sh status

# View logs
./knowledgehub.sh logs [service_name]
```

### Health Monitoring
- **System Health**: http://localhost:3000/api/health
- **Service Status**: `docker compose ps`
- **Resource Usage**: `docker stats`
- **Log Monitoring**: `./knowledgehub.sh logs -f`

### Database Administration
```bash
# Connect to PostgreSQL
docker exec -it knowledgehub-postgres psql -U khuser -d knowledgehub

# Common queries
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM knowledge_sources;
SELECT * FROM jobs WHERE status = 'failed';

# Database backup
./knowledgehub.sh backup

# Database cleanup
docker exec knowledgehub-postgres psql -U khuser -d knowledgehub -c "VACUUM ANALYZE;"
```

## User Management

### Authentication Configuration
Currently using basic authentication. To configure:

1. **Environment Variables**:
```bash
export AUTH_ENABLED=true
export AUTH_SECRET_KEY=your-secret-key
export AUTH_TOKEN_EXPIRE_MINUTES=1440
```

2. **API Key Management**:
```bash
# Generate API key for services
curl -X POST http://localhost:3000/api/auth/api-keys \
  -H "Authorization: Bearer admin-token"
```

### Access Control
- **Admin Panel**: http://localhost:3000/admin
- **User Roles**: admin, user, readonly
- **API Access**: Token-based authentication

## Source Management

### Bulk Source Operations
```bash
# Import sources from file
curl -X POST http://localhost:3000/api/v1/sources/import \
  -H "Content-Type: application/json" \
  -d @sources.json

# Export all sources
curl http://localhost:3000/api/v1/sources/export > sources-backup.json

# Batch refresh all sources
curl -X POST http://localhost:3000/api/v1/scheduler/refresh
```

### Source Health Monitoring
```bash
# Check all source statuses
curl http://localhost:3000/api/v1/sources/ | jq '.sources[] | {name: .name, status: .status, last_scraped: .last_scraped_at}'

# Failed sources report
curl http://localhost:3000/api/v1/sources/?status=failed
```

### Crawling Configuration
Edit docker-compose.yml environment variables:
```yaml
environment:
  - CRAWL_DEPTH=3
  - MAX_CONCURRENT_CRAWLS=5
  - CRAWL_TIMEOUT=300
  - RESPECT_ROBOTS_TXT=true
```

## Performance Monitoring

### Key Metrics
- **Response Times**: API latency monitoring
- **Crawl Success Rate**: Percentage of successful crawls
- **Search Performance**: Query response times
- **Resource Usage**: CPU, memory, disk usage

### Monitoring Endpoints
```bash
# System metrics
curl http://localhost:3000/api/v1/analytics/metrics

# Performance statistics
curl http://localhost:3000/api/v1/analytics/performance

# Search analytics
curl http://localhost:3000/api/v1/analytics/search
```

### Grafana Dashboards
Access monitoring at: http://localhost:3030
- System Overview Dashboard
- Application Performance Dashboard
- Search Analytics Dashboard

### Alerts Configuration
Configure alerts for:
- High error rates (>5%)
- Slow response times (>2s)
- Disk space usage (>80%)
- Failed crawl jobs
- Service downtime

## Backup and Recovery

### Automated Backups
Backups run automatically at 2 AM daily:
```bash
# Check backup status
ls -la /opt/projects/knowledgehub/backups/

# Manual backup
./knowledgehub.sh backup

# Restore from backup
./knowledgehub.sh restore backup-2025-01-15.tar.gz
```

### Backup Components
- **Database**: PostgreSQL dump
- **Vector Store**: Weaviate backup
- **Configuration**: Source definitions
- **Documents**: Full-text content
- **Logs**: Application logs

### Disaster Recovery
1. **Full System Restore**:
```bash
# Stop services
./knowledgehub.sh stop

# Restore from backup
./knowledgehub.sh restore full-backup.tar.gz

# Start services
./knowledgehub.sh start
```

2. **Partial Recovery**:
```bash
# Database only
docker exec -i knowledgehub-postgres psql -U khuser -d knowledgehub < db-backup.sql

# Weaviate only
docker exec knowledgehub-weaviate weaviate-restore backup.json
```

## Security Configuration

### HTTPS Configuration
1. **Certificate Setup**:
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Update nginx configuration
# Add SSL configuration to reverse proxy
```

2. **Environment Variables**:
```bash
export HTTPS_ENABLED=true
export SSL_CERT_PATH=/path/to/cert.pem
export SSL_KEY_PATH=/path/to/key.pem
```

### API Security
- **Rate Limiting**: Configured per IP/user
- **CORS Policy**: Restrict cross-origin requests
- **Input Validation**: Sanitize all user inputs
- **SQL Injection Prevention**: Parameterized queries only

### Secret Management
Using HashiCorp Vault for secrets:
```bash
# Store secret
vault kv put secret/knowledgehub/db password=secure-password

# Retrieve secret
vault kv get -field=password secret/knowledgehub/db
```

### Network Security
- **Firewall Rules**: Restrict port access
- **VPN Access**: Optional VPN-only access
- **Internal Networks**: Use Docker networks
- **Proxy Configuration**: Route through reverse proxy

## Troubleshooting

### Common Issues

#### High Memory Usage
**Symptoms**: Slow performance, OOM errors
**Solutions**:
```bash
# Check memory usage
docker stats --no-stream

# Restart memory-intensive services
docker restart knowledgehub-weaviate knowledgehub-rag

# Adjust memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

#### Database Connection Issues
**Symptoms**: Connection refused, timeout errors
**Solutions**:
```bash
# Check PostgreSQL status
docker logs knowledgehub-postgres

# Test connection
docker exec knowledgehub-postgres pg_isready -U khuser

# Restart database
docker restart knowledgehub-postgres
```

#### Search Performance Issues
**Symptoms**: Slow search results, timeouts
**Solutions**:
```bash
# Check Weaviate status
curl http://localhost:8090/v1/.well-known/ready

# Rebuild search index
curl -X POST http://localhost:3000/api/v1/search/reindex

# Optimize database
docker exec knowledgehub-postgres psql -U khuser -d knowledgehub -c "VACUUM ANALYZE;"
```

### Log Analysis
```bash
# Application logs
docker logs knowledgehub-api --tail=100

# Error logs only
docker logs knowledgehub-api 2>&1 | grep ERROR

# Real-time monitoring
docker logs -f knowledgehub-api

# Export logs
docker logs knowledgehub-api > app-logs.txt
```

### Performance Tuning

#### Database Optimization
```sql
-- Index optimization
CREATE INDEX CONCURRENTLY idx_documents_source_id ON documents(source_id);
CREATE INDEX CONCURRENTLY idx_documents_created_at ON documents(created_at);

-- Query performance
EXPLAIN ANALYZE SELECT * FROM documents WHERE source_id = 'uuid';
```

#### Cache Configuration
```bash
# Redis cache tuning
docker exec knowledgehub-redis redis-cli config set maxmemory 1gb
docker exec knowledgehub-redis redis-cli config set maxmemory-policy allkeys-lru
```

#### Resource Limits
```yaml
# docker-compose.yml adjustments
services:
  api:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M
```

### Emergency Procedures

#### Service Recovery
```bash
# Emergency restart all services
docker compose down && docker compose up -d

# Clean restart (removes containers)
docker compose down -v && docker compose up -d

# Nuclear option (rebuilds everything)
./knowledgehub.sh clean && ./knowledgehub.sh start
```

#### Data Recovery
```bash
# Recover from latest backup
./knowledgehub.sh restore latest

# Recover specific service
docker restore knowledgehub-postgres backup.sql

# Manual data export
pg_dump -h localhost -p 5433 -U khuser knowledgehub > emergency-backup.sql
```

### Maintenance Schedules

#### Daily Tasks
- Check service health
- Review error logs
- Monitor disk space
- Verify backup completion

#### Weekly Tasks
- Update dependencies
- Clean up old logs
- Review performance metrics
- Source health audit

#### Monthly Tasks
- Security updates
- Database optimization
- Backup testing
- Capacity planning

---

## Quick Reference

### Essential Commands
```bash
# Service management
./knowledgehub.sh [start|stop|restart|status]

# Health checks
curl http://localhost:3000/api/health

# Logs
docker logs [container-name]

# Database access
docker exec -it knowledgehub-postgres psql -U khuser -d knowledgehub

# Backup/restore
./knowledgehub.sh backup
./knowledgehub.sh restore [backup-file]
```

### Important Files
- Configuration: `docker-compose.yml`
- Environment: `.env`
- Logs: `logs/`
- Backups: `backups/`
- Scripts: `scripts/`

### Support Contacts
- **System Issues**: Check Docker logs
- **Performance**: Monitor Grafana dashboards
- **Security**: Review audit logs
- **Documentation**: See `/docs` directory