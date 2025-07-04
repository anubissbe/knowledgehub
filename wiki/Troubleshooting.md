# Troubleshooting Guide

This guide covers common issues and their solutions. If you don't find your issue here, check the [FAQ](FAQ) or create a [GitHub issue](https://github.com/anubissbe/knowledgehub/issues).

## Table of Contents

- [Installation Issues](#installation-issues)
- [Service Startup Problems](#service-startup-problems)
- [Crawling Issues](#crawling-issues)
- [Search Problems](#search-problems)
- [Performance Issues](#performance-issues)
- [Database Issues](#database-issues)
- [Network & Connectivity](#network--connectivity)
- [Memory & Resource Issues](#memory--resource-issues)
- [Docker Issues](#docker-issues)
- [API Errors](#api-errors)

## Installation Issues

### Docker Compose Fails to Start

**Symptoms:**
```
ERROR: Service 'api' failed to build
```

**Solutions:**

1. **Check Docker version:**
```bash
docker --version  # Should be 24.0+
docker compose version  # Should be 2.0+
```

2. **Clean Docker cache:**
```bash
docker system prune -a
docker compose build --no-cache
```

3. **Check disk space:**
```bash
df -h  # Ensure 20GB+ free
```

### Port Already in Use

**Error:**
```
Error: bind: address already in use
```

**Solution:**
```bash
# Find process using port
sudo lsof -i :3000  # API port
sudo lsof -i :3101  # Web UI port

# Kill process or change ports in docker-compose.yml
```

### Permission Denied Errors

**Error:**
```
Permission denied: '/opt/projects/knowledgehub/data'
```

**Solution:**
```bash
# Fix permissions
sudo chown -R $USER:$USER /opt/projects/knowledgehub
chmod -R 755 /opt/projects/knowledgehub
```

## Service Startup Problems

### Services Keep Restarting

**Check logs:**
```bash
docker compose logs api
docker compose logs postgres
docker compose logs weaviate
```

**Common causes:**

1. **Database not ready:**
```bash
# Wait for postgres
docker compose up -d postgres
sleep 10
docker compose up -d
```

2. **Environment variables missing:**
```bash
# Check .env file exists
cp .env.example .env
# Edit with correct values
nano .env
```

### Weaviate Won't Start

**Error:**
```
TRANSFORMERS_INFERENCE_API not found
```

**Solution:**
```yaml
# In docker-compose.yml, ensure:
weaviate:
  environment:
    - TRANSFORMERS_INFERENCE_API=http://t2v-transformers:8080
```

### Redis Connection Refused

**Check Redis is running:**
```bash
docker compose ps redis
docker compose logs redis

# Test connection
docker exec knowledgehub-redis redis-cli ping
# Should return: PONG
```

## Crawling Issues

### Crawling Stuck at 0%

**Diagnosis:**
```bash
# Check scraper logs
docker compose logs scraper -f

# Check job status via API
curl http://localhost:3000/api/v1/jobs/{job_id}
```

**Common causes:**

1. **Site blocking crawler:**
   - Add user agent to configuration
   - Increase crawl delay
   - Check robots.txt

2. **JavaScript required:**
   - Ensure Playwright is working
   - Check browser installation

### All Pages Being Re-crawled

**Symptoms:** Incremental crawling takes as long as full crawl

**Check content hashes:**
```bash
# Via API
curl "http://localhost:3000/api/v1/documents/?source_id={source_id}&limit=5" | \
  jq '.documents[].content_hash'
```

**Solution:**
1. Ensure content_hash column exists in database
2. Verify scraper is calculating hashes
3. Force a full refresh once

### Crawl Errors: Timeout

**Error:**
```
TimeoutError: Navigation timeout of 30000 ms exceeded
```

**Solutions:**

1. **Increase timeout:**
```json
{
  "config": {
    "timeout": 60000,  // 60 seconds
    "wait_for_load": true
  }
}
```

2. **Skip problematic pages:**
```json
{
  "exclude_patterns": [
    "**/video/**",
    "**/download/**"
  ]
}
```

## Search Problems

### No Search Results

**Diagnosis checklist:**

1. **Check if content was crawled:**
```bash
curl http://localhost:3000/api/v1/sources/{source_id}
# Check document_count > 0
```

2. **Verify RAG processing completed:**
```bash
curl "http://localhost:3000/api/v1/jobs?job_type=rag&status=completed"
```

3. **Check Weaviate is running:**
```bash
curl http://localhost:8090/v1/meta
```

4. **Verify embeddings exist:**
```bash
curl http://localhost:8090/v1/objects | jq '.objects | length'
```

### Poor Search Results

**Improve search quality:**

1. **Use hybrid search:**
```json
{
  "search_type": "hybrid",
  "query": "your search query"
}
```

2. **Be more specific:**
   - Add context to queries
   - Use natural language questions
   - Include relevant keywords

3. **Check embedding service:**
```bash
docker compose logs embeddings-service
```

### Search Timeout

**Error:**
```
504 Gateway Timeout
```

**Solutions:**

1. **Reduce result limit:**
```json
{
  "limit": 10  // Instead of 100
}
```

2. **Optimize Weaviate:**
```bash
# Check Weaviate performance
curl http://localhost:8090/v1/nodes
```

3. **Clear search cache:**
```bash
docker exec knowledgehub-redis redis-cli FLUSHDB
```

## Performance Issues

### Slow System Response

**Performance checklist:**

1. **Check resource usage:**
```bash
docker stats
```

2. **Database performance:**
```sql
-- Connect to PostgreSQL
docker exec -it knowledgehub-postgres psql -U khuser knowledgehub

-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Run maintenance
VACUUM ANALYZE;
```

3. **Redis memory:**
```bash
docker exec knowledgehub-redis redis-cli info memory
```

### High CPU Usage

**Identify culprit:**
```bash
# Check container CPU usage
docker stats --no-stream

# Check specific service
docker compose logs -f [service_name]
```

**Common causes:**
- Embedding generation (normal during processing)
- Large crawl jobs
- Unoptimized search queries

### Slow Crawling

**Optimize crawl speed:**

1. **Reduce delays:**
```json
{
  "crawl_delay": 0.5,  // Faster
  "incremental_delay": 0.2
}
```

2. **Scale workers:**
```bash
docker compose up -d --scale scraper=3
```

3. **Use incremental crawling:**
   - Automatically enabled after first crawl
   - 95%+ faster for updates

## Database Issues

### PostgreSQL Connection Failed

**Error:**
```
FATAL: password authentication failed
```

**Fix:**
1. Check credentials in .env
2. Verify PostgreSQL is running
3. Reset database:
```bash
docker compose down -v
docker compose up -d postgres
# Wait 10 seconds
docker compose up -d
```

### Database Full

**Check disk usage:**
```bash
# Database size
docker exec knowledgehub-postgres psql -U khuser -c "SELECT pg_database_size('knowledgehub')/1024/1024 as size_mb;"

# Clean old data
docker exec knowledgehub-postgres psql -U khuser knowledgehub -c "DELETE FROM scraping_jobs WHERE created_at < NOW() - INTERVAL '30 days';"
```

### Migration Errors

**Reset database schema:**
```bash
# Backup first!
docker exec knowledgehub-postgres pg_dump -U khuser knowledgehub > backup.sql

# Reset
docker compose down
docker volume rm knowledgehub_postgres_data
docker compose up -d
```

## Network & Connectivity

### Cannot Access Web UI

**Troubleshooting steps:**

1. **Check service is running:**
```bash
docker compose ps web-ui
curl http://localhost:3101
```

2. **Check firewall:**
```bash
# Linux
sudo ufw status
sudo ufw allow 3101

# Windows - Check Windows Defender Firewall
```

3. **Try different browser/incognito mode**

### API Connection Refused

**Check API health:**
```bash
curl http://localhost:3000/health

# If fails, check logs
docker compose logs api
```

### WebSocket Connection Failed

**Browser console error:**
```
WebSocket connection to 'ws://localhost:3000/ws' failed
```

**Solutions:**
1. Ensure API is running
2. Check CORS settings in .env
3. Verify WebSocket URL in frontend config

## Memory & Resource Issues

### Out of Memory Errors

**Symptoms:**
```
Container killed due to OOM
```

**Solutions:**

1. **Increase Docker memory:**
   - Docker Desktop: Settings → Resources → Memory

2. **Limit service memory:**
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

3. **Reduce concurrent operations:**
   - Lower worker count
   - Reduce batch sizes

### Disk Space Issues

**Check disk usage:**
```bash
# Overall
df -h

# Docker specific
docker system df
```

**Clean up:**
```bash
# Remove unused images
docker image prune -a

# Clean build cache
docker builder prune

# Remove old logs
docker compose logs --tail=0 > /dev/null
```

## Docker Issues

### Docker Daemon Not Running

**Error:**
```
Cannot connect to the Docker daemon
```

**Fix:**
```bash
# Linux
sudo systemctl start docker

# macOS/Windows
# Start Docker Desktop
```

### Container Name Conflicts

**Error:**
```
Container name "/knowledgehub-api" is already in use
```

**Solution:**
```bash
# Remove old containers
docker compose down
docker rm -f $(docker ps -aq)
docker compose up -d
```

### Build Cache Issues

**Force rebuild:**
```bash
docker compose build --no-cache
docker compose up -d
```

## API Errors

### 401 Unauthorized

**Causes:**
- Missing API key
- Invalid API key
- CORS issues

**Fix:**
1. Check API key configuration
2. Verify headers in requests
3. Update CORS_ORIGINS in .env

### 422 Validation Error

**Common causes:**
- Invalid request body
- Missing required fields
- Incorrect data types

**Debug:**
```bash
# Check API docs
http://localhost:3000/docs

# Validate request format
```

### 500 Internal Server Error

**Diagnosis:**
```bash
# Check API logs
docker compose logs api --tail=100

# Check dependencies
curl http://localhost:3000/health
```

## Getting More Help

If these solutions don't resolve your issue:

1. **Collect diagnostic information:**
```bash
# System info
docker version
docker compose version
uname -a

# Service logs
docker compose logs > diagnostics.log

# Health check
curl http://localhost:3000/health
```

2. **Search existing issues:**
   - [GitHub Issues](https://github.com/anubissbe/knowledgehub/issues)

3. **Create new issue with:**
   - Error messages
   - Steps to reproduce
   - System information
   - Relevant logs

4. **Community support:**
   - [GitHub Discussions](https://github.com/anubissbe/knowledgehub/discussions)
   - Include diagnostic information
   - Be specific about the problem

Remember: Most issues are configuration-related and can be resolved by checking logs and ensuring all services are running correctly.