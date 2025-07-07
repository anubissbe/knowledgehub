# KnowledgeHub Health Check Configuration

## Overview

This document describes the health check configuration for all KnowledgeHub services. Health checks ensure services are running correctly and enable Docker orchestration features like automatic restarts and load balancing.

## Health Check Strategy

KnowledgeHub uses a **file-based health check strategy** for most services:
- Services create a health file when successfully initialized
- Docker checks for file existence
- Simple, reliable, no external dependencies

## Service Health Check Configurations

### Core Services

#### API Gateway
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```
- HTTP endpoint-based check
- FastAPI provides `/health` endpoint

#### RAG Processor
```yaml
healthcheck:
  test: ["CMD", "/bin/bash", "/tmp/health_check.sh"]
  interval: 30s
  timeout: 10s
  retries: 3
```
- File-based check: `/tmp/health`
- Service creates file during initialization
- Health check script:
  ```bash
  #!/bin/bash
  test -f /tmp/health && echo "healthy" || echo "unhealthy"
  ```

#### MCP Server
```yaml
healthcheck:
  test: ["CMD", "/bin/bash", "/tmp/health_check.sh"]
  interval: 30s
  timeout: 10s
  retries: 3
```
- File-based check: `/tmp/mcp_healthy`
- Created when WebSocket server starts
- Uses fixed image: `knowledgehub-mcp-server:fixed` (includes websockets dependency)

#### AI Service
```yaml
healthcheck:
  test: ["CMD", "python3", "/tmp/health_check.py"]
  interval: 45s
  timeout: 15s
  retries: 5
  start_period: 90s
```
- HTTP endpoint-based check with Python script
- Extended timing for model loading
- Health check script:
  ```python
  #!/usr/bin/env python3
  import requests
  try:
      r = requests.get("http://localhost:8000/health", timeout=10)
      r.raise_for_status()
      print("healthy")
      exit(0)
  except Exception as e:
      print(f"unhealthy: {e}")
      exit(1)
  ```

#### Scraper Services
```yaml
healthcheck:
  test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3014/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```
- HTTP endpoint-based check
- Service provides health endpoint with dependency status

### Data Services

#### PostgreSQL
```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U khuser -d knowledgehub"]
  interval: 10s
  timeout: 5s
  retries: 5
```

#### Redis
```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 10s
  timeout: 5s
  retries: 5
```

#### Weaviate
```yaml
healthcheck:
  test: ["CMD", "wget", "-q", "--spider", "http://localhost:8080/v1/.well-known/ready"]
  interval: 30s
  timeout: 10s
  retries: 3
```

#### MinIO
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/ready"]
  interval: 30s
  timeout: 20s
  retries: 3
```

## Health Check Implementation Guidelines

### 1. File-Based Health Checks

**When to use**: Services that don't have HTTP endpoints

**Implementation**:
```python
# In service initialization
async def initialize(self):
    # ... initialization code ...
    
    # Create health file when ready
    with open('/tmp/health', 'w') as f:
        f.write('healthy')
    
    logger.info("Service initialized and healthy")

# In service cleanup
async def cleanup(self):
    # Remove health file
    try:
        os.remove('/tmp/health')
    except FileNotFoundError:
        pass
```

### 2. HTTP Endpoint Health Checks

**When to use**: Services with web APIs

**Implementation**:
```python
# FastAPI example
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": check_database_connection(),
            "redis": check_redis_connection(),
            "model": check_model_loaded()
        }
    }
```

### 3. Custom Health Check Scripts

**When to use**: Complex health verification logic

**Implementation**:
1. Create script during container startup
2. Make it executable
3. Reference in docker-compose.yml

## Timing Parameters

### Standard Services
- **interval**: 30s - Check frequency
- **timeout**: 10s - Max time for check
- **retries**: 3 - Failures before unhealthy

### Slow-Starting Services (AI, ML)
- **interval**: 45s - Less frequent checks
- **timeout**: 15s - More time for response
- **retries**: 5 - More tolerance
- **start_period**: 90s - Grace period on startup

### Fast Services (Redis, DB)
- **interval**: 10s - Frequent checks
- **timeout**: 5s - Quick response expected
- **retries**: 5 - Quick recovery detection

## Troubleshooting

### Check Health Status
```bash
# Single service
docker inspect <container> --format='{{.State.Health.Status}}'

# All services
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### View Health Logs
```bash
docker inspect <container> --format='{{range .State.Health.Log}}{{.Output}}{{end}}'
```

### Test Health Check Manually
```bash
# File-based
docker exec <container> test -f /tmp/health && echo "healthy"

# HTTP-based
docker exec <container> curl http://localhost:port/health
```

## Best Practices

1. **Keep It Simple**: Prefer file-based checks over complex scripts
2. **Avoid External Tools**: Don't rely on wget/curl in slim containers
3. **Fast Checks**: Health checks should complete quickly
4. **Meaningful Status**: Include service dependencies in health status
5. **Graceful Degradation**: Service can be "healthy" even if some features are degraded
6. **Logging**: Log health check failures for debugging

## Migration from Old Health Checks

If upgrading from older configurations:

1. Stop container: `docker stop <container>`
2. Remove container: `docker rm <container>`
3. Update docker-compose.yml with new health check
4. Recreate container: `docker compose up -d <service>`
5. Verify health: `docker ps`

## References

- [Docker HEALTHCHECK Documentation](https://docs.docker.com/engine/reference/builder/#healthcheck)
- [Health Check Troubleshooting Guide](./HEALTH_CHECK_TROUBLESHOOTING.md)
- [Maintenance Report](../MAINTENANCE_REPORT.md) - See 2025-07-06 update