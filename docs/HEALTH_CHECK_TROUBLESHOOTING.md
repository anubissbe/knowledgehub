# Docker Health Check Troubleshooting Guide

## Overview

This guide provides solutions for common Docker health check issues in the KnowledgeHub system. Health checks are crucial for monitoring service availability and enabling proper orchestration.

## Common Health Check Issues

### 1. Container Shows "Unhealthy" Despite Service Running

**Symptoms**:
- Service responds correctly to API calls
- Docker shows container as "unhealthy"
- `docker ps` shows unhealthy status

**Common Causes**:
- Outdated health check configuration in running container
- Missing dependencies for health check command
- Health check timeout too short
- Service not ready when health check runs

**Diagnosis**:
```bash
# Check current health status
docker inspect <container-name> --format='{{.State.Health.Status}}'

# View health check command
docker inspect <container-name> --format='{{.Config.Healthcheck.Test}}'

# View health check logs
docker inspect <container-name> --format='{{range .State.Health.Log}}{{.Output}}{{end}}'

# Test service manually
curl http://localhost:<port>/health
```

### 2. Health Check Command Not Found

**Error**: `executable file not found in $PATH`

**Common Commands That Fail**:
- `wget`: Not installed in slim Python images
- `curl`: Not installed in slim Python images
- Custom scripts: Not in PATH or not executable

**Solutions**:

#### Option 1: Use Built-in Commands
```yaml
# File-based health check (if service creates a health file)
healthcheck:
  test: ["CMD", "test", "-f", "/tmp/health"]
  
# Python-based health check
healthcheck:
  test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health').raise_for_status()"]
```

#### Option 2: Create Health Check Scripts
```bash
# Create script inside container
docker exec <container> bash -c 'echo "#!/bin/bash
test -f /tmp/health && echo healthy || echo unhealthy" > /tmp/health_check.sh && chmod +x /tmp/health_check.sh'

# Update docker-compose.yml
healthcheck:
  test: ["CMD", "/bin/bash", "/tmp/health_check.sh"]
```

### 3. Intermittent Health Check Failures

**Symptoms**:
- Health status alternates between healthy/unhealthy
- Service works but occasionally marked unhealthy

**Solutions**:

1. **Increase Timing Parameters**:
```yaml
healthcheck:
  test: ["CMD", "your-health-check-command"]
  interval: 45s      # How often to check (increase from 30s)
  timeout: 15s       # Max time for check (increase from 10s)
  retries: 5         # Failures before unhealthy (increase from 3)
  start_period: 90s  # Grace period on startup (increase from 60s)
```

2. **Add Error Handling to Health Check**:
```python
#!/usr/bin/env python3
import requests
import time

max_retries = 3
for i in range(max_retries):
    try:
        r = requests.get("http://localhost:8000/health", timeout=10)
        r.raise_for_status()
        print("healthy")
        exit(0)
    except Exception as e:
        if i < max_retries - 1:
            time.sleep(2)
            continue
        print(f"unhealthy: {e}")
        exit(1)
```

### 4. Container Exits Before Health Check

**Symptoms**:
- Container exits immediately
- Health check never runs
- Missing dependencies error in logs

**Solution**:

1. **Fix Missing Dependencies**:
```bash
# Create temporary container
docker run -d --name temp-fix <image> sleep 3600

# Install missing dependencies
docker exec temp-fix pip install websockets requests

# Commit as new image
docker commit temp-fix <image>:fixed

# Update docker-compose.yml to use fixed image
```

2. **Add Dependencies to Dockerfile**:
```dockerfile
RUN pip install --no-cache-dir websockets requests
```

## KnowledgeHub Specific Health Checks

### RAG Processor
- Creates `/tmp/health` file when initialized
- File-based health check recommended
- Remove file on cleanup

### MCP Server
- Creates `/tmp/mcp_healthy` file when WebSocket server starts
- Requires `websockets` Python package
- File-based health check recommended

### AI Service
- Provides HTTP endpoint at `/health`
- Loads AI models on startup (can be slow)
- Requires longer start_period (90s+)

## Best Practices

1. **Use Simple Health Checks**:
   - Prefer file-based checks over network requests
   - Avoid external dependencies (wget, curl)
   - Use built-in Python or shell commands

2. **Appropriate Timing**:
   - Set start_period for services with slow initialization
   - Increase timeout for services under load
   - Balance interval with monitoring needs

3. **Consistent Patterns**:
   - Use similar health check approaches across services
   - Document health check behavior in service code
   - Test health checks before deployment

4. **Debugging Steps**:
   ```bash
   # 1. Check container logs
   docker logs <container> --tail=50
   
   # 2. Test health check manually
   docker exec <container> <health-check-command>
   
   # 3. Verify service is actually running
   curl http://localhost:<port>/health
   
   # 4. Check health check history
   docker inspect <container> --format='{{json .State.Health}}' | jq
   ```

## Emergency Fixes

### Quick Fix Script
```bash
#!/bin/bash
# Save as fix-health-checks.sh

CONTAINER=$1
HEALTH_FILE=$2

# Create simple health check script
docker exec $CONTAINER bash -c "echo '#!/bin/bash
test -f $HEALTH_FILE && echo healthy || echo unhealthy' > /tmp/health_check.sh && chmod +x /tmp/health_check.sh"

# Test it
docker exec $CONTAINER /tmp/health_check.sh
```

### Force Container Healthy (Development Only)
```bash
# WARNING: Only for development/debugging
docker exec <container> touch /tmp/health
```

## Monitoring Health Status

### Check All Services
```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(healthy|unhealthy|starting)"
```

### Watch Health Status
```bash
watch -n 5 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep knowledgehub'
```

### Health Check Dashboard
```bash
#!/bin/bash
echo "=== KnowledgeHub Health Status ==="
for container in $(docker ps --format "{{.Names}}" | grep knowledgehub); do
  status=$(docker inspect $container --format='{{.State.Health.Status}}' 2>/dev/null || echo "no health check")
  printf "%-30s %s\n" "$container:" "$status"
done
```

## References

- [Docker HEALTHCHECK Documentation](https://docs.docker.com/engine/reference/builder/#healthcheck)
- [Docker Compose Health Check](https://docs.docker.com/compose/compose-file/compose-file-v3/#healthcheck)
- KnowledgeHub MAINTENANCE_REPORT.md for specific fixes applied