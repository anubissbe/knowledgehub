# KnowledgeHub Health Checks

## Overview

Health check endpoints have been added to the RAG Processor and Scraper services to enable monitoring and ensure service availability.

## Implementation Details

### RAG Processor
- **Port**: 3013
- **Endpoint**: `http://192.168.1.24:3013/health`
- **Dependencies Checked**:
  - Redis connectivity
  - Weaviate connectivity
  - API client status

### Scraper Service
- **Port**: 3014
- **Endpoint**: `http://192.168.1.24:3014/health`
- **Dependencies Checked**:
  - Redis connectivity
  - API client status
  - Crawler status

## Response Format

All health check endpoints return JSON with the following structure:

```json
{
  "status": "healthy|unhealthy|degraded",
  "service": "service-name",
  "dependencies": {
    "redis": "healthy|unhealthy|not_initialized",
    "weaviate": "healthy|unhealthy|not_initialized",
    "api": "healthy|unhealthy|not_initialized"
  },
  "processing": {
    "running": true|false,
    "rate_limit_requests_per_minute": 500,
    "batch_size": 50
  },
  "timestamp": "2025-01-05T12:34:56.789Z"
}
```

### Status Codes
- **200 OK**: Service is healthy or degraded but operational
- **503 Service Unavailable**: Service is unhealthy

### Status Values
- **healthy**: All dependencies are operational
- **degraded**: Some dependencies are not initialized but service can operate
- **unhealthy**: Critical dependencies have failed

## Docker Configuration

The `docker-compose.yml` has been updated to:
1. Expose health check ports (3013 for RAG, 3014 for Scraper)
2. Add Docker health check configurations
3. Set `HEALTH_CHECK_PORT` environment variables

## Monitoring Integration

The health check endpoints are integrated with the infrastructure monitoring script:
- `/opt/projects/mcp-infrastructure-docs/scripts/check-all-servers.sh`

## Testing

To test the health checks locally:

```bash
# Test RAG Processor
curl http://localhost:3013/health

# Test Scraper
curl http://localhost:3014/health

# Run the test script
python /opt/projects/knowledgehub/test_health_checks.py
```

## Deployment

1. Build the updated services:
   ```bash
   cd /opt/projects/knowledgehub
   docker compose build rag-processor scraper
   ```

2. Deploy the services:
   ```bash
   docker compose up -d rag-processor scraper
   ```

3. Verify health checks:
   ```bash
   /opt/projects/mcp-infrastructure-docs/scripts/check-all-servers.sh
   ```

## Benefits

1. **Monitoring**: Easy integration with monitoring tools
2. **Debugging**: Quick visibility into service dependencies
3. **Load Balancing**: Health checks can be used by load balancers
4. **Docker Integration**: Native Docker health check support
5. **Automated Recovery**: Services can be automatically restarted if unhealthy