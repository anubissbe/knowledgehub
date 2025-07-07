# KnowledgeHub Log Aggregation System

## Overview

A centralized logging system has been implemented for KnowledgeHub using the Grafana Loki stack, providing:
- Centralized log collection from all services
- Structured JSON logging in production
- Real-time log viewing and searching
- Log retention and rotation
- Grafana dashboards for visualization

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   KnowledgeHub  │     │    Promtail     │     │      Loki       │
│    Services     │────▶│  (Log Shipper)  │────▶│  (Log Storage)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │     Grafana     │
                                                 │ (Visualization) │
                                                 └─────────────────┘
```

### Components

1. **Loki** (Port 3100)
   - Log aggregation and storage system
   - Indexes metadata, stores logs in chunks
   - Provides query API for log retrieval

2. **Promtail** (Port 9080)
   - Log collection agent
   - Discovers Docker containers
   - Parses JSON logs and ships to Loki
   - Applies labels for filtering

3. **Grafana** (Port 3015)
   - Web UI for log exploration
   - Pre-configured dashboards
   - Query interface with LogQL

## Implementation Details

### Service Configuration

All KnowledgeHub services are configured with:
- JSON structured logging in production
- Docker logging driver with size limits
- Service labels for identification

### Log Format

Production logs use JSON format:
```json
{
  "timestamp": "2025-01-05T12:34:56.789Z",
  "level": "INFO",
  "logger": "scraper",
  "message": "Processing job 123",
  "module": "main",
  "function": "process_job",
  "line": 156,
  "service": "scraper",
  "job_id": "123"
}
```

### Log Retention

- Default retention: 30 days
- Log rotation: 10MB per file, max 3 files
- Automatic cleanup of old logs

## Deployment

### Quick Start

```bash
# Deploy the logging stack
./scripts/deploy-logging.sh

# Or manually:
docker compose -f docker-compose.yml -f docker-compose.logging.yml up -d
```

### Access Points

- **Grafana UI**: http://localhost:3015 (admin/admin)
- **Loki API**: http://localhost:3100
- **Promtail Metrics**: http://localhost:9080/metrics

## Usage

### Viewing Logs in Grafana

1. Open Grafana: http://localhost:3015
2. Navigate to the KnowledgeHub Logs Dashboard
3. Use filters to select specific services
4. Search logs using the search box

### Common LogQL Queries

```logql
# All logs from a service
{project="knowledgehub",service="rag-processor"}

# Error logs across all services
{project="knowledgehub"} |= "error"

# JSON parsing with field extraction
{project="knowledgehub"} | json | level="ERROR"

# Logs from last hour with specific text
{project="knowledgehub"} |~ "processing job"

# Rate of logs per service
sum by (service) (rate({project="knowledgehub"}[5m]))
```

### Dashboard Features

The pre-configured dashboard includes:
- Log volume by service (time series)
- Log level distribution (pie chart)
- Service logs viewer with search
- Error logs panel
- Service and search filters

## Configuration Files

### Docker Compose Extension

`docker-compose.logging.yml`:
- Defines Loki, Promtail, and Grafana services
- Configures logging drivers for all services
- Sets up volumes and networks

### Loki Configuration

`config/loki.yaml`:
- Storage configuration (filesystem)
- Retention policies (30 days)
- Query limits and performance tuning

### Promtail Configuration

`config/promtail.yaml`:
- Docker service discovery
- JSON log parsing pipeline
- Label extraction rules
- Timestamp parsing

### Grafana Provisioning

`config/grafana/provisioning/`:
- Automatic datasource configuration
- Pre-built dashboard deployment
- Folder organization

## Monitoring and Maintenance

### Health Checks

```bash
# Check Loki
curl http://localhost:3100/ready

# Check Grafana
curl http://localhost:3015/api/health

# Check Promtail
docker logs knowledgehub-promtail --tail 50
```

### Disk Usage

Monitor log storage:
```bash
du -sh /opt/projects/knowledgehub/logs/*
```

### Troubleshooting

1. **No logs appearing**:
   - Check Promtail is running: `docker ps | grep promtail`
   - Verify Docker socket access
   - Check service labels in docker-compose

2. **Grafana connection issues**:
   - Verify Loki is accessible
   - Check datasource configuration
   - Review Grafana logs

3. **High disk usage**:
   - Adjust retention period in loki.yaml
   - Reduce log verbosity in services
   - Enable log sampling if needed

## Best Practices

1. **Structured Logging**:
   - Use the shared logging module
   - Include relevant context in extra fields
   - Keep messages concise

2. **Log Levels**:
   - DEBUG: Detailed debugging information
   - INFO: General operational messages
   - WARNING: Warning conditions
   - ERROR: Error conditions
   - CRITICAL: Critical failures

3. **Performance**:
   - Avoid logging in tight loops
   - Use appropriate log levels
   - Batch log queries in Grafana

## Integration with Services

All services automatically integrate with the logging system:
- API service logs HTTP requests and responses
- Scraper logs crawl progress and errors
- RAG processor logs chunk processing
- Scheduler logs job execution
- Health checks are logged for monitoring

## Future Enhancements

Potential improvements:
- Alert rules for error patterns
- Log-based metrics and dashboards
- Integration with external monitoring
- Log forwarding to cloud services
- Advanced parsing for specific log types