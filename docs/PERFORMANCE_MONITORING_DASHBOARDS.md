# KnowledgeHub Performance Monitoring Dashboards

**Date**: 2025-07-06  
**Status**: ✅ COMPLETED

## Overview

Created comprehensive Grafana dashboards for monitoring KnowledgeHub performance, resource usage, and system health. The dashboards provide real-time visibility into all aspects of the system.

## Dashboards Created

### 1. KnowledgeHub Overview Dashboard
**File**: `/opt/projects/knowledgehub/dashboards/knowledgehub-overview.json`

**Panels**:
- **Service Health Status**: Shows up/down status of all services
- **Container CPU Usage**: Real-time CPU usage per container
- **Container Memory Usage**: Memory consumption trends
- **Network I/O**: Inbound/outbound traffic per service
- **Disk I/O**: Read/write operations per service
- **Container Restart Count**: Stability indicator
- **System Load Average**: Overall system load
- **Disk Usage**: Root filesystem usage percentage
- **Total Memory Usage**: System-wide memory utilization

### 2. Database Performance Dashboard
**File**: `/opt/projects/knowledgehub/dashboards/knowledgehub-database.json`

**Panels**:
- **PostgreSQL Connections**: Active connection monitoring
- **Redis Operations**: Operations per second
- **Weaviate Vector DB Memory**: Memory usage for vector storage
- **MinIO Storage I/O**: Object storage read/write rates
- **Database Container CPU**: CPU usage across all data stores

### 3. Processing Pipeline Dashboard
**File**: `/opt/projects/knowledgehub/dashboards/knowledgehub-pipeline.json`

**Panels**:
- **Scraper Activity**: Crawl rate and network usage
- **RAG Processor Activity**: CPU usage during processing
- **API Response Times**: 95th percentile and median
- **Processing Queue Depth**: Redis queue size
- **Service Memory Usage**: Memory trends for processing services

## Access Information

- **Grafana URL**: http://localhost:3030 (or http://192.168.1.24:3030)
- **Username**: admin
- **Password**: admin123

## Installation

### Automatic Import
```bash
/opt/projects/knowledgehub/scripts/import-grafana-dashboards.sh
```

### Manual Import
1. Navigate to Grafana UI
2. Go to Dashboards → Import
3. Upload JSON files from `/opt/projects/knowledgehub/dashboards/`
4. Select Prometheus as the data source

## Metrics Sources

### Container Metrics (via cAdvisor)
- CPU usage: `container_cpu_usage_seconds_total`
- Memory usage: `container_memory_usage_bytes`
- Network I/O: `container_network_*_bytes_total`
- Disk I/O: `container_fs_*_bytes_total`

### System Metrics (via Node Exporter)
- Load average: `node_load1`, `node_load5`, `node_load15`
- Memory: `node_memory_*`
- Disk usage: `node_filesystem_*`
- CPU: `node_cpu_seconds_total`

### Service Health (via Prometheus)
- Service up/down: `up{job=~"knowledgehub.*"}`

## Key Performance Indicators

### System Health
- **All services UP**: Green status across all services
- **CPU < 80%**: Healthy CPU utilization
- **Memory < 90%**: Adequate memory headroom
- **Disk < 85%**: Sufficient disk space
- **Load < CPU count**: System not overloaded

### Performance Targets
- **API Response**: < 200ms (95th percentile)
- **Scraper Rate**: > 10 pages/minute
- **RAG Processing**: < 5s per document
- **Queue Depth**: < 1000 items

## Alerting Rules

While not implemented in this phase, recommended alerts:

1. **Service Down**: Any service down for > 5 minutes
2. **High CPU**: CPU > 90% for > 10 minutes
3. **Memory Pressure**: Memory > 95% for > 5 minutes
4. **Disk Full**: Disk > 90%
5. **Processing Backlog**: Queue > 5000 items

## Customization

### Adding New Panels
1. Edit dashboard JSON file
2. Add new panel configuration
3. Re-import dashboard

### Modifying Queries
- Update `expr` field in panel targets
- Adjust `legendFormat` for display names
- Change aggregation functions as needed

### Creating Custom Dashboards
```json
{
  "dashboard": {
    "title": "Your Dashboard Name",
    "panels": [...],
    "refresh": "30s",
    "tags": ["knowledgehub", "custom"]
  }
}
```

## Troubleshooting

### No Data Showing
1. Check Prometheus targets: http://localhost:9090/targets
2. Verify service names match: `docker ps --format "{{.Names}}"`
3. Check time range in Grafana (top right)

### Metrics Missing
1. Some services don't expose metrics endpoints yet
2. Use container metrics from cAdvisor instead
3. Consider adding application metrics in future

### Dashboard Import Fails
1. Check Grafana is running: `docker ps | grep grafana`
2. Verify credentials are correct
3. Ensure Prometheus data source exists

## Future Enhancements

1. **Application Metrics**: Add custom metrics to services
   - Request count, latency, error rate
   - Business metrics (documents processed, chunks created)
   
2. **Log Integration**: Link to Loki for log correlation

3. **Alerting**: Configure alert rules and notification channels

4. **SLO Tracking**: Service Level Objective dashboards

5. **Cost Metrics**: Resource usage cost estimation

## Best Practices

1. **Regular Review**: Check dashboards daily
2. **Baseline Establishment**: Note normal performance patterns
3. **Proactive Monitoring**: Address issues before they escalate
4. **Dashboard Evolution**: Update as system grows
5. **Team Access**: Share dashboard URLs with team

## Quick Links

- **Overview**: http://localhost:3030/d/knowledgehub-overview
- **Database**: http://localhost:3030/d/knowledgehub-database  
- **Pipeline**: http://localhost:3030/d/knowledgehub-pipeline
- **Prometheus**: http://localhost:9090
- **cAdvisor**: http://localhost:8081