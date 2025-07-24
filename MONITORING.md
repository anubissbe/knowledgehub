# KnowledgeHub Production Monitoring System

## Overview

The KnowledgeHub Production Monitoring System provides comprehensive observability, alerting, and automated recovery capabilities for production deployments. It consists of several integrated components working together to ensure system reliability and performance.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────│   Prometheus    │────│    Grafana      │
│   (Metrics)     │    │   (Collection)  │    │  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  AlertManager   │              │
         │              │  (Alerting)     │              │
         │              └─────────────────┘              │
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│ Alert Service   │──────────────┘
                        │ (Processing)    │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   Notifications │
                        │ (Email/Slack/   │
                        │  Webhooks)      │
                        └─────────────────┘
```

## Components

### 1. Prometheus Metrics Collection

**Location**: `api/services/prometheus_metrics.py`

Collects comprehensive metrics including:
- **System Metrics**: CPU, memory, disk, network usage
- **Application Metrics**: HTTP requests, response times, error rates
- **Business Metrics**: Memory operations, AI processing, user sessions
- **Health Metrics**: Service status, uptime, response times

#### Key Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `knowledgehub_http_requests_total` | Counter | Total HTTP requests by method/endpoint/status |
| `knowledgehub_http_request_duration_seconds` | Histogram | Request duration distribution |
| `knowledgehub_service_up` | Gauge | Service health status (1=up, 0=down) |
| `knowledgehub_memory_search_duration_seconds` | Histogram | Memory search performance |
| `knowledgehub_ai_processing_duration_seconds` | Histogram | AI operation duration |
| `knowledgehub_active_users` | Gauge | Current active users |
| `knowledgehub_websocket_connections_active` | Gauge | Active WebSocket connections |
| `system_cpu_usage_percent` | Gauge | CPU usage percentage |
| `system_memory_usage_percent` | Gauge | Memory usage percentage |

#### Usage

The metrics are automatically collected via middleware and exposed at `/metrics` endpoint.

```python
from api.services.prometheus_metrics import prometheus_metrics

# Record custom metrics
prometheus_metrics.record_http_request("GET", "/api/search", 200, 0.15)
prometheus_metrics.record_memory_operation("search", "user123", 0.042)
prometheus_metrics.record_ai_operation("embedding", "codebert", 1.2, True)
```

### 2. Health Monitoring Service

**Location**: `api/services/real_health_monitor.py`

Provides comprehensive health checking for all services:
- **Database Health**: Connection pool, query performance
- **Redis Health**: Connectivity, hit ratios, memory usage
- **Weaviate Health**: Vector database status
- **API Health**: Response times, error rates
- **System Health**: Resource utilization

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health/status` | GET | Overall system health |
| `/api/health/services/{name}` | GET | Specific service health |
| `/api/health/metrics/realtime` | GET | Real-time metrics |
| `/api/health/alerts` | GET | Active alerts |
| `/api/health/uptime/{service}` | GET | Service uptime statistics |

### 3. Alert Management System

**Location**: `api/services/alert_service.py`

Processes alerts from Prometheus AlertManager and coordinates notifications and recovery actions.

#### Features

- **Alert Processing**: Webhook integration with AlertManager
- **Escalation Policies**: Automatic escalation based on severity and time
- **Multi-Channel Notifications**: Email, Slack, webhooks, Teams
- **Alert Correlation**: Deduplication and grouping
- **Recovery Actions**: Automated remediation triggers
- **Statistics Tracking**: Comprehensive alerting metrics

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/alerts/webhook` | POST | AlertManager webhook receiver |
| `/api/alerts/active` | GET | Current active alerts |
| `/api/alerts/acknowledge` | POST | Acknowledge alerts |
| `/api/alerts/resolve` | POST | Manually resolve alerts |
| `/api/alerts/statistics` | GET | Alert processing statistics |

### 4. Grafana Dashboards

**Location**: `grafana/dashboards/`

Pre-configured dashboards for comprehensive visualization:

#### KnowledgeHub Overview Dashboard
- System resource utilization
- Service health status
- Request rates and response times
- Active connections and users
- Error rates by component

#### AI Performance Dashboard
- AI processing duration and throughput
- Memory search performance metrics
- Cache hit ratios and performance
- Performance scores by component

### 5. AlertManager Configuration

**Location**: `alertmanager/alertmanager.yml`

Routing and notification configuration:
- **Critical Alerts**: Immediate email + Slack + webhooks
- **Service Down**: Direct to operations team
- **Performance Issues**: Development team notifications
- **System Resources**: System administration alerts

## Deployment

### Quick Start

1. **Start Monitoring Stack**:
   ```bash
   cd /opt/projects/knowledgehub
   ./scripts/start-monitoring.sh
   ```

2. **Access Dashboards**:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3030 (admin/admin123)
   - AlertManager: http://localhost:9093

3. **Integrate with Application**:
   The monitoring is automatically integrated when the KnowledgeHub API starts.

### Docker Compose Services

The monitoring stack includes:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notifications
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics
- **PostgreSQL Exporter**: Database metrics
- **Redis Exporter**: Redis metrics

### Configuration Files

| File | Purpose |
|------|---------|
| `prometheus/prometheus.yml` | Prometheus scrape configuration |
| `prometheus/rules/knowledgehub-alerts.yml` | Alert rules |
| `grafana/provisioning/` | Dashboard and datasource provisioning |
| `alertmanager/alertmanager.yml` | Alert routing and notifications |

## Alert Rules

### Performance Alerts

- **MemorySearchTooSlow**: Memory search >50ms (95th percentile)
- **HighResponseTime**: API response time >1s
- **AIProcessingTooSlow**: AI operations >10s

### System Alerts

- **ServiceDown**: Service health check failures
- **HighCPUUsage**: CPU usage >80%
- **HighMemoryUsage**: Memory usage >85%
- **LowDiskSpace**: Disk usage >90%

### Application Alerts

- **HighErrorRate**: Error rate >10%
- **DatabaseConnectionsHigh**: >50 active connections
- **WebSocketConnectionsDrop**: Rapid connection loss

## Performance Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Memory Retrieval | <50ms | >50ms (95th percentile) |
| Pattern Matching | <100ms | >100ms (95th percentile) |
| API Response Time | <200ms | >1s (95th percentile) |
| Session Restoration | <1s | >2s |
| System Uptime | >99.9% | <99.5% |

## Custom Metrics Integration

### Adding New Metrics

1. **Define the Metric**:
   ```python
   # In prometheus_metrics.py
   self.custom_metric = Counter(
       'knowledgehub_custom_operations_total',
       'Description of custom metric',
       ['label1', 'label2'],
       registry=self.registry
   )
   ```

2. **Record Metric Data**:
   ```python
   prometheus_metrics.custom_metric.labels(
       label1="value1",
       label2="value2"
   ).inc()
   ```

3. **Create Alert Rule**:
   ```yaml
   # In prometheus/rules/knowledgehub-alerts.yml
   - alert: CustomMetricHigh
     expr: rate(knowledgehub_custom_operations_total[5m]) > 100
     for: 2m
     labels:
       severity: warning
     annotations:
       summary: "High custom operation rate"
   ```

### Custom Recovery Actions

```python
# Register recovery handler
await real_alert_service.add_recovery_handler(
    "ServiceDown",
    custom_recovery_function
)

async def custom_recovery_function(alert, action):
    """Custom recovery logic"""
    if action == "restart_service":
        # Implement service restart logic
        await restart_service(alert.service_name)
```

## Troubleshooting

### Common Issues

1. **Metrics Not Appearing**:
   - Check Prometheus targets: http://localhost:9090/targets
   - Verify application metrics endpoint: http://localhost:3000/metrics
   - Check logs for metric collection errors

2. **Alerts Not Firing**:
   - Verify alert rules in Prometheus: http://localhost:9090/rules
   - Check AlertManager status: http://localhost:9093
   - Review alert rule syntax and thresholds

3. **Dashboard Issues**:
   - Verify Grafana datasource configuration
   - Check dashboard import status
   - Confirm metric names and labels

### Logs and Debugging

```bash
# View monitoring service logs
docker-compose -f docker-compose.monitoring.yml logs -f prometheus
docker-compose -f docker-compose.monitoring.yml logs -f grafana
docker-compose -f docker-compose.monitoring.yml logs -f alertmanager

# Check application monitoring logs
curl http://localhost:3000/api/health/status
curl http://localhost:3000/api/alerts/statistics
```

## Security Considerations

1. **Authentication**: Grafana and AlertManager should use authentication in production
2. **Network Security**: Restrict access to monitoring ports
3. **Data Retention**: Configure appropriate data retention policies
4. **Sensitive Data**: Avoid exposing sensitive information in metric labels
5. **Webhook Security**: Use authentication tokens for webhook endpoints

## Maintenance

### Regular Tasks

1. **Update Dashboards**: Review and update dashboards monthly
2. **Alert Tuning**: Adjust thresholds based on performance data
3. **Data Cleanup**: Monitor storage usage and retention
4. **Security Updates**: Keep monitoring components updated

### Backup and Recovery

1. **Grafana Dashboards**: Export dashboard JSON configurations
2. **Prometheus Configuration**: Backup rules and configuration files
3. **Alert History**: Archive important alert data
4. **Metrics Data**: Configure Prometheus long-term storage if needed

## Integration with CI/CD

The monitoring system can be integrated with CI/CD pipelines for:
- **Performance Regression Testing**: Alert on performance degradation
- **Deployment Health Checks**: Verify service health after deployments
- **Automated Rollbacks**: Trigger rollbacks on critical alerts
- **Capacity Planning**: Monitor resource usage trends

This comprehensive monitoring system ensures the KnowledgeHub platform maintains high availability, performance, and reliability in production environments.