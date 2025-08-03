# KnowledgeHub Service Recovery & Self-Healing System

## Overview

The KnowledgeHub Service Recovery System provides automated service monitoring, failure detection, and self-healing capabilities for production resilience. Built with exponential backoff strategies, dependency awareness, and comprehensive logging, it ensures maximum uptime and automatic recovery from common failure scenarios.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────│ Service Recovery │────│   Monitoring    │
│   Services      │    │    Manager       │    │   Dashboard     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │ Health Checks   │              │
         │              │ & Detection     │              │
         │              └─────────────────┘              │
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│ Recovery Actions│──────────────┘
                        │ & Strategies    │
                        └─────────────────┘
```

## Key Components

### 1. Service Recovery Manager

**Location**: `api/services/service_recovery.py`

Provides comprehensive automated recovery capabilities:

#### Core Features
- **Health Monitoring**: Continuous health checks with configurable intervals
- **Failure Detection**: Smart failure thresholds and consecutive failure tracking
- **Recovery Strategies**: Multiple recovery actions with exponential backoff
- **Dependency Management**: Dependency-aware recovery ordering
- **Statistics Tracking**: Recovery success rates and performance metrics

#### Service States
- **HEALTHY**: Service operating normally
- **DEGRADED**: Service experiencing issues but still functional
- **UNHEALTHY**: Service failing health checks but below failure threshold
- **FAILED**: Service exceeded failure threshold, recovery triggered
- **RECOVERING**: Active recovery process in progress
- **DISABLED**: Monitoring disabled for service

#### Recovery Actions
- **RESTART**: Restart service using Docker or systemd
- **RECONNECT**: Reset database/cache connections
- **RESET_CACHE**: Clear service cache and temporary data
- **SCALE_UP**: Increase service resources/instances
- **FAILOVER**: Switch to backup service instance
- **MANUAL_INTERVENTION**: Flag for manual investigation

### 2. Recovery Configuration

**Default Services**:

| Service | Priority | Max Retries | Actions | Failure Threshold |
|---------|----------|-------------|---------|------------------|
| Database | High | 3 | Reconnect, Restart | 2 failures |
| Redis | Medium | 3 | Reconnect, Restart, Reset Cache | 2 failures |
| Weaviate | High | 2 | Restart | 3 failures |
| Neo4j | Medium | 2 | Restart | 3 failures |
| MinIO | Low | 2 | Restart | 2 failures |

#### Exponential Backoff Strategy
```
Initial Delay: 1-10 seconds (service dependent)
Backoff Multiplier: 2.0
Maximum Delay: 30-300 seconds (service dependent)
Maximum Retries: 2-5 attempts (service dependent)
```

### 3. Health Check Functions

Pre-built health checks for common services:

```python
# Database health check
async def check_database_health() -> bool:
    """Verify PostgreSQL connectivity and basic query execution"""
    
# Redis health check  
async def check_redis_health() -> bool:
    """Verify Redis connectivity and ping response"""
    
# Weaviate health check
async def check_weaviate_health() -> bool:
    """Verify Weaviate cluster status and connectivity"""
    
# Neo4j health check
async def check_neo4j_health() -> bool:
    """Verify Neo4j connectivity and basic query execution"""
    
# MinIO health check
async def check_minio_health() -> bool:
    """Verify MinIO connectivity and bucket access"""
```

## API Endpoints

### Recovery Management

**Base URL**: `/api/recovery`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Overall recovery system status |
| `/services` | GET | Status of all monitored services |
| `/services/{service_name}` | GET | Detailed status of specific service |
| `/statistics` | GET | Recovery system statistics |
| `/services/{service_name}/recover` | POST | Force immediate recovery attempt |
| `/services/{service_name}/enable` | POST | Enable monitoring for service |
| `/services/{service_name}/disable` | POST | Disable monitoring for service |
| `/initialize` | POST | Initialize recovery system with defaults |
| `/health` | GET | Recovery system health status |
| `/alerts` | GET | Recovery system alerts and notifications |
| `/metrics` | GET | Detailed recovery metrics for monitoring |

### Usage Examples

```bash
# Get overall system status
curl http://localhost:3000/api/recovery/status

# Check specific service
curl http://localhost:3000/api/recovery/services/database

# Force recovery for failed service
curl -X POST http://localhost:3000/api/recovery/services/redis/recover \
  -H "Content-Type: application/json" \
  -d '{"service_name": "redis", "reason": "Manual recovery test"}'

# View recovery statistics
curl http://localhost:3000/api/recovery/statistics

# Get system alerts
curl "http://localhost:3000/api/recovery/alerts?severity=critical&limit=20"

# View performance metrics
curl http://localhost:3000/api/recovery/metrics
```

## Deployment

### Quick Start

1. **Start Recovery System**:
   ```bash
   cd /opt/projects/knowledgehub
   ./scripts/start-recovery-system.sh
   ```

2. **Monitor Logs**:
   ```bash
   # View systemd service logs
   sudo journalctl -u knowledgehub-recovery.service -f
   
   # View application logs
   tail -f /var/log/knowledgehub/recovery/recovery.log
   ```

3. **Check Status**:
   ```bash
   # Service status
   sudo systemctl status knowledgehub-recovery.service
   
   # API status
   curl http://localhost:3000/api/recovery/health
   ```

### SystemD Service

The recovery system runs as a systemd service for production reliability:

**Service File**: `/etc/systemd/system/knowledgehub-recovery.service`

#### Service Management
```bash
# Start service
sudo systemctl start knowledgehub-recovery.service

# Stop service  
sudo systemctl stop knowledgehub-recovery.service

# Restart service
sudo systemctl restart knowledgehub-recovery.service

# Enable auto-start
sudo systemctl enable knowledgehub-recovery.service

# View status
sudo systemctl status knowledgehub-recovery.service

# View logs
sudo journalctl -u knowledgehub-recovery.service -f
```

### Configuration Files

#### Recovery Configuration
**Location**: `/opt/projects/knowledgehub/config/recovery.json`

```json
{
    "enabled": true,
    "monitoring_interval": 30,
    "recovery_timeout": 300,
    "max_concurrent_recoveries": 3,
    "notification": {
        "email_enabled": false,
        "slack_enabled": false,
        "webhook_url": null
    },
    "services": {
        "database": {
            "enabled": true,
            "priority": "high",
            "max_retries": 3,
            "recovery_actions": ["reconnect", "restart"]
        },
        "redis": {
            "enabled": true,
            "priority": "medium", 
            "max_retries": 3,
            "recovery_actions": ["reconnect", "restart", "reset_cache"]
        }
    }
}
```

#### Log Configuration
**Location**: `/var/log/knowledgehub/recovery/`

- `recovery.log`: Main recovery system logs
- `health_checks.log`: Health check results and timing
- `recovery_actions.log`: Detailed recovery action logs

### Docker Integration

The recovery system can manage Docker containers:

```bash
# Restart Docker service
docker restart knowledgehub-database

# Scale Docker Compose service
docker-compose scale redis=2

# Check Docker service health
docker inspect knowledgehub-weaviate --format='{{.State.Health.Status}}'
```

## Advanced Features

### 1. Dependency-Aware Recovery

Services are recovered in dependency order:

```python
# Example: Database must be healthy before starting dependent services
strategy = RecoveryStrategy(
    service_name="api-server",
    dependencies=["database", "redis"],  # Wait for these services
    recovery_actions=[RecoveryAction.RESTART]
)
```

### 2. Circuit Breaker Pattern

Prevents cascading failures:
- **Failure Threshold**: Service marked as failed after N consecutive failures
- **Recovery Timeout**: Minimum time between recovery attempts
- **Backoff Strategy**: Exponential delay prevents resource exhaustion

### 3. Performance Monitoring

#### Recovery Statistics
- **Total Recoveries**: Count of all recovery attempts
- **Success Rate**: Percentage of successful recoveries
- **Average Recovery Time**: Mean time to restore service
- **Failure Patterns**: Analysis of common failure types

#### Health Metrics
- **Uptime Percentage**: Service availability over time
- **Failure Rate**: Frequency of service failures
- **Mean Time to Recovery (MTTR)**: Average recovery duration
- **Service Reliability Score**: Overall service health rating

### 4. Alert Generation

Automated alerts for:
- **Critical**: Service failed and recovery attempts unsuccessful
- **High**: Service unhealthy with multiple consecutive failures
- **Medium**: Service recovering or degraded performance
- **Low**: Service warnings and minor issues

### 5. Recovery Analytics

#### Pattern Recognition
- **Failure Clustering**: Identify related service failures
- **Time-based Patterns**: Detect failure patterns by time/date
- **Resource Correlation**: Link failures to resource constraints
- **External Factors**: Correlate with system load and external events

#### Optimization Insights
- **Recovery Effectiveness**: Which actions work best for each service
- **Timing Optimization**: Optimal health check intervals and timeouts
- **Resource Allocation**: Service resource requirements and scaling needs
- **Preventive Measures**: Recommendations to prevent future failures

## Integration with Other Systems

### 1. Prometheus Metrics

Recovery metrics exported to Prometheus:

```
# Recovery attempt counters
recovery_attempts_total{service="database", action="restart", result="success"}

# Service health status
service_health_status{service="redis", state="healthy"}

# Recovery duration histogram
recovery_duration_seconds{service="weaviate", action="restart"}

# Service uptime percentage
service_uptime_percentage{service="neo4j"}
```

### 2. Grafana Dashboards

Pre-configured dashboards for:
- **Service Health Overview**: Real-time status of all services
- **Recovery Analytics**: Recovery success rates and timing
- **Alert Dashboard**: Current alerts and incident history
- **Performance Trends**: Historical service reliability trends

### 3. Log Aggregation

Structured logging with correlation IDs:

```json
{
    "timestamp": "2025-01-20T10:30:00Z",
    "level": "INFO",
    "service": "recovery_manager",
    "event": "recovery_initiated",
    "service_name": "database",
    "attempt": 1,
    "correlation_id": "rec_db_20250120_103000",
    "metadata": {
        "failure_count": 3,
        "last_error": "Connection timeout",
        "recovery_action": "restart"
    }
}
```

## Security Considerations

### 1. Access Control
- **API Security**: Recovery endpoints require appropriate authentication
- **System Permissions**: Recovery actions need elevated system privileges
- **Audit Logging**: All recovery actions logged with user attribution

### 2. Resource Protection
- **Rate Limiting**: Prevent abuse of force recovery endpoints
- **Resource Limits**: Constrain recovery system resource usage
- **Isolation**: Recovery system isolated from services it monitors

### 3. Sensitive Data
- **Credential Protection**: Service credentials securely stored and accessed
- **Log Sanitization**: Sensitive information filtered from logs
- **Secure Communication**: Encrypted communication with monitored services

## Troubleshooting

### Common Issues

1. **Recovery System Not Starting**:
   ```bash
   # Check systemd service status
   sudo systemctl status knowledgehub-recovery.service
   
   # View detailed logs
   sudo journalctl -u knowledgehub-recovery.service -n 50
   
   # Check configuration
   cat /opt/projects/knowledgehub/config/recovery.json
   ```

2. **Services Not Being Monitored**:
   ```bash
   # Verify service registration
   curl http://localhost:3000/api/recovery/services
   
   # Check health check functions
   python3 -c "from api.services.service_recovery import check_database_health; print(check_database_health())"
   ```

3. **Recovery Actions Failing**:
   ```bash
   # Check Docker/systemd permissions
   sudo docker ps
   sudo systemctl list-units --type=service
   
   # Verify service dependencies
   curl http://localhost:3000/api/recovery/status
   ```

4. **High Resource Usage**:
   ```bash
   # Monitor recovery system resource usage
   top -p $(pgrep -f service_recovery)
   
   # Check health check frequency
   grep "health_check_interval" /opt/projects/knowledgehub/config/recovery.json
   ```

### Debugging Commands

```bash
# Test individual health checks
python3 -c "
import asyncio
from api.services.service_recovery import check_redis_health
print(asyncio.run(check_redis_health()))
"

# Force recovery for testing
curl -X POST http://localhost:3000/api/recovery/services/redis/recover \
  -H "Content-Type: application/json" \
  -d '{"service_name": "redis", "reason": "Testing recovery system"}'

# View recovery statistics
curl http://localhost:3000/api/recovery/statistics | jq

# Check alert status
curl "http://localhost:3000/api/recovery/alerts?severity=critical" | jq
```

## Performance Impact

### Overhead Analysis
- **CPU Overhead**: <1% average CPU increase for health monitoring
- **Memory Overhead**: ~20MB for recovery system processes
- **Network Overhead**: Minimal (health check requests only)
- **Storage Overhead**: Log files (rotated daily, 7-day retention)

### Optimization Settings

```json
{
    "health_check_interval": 30,     // Balance responsiveness vs overhead
    "failure_threshold": 3,          // Prevent false positives
    "recovery_timeout": 300,         // Allow sufficient recovery time
    "max_concurrent_recoveries": 3,  // Prevent resource exhaustion
    "log_level": "INFO"              // Reduce verbose logging overhead
}
```

## Best Practices

### 1. Service Design
- **Health Check Endpoints**: Implement lightweight health checks in services
- **Graceful Shutdown**: Services should handle shutdown signals properly
- **State Management**: Services should be able to recover state after restart
- **Resource Cleanup**: Proper cleanup of resources during shutdown

### 2. Monitoring Configuration
- **Appropriate Thresholds**: Set failure thresholds based on service criticality
- **Health Check Timing**: Balance frequency with performance impact
- **Dependency Mapping**: Accurately map service dependencies
- **Recovery Actions**: Choose appropriate recovery actions for each service type

### 3. Production Deployment
- **Resource Monitoring**: Monitor recovery system resource usage
- **Alert Configuration**: Set up appropriate alerting for recovery failures
- **Log Management**: Implement log rotation and retention policies
- **Backup Strategies**: Ensure recovery system has backup/failover capabilities

### 4. Testing and Validation
- **Chaos Engineering**: Regularly test recovery by inducing failures
- **Recovery Drills**: Practice manual recovery procedures
- **Performance Testing**: Validate recovery under load conditions
- **Documentation**: Keep recovery procedures and runbooks updated

This comprehensive service recovery and self-healing system ensures KnowledgeHub maintains maximum uptime and automatically recovers from common failure scenarios, providing production-grade reliability and resilience.