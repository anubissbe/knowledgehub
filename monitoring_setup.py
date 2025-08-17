#\!/usr/bin/env python3
"""
Production Monitoring and Alerting Setup for KnowledgeHub RAG System

Sets up comprehensive monitoring, alerting, and operational runbooks
for the distributed KnowledgeHub RAG system.

Author: Wim De Meyer - Refactoring & Distributed Systems Expert
"""

import json
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ProductionMonitoringSetup:
    """Setup production monitoring and alerting"""
    
    def __init__(self):
        self.monitoring_config = {}
        
    def setup_prometheus_config(self) -> Dict[str, Any]:
        """Generate Prometheus configuration for RAG system monitoring"""
        
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [
                "rag_alerts.yml"
            ],
            "scrape_configs": [
                {
                    "job_name": "knowledgehub-api",
                    "static_configs": [{
                        "targets": ["192.168.1.25:3000"]
                    }],
                    "metrics_path": "/metrics",
                    "scrape_interval": "10s"
                },
                {
                    "job_name": "knowledgehub-ai-service",
                    "static_configs": [{
                        "targets": ["192.168.1.25:8002"]
                    }],
                    "metrics_path": "/metrics",
                    "scrape_interval": "15s"
                },
                {
                    "job_name": "postgres-exporter",
                    "static_configs": [{
                        "targets": ["192.168.1.25:9187"]
                    }]
                },
                {
                    "job_name": "redis-exporter", 
                    "static_configs": [{
                        "targets": ["192.168.1.25:9121"]
                    }]
                },
                {
                    "job_name": "node-exporter",
                    "static_configs": [{
                        "targets": ["192.168.1.25:9100"]
                    }]
                }
            ],
            "alerting": {
                "alertmanagers": [{
                    "static_configs": [{
                        "targets": ["192.168.1.25:9093"]
                    }]
                }]
            }
        }
        
        return prometheus_config
    
    def setup_alerting_rules(self) -> Dict[str, Any]:
        """Generate alerting rules for RAG system"""
        
        alerting_rules = {
            "groups": [
                {
                    "name": "rag_system_alerts",
                    "rules": [
                        {
                            "alert": "RAGSystemDown",
                            "expr": "up{job='knowledgehub-api'} == 0",
                            "for": "1m",
                            "labels": {
                                "severity": "critical"
                            },
                            "annotations": {
                                "summary": "KnowledgeHub API is down",
                                "description": "The KnowledgeHub API has been down for more than 1 minute"
                            }
                        },
                        {
                            "alert": "RAGHighLatency",
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job='knowledgehub-api'}[5m])) > 2",
                            "for": "3m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High RAG query latency",
                                "description": "95th percentile latency is above 2 seconds"
                            }
                        },
                        {
                            "alert": "RAGHighErrorRate",
                            "expr": "rate(http_requests_total{job='knowledgehub-api',status=~'5..'}[5m]) / rate(http_requests_total{job='knowledgehub-api'}[5m]) > 0.1",
                            "for": "2m",
                            "labels": {
                                "severity": "critical"
                            },
                            "annotations": {
                                "summary": "High RAG error rate",
                                "description": "Error rate is above 10%"
                            }
                        },
                        {
                            "alert": "DatabaseConnectionHigh",
                            "expr": "pg_stat_activity_count > 80",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High database connection count",
                                "description": "PostgreSQL connection count is high"
                            }
                        },
                        {
                            "alert": "RedisMemoryHigh",
                            "expr": "redis_memory_used_bytes / redis_config_maxmemory_bytes > 0.9",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "Redis memory usage high",
                                "description": "Redis memory usage is above 90%"
                            }
                        },
                        {
                            "alert": "VectorSearchLatency",
                            "expr": "histogram_quantile(0.95, rate(weaviate_query_duration_seconds_bucket[5m])) > 1",
                            "for": "3m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "High vector search latency",
                                "description": "Weaviate query latency is above 1 second"
                            }
                        }
                    ]
                }
            ]
        }
        
        return alerting_rules
    
    def generate_grafana_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard for RAG system"""
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "KnowledgeHub RAG System Monitoring",
                "tags": ["rag", "knowledgehub"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "API Request Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(http_requests_total{job='knowledgehub-api'}[5m])",
                            "legendFormat": "{{method}} {{endpoint}}"
                        }],
                        "yAxes": [
                            {"label": "Requests/sec"},
                            {"show": False}
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "RAG Query Latency",
                        "type": "graph", 
                        "targets": [{
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job='knowledgehub-api'}[5m]))",
                            "legendFormat": "95th percentile"
                        }],
                        "yAxes": [
                            {"label": "Seconds"},
                            {"show": False}
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Database Connections",
                        "type": "singlestat",
                        "targets": [{
                            "expr": "pg_stat_activity_count",
                            "legendFormat": "Active Connections"
                        }],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Redis Memory Usage",
                        "type": "singlestat",
                        "targets": [{
                            "expr": "redis_memory_used_bytes / redis_config_maxmemory_bytes * 100",
                            "legendFormat": "Memory %"
                        }],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
                    },
                    {
                        "id": 5,
                        "title": "Vector Search Performance",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(weaviate_queries_total[5m])",
                            "legendFormat": "Queries/sec"
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        return dashboard
    
    def create_operational_runbook(self) -> str:
        """Create operational runbook"""
        
        runbook = """
# KnowledgeHub RAG System Operational Runbook

## System Overview
- **Architecture**: Distributed RAG system with microservices
- **Location**: 192.168.1.25 (primary environment)
- **Components**: API, AI Service, PostgreSQL, Redis, Weaviate, Neo4j, TimescaleDB

## Emergency Contacts
- **Primary On-Call**: [Configure based on team]
- **Secondary**: [Configure based on team]
- **Escalation**: [Configure based on team]

## Quick Health Checks

### System Status Check
```bash
curl -s http://192.168.1.25:3000/health | jq '.status'
```

### Database Health
```bash
curl -s http://192.168.1.25:3000/health | jq '.services'
```

### Container Status  
```bash
docker ps --filter "name=knowledgehub" --format "table {{.Names}}\t{{.Status}}"
```

## Common Issues and Resolutions

### 1. API Service Down
**Symptoms**: HTTP 5xx errors, service unreachable
**Resolution**:
```bash
# Check container status
docker ps | grep api

# Restart API service
docker-compose restart api

# Check logs
docker logs knowledgehub-api-1 --tail=50
```

### 2. High Memory Usage
**Symptoms**: Memory alerts, slow performance
**Resolution**:
```bash
# Check memory usage
free -h
docker stats

# Restart services if needed
docker-compose restart api ai-service
```

### 3. Database Connection Issues
**Symptoms**: Database connection errors
**Resolution**:
```bash
# Check PostgreSQL status
docker logs knowledgehub-postgres-1

# Test connection
psql -h 192.168.1.25 -p 5433 -U knowledgehub -d knowledgehub -c "SELECT 1;"

# Restart if needed
docker-compose restart postgres
```

### 4. Vector Search Slow
**Symptoms**: High query latency alerts
**Resolution**:
```bash
# Check Weaviate status
curl -s http://192.168.1.25:8090/v1/meta

# Check memory usage
curl -s http://192.168.1.25:8090/v1/.well-known/ready

# Consider restarting Weaviate
docker-compose restart weaviate
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
docker exec knowledgehub-postgres-1 pg_dump -U knowledgehub knowledgehub > backup_$(date +%Y%m%d).sql

# Restore backup
docker exec -i knowledgehub-postgres-1 psql -U knowledgehub knowledgehub < backup.sql
```

### Configuration Backup
```bash
# Backup docker-compose and configs
tar -czf config_backup_$(date +%Y%m%d).tar.gz docker-compose.yml .env api/config.py
```

## Performance Optimization

### When RAG Queries Are Slow
1. Check database query performance
2. Monitor vector search latency  
3. Review memory usage patterns
4. Consider scaling horizontally

### When Memory Usage Is High
1. Restart services during low-traffic periods
2. Review cache configurations
3. Monitor for memory leaks
4. Consider resource limits

## Scaling Procedures

### Horizontal Scaling
```bash
# Scale API service
docker-compose up --scale api=2

# Load balancer configuration needed
```

### Database Scaling
- Read replicas for query distribution
- Connection pooling optimization
- Query optimization

## Maintenance Windows

### Regular Maintenance (Weekly)
- Check system resource usage
- Review application logs for errors
- Update security patches
- Database maintenance and optimization

### Updates and Deployments
- Use staging environment first
- Coordinate with team for production updates
- Monitor post-deployment metrics
- Have rollback plan ready

## Monitoring URLs
- **Grafana**: http://192.168.1.25:3030
- **Prometheus**: http://192.168.1.25:9090  
- **API Health**: http://192.168.1.25:3000/health
- **WebUI**: http://192.168.1.25:3100

## Log Locations
- **API Logs**: `docker logs knowledgehub-api-1`
- **AI Service Logs**: `docker logs knowledgehub-ai-service-1`
- **Database Logs**: `docker logs knowledgehub-postgres-1`
- **Application Logs**: `/opt/projects/knowledgehub/logs/`

---
*Last Updated: August 2025*
*Maintained by: Systems Engineering Team*
"""
        
        return runbook
    
    def setup_complete_monitoring(self) -> Dict[str, Any]:
        """Setup complete monitoring stack"""
        
        monitoring_setup = {
            "prometheus_config": self.setup_prometheus_config(),
            "alerting_rules": self.setup_alerting_rules(),
            "grafana_dashboard": self.generate_grafana_dashboard(),
            "operational_runbook": self.create_operational_runbook()
        }
        
        return monitoring_setup


def main():
    """Setup production monitoring"""
    print("🔧 Setting up production monitoring for KnowledgeHub RAG System")
    print("=" * 60)
    
    setup = ProductionMonitoringSetup()
    monitoring = setup.setup_complete_monitoring()
    
    # Save configurations
    os.makedirs("/opt/projects/knowledgehub/monitoring", exist_ok=True)
    
    # Prometheus config
    with open("/opt/projects/knowledgehub/monitoring/prometheus.yml", "w") as f:
        json.dump(monitoring["prometheus_config"], f, indent=2)
    
    # Alerting rules
    with open("/opt/projects/knowledgehub/monitoring/rag_alerts.yml", "w") as f:
        json.dump(monitoring["alerting_rules"], f, indent=2)
    
    # Grafana dashboard
    with open("/opt/projects/knowledgehub/monitoring/grafana_dashboard.json", "w") as f:
        json.dump(monitoring["grafana_dashboard"], f, indent=2)
    
    # Operational runbook
    with open("/opt/projects/knowledgehub/OPERATIONAL_RUNBOOK.md", "w") as f:
        f.write(monitoring["operational_runbook"])
    
    print("✅ Monitoring configuration generated:")
    print("   - Prometheus config: monitoring/prometheus.yml")
    print("   - Alert rules: monitoring/rag_alerts.yml")  
    print("   - Grafana dashboard: monitoring/grafana_dashboard.json")
    print("   - Operational runbook: OPERATIONAL_RUNBOOK.md")
    print()
    print("🚀 Next steps:")
    print("   1. Deploy monitoring stack with docker-compose")
    print("   2. Import Grafana dashboard")
    print("   3. Configure alerting channels")
    print("   4. Train team on operational procedures")


if __name__ == "__main__":
    main()
