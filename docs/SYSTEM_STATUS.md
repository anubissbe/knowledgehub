# KnowledgeHub System Status

## Last Updated: July 6, 2025

### Overall Health: ✅ OPERATIONAL

## Service Status (16/16 Healthy)

### Core Services
| Service | Status | Port | Health Check |
|---------|--------|------|--------------|
| PostgreSQL | ✅ Healthy | 5433 | pg_isready |
| Redis | ✅ Healthy | 6381 | redis-cli ping |
| Weaviate | ✅ Healthy | 8090 | HTTP /v1/.well-known/ready |
| MinIO | ✅ Healthy | 9010 | HTTP /minio/health/live |
| API Gateway | ✅ Healthy | 3000 | HTTP /health |
| Frontend | ✅ Healthy | 3100 | Static files served |

### Processing Services
| Service | Status | Port | Health Check |
|---------|--------|------|--------------|
| Scraper 1 | ✅ Healthy | 3014 | HTTP /health |
| Scraper 2 | ✅ Healthy | 3015 | HTTP /health |
| RAG Processor | ✅ Healthy | 3013 | File-based check |
| Scheduler | ✅ Healthy | - | Python execution |
| AI Service | ✅ Healthy | 8002 | HTTP /health |
| MCP Server | ✅ Healthy | 3008 | File-based check |

### Monitoring Stack
| Service | Status | Port | Health Check |
|---------|--------|------|--------------|
| Prometheus | ✅ Healthy | 9090 | HTTP /-/healthy |
| Grafana | ✅ Healthy | 3101 | HTTP /api/health |
| AlertManager | ✅ Healthy | 9093 | HTTP /-/healthy |
| Node Exporter | ✅ Healthy | 9100 | Metrics endpoint |
| cAdvisor | ✅ Healthy | 8080 | HTTP /healthz |

## System Metrics

### Resource Usage
- **Memory**: 7.1 GB / 251.6 GB (2.8%)
- **Storage**: 35 GB / 11 TB (0.3%)
- **CPU**: Normal load
- **Containers**: 16 running

### Performance
- **API Response Time**: ~126ms average
- **Requests/Hour**: ~706
- **WebSocket Connections**: Active
- **Database Queries**: Responsive

### Data Statistics
- **Total Sources**: 12
- **Active Sources**: 12
- **Total Documents**: 1,270
- **Total Chunks**: 5,433
- **Checkmarx Docs**: 693

## Recent Changes

### Deployment Updates
1. Added all optional services (AI, MCP, Monitoring)
2. Fixed health checks for all services
3. Implemented real performance metrics
4. Updated storage reporting to show actual 11TB capacity
5. Fixed frontend API connectivity issues

### Configuration Changes
- AI Service moved to port 8002
- MCP Server moved to port 3008
- Added ACTUAL_STORAGE_TB=11 environment variable
- Updated frontend production environment URLs

## Access URLs

### User Interfaces
- **Main Dashboard**: http://192.168.1.25:3100/dashboard
- **Grafana Monitoring**: http://192.168.1.25:3101
- **Prometheus Metrics**: http://192.168.1.25:9090
- **MinIO Console**: http://192.168.1.25:9011

### API Endpoints
- **Main API**: http://192.168.1.25:3000
- **API Docs**: http://192.168.1.25:3000/docs
- **Health Check**: http://192.168.1.25:3000/health
- **Performance Metrics**: http://192.168.1.25:3000/api/v1/analytics/performance

### WebSocket
- **Notifications**: ws://192.168.1.25:3000/ws/notifications

## Maintenance Notes

### Daily Checks
1. Monitor service health status
2. Check storage usage trends
3. Review error logs for anomalies
4. Verify scraper job completion

### Weekly Tasks
1. Review and optimize slow queries
2. Check for pending system updates
3. Analyze performance trends
4. Clean up old job logs

### Known Issues
- None currently active

### Upcoming Improvements
1. Implement automated backups
2. Add more granular monitoring
3. Optimize vector search performance
4. Enhance AI service capabilities