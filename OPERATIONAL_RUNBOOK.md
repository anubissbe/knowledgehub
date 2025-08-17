
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
docker ps --filter "name=knowledgehub" --format "table {{.Names}}	{{.Status}}"
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
