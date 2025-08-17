
#!/bin/bash
# Health check script for all services

echo "🏥 KnowledgeHub Health Check"
echo "============================"

# Check API
echo -n "API Service: "
curl -s http://localhost:3000/health > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Unhealthy"

# Check PostgreSQL
echo -n "PostgreSQL: "
docker exec knowledgehub-postgres-1 pg_isready > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Unhealthy"

# Check Redis
echo -n "Redis: "
docker exec knowledgehub-redis-1 redis-cli ping > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Unhealthy"

# Check Neo4j
echo -n "Neo4j: "
curl -s http://localhost:7474 > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Unhealthy"

# Check Weaviate
echo -n "Weaviate: "
curl -s http://localhost:8090/v1/.well-known/ready > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Unhealthy"

# Check Qdrant
echo -n "Qdrant: "
curl -s http://localhost:6333 > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Unhealthy"

# Check Zep
echo -n "Zep: "
curl -s http://localhost:8100 > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Unhealthy"

# Check MinIO
echo -n "MinIO: "
curl -s http://localhost:9010/minio/health/live > /dev/null 2>&1 && echo "✅ Healthy" || echo "❌ Unhealthy"

echo "============================"
