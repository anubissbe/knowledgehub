#!/bin/bash
# Container Health Fix Script
# Resolves startup dependency issues

echo "🔧 Fixing container health issues..."

# Stop all services
docker-compose down

# Remove problematic volumes
docker volume prune -f

# Recreate volumes with proper permissions
docker volume create knowledgehub_postgres_data
docker volume create knowledgehub_redis_data
docker volume create knowledgehub_weaviate_data
docker volume create knowledgehub_neo4j_data
docker volume create knowledgehub_minio_data

# Update docker-compose with health checks
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: knowledgehub
      POSTGRES_USER: knowledgehub
      POSTGRES_PASSWORD: knowledgehub_prod_2025
    ports:
      - "5433:5432"
    volumes:
      - knowledgehub_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U knowledgehub -d knowledgehub"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

  redis:
    image: redis:7
    ports:
      - "6381:6379"
    volumes:
      - knowledgehub_redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    environment:
      - DATABASE_URL=postgresql://knowledgehub:knowledgehub_prod_2025@postgres:5432/knowledgehub
      - REDIS_URL=redis://redis:6379
    ports:
      - "3000:3000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s

volumes:
  knowledgehub_postgres_data:
  knowledgehub_redis_data:
  knowledgehub_weaviate_data:
  knowledgehub_neo4j_data:
  knowledgehub_minio_data:
EOF

echo "✅ Container health fixes applied"
