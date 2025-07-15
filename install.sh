#!/bin/bash

# KnowledgeHub Installation Script
# This script sets up KnowledgeHub with all required services

set -e

echo "🚀 KnowledgeHub Installation Script"
echo "=================================="

# Check prerequisites
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install $1 first."
        exit 1
    fi
}

echo "Checking prerequisites..."
check_command docker
check_command docker-compose
check_command git

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file - please edit it with your configuration"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs webui/public

# Pull required Docker images
echo "Pulling Docker images..."
docker-compose pull

# Build custom images
echo "Building KnowledgeHub services..."
docker-compose build

# Start core services first
echo "Starting core services (PostgreSQL, Redis)..."
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until docker-compose exec -T postgres pg_isready -U knowledgehub &>/dev/null; do
  echo -n "."
  sleep 2
done
echo " Ready!"

# Run database migrations
echo "Running database migrations..."
for migration in migrations/*.sql; do
    if [ -f "$migration" ]; then
        echo "Applying $migration..."
        docker-compose exec -T postgres psql -U knowledgehub -d knowledgehub < "$migration"
    fi
done

# Run TimescaleDB migrations
echo "Setting up TimescaleDB..."
docker-compose up -d timescale
sleep 10
docker-compose exec -T timescale psql -U knowledgehub -d knowledgehub_analytics < migrations/003_timescale_analytics.sql || true

# Start remaining services
echo "Starting all services..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 20

# Check service health
echo ""
echo "Checking service status..."
echo "=========================="

check_service() {
    if curl -f -s $2 > /dev/null; then
        echo "✅ $1 is running at $2"
    else
        echo "❌ $1 is not responding at $2"
    fi
}

check_service "API Gateway" "http://localhost:3000/health"
check_service "Web UI" "http://localhost:3100"
check_service "PostgreSQL" "localhost:5433"
check_service "Redis" "localhost:6381"
check_service "Weaviate" "http://localhost:8090/v1/.well-known/ready"
check_service "Neo4j" "http://localhost:7474"
check_service "MinIO" "http://localhost:9010/minio/health/live"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Access KnowledgeHub at:"
echo "  - Web UI: http://localhost:3100"
echo "  - API Docs: http://localhost:3000/docs"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - MinIO Console: http://localhost:9011"
echo ""
echo "Default credentials:"
echo "  - Neo4j: neo4j / knowledgehub123"
echo "  - MinIO: minioadmin / minioadmin"
echo ""
echo "To stop all services: docker-compose down"
echo "To view logs: docker-compose logs -f"
echo ""
echo "⚠️  Remember to update the passwords in your .env file for production use!"