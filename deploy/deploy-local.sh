#!/bin/bash

# KnowledgeHub Local Deployment Script
# Deploys all services locally with proper configuration

set -e  # Exit on error

echo "🚀 KnowledgeHub Local Deployment"
echo "================================"
echo "Time: $(date)"
echo ""

# Configuration
PROJECT_DIR="/opt/projects/knowledgehub"
COMPOSE_FILE="docker-compose.yml"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command result
check_result() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        exit 1
    fi
}

# Change to project directory
cd "$PROJECT_DIR"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check Docker
docker --version > /dev/null 2>&1
check_result "Docker check"

# Check Docker Compose
docker compose version > /dev/null 2>&1
check_result "Docker Compose check"

# Check disk space
AVAILABLE_SPACE=$(df -BG /opt | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 20 ]; then
    echo -e "${YELLOW}⚠️  Warning: Less than 20GB available disk space${NC}"
fi

# Create necessary directories
echo -e "\n📁 Creating necessary directories..."
mkdir -p data/{postgres,redis,weaviate,minio}
mkdir -p logs
mkdir -p backups
check_result "Directory creation"

# Backup existing data if any
if [ -d "data" ] && [ "$(ls -A data)" ]; then
    echo -e "\n📦 Backing up existing data..."
    ./scripts/backup.sh
    check_result "Data backup"
fi

# Build all services
echo -e "\n🔨 Building all services..."
docker compose build --parallel
check_result "Build services"

# Stop existing containers
echo -e "\n🛑 Stopping existing containers..."
docker compose down
check_result "Stop containers"

# Start infrastructure services first
echo -e "\n🚀 Starting infrastructure services..."
docker compose up -d postgres redis weaviate minio
check_result "Start infrastructure"

# Wait for infrastructure to be ready
echo -e "\n⏳ Waiting for infrastructure services..."
sleep 20

# Check infrastructure health
echo -e "\n🏥 Checking infrastructure health..."
docker compose ps postgres redis weaviate minio
pg_isready -h localhost -p 5433 -U khuser
check_result "PostgreSQL health"

redis-cli -p 6381 ping > /dev/null 2>&1
check_result "Redis health"

curl -s http://localhost:8090/v1/.well-known/ready > /dev/null 2>&1
check_result "Weaviate health"

curl -s http://localhost:9010/minio/health/ready > /dev/null 2>&1
check_result "MinIO health"

# Start application services
echo -e "\n🚀 Starting application services..."
docker compose up -d api mcp-server scraper rag-processor scheduler
check_result "Start applications"

# Start frontend
echo -e "\n🎨 Starting frontend..."
docker compose up -d frontend
check_result "Start frontend"

# Start monitoring services (optional)
echo -e "\n📊 Starting monitoring services..."
docker compose -f docker-compose.monitoring.yml up -d || echo "Monitoring services are optional"

# Wait for all services to be ready
echo -e "\n⏳ Waiting for all services to be ready..."
sleep 30

# Run comprehensive health check
echo -e "\n🏥 Running comprehensive health checks..."
./scripts/check-health.sh

# Show service status
echo -e "\n📊 Service Status:"
docker compose ps

# Show access URLs
echo -e "\n🌐 Access URLs:"
echo "   KnowledgeHub UI: http://localhost:3100"
echo "   API Documentation: http://localhost:3000/docs"
echo "   MCP Server: http://localhost:3008"
echo "   Grafana (if enabled): http://localhost:3030"
echo "   Prometheus (if enabled): http://localhost:9090"

echo -e "\n${GREEN}✅ Local deployment complete!${NC}"
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop services: docker compose down"
echo "To restart services: docker compose restart"