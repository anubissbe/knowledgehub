#!/bin/bash

# KnowledgeHub Smart Deployment Script
# Works on any machine with Claude Code
# Handles existing deployments gracefully

set -e  # Exit on error

echo "🚀 KnowledgeHub Smart Deployment"
echo "================================"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo ""

# Configuration
PROJECT_DIR="/opt/projects/knowledgehub"
REQUIRED_SERVICES="postgres redis weaviate minio api mcp-server scraper scheduler frontend"
OPTIONAL_SERVICES="rag-processor ai-service"
MONITORING_SERVICES="prometheus grafana alertmanager node-exporter cadvisor"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to project directory
cd "$PROJECT_DIR"

# Function to check if service is running
is_running() {
    local service=$1
    docker compose ps --services --filter "status=running" | grep -q "^${service}$"
}

# Function to check if service exists
service_exists() {
    local service=$1
    docker compose ps --services | grep -q "^${service}$"
}

# Pre-deployment checks
echo "🔍 Pre-deployment checks..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"

# Check Docker Compose
if ! docker compose version &> /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose v2 is not available${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose found${NC}"

# Check disk space
AVAILABLE_SPACE=$(df -BG /opt | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    echo -e "${RED}❌ Less than 10GB available disk space${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Disk space: ${AVAILABLE_SPACE}GB available${NC}"

# Create necessary directories
echo -e "\n📁 Setting up directories..."
mkdir -p data/{postgres,redis,weaviate,minio}
mkdir -p logs
mkdir -p backups
mkdir -p config

# Check existing deployment
echo -e "\n🔍 Checking existing deployment..."
EXISTING_SERVICES=$(docker compose ps --services 2>/dev/null || echo "")

if [ -n "$EXISTING_SERVICES" ]; then
    echo -e "${YELLOW}Found existing services:${NC}"
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Service}}"
    
    echo -e "\n${YELLOW}What would you like to do?${NC}"
    echo "1) Stop and redeploy all services"
    echo "2) Update only non-running services"
    echo "3) Restart all services"
    echo "4) Cancel deployment"
    
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            echo -e "\n${YELLOW}Stopping all services...${NC}"
            docker compose down
            ;;
        2)
            echo -e "\n${BLUE}Updating non-running services only${NC}"
            ;;
        3)
            echo -e "\n${YELLOW}Restarting all services...${NC}"
            docker compose restart
            echo -e "${GREEN}✅ Services restarted${NC}"
            exit 0
            ;;
        4)
            echo -e "${YELLOW}Deployment cancelled${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
fi

# Build services
echo -e "\n🔨 Building services..."
docker compose build --parallel

# Deploy required services
echo -e "\n🚀 Deploying required services..."

# Start infrastructure first
echo "Starting infrastructure services..."
docker compose up -d postgres redis weaviate minio

# Wait for infrastructure
echo "Waiting for infrastructure to be ready..."
sleep 20

# Check infrastructure health
pg_isready -h localhost -p 5433 -U khuser > /dev/null 2>&1 || echo -e "${YELLOW}⚠️  PostgreSQL may need more time${NC}"
redis-cli -p 6381 ping > /dev/null 2>&1 || echo -e "${YELLOW}⚠️  Redis may need more time${NC}"

# Start application services
echo -e "\nStarting application services..."
docker compose up -d api mcp-server scraper scheduler frontend

# Handle optional services
echo -e "\n📦 Checking optional services..."

# RAG processor (currently has issues with aiohttp)
if docker compose ps rag-processor 2>/dev/null | grep -q "Restarting"; then
    echo -e "${YELLOW}⚠️  RAG processor is having issues, skipping for now${NC}"
    docker compose stop rag-processor
else
    echo "Starting RAG processor..."
    docker compose up -d rag-processor || echo -e "${YELLOW}⚠️  RAG processor failed to start${NC}"
fi

# AI service (optional)
if service_exists "ai-service"; then
    echo "Starting AI service..."
    docker compose up -d ai-service || echo -e "${YELLOW}⚠️  AI service is optional${NC}"
fi

# Monitoring services (optional)
echo -e "\n📊 Deploy monitoring? (y/n)"
read -p "> " deploy_monitoring

if [[ "$deploy_monitoring" =~ ^[Yy]$ ]]; then
    echo "Starting monitoring services..."
    docker compose -f docker-compose.monitoring.yml up -d || echo -e "${YELLOW}⚠️  Some monitoring services may fail${NC}"
fi

# Wait for services to stabilize
echo -e "\n⏳ Waiting for services to stabilize..."
sleep 20

# Health check
echo -e "\n🏥 Running health checks..."
./scripts/check-health.sh || true

# Show final status
echo -e "\n📊 Final deployment status:"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Service}}"

# Show access information
echo -e "\n🌐 Access URLs:"
echo -e "${GREEN}   KnowledgeHub UI: http://localhost:3100${NC}"
echo -e "${GREEN}   API Documentation: http://localhost:3000/docs${NC}"
echo -e "${GREEN}   MCP Server: http://localhost:3008${NC}"

if [[ "$deploy_monitoring" =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}   Grafana: http://localhost:3030${NC}"
    echo -e "${GREEN}   Prometheus: http://localhost:9090${NC}"
fi

echo -e "\n📝 Useful commands:"
echo "   View logs: docker compose logs -f <service>"
echo "   Stop all: docker compose down"
echo "   Restart: docker compose restart"
echo "   Status: docker compose ps"

echo -e "\n${GREEN}✅ Deployment complete!${NC}"

# Create a deployment record
echo "$(date): Deployment completed by $(whoami) on $(hostname)" >> deployments.log