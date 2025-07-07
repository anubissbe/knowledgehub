#!/bin/bash

# KnowledgeHub Production Deployment Script
# Deploys all services to Synology NAS at 192.168.1.24

set -e  # Exit on error

echo "🚀 KnowledgeHub Production Deployment"
echo "===================================="
echo "Target: Synology NAS at 192.168.1.24"
echo "Time: $(date)"
echo ""

# Configuration
SYNOLOGY_HOST="192.168.1.24"
SYNOLOGY_USER="Bert"
SYNOLOGY_PORT="2222"
REMOTE_PATH="/volume1/docker/knowledgehub"
PROJECT_NAME="knowledgehub"

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

# Function to deploy a service
deploy_service() {
    local service=$1
    echo -e "\n${YELLOW}Deploying $service...${NC}"
    
    # Build the service
    echo "Building $service..."
    docker compose build $service
    check_result "Build $service"
    
    # Tag for registry
    docker tag ${PROJECT_NAME}-${service}:latest ${SYNOLOGY_HOST}:5000/${PROJECT_NAME}-${service}:latest
    check_result "Tag $service"
    
    # Push to registry (if available)
    # docker push ${SYNOLOGY_HOST}:5000/${PROJECT_NAME}-${service}:latest
    # check_result "Push $service"
}

# Pre-deployment checks
echo "🔍 Pre-deployment checks..."

# Check SSH connectivity
ssh -p $SYNOLOGY_PORT $SYNOLOGY_USER@$SYNOLOGY_HOST "echo 'SSH connection OK'" > /dev/null 2>&1
check_result "SSH connectivity"

# Check Docker on Synology
ssh -p $SYNOLOGY_PORT $SYNOLOGY_USER@$SYNOLOGY_HOST "docker --version" > /dev/null 2>&1
check_result "Docker on Synology"

# Create remote directory if needed
echo -e "\n📁 Setting up remote directory..."
ssh -p $SYNOLOGY_PORT $SYNOLOGY_USER@$SYNOLOGY_HOST "mkdir -p $REMOTE_PATH"
check_result "Remote directory setup"

# Copy configuration files
echo -e "\n📋 Copying configuration files..."
scp -P $SYNOLOGY_PORT docker-compose.yml $SYNOLOGY_USER@$SYNOLOGY_HOST:$REMOTE_PATH/
check_result "Copy docker-compose.yml"

scp -P $SYNOLOGY_PORT -r config $SYNOLOGY_USER@$SYNOLOGY_HOST:$REMOTE_PATH/
check_result "Copy config directory"

# Build all services
echo -e "\n🔨 Building all services..."
docker compose build
check_result "Build all services"

# Export images and transfer
echo -e "\n📦 Exporting and transferring images..."

# List of services to deploy
SERVICES=(
    "postgres"
    "redis"
    "weaviate"
    "minio"
    "api"
    "mcp-server"
    "scraper"
    "rag-processor"
    "scheduler"
    "frontend"
)

for service in "${SERVICES[@]}"; do
    if docker images | grep -q "${PROJECT_NAME}-${service}"; then
        echo "Exporting $service..."
        docker save ${PROJECT_NAME}-${service}:latest | gzip > /tmp/${PROJECT_NAME}-${service}.tar.gz
        check_result "Export $service"
        
        echo "Transferring $service..."
        scp -P $SYNOLOGY_PORT /tmp/${PROJECT_NAME}-${service}.tar.gz $SYNOLOGY_USER@$SYNOLOGY_HOST:$REMOTE_PATH/
        check_result "Transfer $service"
        
        echo "Loading $service on Synology..."
        ssh -p $SYNOLOGY_PORT $SYNOLOGY_USER@$SYNOLOGY_HOST "cd $REMOTE_PATH && gunzip -c ${PROJECT_NAME}-${service}.tar.gz | docker load"
        check_result "Load $service"
        
        # Cleanup
        rm -f /tmp/${PROJECT_NAME}-${service}.tar.gz
    fi
done

# Deploy on Synology
echo -e "\n🚀 Starting services on Synology..."
ssh -p $SYNOLOGY_PORT $SYNOLOGY_USER@$SYNOLOGY_HOST "cd $REMOTE_PATH && docker compose down"
ssh -p $SYNOLOGY_PORT $SYNOLOGY_USER@$SYNOLOGY_HOST "cd $REMOTE_PATH && docker compose up -d"
check_result "Start services"

# Wait for services to be healthy
echo -e "\n⏳ Waiting for services to be healthy..."
sleep 30

# Health checks
echo -e "\n🏥 Running health checks..."

# Check each service
HEALTH_ENDPOINTS=(
    "http://${SYNOLOGY_HOST}:5433" # PostgreSQL
    "http://${SYNOLOGY_HOST}:6381" # Redis
    "http://${SYNOLOGY_HOST}:8090/v1/.well-known/ready" # Weaviate
    "http://${SYNOLOGY_HOST}:9010/minio/health/ready" # MinIO
    "http://${SYNOLOGY_HOST}:3000/api/v1/sources/" # API
    "http://${SYNOLOGY_HOST}:3008/health" # MCP Server
    "http://${SYNOLOGY_HOST}:3014/health" # Scraper
    "http://${SYNOLOGY_HOST}:3100" # Frontend
)

all_healthy=true
for endpoint in "${HEALTH_ENDPOINTS[@]}"; do
    if curl -s -f -m 5 "$endpoint" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $endpoint is healthy${NC}"
    else
        echo -e "${RED}❌ $endpoint is not responding${NC}"
        all_healthy=false
    fi
done

# Summary
echo -e "\n📊 Deployment Summary"
echo "===================="
if [ "$all_healthy" = true ]; then
    echo -e "${GREEN}✅ All services deployed successfully!${NC}"
    echo -e "\nAccess KnowledgeHub at: http://${SYNOLOGY_HOST}:3100"
else
    echo -e "${YELLOW}⚠️  Some services may need attention${NC}"
fi

echo -e "\nDeployment completed at $(date)"