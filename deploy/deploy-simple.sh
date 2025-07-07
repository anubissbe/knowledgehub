#!/bin/bash

# Simple deployment script for KnowledgeHub
# This version uses docker compose directly on Synology

set -e

echo "🚀 KnowledgeHub Simple Deployment"
echo "================================"

# Configuration
SYNOLOGY_HOST="192.168.1.24"
SYNOLOGY_USER="Bert"
SYNOLOGY_PORT="2222"
REMOTE_PATH="/volume1/docker/knowledgehub"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Pre-deployment backup
echo -e "\n📦 Creating backup..."
./scripts/backup.sh

# Copy files to Synology
echo -e "\n📋 Copying files to Synology..."
rsync -avz --progress \
    -e "ssh -p $SYNOLOGY_PORT" \
    --exclude 'data/' \
    --exclude 'logs/' \
    --exclude 'node_modules/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.env.local' \
    ./ $SYNOLOGY_USER@$SYNOLOGY_HOST:$REMOTE_PATH/

echo -e "${GREEN}✅ Files copied${NC}"

# Deploy on Synology
echo -e "\n🚀 Deploying on Synology..."
ssh -p $SYNOLOGY_PORT $SYNOLOGY_USER@$SYNOLOGY_HOST << 'ENDSSH'
cd /volume1/docker/knowledgehub

# Stop existing services
echo "Stopping existing services..."
docker compose down

# Pull latest images
echo "Pulling latest images..."
docker compose pull

# Start services
echo "Starting services..."
docker compose up -d

# Show status
echo ""
docker compose ps
ENDSSH

# Wait for services
echo -e "\n⏳ Waiting for services to start..."
sleep 30

# Health check
echo -e "\n🏥 Running health checks..."
./scripts/check-health.sh

echo -e "\n${GREEN}✅ Deployment complete!${NC}"
echo "Access KnowledgeHub at: http://${SYNOLOGY_HOST}:3100"