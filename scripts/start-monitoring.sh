#!/bin/bash

# KnowledgeHub Monitoring Stack Startup Script
# Starts Prometheus, Grafana, AlertManager, and related monitoring services

set -e

echo "🔧 Starting KnowledgeHub Monitoring Stack..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating monitoring directories..."
mkdir -p prometheus/data
mkdir -p grafana/data
mkdir -p alertmanager/data
mkdir -p loki/data

# Set proper permissions
sudo chown -R 472:472 grafana/data 2>/dev/null || echo "⚠️  Could not set Grafana permissions (may need sudo)"
sudo chown -R 65534:65534 prometheus/data 2>/dev/null || echo "⚠️  Could not set Prometheus permissions (may need sudo)"

# Create network if it doesn't exist
echo "🌐 Creating monitoring network..."
docker network create monitoring 2>/dev/null || echo "Network 'monitoring' already exists"
docker network create knowledgehub-network 2>/dev/null || echo "Network 'knowledgehub-network' already exists"

# Start monitoring stack
echo "🚀 Starting monitoring services..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to become healthy..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

check_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo "Checking $service_name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        echo "⏳ Attempt $attempt/$max_attempts for $service_name..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start properly"
    return 1
}

# Check each service
check_service "Prometheus" "http://localhost:9090/-/healthy"
check_service "Grafana" "http://localhost:3030/api/health"
check_service "AlertManager" "http://localhost:9093/-/healthy"
check_service "Node Exporter" "http://localhost:9100/metrics"

echo ""
echo "🎉 Monitoring stack started successfully!"
echo ""
echo "📊 Access URLs:"
echo "  Prometheus:    http://localhost:9090"
echo "  Grafana:       http://localhost:3030 (admin/admin123)"
echo "  AlertManager:  http://localhost:9093"
echo "  Node Exporter: http://localhost:9100"
echo ""
echo "📈 Grafana Dashboards:"
echo "  - KnowledgeHub Overview: http://localhost:3030/d/knowledgehub-overview"
echo "  - AI Performance: http://localhost:3030/d/knowledgehub-ai-performance"
echo ""
echo "🔧 To stop the monitoring stack:"
echo "  ./scripts/stop-monitoring.sh"
echo ""
echo "📝 View logs:"
echo "  docker-compose -f docker-compose.monitoring.yml logs -f [service_name]"