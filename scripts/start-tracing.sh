#!/bin/bash

# KnowledgeHub Distributed Tracing Stack Startup Script
# Starts Jaeger, OpenTelemetry Collector, Tempo, and related tracing services

set -e

echo "🔍 Starting KnowledgeHub Distributed Tracing Stack..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating tracing directories..."
mkdir -p otel-collector
mkdir -p tempo
mkdir -p grafana-tracing/provisioning/datasources
mkdir -p grafana-tracing/provisioning/dashboards
mkdir -p grafana-tracing/dashboards

# Create networks
echo "🌐 Creating tracing networks..."
docker network create tracing 2>/dev/null || echo "Network 'tracing' already exists"
docker network create monitoring 2>/dev/null || echo "Network 'monitoring' already exists"

# Set proper permissions for volumes
echo "🔧 Setting up volume permissions..."
sudo chown -R 472:472 grafana-tracing/ 2>/dev/null || echo "⚠️  Could not set Grafana permissions"

# Start tracing stack
echo "🚀 Starting tracing services..."
docker-compose -f docker-compose.tracing.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to become healthy..."
sleep 45

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
check_service "Jaeger UI" "http://localhost:16686/"
check_service "OpenTelemetry Collector" "http://localhost:13133/"
check_service "Tempo" "http://localhost:3200/ready"
check_service "Zipkin" "http://localhost:9411/health"
check_service "Grafana Tracing" "http://localhost:3031/api/health"

echo ""
echo "🎉 Distributed tracing stack started successfully!"
echo ""
echo "🔍 Access URLs:"
echo "  Jaeger UI:         http://localhost:16686"
echo "  Grafana Tracing:   http://localhost:3031 (admin/admin123)"
echo "  Zipkin:            http://localhost:9411"
echo "  Tempo:             http://localhost:3200"
echo "  OTEL Collector:    http://localhost:13133"
echo ""
echo "📊 Tracing Endpoints:"
echo "  OTLP gRPC:         localhost:4317"
echo "  OTLP HTTP:         localhost:4318"
echo "  Jaeger gRPC:       localhost:14250"
echo "  Jaeger HTTP:       localhost:14268"
echo "  Zipkin HTTP:       localhost:9411"
echo ""
echo "📈 Key Features:"
echo "  - Distributed request tracing across all services"
echo "  - Performance bottleneck identification"
echo "  - Service dependency mapping"
echo "  - Real-time trace visualization"
echo "  - Memory operation performance tracking (<50ms target)"
echo "  - AI operation duration analysis"
echo "  - Database query performance monitoring"
echo ""
echo "🔧 Integration:"
echo "  Applications automatically send traces to:"
echo "  - Jaeger via localhost:14268 (HTTP) or localhost:14250 (gRPC)"
echo "  - OTLP via localhost:4317 (gRPC) or localhost:4318 (HTTP)"
echo ""
echo "📝 View traces:"
echo "  1. Open Jaeger UI: http://localhost:16686"
echo "  2. Select 'knowledgehub-api' service"
echo "  3. Click 'Find Traces' to see distributed traces"
echo ""
echo "🛑 To stop the tracing stack:"
echo "  ./scripts/stop-tracing.sh"
echo ""
echo "📋 View logs:"
echo "  docker-compose -f docker-compose.tracing.yml logs -f [service_name]"