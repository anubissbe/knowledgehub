#!/bin/bash

# ProjectHub Proxy Deployment Script

set -e

echo "🚀 Deploying ProjectHub Proxy..."

# Stop existing proxy if running
echo "Stopping existing proxy..."
docker-compose down 2>/dev/null || true

# Build and start the proxy
echo "Building and starting proxy..."
docker-compose build
docker-compose up -d

# Wait for health check
echo "Waiting for proxy to be healthy..."
for i in {1..30}; do
    if curl -f http://localhost:3109/health > /dev/null 2>&1; then
        echo "✅ Proxy is healthy and ready!"
        break
    fi
    echo "Waiting for proxy to start... ($i/30)"
    sleep 2
done

# Test the proxy
echo "Testing proxy functionality..."
if curl -f http://localhost:3109/health > /dev/null 2>&1; then
    echo "✅ Proxy deployed successfully!"
    echo ""
    echo "🌐 Proxy is running at: http://localhost:3109"
    echo "📊 Health check: http://localhost:3109/health"
    echo "🔧 Use http://localhost:3109/api instead of http://192.168.1.24:3009/api"
    echo ""
    echo "Example usage:"
    echo "  curl -X PUT http://localhost:3109/api/tasks/TASK_ID \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"status\": \"completed\"}'"
else
    echo "❌ Proxy deployment failed!"
    exit 1
fi