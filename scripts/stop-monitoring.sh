#!/bin/bash

# KnowledgeHub Monitoring Stack Shutdown Script
# Stops all monitoring services gracefully

set -e

echo "🛑 Stopping KnowledgeHub Monitoring Stack..."

# Stop all monitoring services
echo "⏹️  Stopping monitoring services..."
docker-compose -f docker-compose.monitoring.yml down

# Optional: Remove volumes (uncomment if you want to clean up data)
# echo "🗑️  Removing monitoring data volumes..."
# docker-compose -f docker-compose.monitoring.yml down -v

echo ""
echo "✅ Monitoring stack stopped successfully!"
echo ""
echo "💡 To restart the monitoring stack:"
echo "  ./scripts/start-monitoring.sh"
echo ""
echo "🗑️  To completely remove monitoring data:"
echo "  docker-compose -f docker-compose.monitoring.yml down -v"