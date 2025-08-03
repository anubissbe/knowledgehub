#!/bin/bash

# KnowledgeHub Distributed Tracing Stack Shutdown Script
# Stops all tracing services gracefully

set -e

echo "🛑 Stopping KnowledgeHub Distributed Tracing Stack..."

# Stop all tracing services
echo "⏹️  Stopping tracing services..."
docker-compose -f docker-compose.tracing.yml down

# Optional: Remove volumes (uncomment if you want to clean up data)
# echo "🗑️  Removing tracing data volumes..."
# docker-compose -f docker-compose.tracing.yml down -v

echo ""
echo "✅ Distributed tracing stack stopped successfully!"
echo ""
echo "💡 To restart the tracing stack:"
echo "  ./scripts/start-tracing.sh"
echo ""
echo "🗑️  To completely remove tracing data:"
echo "  docker-compose -f docker-compose.tracing.yml down -v"
echo ""
echo "📝 Trace data is preserved in volumes for analysis"