#!/bin/bash
# Deploy the RAG system components

set -e

echo "🚀 Deploying KnowledgeHub RAG System..."

# Check if running from correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Must run from KnowledgeHub root directory"
    exit 1
fi

# Build secure hook executor
echo "🔨 Building secure hook executor..."
cd secure-hook-executor
docker build -t knowledgehub-hook-executor .
cd ..

# Start Qdrant if not already running
echo "🗄️  Starting Qdrant vector database..."
docker-compose up -d qdrant

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
timeout=30
while ! curl -f http://localhost:6333/health >/dev/null 2>&1; do
    sleep 1
    timeout=$((timeout - 1))
    if [ $timeout -eq 0 ]; then
        echo "❌ Qdrant failed to start"
        exit 1
    fi
done
echo "✅ Qdrant is ready"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Restart API service to load new modules
echo "🔄 Restarting API service..."
docker-compose restart api

# Wait for API to be ready
echo "⏳ Waiting for API to be ready..."
timeout=60
while ! curl -f http://localhost:3000/api/rag/health >/dev/null 2>&1; do
    sleep 1
    timeout=$((timeout - 1))
    if [ $timeout -eq 0 ]; then
        echo "❌ API failed to start with RAG module"
        exit 1
    fi
done

echo "✅ RAG system deployed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Test the RAG pipeline: curl -X POST http://localhost:3000/api/rag/test -H 'X-API-Key: your-key'"
echo "2. Start documentation scraping: curl -X POST http://localhost:3000/api/rag/scrape/schedule -H 'X-API-Key: admin-key'"
echo "3. Configure Claude Code with the settings in claude-code-config/settings.json"
echo ""
echo "📚 Documentation: See RAG_IMPLEMENTATION_GUIDE.md for full details"