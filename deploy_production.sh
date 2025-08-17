#!/bin/bash
# KnowledgeHub Production Deployment Workflow

set -e

echo "🚀 KnowledgeHub Production Deployment Workflow"
echo "=============================================="

# Step 1: Pre-deployment validation
echo "📋 Step 1: Pre-deployment validation..."
python3 -c "
import os
required_files = ['.env.production', 'docker-compose.production.yml']
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f'Required file missing: {file}')
print('✅ Pre-deployment validation passed')
"

# Step 2: Container health preparation
echo "🔧 Step 2: Preparing container health..."
if [ -f "./fix_container_health.sh" ]; then
    ./fix_container_health.sh
    echo "✅ Container health preparation complete"
else
    echo "⚠️ Container health script not found"
fi

# Step 3: Deploy production stack
echo "🚀 Step 3: Deploying production stack..."
cp .env.production .env
docker-compose -f docker-compose.production.yml up -d
sleep 30  # Allow services to start

# Step 4: Health validation
echo "🔍 Step 4: Validating deployment health..."
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f -s http://192.168.1.25:3000/health > /dev/null; then
        echo "✅ Health check passed on attempt $attempt"
        break
    else
        echo "⏳ Health check failed, attempt $attempt/$max_attempts"
        sleep 10
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Health check failed after $max_attempts attempts"
    exit 1
fi

# Step 5: Production validation
echo "🎯 Step 5: Running production validation..."
if [ -f "./deploy_validate_rag.py" ]; then
    python3 deploy_validate_rag.py
    echo "✅ Production validation complete"
else
    echo "⚠️ Production validation script not found"
fi

# Step 6: Final status
echo "🏁 Deployment Complete!"
echo "========================"
echo "API: http://192.168.1.25:3000"
echo "WebUI: http://192.168.1.25:3100"
echo "Health: http://192.168.1.25:3000/health"
echo ""
echo "🎉 KnowledgeHub is now production ready!"
