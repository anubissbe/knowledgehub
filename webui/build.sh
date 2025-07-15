#!/bin/bash

# Build script for KnowledgeHub WebUI
# This ensures the React app is built before Docker image creation

set -e

echo "🚀 Building KnowledgeHub WebUI..."

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Build the React application
echo "🔨 Building React application..."
npm run build

echo "✅ Build complete! The build directory is ready for Docker."