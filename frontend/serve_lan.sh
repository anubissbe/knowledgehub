#!/bin/bash
# Serve the production build on all interfaces for LAN access

cd /opt/projects/knowledgehub/frontend

echo "🚀 Starting KnowledgeHub frontend on LAN..."
echo "📡 Will be accessible at: http://192.168.1.25:3100"

# Use npx serve with host 0.0.0.0 to bind to all interfaces
npx serve -s dist -l tcp://0.0.0.0:3100

# Alternative: python simple server
# python3 -m http.server 3100 --bind 0.0.0.0 --directory dist