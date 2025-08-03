#!/bin/bash

echo "🧪 Testing KnowledgeHub LAN Access..."
echo "======================================"

API_KEY="knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM"
API_URL="http://192.168.1.25:3000"
ORIGIN="http://192.168.1.25:3100"

echo -e "\n📡 Testing API Health..."
curl -s "$API_URL/health" | jq -r '.status' || echo "❌ Failed"

echo -e "\n🔍 Testing Sources API..."
curl -s -H "X-API-Key: $API_KEY" -H "Origin: $ORIGIN" "$API_URL/api/v1/sources/" | jq -r '.total // "Failed"' | xargs -I {} echo "✅ Found {} sources"

echo -e "\n🔍 Testing Search API..."
curl -s -H "X-API-Key: $API_KEY" -H "Origin: $ORIGIN" "$API_URL/api/public/search?q=react&limit=1" | jq -r '.results | length // "Failed"' | xargs -I {} echo "✅ Search returned {} results"

echo -e "\n🧠 Testing Memory Stats API..."
curl -s -H "X-API-Key: $API_KEY" -H "Origin: $ORIGIN" "$API_URL/api/claude-auto/memory/stats" | jq -r '.total_memories // "Failed"' | xargs -I {} echo "✅ Found {} memories"

echo -e "\n🤖 Testing AI Intelligence API..."
curl -s -H "X-API-Key: $API_KEY" -H "Origin: $ORIGIN" "$API_URL/api/claude-auto/session/current" | jq -r '.user_id // "Failed"' | xargs -I {} echo "✅ Session user: {}"

echo -e "\n📊 Testing Decision API..."
curl -s -H "X-API-Key: $API_KEY" -H "Origin: $ORIGIN" "$API_URL/api/decisions/categories" | jq -r '. | length // "Failed"' | xargs -I {} echo "✅ Found {} decision categories"

echo -e "\n✨ All tests complete!"