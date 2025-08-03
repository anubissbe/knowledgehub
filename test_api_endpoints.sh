#!/bin/bash

echo "=== Testing KnowledgeHub API Endpoints ==="
echo "Base URL: http://192.168.1.25:3000"
echo "========================================="

API_KEY="knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM"
BASE_URL="http://192.168.1.25:3000"

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local description=$3
    local data=$4
    
    echo -e "\n📍 Testing: $description"
    echo "   Endpoint: $method $endpoint"
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" -H "X-API-Key: $API_KEY" "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d "$data" "$BASE_URL$endpoint")
    fi
    
    http_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" == "200" ] || [ "$http_code" == "201" ]; then
        echo "   ✅ Status: $http_code"
        echo "   Response preview: $(echo "$body" | head -c 200)..."
    else
        echo "   ❌ Status: $http_code"
        echo "   Error: $body"
    fi
}

# Test health endpoint
test_endpoint "GET" "/health" "Health Check" ""

# Test memory endpoints
test_endpoint "GET" "/api/memory/stats" "Memory Statistics" ""
test_endpoint "GET" "/api/memory/search?q=test&limit=5" "Memory Search" ""

# Test AI Intelligence endpoints
test_endpoint "GET" "/api/claude-auto/session/123" "Session Continuity" ""
test_endpoint "GET" "/api/proactive/suggestions/123?context=testing" "Proactive Suggestions" ""
test_endpoint "GET" "/api/performance/metrics/123" "Performance Metrics" ""
test_endpoint "GET" "/api/mistake-learning/patterns/123" "Error Patterns" ""

# Test project context
test_endpoint "GET" "/api/project-context/list/123" "Project List" ""

# Test sources
test_endpoint "GET" "/api/sources" "List Sources" ""

echo -e "\n========================================="
echo "Testing complete!"