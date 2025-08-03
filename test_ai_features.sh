#\!/bin/bash

echo "=== KnowledgeHub AI Intelligence Feature Test Report ==="
echo "Date: $(date)"
echo "Base URL: http://192.168.1.25:3000"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    
    echo -n "Testing $name... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" -X GET "http://192.168.1.25:3000$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X POST "http://192.168.1.25:3000$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    http_code=$(echo "$response"  < /dev/null |  tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        echo -e "${GREEN}✓ SUCCESS${NC} (HTTP $http_code)"
        echo "Response: $(echo "$body" | jq -c . 2>/dev/null || echo "$body" | head -c 100)"
    elif [ "$http_code" = "404" ]; then
        echo -e "${YELLOW}⚠ NOT FOUND${NC} (HTTP $http_code)"
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $http_code)"
        echo "Error: $(echo "$body" | jq -c . 2>/dev/null || echo "$body" | head -c 100)"
    fi
    echo ""
}

echo "=== 1. Memory Operations ==="
test_endpoint "Create Memory" "POST" "/api/memory" '{
    "content": "Test memory from AI feature test",
    "type": "test",
    "metadata": {
        "test": true,
        "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
    }
}'

test_endpoint "Search Memory" "POST" "/api/memory/search" '{
    "query": "test",
    "limit": 5
}'

test_endpoint "Memory Stats" "GET" "/api/claude-auto/memory/stats"

echo "=== 2. Error Tracking and Learning ==="
test_endpoint "Track Error" "POST" "/api/mistake-learning/errors" '{
    "error_type": "TestError",
    "error_message": "This is a test error",
    "context": {
        "file": "test.py",
        "line": 42
    },
    "solution": "Fixed by adding error handling",
    "successful": true
}'

test_endpoint "Find Similar Errors" "POST" "/api/mistake-learning/find-similar" '{
    "error_message": "test error"
}'

test_endpoint "Get Lessons Learned" "GET" "/api/mistake-learning/lessons"

echo "=== 3. Decision Recording ==="
test_endpoint "Record Decision" "POST" "/api/decisions" '{
    "decision": "Use PostgreSQL for persistence",
    "reasoning": "Better for complex queries and relationships",
    "alternatives_considered": ["Redis", "MongoDB", "SQLite"],
    "context": {
        "project": "memory-system",
        "component": "storage"
    },
    "confidence": 0.9
}'

test_endpoint "Search Decisions" "POST" "/api/decisions/search" '{
    "query": "database",
    "limit": 5
}'

echo "=== 4. Performance Tracking ==="
test_endpoint "Track Performance" "POST" "/api/performance/track" '{
    "command": "pytest",
    "execution_time": 5.23,
    "success": true,
    "context": {
        "tests_run": 42,
        "tests_passed": 42
    }
}'

test_endpoint "Get Performance Stats" "GET" "/api/performance/stats"

test_endpoint "Get Recommendations" "GET" "/api/performance/recommendations"

echo "=== 5. Code Evolution Tracking ==="
test_endpoint "Track Code Change" "POST" "/api/code-evolution/track" '{
    "file_path": "/opt/projects/test.py",
    "change_type": "refactor",
    "description": "Extracted helper functions",
    "diff_summary": "Moved validation logic to separate functions",
    "metrics": {
        "lines_added": 50,
        "lines_removed": 30
    }
}'

test_endpoint "Get Code Patterns" "GET" "/api/code-evolution/patterns"

echo "=== 6. Pattern Recognition ==="
test_endpoint "Analyze Patterns" "POST" "/api/claude-auto/patterns/analyze" '{
    "context": "user frequently uses pytest for testing",
    "type": "workflow"
}'

test_endpoint "Get User Patterns" "GET" "/api/claude-auto/patterns"

echo "=== 7. Real-time Streaming ==="
echo -n "Testing SSE Endpoint... "
# Test SSE endpoint with timeout
timeout 2s curl -s -N "http://192.168.1.25:3000/api/claude-auto/stream/events" 2>&1 | head -n5
if [ $? -eq 124 ]; then
    echo -e "${GREEN}✓ STREAMING ACTIVE${NC}"
else
    echo -e "${RED}✗ STREAMING FAILED${NC}"
fi
echo ""

echo "=== 8. Search Functionality ==="
test_endpoint "Universal Search" "POST" "/api/search" '{
    "query": "memory system",
    "types": ["memory", "decision", "error"],
    "limit": 10
}'

echo "=== 9. Session Management ==="
test_endpoint "Get Session Info" "GET" "/api/memory/session/session-1752928403"

test_endpoint "Create Session Link" "POST" "/api/claude-auto/session/link" '{
    "from_session": "session-old",
    "to_session": "session-1752928403",
    "handoff_message": "Continuing work on memory system"
}'

test_endpoint "Get Session Context" "GET" "/api/memory/context/quick/default-user"

echo "=== 10. Task Predictions ==="
test_endpoint "Get Next Tasks" "GET" "/api/proactive/next-tasks"

test_endpoint "Get Task Suggestions" "POST" "/api/proactive/suggest" '{
    "context": "Working on memory system integration",
    "recent_actions": ["created API endpoints", "added tests"]
}'

echo ""
echo "=== Test Summary ==="
echo "Test completed at: $(date)"
