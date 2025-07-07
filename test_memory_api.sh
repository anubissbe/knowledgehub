#!/bin/bash
# Test Memory System API Endpoints

API_URL="http://localhost:3000"
echo "Testing Memory System API..."

# Test 1: Start a session
echo -e "\n1. Starting a new session..."
SESSION_RESPONSE=$(curl -s -X POST "$API_URL/api/memory/session/start" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test@example.com",
    "project_id": "550e8400-e29b-41d4-a716-446655440000",
    "metadata": {"client": "test_script", "version": "1.0.0"},
    "tags": ["test", "development"]
  }')

if [[ $? -eq 0 ]]; then
  echo "✅ Session creation endpoint accessible"
  SESSION_ID=$(echo $SESSION_RESPONSE | jq -r '.id' 2>/dev/null)
  if [[ -n "$SESSION_ID" && "$SESSION_ID" != "null" ]]; then
    echo "✅ Session created: $SESSION_ID"
  else
    echo "❌ Failed to create session"
    echo "Response: $SESSION_RESPONSE"
  fi
else
  echo "❌ Cannot reach session endpoint"
fi

# Test 2: Create a memory
if [[ -n "$SESSION_ID" && "$SESSION_ID" != "null" ]]; then
  echo -e "\n2. Creating a memory..."
  MEMORY_RESPONSE=$(curl -s -X POST "$API_URL/api/memory/memories/" \
    -H "Content-Type: application/json" \
    -d "{
      \"session_id\": \"$SESSION_ID\",
      \"content\": \"User prefers functional components over class components\",
      \"memory_type\": \"preference\",
      \"importance\": 0.8,
      \"entities\": [\"React\", \"components\"]
    }")
  
  if [[ $? -eq 0 ]]; then
    echo "✅ Memory creation endpoint accessible"
    MEMORY_ID=$(echo $MEMORY_RESPONSE | jq -r '.id' 2>/dev/null)
    if [[ -n "$MEMORY_ID" && "$MEMORY_ID" != "null" ]]; then
      echo "✅ Memory created: $MEMORY_ID"
    else
      echo "❌ Failed to create memory"
      echo "Response: $MEMORY_RESPONSE"
    fi
  else
    echo "❌ Cannot reach memory endpoint"
  fi
fi

# Test 3: Search memories
echo -e "\n3. Testing memory search..."
SEARCH_RESPONSE=$(curl -s -X POST "$API_URL/api/memory/memories/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "React",
    "user_id": "test@example.com",
    "limit": 10
  }')

if [[ $? -eq 0 ]]; then
  echo "✅ Search endpoint accessible"
  RESULT_COUNT=$(echo $SEARCH_RESPONSE | jq '.total' 2>/dev/null)
  if [[ -n "$RESULT_COUNT" ]]; then
    echo "✅ Search returned $RESULT_COUNT results"
  else
    echo "❌ Search failed"
    echo "Response: $SEARCH_RESPONSE"
  fi
else
  echo "❌ Cannot reach search endpoint"
fi

# Test 4: Get user sessions
echo -e "\n4. Getting user sessions..."
SESSIONS_RESPONSE=$(curl -s "$API_URL/api/memory/session/user/test@example.com")

if [[ $? -eq 0 ]]; then
  echo "✅ User sessions endpoint accessible"
  SESSION_COUNT=$(echo $SESSIONS_RESPONSE | jq 'length' 2>/dev/null)
  if [[ -n "$SESSION_COUNT" ]]; then
    echo "✅ Found $SESSION_COUNT sessions for user"
  else
    echo "❌ Failed to get sessions"
    echo "Response: $SESSIONS_RESPONSE"
  fi
else
  echo "❌ Cannot reach user sessions endpoint"
fi

echo -e "\n✅ Memory API test complete!"