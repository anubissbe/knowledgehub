#!/usr/bin/env python3
"""Test with correct endpoint paths"""

import httpx
import json

API_BASE = "http://192.168.1.25:3000"
USER_ID = "test_user_correct"

client = httpx.Client(base_url=API_BASE, timeout=30)

print("Testing correct endpoint paths...")

# Test 1: Memory creation with v1 prefix
print("\n1. Testing memory creation (v1 prefix)...")
try:
    response = client.post("/api/v1/memories", json={
        "user_id": USER_ID,
        "content": "Test memory with correct endpoint",
        "memory_type": "technical",  # Using memory_type instead of type
        "tags": ["test"],
        "project": "test-project"
    })
    print(f"Memory creation: {response.status_code}")
    if response.status_code == 200:
        print(f"Memory ID: {response.json().get('id')}")
    else:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Search memories
print("\n2. Testing memory search...")
try:
    response = client.get("/api/v1/search", params={
        "user_id": USER_ID,
        "query": "test memory"
    })
    print(f"Memory search: {response.status_code}")
    if response.status_code == 200:
        print(f"Results: {len(response.json())} found")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Claude Auto (no prefix)
print("\n3. Testing claude-auto init-session...")
try:
    response = client.post("/init-session", json={
        "user_id": USER_ID,
        "metadata": {"test": "true"}
    })
    print(f"Claude auto init: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Mistake Learning (no prefix)
print("\n4. Testing mistake learning...")
try:
    response = client.post("/errors", json={
        "user_id": USER_ID,
        "error_type": "TestError",
        "error_message": "Test error message",
        "context": {"file": "test.py"},
        "solution": "Test solution"
    })
    print(f"Error tracking: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Test 5: Decision recording (no prefix)
print("\n5. Testing decision recording...")
try:
    response = client.post("/decisions/record", json={
        "user_id": USER_ID,
        "decision": "Test decision",
        "reasoning": "Test reasoning",
        "alternatives": ["option1", "option2"],
        "confidence": 0.8
    })
    print(f"Decision recording: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Test 6: Proactive predictions (no prefix)
print("\n6. Testing proactive predictions...")
try:
    response = client.get("/predict-tasks", params={
        "user_id": USER_ID,
        "context": json.dumps({"current_file": "test.py"})
    })
    print(f"Proactive predictions: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Test 7: Code evolution (no prefix)
print("\n7. Testing code evolution...")
try:
    response = client.post("/track-change", json={
        "user_id": USER_ID,
        "file_path": "/test.py",
        "change_type": "refactor",
        "description": "Test change"
    })
    print(f"Code evolution: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Test memory-specific endpoints
print("\n8. Testing memory-specific endpoints...")
try:
    # Memory stats
    response = client.get("/api/memory/stats", params={"user_id": USER_ID})
    print(f"Memory stats: {response.status_code}")
    
    # Memory search (different endpoint)
    response = client.get("/api/memory/search", params={
        "user_id": USER_ID,
        "query": "test"
    })
    print(f"Memory search (memory prefix): {response.status_code}")
    
    # Memory create (different endpoint)
    response = client.post("/api/memory/memories", json={
        "user_id": USER_ID,
        "content": "Test memory via memory endpoint",
        "type": "technical"
    })
    print(f"Memory create (memory prefix): {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

print("\nChecking available endpoints via OpenAPI...")
try:
    response = client.get("/openapi.json")
    if response.status_code == 200:
        openapi = response.json()
        paths = list(openapi.get("paths", {}).keys())
        print(f"\nFound {len(paths)} endpoints")
        
        # Show AI-related endpoints
        ai_endpoints = [p for p in paths if any(x in p for x in ["claude", "decision", "error", "mistake", "proactive", "evolution"])]
        print(f"\nAI-related endpoints ({len(ai_endpoints)}):")
        for ep in sorted(ai_endpoints)[:20]:
            print(f"  {ep}")
except Exception as e:
    print(f"Error getting OpenAPI: {e}")