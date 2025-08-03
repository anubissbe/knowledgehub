#!/usr/bin/env python3
"""Test KnowledgeHub API Endpoints"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:3000"

def test_endpoint(method, path, data=None, description=""):
    """Test an API endpoint"""
    url = f"{BASE_URL}{path}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Content-Type': 'application/json'
    }
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=5)
        
        status = "✅" if response.status_code in [200, 201] else "❌"
        print(f"{status} {method} {path}: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, dict):
                    keys = list(data.keys())[:3]
                    print(f"   Response keys: {keys}")
                elif isinstance(data, list):
                    print(f"   Response: List with {len(data)} items")
                return True
            except:
                print(f"   Response: {response.text[:100]}")
        else:
            print(f"   Error: {response.text[:100]}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ {method} {path}: {e}")
        return False

def main():
    print("🧪 Testing KnowledgeHub API Endpoints")
    print("=" * 60)
    
    # Test health endpoints
    print("\n📋 Health Endpoints:")
    endpoints = [
        ("GET", "/health"),
        ("GET", "/api/claude-auto/health"),
        ("GET", "/api/mistake-learning/health"),
        ("GET", "/api/proactive/health"),
        ("GET", "/api/performance/health"),
        ("GET", "/api/decisions/health"),
        ("GET", "/api/patterns/health"),
    ]
    
    for method, path in endpoints:
        test_endpoint(method, path)
    
    # Test memory endpoints
    print("\n💾 Memory Endpoints:")
    memory_endpoints = [
        ("GET", "/api/memory/memories/recent?limit=5"),  # Use correct endpoint
        ("GET", "/api/memory/session/active"),  # Use correct endpoint
        ("GET", "/api/memory/context/quick/test-user"),
    ]
    
    for method, path in memory_endpoints:
        test_endpoint(method, path)
    
    # Test AI feature endpoints
    print("\n🤖 AI Feature Endpoints:")
    ai_endpoints = [
        # Session management
        ("GET", "/api/claude-auto/session/current"),
        ("GET", "/api/claude-auto/memory/stats"),
        ("GET", "/api/claude-auto/tasks/predict"),
        
        # Mistake learning
        ("GET", "/api/mistake-learning/patterns"),
        ("GET", "/api/mistake-learning/lessons"),
        
        # Performance
        ("GET", "/api/performance/recommendations"),
        ("GET", "/api/performance/optimization-history"),
        
        # Decisions
        ("GET", "/api/decisions/confidence-report"),
        ("GET", "/api/decisions/search?query=test"),  # Add required query parameter
        ("GET", "/api/decisions/history"),  # Test the new history endpoint
        
        # Patterns
        ("GET", "/api/patterns/statistics"),  # Use correct endpoint
        ("GET", "/api/patterns/recent"),
        
        # Proactive
        ("GET", "/api/proactive/analyze?session_id=test"),  # Add required parameter
        ("GET", "/api/proactive/suggestions?session_id=test"),  # Use correct endpoint
        
        # Code Evolution 
        ("GET", "/api/code-evolution/files/test.py/history"),  # Test file history
        
        # Workflow
        ("GET", "/api/claude-workflow/patterns"),  # Test workflow patterns
    ]
    
    for method, path in ai_endpoints:
        test_endpoint(method, path)
    
    # Test creating sample data
    print("\n📝 Creating Sample Data:")
    
    # Track a mistake
    mistake_data = {
        "error_type": "TestError",
        "error_message": "This is a test error",
        "context": {"file": "test.py", "line": 42},
        "project_id": "test-project"
    }
    test_endpoint("POST", "/api/mistake-learning/track", mistake_data, "Track mistake")
    
    # Record a decision
    decision_data = {
        "decision_title": "Test Decision",
        "chosen_solution": "Use FastAPI for the API",
        "reasoning": "FastAPI provides async support and automatic documentation",
        "alternatives": ["Flask", "Django"],
        "confidence": 0.85
    }
    test_endpoint("POST", "/api/decisions/record", decision_data, "Record decision")
    
    # Capture workflow conversation
    workflow_data = {
        "messages": [
            {"role": "user", "content": "Fix the error in main.py"},
            {"role": "assistant", "content": "I found and fixed the ImportError"}
        ],
        "context": {"session_id": "test-session", "project": "test-project"}
    }
    test_endpoint("POST", "/api/claude-workflow/capture-conversation", workflow_data, "Capture workflow")
    
    print("\n" + "=" * 60)
    print("✨ API Endpoint Test Complete")

if __name__ == "__main__":
    main()