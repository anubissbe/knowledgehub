#!/usr/bin/env python3
"""Test KnowledgeHub API Endpoints"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:3000"

def test_endpoint(method, path, data=None, description=""):
    """Test an API endpoint"""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        
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
        ("GET", "/api/memory/memories"),
        ("GET", "/api/memory/memories?limit=5"),
        ("GET", "/api/memory/session/current"),
        ("GET", "/api/memory/context/quick/test-user"),
    ]
    
    for method, path in memory_endpoints:
        test_endpoint(method, path)
    
    # Test AI feature endpoints
    print("\n🤖 AI Feature Endpoints:")
    ai_endpoints = [
        # Session management
        ("GET", "/api/claude-auto/session/recent"),
        ("GET", "/api/claude-auto/session/current"),
        ("GET", "/api/claude-auto/memory/stats"),
        
        # Mistake learning
        ("GET", "/api/mistake-learning/patterns"),
        ("GET", "/api/mistake-learning/lessons"),
        
        # Performance
        ("GET", "/api/performance/recommendations"),
        ("GET", "/api/performance/optimization-history"),
        
        # Decisions
        ("GET", "/api/decisions/confidence-report"),
        ("GET", "/api/decisions/search"),
        
        # Patterns
        ("GET", "/api/patterns/analyzed"),
        ("GET", "/api/patterns/stats"),
        
        # Proactive
        ("GET", "/api/proactive/health"),
        ("GET", "/api/proactive/next-actions"),
    ]
    
    for method, path in ai_endpoints:
        test_endpoint(method, path)
    
    # Test creating sample data
    print("\n📝 Creating Sample Data:")
    
    # Create a memory
    memory_data = {
        "content": "Test memory from API verification",
        "memory_type": "test",
        "user_id": "test-user",
        "importance": 0.7,
        "tags": ["test", "api", "verification"]
    }
    test_endpoint("POST", "/api/memory/memories", memory_data, "Create memory")
    
    # Track a mistake
    mistake_data = {
        "error_type": "TestError",
        "error_message": "This is a test error",
        "context": {"file": "test.py", "line": 42},
        "project_id": "test-project"
    }
    test_endpoint("POST", "/api/mistake-learning/track", mistake_data, "Track mistake")
    
    print("\n" + "=" * 60)
    print("✨ API Endpoint Test Complete")

if __name__ == "__main__":
    main()