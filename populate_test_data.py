#!/usr/bin/env python3
"""Populate KnowledgeHub with test data"""

import requests
import json
from datetime import datetime, timedelta
import random

BASE_URL = "http://localhost:3000"

def create_test_data():
    """Create test data for all AI features"""
    
    print("📝 Creating test data...")
    
    # 1. Create memories
    print("\n💾 Creating memories...")
    memories = [
        {
            "content": "Implemented new authentication system using JWT tokens",
            "memory_type": "implementation",
            "importance": 0.9,
            "tags": ["auth", "security", "jwt"]
        },
        {
            "content": "Fixed bug in user registration flow - email validation was failing",
            "memory_type": "bug_fix",
            "importance": 0.7,
            "tags": ["bug", "registration", "validation"]
        },
        {
            "content": "Optimized database queries for user search - 50% performance improvement",
            "memory_type": "optimization",
            "importance": 0.8,
            "tags": ["performance", "database", "optimization"]
        },
        {
            "content": "Added dark mode support to the UI with theme context",
            "memory_type": "feature",
            "importance": 0.6,
            "tags": ["ui", "theme", "dark-mode"]
        },
        {
            "content": "Refactored API error handling to use consistent response format",
            "memory_type": "refactoring",
            "importance": 0.7,
            "tags": ["api", "error-handling", "refactoring"]
        }
    ]
    
    for memory in memories:
        try:
            response = requests.post(f"{BASE_URL}/api/memory/create", json=memory)
            if response.status_code in [200, 201]:
                print(f"  ✅ Created memory: {memory['content'][:50]}...")
            else:
                print(f"  ❌ Failed to create memory: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error creating memory: {e}")
    
    # 2. Track mistakes
    print("\n🐛 Tracking mistakes...")
    mistakes = [
        {
            "error_type": "ImportError",
            "error_message": "No module named 'redis'",
            "context": {"file": "cache_service.py", "line": 5},
            "attempted_solution": "pip install redis",
            "successful_solution": "Added redis to requirements.txt and rebuilt container"
        },
        {
            "error_type": "DatabaseError",
            "error_message": "relation 'users' does not exist",
            "context": {"file": "user_service.py", "line": 42},
            "attempted_solution": "Check database connection",
            "successful_solution": "Run database migrations: alembic upgrade head"
        },
        {
            "error_type": "TypeError",
            "error_message": "Cannot read property 'map' of undefined",
            "context": {"file": "UserList.tsx", "line": 28},
            "attempted_solution": "Check if data exists",
            "successful_solution": "Add null check: data?.users?.map() || []"
        }
    ]
    
    for mistake in mistakes:
        try:
            response = requests.post(f"{BASE_URL}/api/mistakes/track", json=mistake)
            if response.status_code in [200, 201]:
                print(f"  ✅ Tracked mistake: {mistake['error_type']}")
            else:
                print(f"  ❌ Failed to track mistake: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error tracking mistake: {e}")
    
    # 3. Record decisions
    print("\n🎯 Recording decisions...")
    decisions = [
        {
            "decision": "Use PostgreSQL for main database",
            "reasoning": "Need ACID compliance and complex queries",
            "alternatives": ["MongoDB", "MySQL", "SQLite"],
            "confidence": 0.9
        },
        {
            "decision": "Implement authentication with JWT",
            "reasoning": "Stateless, scalable, and works well with microservices",
            "alternatives": ["Session-based auth", "OAuth only"],
            "confidence": 0.85
        },
        {
            "decision": "Use React with TypeScript for frontend",
            "reasoning": "Type safety and better developer experience",
            "alternatives": ["Vue.js", "Plain JavaScript", "Angular"],
            "confidence": 0.95
        }
    ]
    
    for decision in decisions:
        try:
            response = requests.post(f"{BASE_URL}/api/decisions/record", json=decision)
            if response.status_code in [200, 201]:
                print(f"  ✅ Recorded decision: {decision['decision']}")
            else:
                print(f"  ❌ Failed to record decision: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error recording decision: {e}")
    
    # 4. Track performance metrics
    print("\n📊 Tracking performance metrics...")
    metrics = [
        {
            "metric_name": "api_response_time",
            "value": 0.125,
            "unit": "seconds",
            "context": {"endpoint": "/api/users", "method": "GET"}
        },
        {
            "metric_name": "database_query_time",
            "value": 0.045,
            "unit": "seconds",
            "context": {"query": "SELECT * FROM users WHERE active = true"}
        },
        {
            "metric_name": "build_time",
            "value": 45.2,
            "unit": "seconds",
            "context": {"project": "frontend", "type": "production"}
        }
    ]
    
    for metric in metrics:
        try:
            response = requests.post(f"{BASE_URL}/api/performance/track", json=metric)
            if response.status_code in [200, 201]:
                print(f"  ✅ Tracked metric: {metric['metric_name']}")
            else:
                print(f"  ❌ Failed to track metric: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error tracking metric: {e}")
    
    # 5. Track code evolution
    print("\n📈 Tracking code evolution...")
    code_changes = [
        {
            "file_path": "src/services/auth.py",
            "change_type": "refactor",
            "description": "Extracted token validation to separate function",
            "metrics": {"lines_added": 25, "lines_removed": 10, "complexity_reduced": 3}
        },
        {
            "file_path": "src/components/UserProfile.tsx",
            "change_type": "feature",
            "description": "Added avatar upload functionality",
            "metrics": {"lines_added": 50, "lines_removed": 5, "new_features": 1}
        }
    ]
    
    for change in code_changes:
        try:
            response = requests.post(f"{BASE_URL}/api/code-evolution/track", json=change)
            if response.status_code in [200, 201]:
                print(f"  ✅ Tracked code change: {change['file_path']}")
            else:
                print(f"  ❌ Failed to track code change: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error tracking code change: {e}")
    
    print("\n✨ Test data creation complete!")
    
    # Test retrieval
    print("\n🔍 Testing data retrieval...")
    
    endpoints = [
        ("/api/memory/stats", "Memory stats"),
        ("/api/mistakes/patterns", "Mistake patterns"),
        ("/api/performance/report", "Performance report"),
        ("/api/decisions/suggest", "Decision suggestions"),
        ("/api/code-evolution/history", "Code evolution history")
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ {description}: {json.dumps(data, indent=2)[:100]}...")
            else:
                print(f"  ❌ {description}: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error fetching {description}: {e}")

if __name__ == "__main__":
    create_test_data()