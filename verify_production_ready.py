#!/usr/bin/env python3
"""Comprehensive production readiness verification"""

import requests
import json
import sys

API_KEY = "knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM"
BASE_URL = "http://localhost:3000"
HEADERS = {
    'X-API-Key': API_KEY,
    'User-Agent': 'Production Verification Script'
}

def check_endpoint(name, method, path, **kwargs):
    """Check if an endpoint is working"""
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=5, **kwargs)
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{path}", headers=HEADERS, timeout=5, **kwargs)
        
        if response.status_code in [200, 201]:
            print(f"✅ {name}: Working")
            return True
        else:
            print(f"❌ {name}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {str(e)}")
        return False

def main():
    print("=== KnowledgeHub Production Readiness Verification ===\n")
    
    passed = 0
    total = 0
    
    # Core Health
    print("--- Core System Health ---")
    endpoints = [
        ("API Health", "GET", "/health"),
        ("Detailed Health", "GET", "/api/monitoring/health/detailed"),
        ("System Metrics", "GET", "/api/monitoring/metrics"),
        ("AI Features Status", "GET", "/api/monitoring/ai-features/status"),
    ]
    
    for name, method, path in endpoints:
        if check_endpoint(name, method, path):
            passed += 1
        total += 1
    
    # AI Intelligence Features
    print("\n--- AI Intelligence Features ---")
    
    # Session Continuity
    if check_endpoint("Session Current", "GET", "/api/claude-auto/session/current"):
        passed += 1
    total += 1
    
    if check_endpoint("Memory Stats", "GET", "/api/claude-auto/memory/stats"):
        passed += 1
    total += 1
    
    # Project Context
    if check_endpoint("Project List", "GET", "/api/project-context/list"):
        passed += 1
    total += 1
    
    # Mistake Learning
    if check_endpoint("Lessons", "GET", "/api/mistake-learning/lessons"):
        passed += 1
    total += 1
    
    if check_endpoint("Patterns", "GET", "/api/mistake-learning/patterns"):
        passed += 1
    total += 1
    
    # Decision Reasoning
    decision_data = {
        "alternatives": [],
        "context": {"test": True},
        "category": "testing",
        "impact": "low"
    }
    if check_endpoint("Decision Search", "GET", "/api/decisions/search", params={"query": "test"}):
        passed += 1
    total += 1
    
    # Code Evolution
    if check_endpoint("Code History", "GET", "/api/code-evolution/history", params={"file_path": "test.py"}):
        passed += 1
    total += 1
    
    # Performance Intelligence
    if check_endpoint("Performance Stats", "GET", "/api/performance/stats"):
        passed += 1
    total += 1
    
    if check_endpoint("Recommendations", "GET", "/api/performance/recommendations"):
        passed += 1
    total += 1
    
    # Pattern Recognition
    pattern_data = {"code": "def test(): pass", "language": "python"}
    if check_endpoint("Pattern Analysis", "POST", "/api/patterns/analyze", json=pattern_data):
        passed += 1
    total += 1
    
    # Real-time Features
    print("\n--- Real-time Features ---")
    if check_endpoint("WebSocket Health", "GET", "/ws/notifications"):
        passed += 1
    total += 1
    
    if check_endpoint("SSE Stream", "GET", "/api/realtime/stream"):
        passed += 1
    total += 1
    
    # Background Services
    print("\n--- Background Services ---")
    try:
        response = requests.get(f"{BASE_URL}/api/monitoring/health/detailed", headers=HEADERS, timeout=5)
        data = response.json()
        components = data.get("components", {})
        
        services = ["database", "cache", "background_jobs", "pattern_workers", "realtime_pipeline"]
        for service in services:
            status = components.get(service, {}).get("status", "unknown")
            if status == "healthy":
                print(f"✅ {service.replace('_', ' ').title()}: Healthy")
                passed += 1
            else:
                print(f"❌ {service.replace('_', ' ').title()}: {status}")
            total += 1
    except Exception as e:
        print(f"❌ Error checking background services: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Production Readiness: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    print(f"Status: {'✅ PRODUCTION READY' if passed == total else '⚠️  NEEDS ATTENTION'}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)