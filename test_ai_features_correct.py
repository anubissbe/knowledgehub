#!/usr/bin/env python3
"""Test AI Intelligence features with correct endpoint paths"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:3000"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Content-Type": "application/json"
}

def test_endpoint(method, path, data=None, params=None):
    """Test an endpoint and return the result"""
    url = f"{BASE_URL}{path}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, headers=HEADERS, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=HEADERS, timeout=5)
        else:
            return None, f"Unknown method: {method}"
            
        return response.status_code, response.text
    except Exception as e:
        return None, str(e)

def main():
    print("Testing AI Intelligence Features with Correct Paths")
    print("=" * 60)
    
    # Test cases with corrected paths
    tests = [
        # Claude Auto endpoints
        ("GET", "/api/claude-auto/health", None, None, "Claude Auto Health"),
        ("POST", "/api/claude-auto/session/start", {"cwd": "/opt/projects/test"}, None, "Session Start"),
        ("POST", "/api/claude-auto/session/handoff", {"session_id": "test", "message": "Test handoff"}, None, "Session Handoff"),
        ("POST", "/api/claude-auto/error/record", {"error_type": "test", "message": "Test error", "context": {}}, None, "Error Record"),
        ("GET", "/api/claude-auto/error/similar", None, {"error_type": "test"}, "Similar Errors"),
        ("GET", "/api/claude-auto/tasks/predict", None, {"session_id": "test"}, "Predict Tasks"),
        ("GET", "/api/claude-auto/session/current", None, {"user_id": "test"}, "Current Session"),
        ("GET", "/api/claude-auto/memory/stats", None, {"user_id": "test"}, "Memory Stats"),
        
        # Check other routers
        ("GET", "/api/project-context/health", None, None, "Project Context Health"),
        ("GET", "/api/mistake-learning/health", None, None, "Mistake Learning Health"),
        ("GET", "/api/proactive/health", None, None, "Proactive Health"),
        ("GET", "/api/claude-workflow/health", None, None, "Workflow Health"),
    ]
    
    successful = 0
    failed = []
    
    for method, path, data, params, description in tests:
        print(f"\n{description}:")
        print(f"  {method} {path}")
        
        status, response = test_endpoint(method, path, data, params)
        
        if status and status in [200, 201]:
            print(f"  ✅ SUCCESS ({status})")
            try:
                response_data = json.loads(response)
                print(f"     Response: {json.dumps(response_data, indent=2)[:200]}...")
            except:
                print(f"     Response: {response[:200]}...")
            successful += 1
        else:
            print(f"  ❌ FAILED ({status})")
            print(f"     Error: {response[:200]}...")
            failed.append((description, path, status, response))
    
    print("\n" + "=" * 60)
    print(f"Summary: {successful}/{len(tests)} tests passed ({successful/len(tests)*100:.1f}%)")
    
    if failed:
        print("\nFailed tests:")
        for desc, path, status, error in failed:
            print(f"  - {desc} ({path}): {status}")

if __name__ == "__main__":
    main()