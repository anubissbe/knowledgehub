#!/usr/bin/env python3
"""Test proactive assistance endpoints"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:3001"

def test_endpoint(method, path, data=None, params=None):
    """Test an endpoint and print results"""
    url = f"{BASE_URL}{path}"
    print(f"\nTesting: {method} {path}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2)[:200] + "...")
        else:
            print("Error:", response.text[:200])
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Exception: {e}")
        return False

def main():
    """Test all proactive assistance endpoints"""
    
    print("=== Testing Proactive Assistance Endpoints ===")
    
    # Test health check
    test_endpoint("GET", "/api/proactive/health")
    
    # Test analyze endpoint (GET)
    test_endpoint("GET", "/api/proactive/analyze", params={
        "session_id": "test-session-123",
        "project_id": "knowledgehub"
    })
    
    # Test analyze endpoint (POST)
    test_endpoint("POST", "/api/proactive/analyze", data={
        "session_id": "test-session-123",
        "project_id": "knowledgehub",
        "context": {"current_file": "test.py"}
    })
    
    # Test brief endpoint
    test_endpoint("GET", "/api/proactive/brief", params={
        "session_id": "test-session-123"
    })
    
    # Test incomplete tasks
    test_endpoint("GET", "/api/proactive/incomplete-tasks", params={
        "session_id": "test-session-123"
    })
    
    # Test predictions
    test_endpoint("GET", "/api/proactive/predictions", params={
        "session_id": "test-session-123"
    })
    
    # Test reminders
    test_endpoint("GET", "/api/proactive/reminders", params={
        "session_id": "test-session-123"
    })
    
    # Test check interrupt
    test_endpoint("POST", "/api/proactive/check-interrupt", 
                 params={"action": "delete_file"},
                 data={"context": {"file": "important.py"}})
    
    # Test context
    test_endpoint("GET", "/api/proactive/context", params={
        "session_id": "test-session-123"
    })
    
    # Test suggestions (GET)
    test_endpoint("GET", "/api/proactive/suggestions", params={
        "session_id": "test-session-123",
        "limit": 3
    })
    
    # Test suggestions (POST)
    test_endpoint("POST", "/api/proactive/suggestions", data={
        "session_id": "test-session-123",
        "project_id": "knowledgehub"
    })

if __name__ == "__main__":
    main()