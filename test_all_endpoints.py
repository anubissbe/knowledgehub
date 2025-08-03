#!/usr/bin/env python3
"""Comprehensive test script for KnowledgeHub endpoints"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Tuple, List

BASE_URL = "http://localhost:3000"

def test_endpoint(method: str, path: str, data: Dict = None, params: Dict = None) -> Tuple[bool, str]:
    """Test a single endpoint and return success status and response"""
    try:
        url = f"{BASE_URL}{path}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json"
        }
        
        if method == "GET":
            response = requests.get(url, params=params, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=5)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=5)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=5)
        else:
            return False, f"Unknown method: {method}"
        
        if response.status_code in [200, 201, 202]:
            return True, f"{response.status_code} - {len(response.text)} bytes"
        else:
            return False, f"{response.status_code} - {response.text[:100]}"
    
    except requests.exceptions.Timeout:
        return False, "TIMEOUT"
    except requests.exceptions.ConnectionError:
        return False, "CONNECTION_ERROR"
    except Exception as e:
        return False, str(e)


def main():
    """Run comprehensive tests on all endpoints"""
    
    print("=" * 80)
    print(f"KnowledgeHub Comprehensive Test Report - {datetime.now()}")
    print("=" * 80)
    print()
    
    # Define all endpoints to test
    test_cases = [
        # Health checks
        ("GET", "/health", None, None, "Health Check"),
        ("GET", "/api/health", None, None, "API Health"),
        
        # AI Intelligence Features
        ("POST", "/api/claude-auto/session-init", {"user_id": "test", "metadata": {}}, None, "Session Continuity - Init"),
        ("GET", "/api/claude-auto/context-restoration", None, {"user_id": "test"}, "Session Continuity - Restore"),
        ("POST", "/api/project-context/register", {"project_id": "test", "config": {}}, None, "Project Context - Register"),
        ("GET", "/api/project-context/test/summary", None, None, "Project Context - Summary"),
        ("POST", "/api/mistake-learning/track", {"error_type": "test", "context": {}, "resolution": "test"}, None, "Mistake Learning - Track"),
        ("GET", "/api/mistake-learning/similar", None, {"error_type": "test"}, "Mistake Learning - Similar"),
        ("GET", "/api/proactive/next-tasks", None, {"session_id": "test"}, "Proactive Assistance - Tasks"),
        ("POST", "/api/decisions/record", {"decision": "test", "alternatives": [], "reasoning": "test"}, None, "Decision Reasoning - Record"),
        ("GET", "/api/decisions/history", None, {"user_id": "test"}, "Decision Reasoning - History"),
        ("POST", "/api/code-evolution/track", {"file_path": "test.py", "changes": {}}, None, "Code Evolution - Track"),
        ("GET", "/api/code-evolution/history/test.py", None, None, "Code Evolution - History"),
        ("POST", "/api/performance/track", {"operation": "test", "metrics": {}}, None, "Performance Tracking - Track"),
        ("GET", "/api/performance/recommendations", None, {"operation": "test"}, "Performance Tracking - Recommend"),
        ("POST", "/api/claude-workflow/capture", {"workflow": "test", "steps": []}, None, "Workflow Integration - Capture"),
        ("GET", "/api/claude-workflow/patterns", None, {"user_id": "test"}, "Workflow Integration - Patterns"),
        
        # Search endpoints  
        ("POST", "/api/v1/search", {"query": "test", "filters": {}}, None, "Document Search"),
        ("POST", "/api/v1/search/unified", {"query": "test", "search_type": "keyword", "memory_user_id": "test"}, None, "Unified Search"),
        
        # Analytics endpoints (using correct path)
        ("GET", "/api/api/v1/analytics/performance", None, None, "Analytics - Performance"),
        ("GET", "/api/api/v1/analytics/trends", None, None, "Analytics - Trends"),
        
        # Memory endpoints
        ("POST", "/api/memory/session/start", {"user_id": "test"}, None, "Memory - Start Session"),
        ("POST", "/api/memory/create", {"session_id": "test", "content": "test memory"}, None, "Memory - Create"),
        ("GET", "/api/memory/session/test/memories", None, None, "Memory - List"),
        
        # Sources and Jobs
        ("GET", "/api/sources", None, None, "Sources - List"),
        ("GET", "/api/jobs", None, None, "Jobs - List"),
    ]
    
    # Track results
    total_tests = len(test_cases)
    successful_tests = 0
    failed_tests = []
    
    # Run tests
    for method, path, data, params, description in test_cases:
        print(f"Testing: {description}")
        print(f"  {method} {path}")
        
        success, response = test_endpoint(method, path, data, params)
        
        if success:
            print(f"  ✅ SUCCESS: {response}")
            successful_tests += 1
        else:
            print(f"  ❌ FAILED: {response}")
            failed_tests.append((description, path, response))
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
    print(f"Failed: {len(failed_tests)} ({len(failed_tests)/total_tests*100:.1f}%)")
    print()
    
    if failed_tests:
        print("FAILED TESTS:")
        print("-" * 80)
        for description, path, error in failed_tests:
            print(f"• {description} ({path})")
            print(f"  Error: {error}")
        print()
    
    # Feature status
    print("FEATURE STATUS:")
    print("-" * 80)
    
    ai_features = [
        "Session Continuity", "Project Context", "Mistake Learning", 
        "Proactive Assistance", "Decision Reasoning", "Code Evolution",
        "Performance Tracking", "Workflow Integration"
    ]
    
    for feature in ai_features:
        feature_tests = [t for t in test_cases if feature in t[4]]
        feature_success = sum(1 for t in feature_tests if test_endpoint(t[0], t[1], t[2], t[3])[0])
        if feature_tests:
            print(f"• {feature}: {feature_success}/{len(feature_tests)} working ({feature_success/len(feature_tests)*100:.0f}%)")
    
    print()
    print(f"Overall functionality: {successful_tests/total_tests*100:.1f}%")


if __name__ == "__main__":
    main()