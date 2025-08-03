#!/usr/bin/env python3
"""Final comprehensive test of AI Intelligence features with correct parameters"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:3000"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Content-Type": "application/json"
}

def test_endpoint(method, path, data=None, params=None):
    """Test an endpoint and return status and response"""
    url = f"{BASE_URL}{path}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, headers=HEADERS, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=HEADERS, timeout=5)
        else:
            return None, "Unknown method"
            
        return response.status_code, response.text
    except Exception as e:
        return None, str(e)

def main():
    print("=" * 80)
    print("KnowledgeHub AI Intelligence Features - Final Comprehensive Test")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 80)
    
    # Define all endpoints with correct parameters
    test_cases = [
        # Health checks
        ("GET", "/health", None, None, "Health Check"),
        
        # Claude Auto (Session Management) - Working
        ("GET", "/api/claude-auto/health", None, None, "Claude Auto Health"),
        ("POST", "/api/claude-auto/session/start", {"cwd": "/opt/projects/knowledgehub"}, None, "Session Start"),
        ("GET", "/api/claude-auto/session/current", None, {"user_id": "test"}, "Current Session"),
        ("GET", "/api/claude-auto/memory/stats", None, {"user_id": "test"}, "Memory Stats"),
        ("GET", "/api/claude-auto/tasks/predict", None, {"session_id": "test"}, "Predict Tasks"),
        
        # Project Context - Working
        ("GET", "/api/project-context/health", None, None, "Project Context Health"),
        ("POST", "/api/project-context/switch", {"project_path": "/opt/projects/knowledgehub"}, None, "Switch Project"),
        ("GET", "/api/project-context/current", None, None, "Current Project"),
        ("GET", "/api/project-context/conventions/knowledgehub", None, None, "Get Conventions"),
        
        # Mistake Learning - Working with correct params
        ("GET", "/api/mistake-learning/health", None, None, "Mistake Learning Health"),
        ("POST", "/api/mistake-learning/track", {
            "error_type": "ImportError",
            "error_message": "No module named 'test'",
            "context": {"file": "test.py", "line": 10},
            "successful_solution": "pip install test"
        }, None, "Track Mistake"),
        ("GET", "/api/mistake-learning/lessons", None, None, "Get Lessons"),
        ("GET", "/api/mistake-learning/patterns", None, None, "Get Patterns"),
        ("POST", "/api/mistake-learning/search", {"query": "import error", "limit": 5}, None, "Search Mistakes"),
        
        # Proactive Assistant - Working
        ("GET", "/api/proactive/health", None, None, "Proactive Health"),
        ("POST", "/api/proactive/analyze", {"context": {"current_file": "test.py"}}, None, "Analyze Context"),
        ("GET", "/api/proactive/suggestions", None, None, "Get Suggestions"),
        
        # Decision Reasoning - With new history endpoint
        ("GET", "/api/decisions/health", None, None, "Decisions Health"),
        ("POST", "/api/decisions/record", {
            "decision_title": "Use FastAPI for API",
            "chosen_solution": "FastAPI",
            "alternatives": ["Flask", "Django"],
            "reasoning": "Better performance and type hints",
            "confidence_score": 0.9,
            "tags": ["architecture", "framework"]
        }, None, "Record Decision"),
        ("GET", "/api/decisions/history", None, {"limit": 10}, "Decision History"),
        
        # Code Evolution - With new file history endpoint
        ("GET", "/api/code-evolution/health", None, None, "Code Evolution Health"),
        ("POST", "/api/code-evolution/track", {
            "file_path": "test.py",
            "change_type": "refactor",
            "description": "Extract method"
        }, None, "Track Code Change"),
        ("GET", "/api/code-evolution/files/test.py/history", None, None, "File History"),
        ("GET", "/api/code-evolution/patterns/analytics", None, None, "Pattern Analytics"),
        
        # Performance Tracking - Working
        ("POST", "/api/performance/track", {"operation": "api_call", "metrics": {"duration": 100}}, None, "Track Performance"),
        ("GET", "/api/performance/recommendations", None, {"operation": "api_call"}, "Performance Recommendations"),
        
        # Claude Workflow - With new endpoints
        ("GET", "/api/claude-workflow/health", None, None, "Workflow Health"),
        ("POST", "/api/claude-workflow/capture-conversation", {
            "messages": [{"role": "user", "content": "test"}],
            "context": {"project": "test"}
        }, None, "Capture Conversation"),
        ("POST", "/api/claude-workflow/extract-memories", {
            "conversation_id": "test-123",
            "auto_categorize": True
        }, None, "Extract Memories"),
        ("GET", "/api/claude-workflow/patterns", None, None, "Workflow Patterns"),
        
        # Search endpoints
        ("POST", "/api/v1/search", {"query": "test", "filters": {}}, None, "Document Search"),
        ("POST", "/api/v1/search/unified", {
            "query": "test",
            "search_type": "keyword",
            "memory_user_id": "test"
        }, None, "Unified Search"),
        
        # Analytics
        ("GET", "/api/api/v1/analytics/performance", None, None, "Analytics Performance"),
        ("GET", "/api/api/v1/analytics/trends", None, None, "Analytics Trends"),
        
        # Memory
        ("POST", "/api/memory/session/start", {"user_id": "test"}, None, "Memory Session Start"),
        
        # Sources and Jobs (correct paths)
        ("GET", "/api/v1/sources", None, None, "Sources List"),
        ("GET", "/api/v1/jobs", None, None, "Jobs List"),
    ]
    
    # Track results
    successful = 0
    failed = []
    
    for method, path, data, params, description in test_cases:
        status, response = test_endpoint(method, path, data, params)
        
        if status and status in [200, 201]:
            print(f"✅ {description}: SUCCESS ({status})")
            successful += 1
        else:
            print(f"❌ {description}: FAILED ({status})")
            failed.append((description, path, status, response[:100] if response else ""))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Successful: {successful} ({successful/len(test_cases)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(test_cases)*100:.1f}%)")
    
    if failed:
        print("\nFailed tests:")
        for desc, path, status, error in failed:
            print(f"  - {desc} ({path}): {status}")
            if error:
                print(f"    Error: {error}")
    
    # Feature breakdown
    print("\nFEATURE STATUS:")
    print("-" * 80)
    
    features = {
        "Session Management": ["Claude Auto", "Session Start", "Current Session", "Memory Stats", "Predict Tasks"],
        "Project Context": ["Project Context", "Switch Project", "Current Project", "Get Conventions"],
        "Mistake Learning": ["Mistake Learning", "Track Mistake", "Get Lessons", "Get Patterns", "Search Mistakes"],
        "Proactive Assistant": ["Proactive", "Analyze Context", "Get Suggestions"],
        "Decision Reasoning": ["Decisions", "Record Decision", "Decision History"],
        "Code Evolution": ["Code Evolution", "Track Code Change", "File History", "Pattern Analytics"],
        "Performance Tracking": ["Track Performance", "Performance Recommendations"],
        "Workflow Integration": ["Workflow", "Capture Conversation", "Extract Memories", "Workflow Patterns"]
    }
    
    for feature, keywords in features.items():
        feature_tests = [t for t in test_cases if any(k in t[4] for k in keywords)]
        feature_success = sum(1 for t in feature_tests if test_endpoint(t[0], t[1], t[2], t[3])[0] in [200, 201])
        if feature_tests:
            percentage = feature_success/len(feature_tests)*100
            status = "✅" if percentage == 100 else "⚠️" if percentage > 0 else "❌"
            print(f"{status} {feature}: {feature_success}/{len(feature_tests)} ({percentage:.0f}%)")

if __name__ == "__main__":
    # Wait for API to be ready
    time.sleep(3)
    main()