#!/usr/bin/env python3
"""Complete test of all AI Intelligence features with real data"""

import requests
import json
import time
import uuid
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

def test_claude_auto():
    """Test Claude Auto session management features"""
    print("\n=== Testing Claude Auto (Session Management) ===")
    
    # Start a session
    status, response = test_endpoint("POST", "/api/claude-auto/session/start", 
                                   {"cwd": "/opt/projects/knowledgehub"})
    print(f"Session Start: {'✅' if status == 200 else '❌'} ({status})")
    
    session_data = None
    if status == 200:
        session_data = json.loads(response)
        session_id = session_data.get("session", {}).get("session_id")
        print(f"  Session ID: {session_id}")
    
    # Record an error (with query params)
    status, response = test_endpoint("POST", "/api/claude-auto/error/record", 
                                   params={
                                       "error_type": "ImportError",
                                       "error_message": "No module named 'test_module'",
                                       "solution": "pip install test_module",
                                       "worked": "true"
                                   })
    print(f"Error Record: {'✅' if status == 200 else '❌'} ({status})")
    
    # Find similar errors
    status, response = test_endpoint("GET", "/api/claude-auto/error/similar",
                                   params={"error_message": "No module named"})
    print(f"Similar Errors: {'✅' if status == 200 else '❌'} ({status})")
    
    # Predict tasks
    if session_data:
        status, response = test_endpoint("GET", "/api/claude-auto/tasks/predict",
                                       params={"session_id": session_id})
        print(f"Predict Tasks: {'✅' if status == 200 else '❌'} ({status})")
    
    # Create handoff
    status, response = test_endpoint("POST", "/api/claude-auto/session/handoff",
                                   params={
                                       "content": "Fixed import errors and added session management",
                                       "next_tasks": ["Add unit tests", "Update documentation"],
                                       "unresolved_issues": ["WebSocket connection issues"]
                                   })
    print(f"Session Handoff: {'✅' if status == 200 else '❌'} ({status})")
    
    return session_data

def test_project_context():
    """Test Project Context features"""
    print("\n=== Testing Project Context ===")
    
    # Switch project
    status, response = test_endpoint("POST", "/api/project-context/switch",
                                   {"project_path": "/opt/projects/knowledgehub"})
    print(f"Switch Project: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get current project
    status, response = test_endpoint("GET", "/api/project-context/current")
    print(f"Current Project: {'✅' if status == 200 else '❌'} ({status})")
    
    # Add preference
    status, response = test_endpoint("POST", "/api/project-context/preference",
                                   {"preference_type": "code_style",
                                    "preference_value": "black formatter"})
    print(f"Add Preference: {'✅' if status == 200 else '❌'} ({status})")
    
    # Add pattern
    status, response = test_endpoint("POST", "/api/project-context/pattern",
                                   {"pattern_type": "import_order",
                                    "pattern_value": "stdlib, third-party, local"})
    print(f"Add Pattern: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get conventions
    status, response = test_endpoint("GET", "/api/project-context/conventions/knowledgehub")
    print(f"Get Conventions: {'✅' if status == 200 else '❌'} ({status})")

def test_mistake_learning():
    """Test Mistake Learning features"""
    print("\n=== Testing Mistake Learning ===")
    
    # Track a mistake
    status, response = test_endpoint("POST", "/api/mistake-learning/track",
                                   {"error_type": "TypeError",
                                    "context": {"file": "test.py", "line": 42},
                                    "resolution": "Added type checking",
                                    "lesson_learned": "Always validate input types"})
    print(f"Track Mistake: {'✅' if status == 200 else '❌'} ({status})")
    
    # Check action before doing it
    status, response = test_endpoint("POST", "/api/mistake-learning/check-action",
                                   {"action_type": "file_operation",
                                    "action_details": {"operation": "delete", "path": "/tmp/test"}})
    print(f"Check Action: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get lessons learned
    status, response = test_endpoint("GET", "/api/mistake-learning/lessons")
    print(f"Get Lessons: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get patterns
    status, response = test_endpoint("GET", "/api/mistake-learning/patterns")
    print(f"Get Patterns: {'✅' if status == 200 else '❌'} ({status})")
    
    # Search for similar mistakes
    status, response = test_endpoint("POST", "/api/mistake-learning/search",
                                   {"query": "type error",
                                    "limit": 5})
    print(f"Search Mistakes: {'✅' if status == 200 else '❌'} ({status})")

def test_proactive_assistant():
    """Test Proactive Assistant features"""
    print("\n=== Testing Proactive Assistant ===")
    
    # Analyze context for suggestions
    status, response = test_endpoint("POST", "/api/proactive/analyze",
                                   {"context": {"current_file": "test.py",
                                               "recent_actions": ["added function", "fixed import"]}})
    print(f"Analyze Context: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get suggestions
    status, response = test_endpoint("GET", "/api/proactive/suggestions",
                                   params={"context_type": "coding"})
    print(f"Get Suggestions: {'✅' if status == 200 else '❌'} ({status})")
    
    # Record action
    status, response = test_endpoint("POST", "/api/proactive/action",
                                   {"action": "accepted_suggestion",
                                    "suggestion_id": "test_123"})
    print(f"Record Action: {'✅' if status == 200 else '❌'} ({status})")

def test_decision_reasoning():
    """Test Decision Reasoning features"""
    print("\n=== Testing Decision Reasoning ===")
    
    # Record a decision
    status, response = test_endpoint("POST", "/api/decisions/record",
                                   {"decision_title": "Use PostgreSQL for persistence",
                                    "chosen_solution": "PostgreSQL with TimescaleDB",
                                    "alternatives": ["MongoDB", "Redis", "SQLite"],
                                    "reasoning": "Need time-series support and SQL queries",
                                    "confidence_score": 0.85,
                                    "tags": ["database", "architecture"]})
    print(f"Record Decision: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get decision history
    status, response = test_endpoint("GET", "/api/enhanced/decisions",
                                   params={"limit": 10})
    print(f"Decision History: {'✅' if status == 200 else '❌'} ({status})")
    
    # Analyze decision
    status, response = test_endpoint("GET", "/api/enhanced/decisions/analytics")
    print(f"Decision Analytics: {'✅' if status == 200 else '❌'} ({status})")

def test_code_evolution():
    """Test Code Evolution features"""
    print("\n=== Testing Code Evolution ===")
    
    # Track code change
    status, response = test_endpoint("POST", "/api/code-evolution/track",
                                   {"file_path": "test.py",
                                    "change_type": "refactor",
                                    "description": "Extract method for better readability",
                                    "before_snippet": "def long_function(): pass",
                                    "after_snippet": "def short_function(): pass"})
    print(f"Track Code Change: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get file history
    status, response = test_endpoint("GET", "/api/code-evolution/files/test.py/history")
    print(f"File History: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get patterns
    status, response = test_endpoint("GET", "/api/code-evolution/patterns/analytics")
    print(f"Pattern Analytics: {'✅' if status == 200 else '❌'} ({status})")

def test_claude_workflow():
    """Test Claude Workflow Integration"""
    print("\n=== Testing Claude Workflow Integration ===")
    
    # Capture conversation
    status, response = test_endpoint("POST", "/api/claude-workflow/capture-conversation",
                                   {"messages": [
                                       {"role": "user", "content": "How do I add caching?"},
                                       {"role": "assistant", "content": "Use Redis for caching"}
                                   ],
                                    "context": {"project": "knowledgehub"}})
    print(f"Capture Conversation: {'✅' if status == 200 else '❌'} ({status})")
    
    # Extract memories
    status, response = test_endpoint("POST", "/api/claude-workflow/extract-memories",
                                   {"conversation_id": str(uuid.uuid4()),
                                    "auto_categorize": True})
    print(f"Extract Memories: {'✅' if status == 200 else '❌'} ({status})")
    
    # Get patterns
    status, response = test_endpoint("GET", "/api/claude-workflow/patterns")
    print(f"Workflow Patterns: {'✅' if status == 200 else '❌'} ({status})")

def main():
    print("=" * 60)
    print("KnowledgeHub AI Intelligence Features - Complete Test")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    # Run all tests
    session_data = test_claude_auto()
    test_project_context()
    test_mistake_learning()
    test_proactive_assistant()
    test_decision_reasoning()
    test_code_evolution()
    test_claude_workflow()
    
    print("\n" + "=" * 60)
    print("Test Complete!")

if __name__ == "__main__":
    main()