#!/usr/bin/env python3
"""Test task endpoints"""

import requests
import json
import uuid

# Test /tasks endpoint (GET)
print("=== Testing /tasks (GET) ===")
try:
    response = requests.get("http://localhost:3001/tasks", headers={"User-Agent": "TestClient/1.0"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Testing /tasks (POST) ===")
# Create a test task
task_data = {
    "title": "Test Task",
    "description": "This is a test task",
    "priority": "high",
    "created_by": "test_user"
}

try:
    response = requests.post("http://localhost:3001/tasks", 
                            json=task_data, 
                            headers={"User-Agent": "TestClient/1.0"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        task = response.json()
        task_id = task.get('id')
        print(f"Created task ID: {task_id}")
        
        print(f"\n=== Testing /task/{task_id} (GET) ===")
        # Test get specific task
        response = requests.get(f"http://localhost:3001/task/{task_id}", 
                               headers={"User-Agent": "TestClient/1.0"})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")

print("\n=== Testing /tasks/stats ===")
try:
    response = requests.get("http://localhost:3001/tasks/stats", 
                           headers={"User-Agent": "TestClient/1.0"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")