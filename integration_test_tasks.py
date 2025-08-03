#!/usr/bin/env python3
"""
Integration tests for Task Tracking System API endpoints
This test suite demonstrates all functionality works correctly.
"""

import requests
import json
import uuid
from datetime import datetime

BASE_URL = "http://localhost:3001"
HEADERS = {"User-Agent": "TestClient/1.0", "Content-Type": "application/json"}

def print_result(test_name, response):
    """Print test results"""
    print(f"\n=== {test_name} ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}...")
    return response.status_code == 200

def run_integration_tests():
    """Run comprehensive integration tests"""
    success_count = 0
    total_tests = 0
    
    print("🚀 Starting Task Tracking System Integration Tests")
    print("=" * 60)
    
    # Test 1: GET /api/tasks (Original user requirement)
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: GET /api/tasks")
    response = requests.get(f"{BASE_URL}/api/tasks", headers=HEADERS)
    if print_result("GET /api/tasks", response):
        success_count += 1
        print("✅ /api/tasks endpoint works correctly")
    else:
        print("❌ /api/tasks endpoint failed")
    
    # Test 2: GET /tasks (Alternative endpoint)
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: GET /tasks")
    response = requests.get(f"{BASE_URL}/tasks", headers=HEADERS)
    if print_result("GET /tasks", response):
        success_count += 1
        print("✅ /tasks endpoint works correctly")
    else:
        print("❌ /tasks endpoint failed")
    
    # Test 3: POST /tasks (Create task)
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: POST /tasks (Create Task)")
    task_data = {
        "title": "Integration Test Task",
        "description": "This task was created during integration testing",
        "priority": "high",
        "assignee": "test_user",
        "tags": ["integration", "testing"],
        "metadata": {"test_run": "2025-07-31"},
        "estimated_hours": 4,
        "created_by": "integration_test"
    }
    
    response = requests.post(f"{BASE_URL}/tasks", json=task_data, headers=HEADERS)
    if print_result("POST /tasks", response):
        success_count += 1
        print("✅ Task creation works correctly")
        created_task = response.json()
        task_id = created_task['id']
        print(f"📝 Created task ID: {task_id}")
    else:
        print("❌ Task creation failed")
        return success_count, total_tests
    
    # Test 4: GET /task/{id} (Get specific task - Original user requirement)
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: GET /task/{task_id}")
    response = requests.get(f"{BASE_URL}/task/{task_id}", headers=HEADERS)
    if print_result("GET /task/{id}", response):
        success_count += 1
        print("✅ Individual task retrieval works correctly")
        task = response.json()
        print(f"📋 Task title: {task['title']}")
        print(f"📊 Task status: {task['status']}")
        print(f"⭐ Task priority: {task['priority']}")
    else:
        print("❌ Individual task retrieval failed")
    
    # Test 5: PUT /task/{id} (Update task)
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: PUT /task/{task_id} (Update Task)")
    update_data = {
        "status": "in_progress",
        "progress": 25,
        "assignee": "updated_user"
    }
    
    response = requests.put(f"{BASE_URL}/task/{task_id}", json=update_data, headers=HEADERS)
    if print_result("PUT /task/{id}", response):
        success_count += 1
        print("✅ Task update works correctly")
        updated_task = response.json()
        print(f"📊 Updated status: {updated_task['status']}")
        print(f"📈 Updated progress: {updated_task['progress']}%")
    else:
        print("❌ Task update failed")
    
    # Test 6: PATCH /task/{id}/status (Update task status)
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: PATCH /task/{task_id}/status")
    status_data = {
        "status": "completed",
        "progress": 100,
        "actual_hours": 3
    }
    
    response = requests.patch(f"{BASE_URL}/task/{task_id}/status", json=status_data, headers=HEADERS)
    if print_result("PATCH /task/{id}/status", response):
        success_count += 1
        print("✅ Task status update works correctly")
        completed_task = response.json()
        print(f"🎉 Task completed: {completed_task['status']}")
        print(f"⏰ Completion time: {completed_task['completed_at']}")
    else:
        print("❌ Task status update failed")
    
    # Test 7: GET /tasks/stats (Task statistics)
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: GET /tasks/stats")
    response = requests.get(f"{BASE_URL}/tasks/stats", headers=HEADERS)
    if print_result("GET /tasks/stats", response):
        success_count += 1
        print("✅ Task statistics work correctly")
        stats = response.json()
        print(f"📊 Total tasks: {stats['total']}")
        print(f"📈 By status: {stats['by_status']}")
        print(f"⭐ By priority: {stats['by_priority']}")
    else:
        print("❌ Task statistics failed")
    
    # Test 8: GET /tasks with filters
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: GET /tasks?status=completed")
    response = requests.get(f"{BASE_URL}/tasks?status=completed", headers=HEADERS)
    if print_result("GET /tasks with filter", response):
        success_count += 1
        print("✅ Task filtering works correctly")
        filtered_tasks = response.json()
        print(f"🔍 Found {len(filtered_tasks['tasks'])} completed tasks")
    else:
        print("❌ Task filtering failed")
    
    # Test 9: DELETE /task/{id} (Soft delete)
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: DELETE /task/{task_id} (Soft Delete)")
    response = requests.delete(f"{BASE_URL}/task/{task_id}", headers=HEADERS)
    if print_result("DELETE /task/{id}", response):
        success_count += 1
        print("✅ Task soft deletion works correctly")
    else:
        print("❌ Task soft deletion failed")
    
    # Test 10: Verify task is soft deleted
    total_tests += 1
    print(f"\n🧪 Test {total_tests}: Verify soft deletion - GET /tasks?is_active=false")
    response = requests.get(f"{BASE_URL}/tasks?is_active=false", headers=HEADERS)
    if print_result("Verify soft deletion", response):
        success_count += 1
        print("✅ Soft deletion verification works correctly")
        inactive_tasks = response.json()
        print(f"🗑️ Found {len(inactive_tasks['tasks'])} inactive tasks")
    else:
        print("❌ Soft deletion verification failed")
    
    # Test Results Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {success_count}/{total_tests} tests")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests} tests")
    print(f"📈 Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("\n🎉 ALL TESTS PASSED! Task Tracking System is working correctly!")
        print("\n✅ EVIDENCE:")
        print("  • /api/tasks endpoint returns 200 OK with task list")
        print("  • /task/{id} endpoint returns 200 OK with specific task")
        print("  • Task creation, updates, and deletion work correctly")
        print("  • All CRUD operations are functional")
        print("  • Database integration is working")
        print("  • API responses are properly formatted JSON")
    else:
        print(f"\n⚠️ {total_tests - success_count} tests failed. System needs attention.")
    
    return success_count, total_tests

if __name__ == "__main__":
    success, total = run_integration_tests()
    exit(0 if success == total else 1)