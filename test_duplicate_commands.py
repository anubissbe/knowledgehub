#!/usr/bin/env python3
"""Test that identical commands are handled correctly"""

import requests
import time
import json

API_BASE = "http://localhost:3000"

def test_identical_commands():
    """Test tracking the exact same command multiple times"""
    
    print("Testing Identical Command Tracking...")
    print("=" * 60)
    
    # Use identical data for all requests
    identical_data = {
        "command_type": "git_status",
        "command_details": {"command": "git status", "directory": "/opt/projects"},
        "execution_time": 1.234,  # Same time
        "success": True,
        "output_size": 500,
        "context": {"test": "duplicate_handling"}  # Same context
    }
    
    memory_ids = []
    
    for i in range(3):
        print(f"\nAttempt {i+1}: Tracking identical command...")
        
        try:
            response = requests.post(
                f"{API_BASE}/api/performance/track",
                json=identical_data
            )
            
            if response.status_code == 200:
                result = response.json()
                memory_id = result.get('memory_id')
                memory_ids.append(memory_id)
                print(f"✅ Success: Memory ID = {memory_id}")
                print(f"   Execution ID = {result.get('execution_id')}")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Small delay
        time.sleep(0.1)
    
    print("\n" + "-" * 60)
    print("Results:")
    print(f"Total attempts: {len(memory_ids)}")
    print(f"Unique memory IDs: {len(set(memory_ids))}")
    
    if len(memory_ids) == len(set(memory_ids)):
        print("✅ Each tracking created a unique memory (as expected with timestamp in hash)")
    else:
        print("⚠️  Some memory IDs are duplicated (indicates reuse of existing records)")
    
    # Test workflow integration duplicate handling
    print("\n\nTesting Workflow Integration Duplicate Handling...")
    print("=" * 60)
    
    workflow_data = {
        "command": "test command",
        "status": "success",
        "exit_code": 0,
        "output": "test output",
        "execution_time": 1.5,
        "session_id": "test-session"
    }
    
    for i in range(3):
        print(f"\nAttempt {i+1}: Auto-capturing command...")
        
        try:
            response = requests.post(
                f"{API_BASE}/api/claude-workflow/auto-capture/command",
                json=workflow_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result.get('status')}")
                if 'memory_id' in result:
                    print(f"   Memory ID: {result['memory_id']}")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        time.sleep(0.1)

if __name__ == "__main__":
    test_identical_commands()