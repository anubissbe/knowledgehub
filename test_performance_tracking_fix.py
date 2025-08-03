#!/usr/bin/env python3
"""Test script to verify the performance tracking duplicate key fix"""

import sys
import requests
import time
import json
from datetime import datetime

API_BASE = "http://localhost:3000"

def test_performance_tracking():
    """Test that performance tracking handles duplicates correctly"""
    
    print("Testing Performance Tracking Duplicate Key Fix...")
    print("=" * 60)
    
    # Track the same command multiple times to test duplicate handling
    command_type = "test_command"
    command_details = {"action": "test", "target": "duplicate_fix"}
    
    results = []
    
    for i in range(5):
        print(f"\nTest {i+1}: Tracking performance metrics...")
        
        # Track performance
        data = {
            "command_type": command_type,
            "command_details": command_details,
            "execution_time": 1.5 + (i * 0.1),  # Slightly different times
            "success": True,
            "output_size": 1000,
            "context": {
                "test_run": i + 1,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            response = requests.post(
                f"{API_BASE}/api/performance/track",
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append(result)
                print(f"✅ Success: {result}")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Test batch tracking
    print("\n\nTesting Batch Performance Tracking...")
    print("-" * 60)
    
    batch_data = []
    for i in range(3):
        batch_data.append({
            "command_type": f"batch_command_{i}",
            "command_details": {"batch": True, "index": i},
            "execution_time": 2.0 + (i * 0.5),
            "success": True,
            "context": {"batch_test": True}
        })
    
    try:
        response = requests.post(
            f"{API_BASE}/api/performance/track/batch",
            json={"metrics": batch_data}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch tracking success: {result}")
        else:
            print(f"❌ Batch tracking error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch tracking exception: {e}")
    
    # Get performance report
    print("\n\nGetting Performance Report...")
    print("-" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/api/performance/report")
        
        if response.status_code == 200:
            report = response.json()
            print(f"✅ Performance Report:")
            print(f"   Total commands: {report['summary']['total_commands']}")
            print(f"   Success rate: {report['summary'].get('success_rate', 0):.2%}")
            print(f"   Average execution time: {report['summary']['average_execution_time']:.2f}s")
            print(f"   Categories: {dict(report['summary']['categories'])}")
        else:
            print(f"❌ Report error: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Report exception: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    
    # Summary
    if results:
        print(f"\nSummary: Successfully tracked {len(results)} performance metrics")
        print("The duplicate key constraint issue should be resolved.")
    else:
        print("\nNo successful tracking - please check the API is running")

if __name__ == "__main__":
    test_performance_tracking()