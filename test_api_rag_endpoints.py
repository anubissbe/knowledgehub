#!/usr/bin/env python3
"""
Test RAG API Endpoints directly 
Tests the RAG endpoints that might be available through the API
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

# API Configuration  
BASE_URL = "http://localhost:3000"
TIMEOUT = 10

def test_api_health():
    """Test basic API health"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}"
    except requests.RequestException as e:
        return False, str(e)

def test_rag_endpoints():
    """Test available RAG endpoints"""
    endpoints_to_test = [
        "/api/rag/health",
        "/api/rag/test", 
        "/api/rag/index/stats",
        "/docs",  # Check if swagger/docs are available
    ]
    
    results = {}
    
    for endpoint in endpoints_to_test:
        try:
            url = f"{BASE_URL}{endpoint}"
            print(f"Testing: {url}")
            
            if endpoint == "/api/rag/test":
                # This needs authentication, try without first
                response = requests.post(url, 
                    json={}, 
                    timeout=TIMEOUT,
                    headers={"Content-Type": "application/json"}
                )
            else:
                response = requests.get(url, timeout=TIMEOUT)
            
            if response.status_code == 200:
                results[endpoint] = {
                    "status": "success",
                    "status_code": response.status_code,
                    "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:500]
                }
                print(f"  ✅ {endpoint}: Success")
            else:
                results[endpoint] = {
                    "status": "failed",  
                    "status_code": response.status_code,
                    "response": response.text[:500]
                }
                print(f"  ❌ {endpoint}: HTTP {response.status_code}")
                
        except requests.RequestException as e:
            results[endpoint] = {
                "status": "error",
                "error": str(e)
            }
            print(f"  ❌ {endpoint}: {e}")
    
    return results

def test_weaviate_directly():
    """Test Weaviate directly"""
    try:
        # Test meta endpoint
        response = requests.get("http://localhost:8090/v1/meta", timeout=5)
        if response.status_code == 200:
            meta = response.json()
            print(f"✅ Weaviate accessible - Version: {meta.get('version', 'unknown')}")
            
            # Test schema
            schema_response = requests.get("http://localhost:8090/v1/schema", timeout=5)
            if schema_response.status_code == 200:
                schema = schema_response.json()
                classes = [cls["class"] for cls in schema.get("classes", [])]
                print(f"   Available classes: {classes}")
                return True, {"version": meta.get("version"), "classes": classes}
            else:
                return True, {"version": meta.get("version"), "classes": []}
        else:
            return False, f"HTTP {response.status_code}"
    except requests.RequestException as e:
        return False, str(e)

def check_postgres_via_api():
    """Check if we can access any database endpoints"""
    db_endpoints = [
        "/api/memories",
        "/api/memories/search",
        "/api/chunks"
    ]
    
    results = {}
    for endpoint in db_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=TIMEOUT)
            results[endpoint] = {
                "accessible": response.status_code != 404,
                "status_code": response.status_code
            }
            print(f"  {endpoint}: HTTP {response.status_code} ({'accessible' if response.status_code != 404 else 'not found'})")
        except requests.RequestException as e:
            results[endpoint] = {"accessible": False, "error": str(e)}
            print(f"  {endpoint}: Error - {e}")
    
    return results

def main():
    """Run all API-based tests"""
    print("🔍 KnowledgeHub RAG API Endpoints Test")
    print("=====================================")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: Basic API Health
    print("\n=== Testing Basic API Health ===")
    success, result = test_api_health()
    results["tests"]["api_health"] = {"success": success, "result": result}
    
    if success:
        print("✅ API is responding")
        
        # Test 2: RAG Endpoints
        print("\n=== Testing RAG Endpoints ===")
        rag_results = test_rag_endpoints()
        results["tests"]["rag_endpoints"] = rag_results
        
        # Test 3: Database Endpoints
        print("\n=== Testing Database Endpoints ===") 
        db_results = check_postgres_via_api()
        results["tests"]["database_endpoints"] = db_results
        
    else:
        print(f"❌ API not responding: {result}")
        print("   Cannot test RAG endpoints without API access")
    
    # Test 4: Direct Weaviate Access
    print("\n=== Testing Direct Weaviate Access ===")
    success, result = test_weaviate_directly()
    results["tests"]["weaviate_direct"] = {"success": success, "result": result}
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 API TEST RESULTS SUMMARY")  
    print(f"{'='*50}")
    
    for test_name, test_data in results["tests"].items():
        if isinstance(test_data, dict) and "success" in test_data:
            status = "✅ PASSED" if test_data["success"] else "❌ FAILED"
            print(f"{test_name:25} {status}")
        else:
            # For complex results like endpoints, count successes
            if isinstance(test_data, dict):
                successful = sum(1 for k, v in test_data.items() 
                               if isinstance(v, dict) and v.get("status") == "success")
                total = len(test_data)
                print(f"{test_name:25} {successful}/{total} endpoints accessible")
    
    # Save results
    with open("api_rag_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: api_rag_test_results.json")
    
    return results

if __name__ == "__main__":
    main()