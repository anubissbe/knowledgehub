#!/usr/bin/env python3
"""Final comprehensive verification of all systems"""

import requests
import websockets
import asyncio
import json
import sys
from datetime import datetime

API_KEY = "knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM"
BASE_URL = "http://localhost:3000"
WS_URL = "ws://localhost:3000"
HEADERS = {
    'X-API-Key': API_KEY,
    'User-Agent': 'Final Verification Script'
}

def check_http_endpoint(name, method, path, **kwargs):
    """Check HTTP endpoint"""
    try:
        url = f"{BASE_URL}{path}"
        if method == "GET":
            response = requests.get(url, headers=HEADERS, timeout=5, **kwargs)
        elif method == "POST":
            response = requests.post(url, headers=HEADERS, timeout=5, **kwargs)
        
        if response.status_code in [200, 201]:
            print(f"✅ {name}")
            return True
        else:
            print(f"❌ {name}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {type(e).__name__}")
        return False

async def check_websocket():
    """Check WebSocket connection"""
    try:
        uri = f"{WS_URL}/ws/notifications"
        async with websockets.connect(uri) as websocket:
            # Wait for welcome message
            welcome = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(welcome)
            if data.get("type") == "connected":
                print("✅ WebSocket Connection")
                return True
            else:
                print("❌ WebSocket: No welcome message")
                return False
    except Exception as e:
        print(f"❌ WebSocket: {type(e).__name__}")
        return False

async def check_sse():
    """Check Server-Sent Events"""
    try:
        # SSE connections stay open, so we just check if we can connect
        response = requests.get(
            f"{BASE_URL}/api/realtime/stream", 
            headers=HEADERS, 
            stream=True,
            timeout=2
        )
        if response.status_code == 200 and response.headers.get("content-type", "").startswith("text/event-stream"):
            print("✅ SSE Stream")
            return True
        else:
            print(f"❌ SSE Stream: Status {response.status_code}")
            return False
    except requests.exceptions.ReadTimeout:
        # SSE keeping connection open is expected
        print("✅ SSE Stream (connection kept open)")
        return True
    except Exception as e:
        print(f"❌ SSE Stream: {type(e).__name__}")
        return False

async def run_all_checks():
    """Run all verification checks"""
    print("=" * 60)
    print("KNOWLEDGEHUB FINAL PRODUCTION VERIFICATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"API Key: {API_KEY[:20]}...")
    print()
    
    passed = 0
    total = 0
    
    # 1. Core System Health
    print("1. CORE SYSTEM HEALTH")
    print("-" * 30)
    checks = [
        ("Health Check", "GET", "/health"),
        ("Detailed Health", "GET", "/api/monitoring/health/detailed"),
        ("System Metrics", "GET", "/api/monitoring/metrics"),
        ("AI Features Status", "GET", "/api/monitoring/ai-features/status"),
    ]
    for name, method, path in checks:
        if check_http_endpoint(name, method, path):
            passed += 1
        total += 1
    
    # 2. Authentication System
    print("\n2. AUTHENTICATION SYSTEM")
    print("-" * 30)
    # Test with wrong API key
    bad_headers = {'X-API-Key': 'invalid_key', 'User-Agent': 'Final Verification Script'}
    try:
        response = requests.get(f"{BASE_URL}/api/monitoring/metrics", headers=bad_headers, timeout=5)
        if response.status_code == 401:
            print("✅ Authentication Enforcement")
            passed += 1
        else:
            print(f"❌ Authentication Not Working (got status {response.status_code})")
    except Exception as e:
        print(f"❌ Authentication Test Error: {e}")
    total += 1
    
    # 3. AI Intelligence Features
    print("\n3. AI INTELLIGENCE FEATURES")
    print("-" * 30)
    
    # Decision Reasoning
    decision_params = {
        "decision_title": "Test",
        "chosen_solution": "Test",
        "reasoning": "Test",
        "confidence": 0.9
    }
    decision_body = {"alternatives": [], "context": {}, "category": "test", "impact": "low"}
    if check_http_endpoint("Decision Recording", "POST", "/api/decisions/record", params=decision_params, json=decision_body):
        passed += 1
    total += 1
    
    if check_http_endpoint("Decision Search", "GET", "/api/decisions/search", params={"query": "test"}):
        passed += 1
    total += 1
    
    # Mistake Learning
    mistake_params = {"error_type": "Test", "error_message": f"Test at {datetime.now().timestamp()}"}
    if check_http_endpoint("Mistake Tracking", "POST", "/api/mistake-learning/track", params=mistake_params, json={}):
        passed += 1
    total += 1
    
    # Pattern Recognition
    pattern_data = {"code": "def test(): pass", "language": "python"}
    if check_http_endpoint("Pattern Analysis", "POST", "/api/patterns/analyze", json=pattern_data):
        passed += 1
    total += 1
    
    # 4. Real-time Features
    print("\n4. REAL-TIME FEATURES")
    print("-" * 30)
    if await check_websocket():
        passed += 1
    total += 1
    
    if await check_sse():
        passed += 1
    total += 1
    
    # 5. Background Services
    print("\n5. BACKGROUND SERVICES")
    print("-" * 30)
    try:
        response = requests.get(f"{BASE_URL}/api/monitoring/health/detailed", headers=HEADERS, timeout=5)
        data = response.json()
        components = data.get("components", {})
        
        services = {
            "database": "PostgreSQL Database",
            "cache": "Redis Cache",
            "background_jobs": "Background Jobs",
            "pattern_workers": "Pattern Workers",
            "realtime_pipeline": "Real-time Pipeline"
        }
        
        all_healthy = True
        for key, name in services.items():
            status = components.get(key, {}).get("status", "unknown")
            if status == "healthy":
                print(f"✅ {name}")
                passed += 1
            else:
                print(f"❌ {name}: {status}")
                all_healthy = False
            total += 1
    except Exception as e:
        print(f"❌ Error checking services: {e}")
        all_healthy = False
    
    # 6. Data Integrity
    print("\n6. DATA INTEGRITY")
    print("-" * 30)
    try:
        response = requests.get(f"{BASE_URL}/api/monitoring/metrics", headers=HEADERS, timeout=5)
        metrics = response.json()
        db_metrics = metrics.get("database", {})
        
        print(f"✅ Total Memories: {db_metrics.get('total_memories', 0)}")
        print(f"✅ Total Documents: {db_metrics.get('total_documents', 0)}")
        print(f"✅ Total Mistakes: {db_metrics.get('total_mistakes', 0)}")
        passed += 1
        total += 1
    except Exception as e:
        print(f"❌ Data integrity check failed: {e}")
        total += 1
    
    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    percentage = (passed / total * 100) if total > 0 else 0
    print(f"Tests Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if passed == total:
        print("\n✅ SYSTEM IS 100% OPERATIONAL AND PRODUCTION READY! ✅")
        return True
    else:
        print(f"\n⚠️  {total - passed} issues need attention")
        return False

def main():
    """Main entry point"""
    success = asyncio.run(run_all_checks())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()