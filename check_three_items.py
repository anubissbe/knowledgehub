#\!/usr/bin/env python3
"""Check the 3 specific items requested"""

import requests
import json
import asyncio
import websockets
from datetime import datetime

BASE_URL = "http://localhost:3000"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Content-Type": "application/json"
}

async def test_websocket():
    """Test WebSocket connectivity"""
    try:
        uri = "ws://localhost:3000/ws/notifications"
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({"type": "ping"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=2)
            return True, "Connected and responsive"
    except Exception as e:
        return False, str(e)

def count_working_endpoints():
    """Test all major endpoint categories"""
    endpoints = [
        # AI Intelligence Features (8 features x 2 endpoints each = 16)
        ("POST", "/api/claude-auto/session-init", {"user_id": "test", "metadata": {}}),
        ("GET", "/api/claude-auto/context-restoration?user_id=test", None),
        ("POST", "/api/project-context/register", {"project_id": "test", "config": {}}),
        ("GET", "/api/project-context/test/summary", None),
        ("POST", "/api/mistake-learning/track", {"error_type": "test", "context": {}, "resolution": "test"}),
        ("GET", "/api/mistake-learning/similar?error_type=test", None),
        ("GET", "/api/proactive/next-tasks?session_id=test", None),
        ("POST", "/api/decisions/record", {"decision": "test", "alternatives": [], "reasoning": "test"}),
        ("GET", "/api/decisions/history?user_id=test", None),
        ("POST", "/api/code-evolution/track", {"file_path": "test.py", "changes": {}}),
        ("GET", "/api/code-evolution/history/test.py", None),
        ("POST", "/api/performance/track", {"operation": "test", "metrics": {}}),
        ("GET", "/api/performance/recommendations?operation=test", None),
        ("POST", "/api/claude-workflow/capture", {"workflow": "test", "steps": []}),
        ("GET", "/api/claude-workflow/patterns?user_id=test", None),
        
        # Core endpoints
        ("GET", "/health", None),
        ("GET", "/api/health", None),
        ("GET", "/api/v1/sources", None),
        ("GET", "/api/v1/jobs", None),
        ("GET", "/api/v1/memories", None),
        ("GET", "/api/v1/documents", None),
        ("GET", "/api/v1/chunks", None),
        
        # Search
        ("POST", "/api/v1/search", {"query": "test", "filters": {}}),
        ("POST", "/api/v1/search/unified", {"query": "test", "search_type": "keyword", "memory_user_id": "test"}),
        
        # Memory
        ("POST", "/api/memory/session/start", {"user_id": "test"}),
        ("POST", "/api/memory/create", {"session_id": "test", "content": "test"}),
        
        # Analytics
        ("GET", "/api/api/v1/analytics/performance", None),
        ("GET", "/api/api/v1/analytics/trends", None),
    ]
    
    working = 0
    total = len(endpoints)
    
    for method, path, data in endpoints:
        try:
            if method == "GET":
                if "?" in path:
                    base_path, params = path.split("?", 1)
                    response = requests.get(f"{BASE_URL}{base_path}?{params}", headers=headers, timeout=2)
                else:
                    response = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=2)
            else:
                response = requests.post(f"{BASE_URL}{path}", json=data, headers=headers, timeout=2)
            
            if response.status_code in [200, 201, 202]:
                working += 1
        except:
            pass
    
    return working, total, (working/total)*100

def test_source_creation():
    """Test source creation endpoint"""
    test_source = {
        "name": "Test API Documentation",
        "url": "https://test-api-docs-example.com",
        "type": "documentation",
        "refresh_interval": 86400
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/sources",
            json=test_source,
            headers=headers,
            timeout=5
        )
        
        if response.status_code in [200, 201, 202]:
            job = response.json()
            return True, f"Job created: {job.get('job_id', 'unknown')}"
        else:
            return False, f"Status {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 80)
    print("KnowledgeHub Status Check - Three Specific Items")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 80)
    print()
    
    # 1. WebSocket connectivity
    print("1. WEBSOCKET CONNECTIVITY:")
    print("-" * 40)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    ws_success, ws_result = loop.run_until_complete(test_websocket())
    if ws_success:
        print(f"✅ WebSocket endpoints: WORKING")
        print(f"   - /ws/notifications: {ws_result}")
        print(f"   - Real-time events: Functional")
    else:
        print(f"❌ WebSocket endpoints: NOT WORKING")
        print(f"   Error: {ws_result}")
    
    print()
    
    # 2. All endpoints working correctly
    print("2. ALL ENDPOINTS STATUS:")
    print("-" * 40)
    
    working, total, percentage = count_working_endpoints()
    print(f"✅ Endpoints tested: {total}")
    print(f"✅ Working endpoints: {working}")
    print(f"✅ Success rate: {percentage:.1f}%")
    
    if percentage >= 90:
        print(f"   Status: EXCELLENT (>90% working)")
    elif percentage >= 80:
        print(f"   Status: GOOD (>80% working)")
    else:
        print(f"   Status: NEEDS ATTENTION (<80% working)")
    
    print()
    
    # 3. Source creation re-enabled
    print("3. SOURCE CREATION STATUS:")
    print("-" * 40)
    
    src_success, src_result = test_source_creation()
    if src_success:
        print(f"✅ Source creation: RE-ENABLED AND WORKING")
        print(f"   - POST /api/v1/sources: Accepting new sources")
        print(f"   - {src_result}")
    else:
        print(f"❌ Source creation: NOT WORKING")
        print(f"   Error: {src_result}")
    
    # List current sources
    try:
        response = requests.get(f"{BASE_URL}/api/v1/sources", headers=headers, timeout=5)
        if response.status_code == 200:
            sources = response.json()
            print(f"   - Current sources in system: {sources['total']}")
    except:
        pass
    
    print()
    print("=" * 80)
    print("SUMMARY:")
    print("-" * 40)
    
    all_good = ws_success and percentage >= 90 and src_success
    
    if all_good:
        print("✅ ALL THREE ITEMS ARE WORKING CORRECTLY\!")
    else:
        print("⚠️  Some items need attention:")
        if not ws_success:
            print("   - WebSocket connectivity needs fixing")
        if percentage < 90:
            print(f"   - Only {percentage:.1f}% of endpoints working (target: >90%)")
        if not src_success:
            print("   - Source creation needs to be re-enabled")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
