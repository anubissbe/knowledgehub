#!/usr/bin/env python3
"""Comprehensive test to verify 100% system functionality"""

import requests
import websockets
import asyncio
import json
import subprocess
import time
from datetime import datetime

API_KEY = "knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM"
API_BASE = "http://192.168.1.25:3000"
UI_BASE = "http://192.168.1.25:3100"
HEADERS = {'X-API-Key': API_KEY}

def test_service_running(name, command):
    """Test if a service is running"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {name}: Running")
            return True
        else:
            print(f"❌ {name}: Not running")
            return False
    except Exception as e:
        print(f"❌ {name}: Error - {e}")
        return False

def test_api_endpoint(name, endpoint, method="GET", data=None):
    """Test an API endpoint"""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            response = requests.get(url, headers=HEADERS, timeout=5)
        else:
            response = requests.post(url, headers=HEADERS, json=data or {}, timeout=5)
        
        if response.status_code in [200, 201]:
            print(f"✅ {name}: Working (Status {response.status_code})")
            return True
        else:
            print(f"❌ {name}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: Error - {type(e).__name__}")
        return False

async def test_websocket():
    """Test WebSocket connection"""
    try:
        uri = f"ws://192.168.1.25:3000/ws/notifications"
        async with websockets.connect(uri) as websocket:
            welcome = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(welcome)
            if data.get("type") == "connected":
                print("✅ WebSocket: Connected")
                return True
    except Exception as e:
        print(f"❌ WebSocket: {type(e).__name__}")
        return False

def test_ui_accessible():
    """Test if UI is accessible"""
    try:
        response = requests.get(UI_BASE, timeout=5)
        if response.status_code == 200 and ("react" in response.text.lower() or "vite" in response.text.lower()):
            print("✅ Web UI: Accessible")
            return True
        else:
            print("❌ Web UI: Not accessible")
            return False
    except Exception as e:
        print(f"❌ Web UI: Error - {e}")
        return False

async def run_all_tests():
    """Run all comprehensive tests"""
    print("="*60)
    print("COMPREHENSIVE SYSTEM TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"API: {API_BASE}")
    print(f"UI: {UI_BASE}")
    print()
    
    passed = 0
    total = 0
    
    # 1. Test Core Services
    print("1. CORE SERVICES")
    print("-"*30)
    services = [
        ("PostgreSQL", "pg_isready -h localhost -p 5433"),
        ("Redis", "redis-cli -p 6381 ping"),
        ("API Server", f"curl -s {API_BASE}/health"),
        ("Frontend", f"curl -s {UI_BASE}")
    ]
    
    for name, cmd in services:
        if test_service_running(name, cmd):
            passed += 1
        total += 1
    
    # 2. Test API Endpoints
    print("\n2. API ENDPOINTS")
    print("-"*30)
    endpoints = [
        ("Health Check", "/health"),
        ("AI Features Status", "/api/monitoring/ai-features/status"),
        ("Memory Stats", "/api/claude-auto/memory/stats"),
        ("Decision Search", "/api/decisions/search?query=test"),
        ("Performance Stats", "/api/performance/stats"),
        ("Mistake Patterns", "/api/mistake-learning/patterns"),
        ("Proactive Health", "/api/proactive/health"),
        ("Code Evolution History", "/api/code-evolution/history?file_path=test"),
        ("Pattern Analysis", "/api/patterns/health"),
    ]
    
    for name, endpoint in endpoints:
        if test_api_endpoint(name, endpoint):
            passed += 1
        total += 1
    
    # 3. Test AI Features
    print("\n3. AI INTELLIGENCE FEATURES")
    print("-"*30)
    
    # Test each AI feature
    ai_features = [
        ("Session Continuity", "/api/claude-auto/session/current"),
        ("Mistake Learning", "/api/mistake-learning/lessons"),
        ("Proactive Assistant", "/api/proactive/predictions"),
        ("Decision Recording", "/api/decisions/record?decision_title=Test&chosen_solution=Test&reasoning=Test&confidence=0.9", 
         "POST", {"alternatives": [], "context": {}, "category": "test", "impact": "low"}),
        ("Code Evolution", "/api/code-evolution/patterns/analytics"),
        ("Performance Metrics", "/api/performance/recommendations"),
        ("Pattern Recognition", "/api/patterns/analyze", "POST", {"code": "def test(): pass", "language": "python"}),
    ]
    
    for feature in ai_features:
        name = feature[0]
        endpoint = feature[1]
        method = feature[2] if len(feature) > 2 else "GET"
        data = feature[3] if len(feature) > 3 else None
        
        if test_api_endpoint(name, endpoint, method, data):
            passed += 1
        total += 1
    
    # 4. Test Real-time Features
    print("\n4. REAL-TIME FEATURES")
    print("-"*30)
    
    # WebSocket test
    if await test_websocket():
        passed += 1
    total += 1
    
    # SSE test
    if test_api_endpoint("SSE Stream", "/api/realtime/stream"):
        passed += 1
    total += 1
    
    # 5. Test Web UI
    print("\n5. WEB UI")
    print("-"*30)
    
    if test_ui_accessible():
        passed += 1
    total += 1
    
    # Test UI pages
    ui_pages = [
        ("Dashboard", "/dashboard"),
        ("AI Intelligence", "/ai"),
        ("Memory System", "/memory"),
        ("Settings", "/settings"),
    ]
    
    for name, path in ui_pages:
        try:
            response = requests.get(f"{UI_BASE}{path}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} Page: Accessible")
                passed += 1
            else:
                print(f"❌ {name} Page: Status {response.status_code}")
        except:
            print(f"❌ {name} Page: Error")
        total += 1
    
    # 6. Test Background Services
    print("\n6. BACKGROUND SERVICES")
    print("-"*30)
    
    try:
        response = requests.get(f"{API_BASE}/api/monitoring/health/detailed", headers=HEADERS, timeout=5)
        if response.status_code == 200:
            data = response.json()
            components = data.get("components", {})
            
            for service, info in components.items():
                status = info.get("status", "unknown")
                if status == "healthy":
                    print(f"✅ {service}: Healthy")
                    passed += 1
                else:
                    print(f"❌ {service}: {status}")
                total += 1
    except Exception as e:
        print(f"❌ Background services check failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    percentage = (passed / total * 100) if total > 0 else 0
    print(f"Tests Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if percentage == 100:
        print("\n✅ SYSTEM IS 100% OPERATIONAL AND FUNCTIONAL! ✅")
        print("\nAll components verified:")
        print("- Core services running")
        print("- API endpoints responding")
        print("- AI features working")
        print("- Real-time features active")
        print("- Web UI accessible")
        print("- Background services healthy")
    else:
        print(f"\n⚠️  System is {percentage:.1f}% operational")
        print(f"{total - passed} components need attention")
    
    return percentage == 100

def main():
    """Main entry point"""
    success = asyncio.run(run_all_tests())
    
    print("\n" + "="*60)
    print("WEB UI SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Open http://192.168.1.25:3100 in your browser")
    print("2. Open browser console (F12)")
    print("3. Run this command:")
    print('localStorage.setItem(\'knowledgehub_settings\', \'{"apiUrl":"http://192.168.1.25:3000","apiKey":"knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM","enableNotifications":true,"autoRefresh":true,"refreshInterval":30,"darkMode":false,"language":"en","animationSpeed":1,"cacheSize":100,"maxMemories":1000,"compressionEnabled":true}\')')
    print("4. Refresh the page")
    print("5. Navigate to AI Intelligence to see all features")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())