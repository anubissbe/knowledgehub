#!/usr/bin/env python3
"""
Test ACTUAL KnowledgeHub functionality - what really works vs what doesn't
"""

import httpx
import json
import time
import asyncio
import websockets
from datetime import datetime
from typing import Dict, Tuple

API_BASE = "http://192.168.1.25:3000"
USER_ID = "real_test_user"

class RealFunctionalityTester:
    def __init__(self):
        self.client = httpx.Client(base_url=API_BASE, timeout=30)
        self.results = {}
        
    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        
    def test_1_memory_creation_real(self) -> Tuple[str, float, str]:
        """Test 1: Real Memory Creation"""
        self.log("Testing REAL memory creation...")
        
        # Try multiple endpoint variations
        endpoints = [
            ("/api/v1/memories", {"memory_type": "technical"}),
            ("/api/memory/memories", {"type": "technical"}),
            ("/api/memories", {"type": "technical"}),
            ("/memories", {"type": "technical"})
        ]
        
        for endpoint, type_field in endpoints:
            try:
                payload = {
                    "user_id": USER_ID,
                    "content": f"Test memory at {endpoint}",
                    "tags": ["test"],
                    **type_field
                }
                response = self.client.post(endpoint, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return "✅", 1.0, f"Memory creation works at {endpoint}"
                elif response.status_code == 307:
                    # Follow redirect
                    location = response.headers.get("location", "")
                    if location:
                        response2 = self.client.post(location, json=payload)
                        if response2.status_code == 200:
                            return "✅", 1.0, f"Memory creation works via redirect to {location}"
            except:
                pass
                
        # Try local memory system
        try:
            local_response = httpx.post("http://localhost:8001/api/memory", json={
                "content": "Test local memory",
                "type": "technical",
                "user_id": USER_ID
            })
            if local_response.status_code == 200:
                return "⚠️", 0.5, "Only local memory system works (port 8001)"
        except:
            pass
            
        return "❌", 0.0, "No memory creation endpoints work"
        
    def test_2_memory_search_real(self) -> Tuple[str, float, str]:
        """Test 2: Real Memory Search"""
        self.log("Testing REAL memory search...")
        
        endpoints = [
            "/api/v1/search",
            "/api/memory/search", 
            "/api/memory/vector/search",
            "/api/memories/search"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.client.get(endpoint, params={
                    "user_id": USER_ID,
                    "query": "test"
                })
                if response.status_code == 200:
                    results = response.json()
                    if isinstance(results, list):
                        return "✅", 1.0, f"Search works at {endpoint} - {len(results)} results"
            except:
                pass
                
        return "❌", 0.0, "No search endpoints work"
        
    def test_3_session_management_real(self) -> Tuple[str, float, str]:
        """Test 3: Real Session Management"""
        self.log("Testing REAL session management...")
        
        # Check if sessions are tracked at all
        try:
            response = self.client.get("/api/memory/stats", params={"user_id": USER_ID})
            if response.status_code == 200:
                stats = response.json()
                if "sessions" in stats or "session_count" in stats:
                    return "⚠️", 0.5, "Session stats available but no session creation"
        except:
            pass
            
        return "❌", 0.0, "No session management functionality"
        
    def test_4_error_learning_real(self) -> Tuple[str, float, str]:
        """Test 4: Real Error Learning"""
        self.log("Testing REAL error learning...")
        
        # Check if error tracking exists anywhere
        endpoints = [
            "/api/mistake-learning/errors",
            "/api/errors",
            "/errors",
            "/api/v1/errors"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.client.post(endpoint, json={
                    "user_id": USER_ID,
                    "error_type": "TestError",
                    "error_message": "Test error"
                })
                if response.status_code in [200, 201]:
                    return "✅", 1.0, f"Error tracking works at {endpoint}"
            except:
                pass
                
        return "❌", 0.0, "No error learning functionality exists"
        
    def test_5_decision_recording_real(self) -> Tuple[str, float, str]:
        """Test 5: Real Decision Recording"""
        self.log("Testing REAL decision recording...")
        
        # The decision tables exist in the database, check if endpoints work
        endpoints = [
            "/api/decisions/record",
            "/api/decisions",
            "/decisions/record",
            "/decisions"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.client.post(endpoint, json={
                    "user_id": USER_ID,
                    "decision": "Test decision",
                    "reasoning": "Test reasoning",
                    "alternatives": ["A", "B"],
                    "confidence": 0.8
                })
                if response.status_code in [200, 201]:
                    return "✅", 1.0, f"Decision recording works at {endpoint}"
            except:
                pass
                
        return "❌", 0.0, "Decision recording endpoints not implemented"
        
    def test_6_code_evolution_real(self) -> Tuple[str, float, str]:
        """Test 6: Real Code Evolution"""
        self.log("Testing REAL code evolution...")
        
        # Check if code tracking exists
        endpoints = [
            "/api/code-evolution/track-change",
            "/api/code-evolution/changes",
            "/track-change",
            "/api/v1/code-changes"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.client.post(endpoint, json={
                    "user_id": USER_ID,
                    "file_path": "/test.py",
                    "change_type": "refactor",
                    "description": "Test change"
                })
                if response.status_code in [200, 201]:
                    return "✅", 1.0, f"Code evolution works at {endpoint}"
            except:
                pass
                
        return "❌", 0.0, "Code evolution not implemented"
        
    def test_7_proactive_predictions_real(self) -> Tuple[str, float, str]:
        """Test 7: Real Proactive Predictions"""
        self.log("Testing REAL proactive predictions...")
        
        # This is likely just mock data
        endpoints = [
            "/api/proactive/predict-tasks",
            "/predict-tasks",
            "/api/ai/predictions"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.client.get(endpoint, params={
                    "user_id": USER_ID,
                    "context": json.dumps({"test": "true"})
                })
                if response.status_code == 200:
                    data = response.json()
                    if data.get("tasks") and len(data["tasks"]) > 0:
                        # Check if it's real or mock
                        tasks = data["tasks"]
                        if all(isinstance(t, str) and len(t) > 10 for t in tasks[:3]):
                            return "⚠️", 0.5, "Returns data but likely mock predictions"
            except:
                pass
                
        return "❌", 0.0, "No proactive prediction functionality"
        
    def test_8_web_ui_real(self) -> Tuple[str, float, str]:
        """Test 8: Real Web UI Functionality"""
        self.log("Testing REAL Web UI...")
        
        try:
            # Check if UI connects to real API
            ui_response = httpx.get("http://192.168.1.25:3100/")
            if ui_response.status_code == 200:
                # Check API health
                api_health = self.client.get("/health")
                if api_health.status_code == 200:
                    health_data = api_health.json()
                    if health_data.get("services", {}).get("database") == "operational":
                        # Check if real data endpoints work
                        stats = self.client.get("/api/memory/stats", params={"user_id": USER_ID})
                        if stats.status_code == 200:
                            return "✅", 1.0, "Web UI works and connects to real API with database"
                        else:
                            return "⚠️", 0.5, "Web UI works but limited API functionality"
                return "⚠️", 0.5, "Web UI accessible but API partially functional"
        except:
            pass
            
        return "❌", 0.0, "Web UI not functional"
        
    async def test_9_websocket_real(self) -> Tuple[str, float, str]:
        """Test 9: Real WebSocket Functionality"""
        self.log("Testing REAL WebSocket...")
        
        try:
            async with websockets.connect("ws://192.168.1.25:3000/ws") as ws:
                await ws.send(json.dumps({"type": "ping"}))
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                return "⚠️", 0.5, "WebSocket connects but likely no real events"
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 403:
                return "❌", 0.0, "WebSocket returns 403 Forbidden"
        except:
            pass
            
        return "❌", 0.0, "WebSocket not functional"
        
    def test_10_cross_session_real(self) -> Tuple[str, float, str]:
        """Test 10: Real Cross-Session Continuity"""
        self.log("Testing REAL cross-session continuity...")
        
        # Check if context API works
        try:
            response = self.client.get(f"/api/memory/context/quick/{USER_ID}")
            if response.status_code == 200:
                context = response.json()
                if isinstance(context, dict) and len(context) > 0:
                    return "⚠️", 0.5, "Context retrieval works but no session linking"
        except:
            pass
            
        return "❌", 0.0, "No cross-session continuity functionality"
        
    async def run_all_tests(self):
        """Run all tests"""
        tests = [
            ("Memory Creation", self.test_1_memory_creation_real),
            ("Memory Search", self.test_2_memory_search_real),
            ("Session Management", self.test_3_session_management_real),
            ("Error Learning", self.test_4_error_learning_real),
            ("Decision Recording", self.test_5_decision_recording_real),
            ("Code Evolution", self.test_6_code_evolution_real),
            ("Proactive Predictions", self.test_7_proactive_predictions_real),
            ("Web UI", self.test_8_web_ui_real),
            ("WebSocket Events", self.test_9_websocket_real),
            ("Cross-session Continuity", self.test_10_cross_session_real)
        ]
        
        total_score = 0.0
        
        print("\n" + "="*80)
        print("KNOWLEDGEHUB REAL FUNCTIONALITY ASSESSMENT")
        print("="*80 + "\n")
        
        for name, test_func in tests:
            if asyncio.iscoroutinefunction(test_func):
                status, score, details = await test_func()
            else:
                status, score, details = test_func()
                
            total_score += score
            self.results[name] = {
                "status": status,
                "score": score,
                "details": details
            }
            
            print(f"{status} {name}: {details}")
            
        overall_percentage = (total_score / len(tests)) * 100
        
        print("\n" + "="*80)
        print("ACTUAL FUNCTIONALITY SUMMARY")
        print("="*80)
        
        print("\nWorking Features:")
        for name, result in self.results.items():
            if result["score"] >= 0.5:
                print(f"  - {name}: {result['score']*100:.0f}%")
                
        print("\nNon-Working Features:")
        for name, result in self.results.items():
            if result["score"] < 0.5:
                print(f"  - {name}: {result['score']*100:.0f}%")
                
        print(f"\nOVERALL ACTUAL FUNCTIONALITY: {overall_percentage:.1f}%")
        print("="*80)
        
        # Save report
        with open("knowledgehub_real_functionality_report.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "overall_percentage": overall_percentage,
                "results": self.results
            }, f, indent=2)
            
        return overall_percentage

async def main():
    tester = RealFunctionalityTester()
    return await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())