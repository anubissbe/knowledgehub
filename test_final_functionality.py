#!/usr/bin/env python3
"""
Final comprehensive test of KnowledgeHub functionality with correct endpoints
"""

import httpx
import json
import time
import asyncio
import websockets
from datetime import datetime
from typing import Dict, Tuple
import uuid

API_BASE = "http://192.168.1.25:3000"
USER_ID = f"test_user_{uuid.uuid4().hex[:8]}"

class FinalFunctionalityTester:
    def __init__(self):
        self.client = httpx.Client(base_url=API_BASE, timeout=30)
        self.results = {}
        self.test_data = {}
        
    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        
    def test_1_memory_creation(self) -> Tuple[str, float, str]:
        """Test 1: Memory Creation with Embeddings"""
        self.log("Testing Memory Creation...")
        
        # Try the v1 memories endpoint with correct field name
        try:
            response = self.client.post("/api/v1/memories", json={
                "user_id": USER_ID,
                "content": "Test memory for functionality assessment",
                "memory_type": "technical",  # This is the correct field name
                "tags": ["test", "assessment"],
                "project": "knowledgehub-test",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "test": True
                }
            }, follow_redirects=True)
            
            if response.status_code == 200:
                data = response.json()
                self.test_data["memory_id"] = data.get("id")
                
                # Check if embeddings were generated
                if data.get("embedding_generated") or data.get("has_embedding"):
                    return "✅", 1.0, f"Memory created with embeddings (ID: {data.get('id')})"
                else:
                    return "⚠️", 0.5, "Memory created but no embeddings confirmation"
            else:
                return "❌", 0.0, f"Memory creation failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_2_memory_search(self) -> Tuple[str, float, str]:
        """Test 2: Memory Search - Vector Similarity"""
        self.log("Testing Memory Search...")
        
        # Create a few test memories first
        memories = [
            "Python error handling best practices",
            "JavaScript async/await patterns",
            "Database indexing strategies"
        ]
        
        for content in memories:
            self.client.post("/api/v1/memories", json={
                "user_id": USER_ID,
                "content": content,
                "memory_type": "technical",
                "project": "knowledgehub-test"
            }, follow_redirects=True)
            
        time.sleep(2)  # Wait for indexing
        
        # Try search
        try:
            response = self.client.get("/api/v1/search", params={
                "user_id": USER_ID,
                "query": "Python error handling",
                "limit": 5
            }, follow_redirects=True)
            
            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list) and len(results) > 0:
                    # Check if most relevant result contains Python
                    if any("Python" in r.get("content", "") for r in results[:2]):
                        return "✅", 1.0, f"Vector search working - found {len(results)} relevant results"
                    else:
                        return "⚠️", 0.5, "Search returns results but relevance unclear"
                else:
                    return "❌", 0.0, "Search returns empty results"
            else:
                return "❌", 0.0, f"Search failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_3_session_management(self) -> Tuple[str, float, str]:
        """Test 3: Session Management"""
        self.log("Testing Session Management...")
        
        try:
            # Start a session using claude-auto
            response = self.client.post("/api/claude-auto/session/start", json={
                "cwd": "/opt/projects/knowledgehub-test",
                "user_id": USER_ID
            })
            
            if response.status_code == 200:
                data = response.json()
                self.test_data["session_id"] = data.get("session", {}).get("session_id")
                
                # Check if context was restored
                context = data.get("context", {})
                if context and data.get("instructions", {}).get("context_restored"):
                    # Try to link a memory to this session
                    memory_response = self.client.post("/api/v1/memories", json={
                        "user_id": USER_ID,
                        "content": "Session-linked test memory",
                        "memory_type": "session",
                        "session_id": self.test_data["session_id"],
                        "project": "knowledgehub-test"
                    }, follow_redirects=True)
                    
                    if memory_response.status_code == 200:
                        return "✅", 1.0, f"Session management working - ID: {self.test_data['session_id']}"
                    else:
                        return "⚠️", 0.5, "Session created but memory linking failed"
                else:
                    return "⚠️", 0.5, "Session created but no context restoration"
            else:
                return "❌", 0.0, f"Session creation failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_4_error_learning(self) -> Tuple[str, float, str]:
        """Test 4: Error Learning"""
        self.log("Testing Error Learning...")
        
        # Check if mistake_learning router is available
        try:
            # Try the health endpoint first
            health = self.client.get("/api/mistake-learning/health")
            if health.status_code == 404:
                return "❌", 0.0, "Error learning endpoints not implemented"
                
            # Try to record an error
            response = self.client.post("/api/mistake-learning/errors", json={
                "user_id": USER_ID,
                "error_type": "ImportError",
                "error_message": "No module named 'test_module'",
                "context": {
                    "file": "test.py",
                    "line": 5,
                    "code": "import test_module"
                },
                "solution": "pip install test_module",
                "tags": ["python", "import", "dependency"]
            })
            
            if response.status_code in [200, 201]:
                self.test_data["error_id"] = response.json().get("id")
                return "✅", 1.0, "Error learning working - errors tracked with solutions"
            else:
                return "❌", 0.0, f"Error tracking failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, "Error learning not implemented"
            
    def test_5_decision_recording(self) -> Tuple[str, float, str]:
        """Test 5: Decision Recording"""
        self.log("Testing Decision Recording...")
        
        try:
            # Check if decision endpoints exist
            response = self.client.post("/api/decisions", json={
                "user_id": USER_ID,
                "decision": "Use PostgreSQL for database",
                "reasoning": "Need JSONB support and full-text search",
                "alternatives": ["MongoDB", "MySQL", "SQLite"],
                "context": {
                    "project": "knowledgehub-test",
                    "factors": ["scalability", "features", "performance"]
                },
                "confidence": 0.9,
                "tags": ["database", "architecture"]
            })
            
            if response.status_code in [200, 201]:
                self.test_data["decision_id"] = response.json().get("id")
                return "✅", 1.0, "Decision recording working"
            elif response.status_code == 404:
                return "❌", 0.0, "Decision recording endpoints not implemented"
            else:
                return "❌", 0.0, f"Decision recording failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, "Decision recording not available"
            
    def test_6_code_evolution(self) -> Tuple[str, float, str]:
        """Test 6: Code Evolution Tracking"""
        self.log("Testing Code Evolution...")
        
        try:
            # Check if code evolution endpoints exist
            response = self.client.post("/api/code-evolution/changes", json={
                "user_id": USER_ID,
                "file_path": "/test/example.py",
                "change_type": "refactor",
                "description": "Refactored database connection pooling",
                "before_snippet": "conn = psycopg2.connect(DB_URL)",
                "after_snippet": "conn = pool.getconn()",
                "impact": "40% performance improvement",
                "tags": ["performance", "database"]
            })
            
            if response.status_code in [200, 201]:
                self.test_data["code_change_id"] = response.json().get("id")
                return "✅", 1.0, "Code evolution tracking working"
            elif response.status_code == 404:
                return "❌", 0.0, "Code evolution endpoints not implemented"
            else:
                return "❌", 0.0, f"Code evolution failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, "Code evolution not available"
            
    def test_7_proactive_predictions(self) -> Tuple[str, float, str]:
        """Test 7: Proactive AI Predictions"""
        self.log("Testing Proactive Predictions...")
        
        try:
            # Check if proactive endpoints exist
            response = self.client.get("/api/proactive/predict-tasks", params={
                "user_id": USER_ID,
                "context": json.dumps({
                    "current_file": "test.py",
                    "recent_actions": ["created function", "added docstring"],
                    "project": "knowledgehub-test"
                })
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get("tasks") and isinstance(data["tasks"], list):
                    # Check if predictions are contextual
                    tasks = data["tasks"]
                    if len(tasks) > 0:
                        return "✅", 1.0, f"Proactive predictions working - {len(tasks)} suggestions"
                    else:
                        return "⚠️", 0.5, "Predictions endpoint works but no suggestions"
                else:
                    return "⚠️", 0.5, "Predictions endpoint returns invalid format"
            elif response.status_code == 404:
                return "❌", 0.0, "Proactive predictions not implemented"
            else:
                return "❌", 0.0, f"Predictions failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, "Proactive predictions not available"
            
    def test_8_web_ui(self) -> Tuple[str, float, str]:
        """Test 8: Web UI Real Data Display"""
        self.log("Testing Web UI...")
        
        try:
            # Check UI accessibility
            ui_response = httpx.get("http://192.168.1.25:3100/")
            if ui_response.status_code != 200:
                return "❌", 0.0, "Web UI not accessible"
                
            # Check critical API endpoints the UI needs
            working_endpoints = 0
            total_endpoints = 0
            
            endpoints = [
                ("/api/memory/stats", {"user_id": USER_ID}),
                ("/health", {}),
                ("/api/v1/sources", {}),
                ("/api/claude-auto/health", {})
            ]
            
            for endpoint, params in endpoints:
                total_endpoints += 1
                try:
                    resp = self.client.get(endpoint, params=params)
                    if resp.status_code == 200:
                        working_endpoints += 1
                except:
                    pass
                    
            if working_endpoints == total_endpoints:
                return "✅", 1.0, "Web UI fully functional with all API endpoints"
            elif working_endpoints > 0:
                percentage = working_endpoints / total_endpoints
                return "⚠️", percentage, f"Web UI partially working ({working_endpoints}/{total_endpoints} endpoints)"
            else:
                return "❌", 0.0, "Web UI accessible but API not functional"
                
        except Exception as e:
            return "❌", 0.0, f"Web UI error: {str(e)}"
            
    async def test_9_websocket_events(self) -> Tuple[str, float, str]:
        """Test 9: WebSocket Real-time Events"""
        self.log("Testing WebSocket Events...")
        
        try:
            # Try to connect with authentication
            headers = {
                "Authorization": "Bearer test_token"
            }
            
            async with websockets.connect(
                "ws://192.168.1.25:3000/ws",
                extra_headers=headers
            ) as ws:
                # Send subscribe message
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "user_id": USER_ID,
                    "events": ["memory_created", "error_tracked"]
                }))
                
                # Wait for confirmation
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                data = json.loads(response)
                
                if data.get("type") == "subscription_confirmed":
                    return "✅", 1.0, "WebSocket working with event subscription"
                else:
                    return "⚠️", 0.5, "WebSocket connects but subscription unclear"
                    
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 403:
                return "❌", 0.0, "WebSocket requires authentication (403)"
            else:
                return "❌", 0.0, f"WebSocket error: {e.status_code}"
        except Exception as e:
            return "❌", 0.0, f"WebSocket not functional: {str(e)}"
            
    def test_10_cross_session_continuity(self) -> Tuple[str, float, str]:
        """Test 10: Cross-session Context Continuity"""
        self.log("Testing Cross-session Continuity...")
        
        try:
            # Create first session with context
            session1 = self.client.post("/api/claude-auto/session/start", json={
                "cwd": "/opt/projects/test-continuity",
                "user_id": USER_ID
            })
            
            if session1.status_code == 200:
                session1_id = session1.json()["session"]["session_id"]
                
                # Add context to session 1
                self.client.post("/api/v1/memories", json={
                    "user_id": USER_ID,
                    "content": "Using JWT authentication strategy",
                    "memory_type": "decision",
                    "session_id": session1_id,
                    "project": "test-continuity"
                }, follow_redirects=True)
                
                # Wait a bit
                time.sleep(1)
                
                # Create second session - should restore context
                session2 = self.client.post("/api/claude-auto/session/start", json={
                    "cwd": "/opt/projects/test-continuity",
                    "user_id": USER_ID
                })
                
                if session2.status_code == 200:
                    session2_data = session2.json()
                    context = session2_data.get("context", {})
                    
                    # Check if previous session is referenced
                    if session2_data["session"]["previous_session"] == session1_id:
                        # Check if context includes previous memories
                        memories = context.get("memories", [])
                        project_memories = context.get("project_memories", [])
                        
                        if memories or project_memories:
                            return "✅", 1.0, "Cross-session continuity fully working"
                        else:
                            return "⚠️", 0.5, "Sessions linked but context not fully restored"
                    else:
                        return "⚠️", 0.5, "New session created but not linked to previous"
                else:
                    return "❌", 0.0, "Failed to create second session"
            else:
                return "❌", 0.0, "Failed to create first session"
                
        except Exception as e:
            return "❌", 0.0, f"Cross-session error: {str(e)}"
            
    async def run_all_tests(self):
        """Run all tests and calculate overall percentage"""
        tests = [
            ("Memory Creation", self.test_1_memory_creation),
            ("Memory Search", self.test_2_memory_search),
            ("Session Management", self.test_3_session_management),
            ("Error Learning", self.test_4_error_learning),
            ("Decision Recording", self.test_5_decision_recording),
            ("Code Evolution", self.test_6_code_evolution),
            ("Proactive Predictions", self.test_7_proactive_predictions),
            ("Web UI", self.test_8_web_ui),
            ("WebSocket Events", self.test_9_websocket_events),
            ("Cross-session Continuity", self.test_10_cross_session_continuity)
        ]
        
        total_score = 0.0
        detailed_results = []
        
        print("\n" + "="*80)
        print("KNOWLEDGEHUB FINAL FUNCTIONALITY ASSESSMENT")
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
            detailed_results.append((name, score))
            
        overall_percentage = (total_score / len(tests)) * 100
        
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        
        print("\n✅ WORKING Features:")
        for name, score in detailed_results:
            if score >= 1.0:
                print(f"  - {name}: 100%")
                
        print("\n⚠️ PARTIALLY Working Features:")
        for name, score in detailed_results:
            if 0 < score < 1.0:
                print(f"  - {name}: {score*100:.0f}%")
                
        print("\n❌ NOT Working Features:")
        for name, score in detailed_results:
            if score == 0:
                print(f"  - {name}: 0%")
                
        print(f"\n🎯 OVERALL ACTUAL FUNCTIONALITY: {overall_percentage:.1f}%")
        print("="*80)
        
        # Save detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_percentage": overall_percentage,
            "test_results": self.results,
            "test_data": self.test_data,
            "user_id": USER_ID
        }
        
        with open("knowledgehub_final_functionality_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        return overall_percentage

async def main():
    tester = FinalFunctionalityTester()
    percentage = await tester.run_all_tests()
    return percentage

if __name__ == "__main__":
    result = asyncio.run(main())