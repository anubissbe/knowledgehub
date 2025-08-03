#!/usr/bin/env python3
"""
Comprehensive test of KnowledgeHub actual functionality
Tests each feature thoroughly to determine real working percentage
"""

import asyncio
import json
import time
import uuid
import httpx
import websockets
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
API_BASE = "http://192.168.1.25:3000"
WS_URL = "ws://192.168.1.25:3000/ws"
USER_ID = "test_user_" + str(uuid.uuid4())[:8]
TIMEOUT = 30

class KnowledgeHubTester:
    def __init__(self):
        self.client = httpx.Client(base_url=API_BASE, timeout=TIMEOUT)
        self.results = {}
        self.test_data = {
            "memory_id": None,
            "session_id": None,
            "error_id": None,
            "decision_id": None,
            "code_change_id": None
        }
        
    def log(self, message: str):
        """Log with timestamp"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    async def test_1_memory_creation(self) -> Tuple[str, float, str]:
        """Test 1: Memory Creation with Embeddings"""
        self.log("Testing Memory Creation...")
        try:
            # Create a memory
            response = self.client.post("/api/memory/memories", json={
                "user_id": USER_ID,
                "content": "This is a test memory for KnowledgeHub functionality testing",
                "type": "technical",
                "tags": ["test", "functionality"],
                "project": "knowledgehub-test",
                "metadata": {
                    "source": "functionality_test",
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            if response.status_code == 200:
                data = response.json()
                self.test_data["memory_id"] = data.get("id")
                
                # Verify the memory has embeddings
                if data.get("embedding_generated"):
                    # Check if we can retrieve the memory
                    get_response = self.client.get(f"/api/memory/memories/{self.test_data['memory_id']}")
                    if get_response.status_code == 200:
                        memory_data = get_response.json()
                        if memory_data.get("content") == "This is a test memory for KnowledgeHub functionality testing":
                            return "✅", 1.0, "Memory created with embeddings and retrieved successfully"
                        else:
                            return "⚠️", 0.5, "Memory created but content mismatch"
                    else:
                        return "⚠️", 0.5, "Memory created but retrieval failed"
                else:
                    return "⚠️", 0.5, "Memory created but no embeddings generated"
            else:
                return "❌", 0.0, f"Failed to create memory: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_2_memory_search(self) -> Tuple[str, float, str]:
        """Test 2: Memory Search with Vector Similarity"""
        self.log("Testing Memory Search...")
        try:
            # First create a few more memories for search
            memories = [
                "Python programming best practices for error handling",
                "JavaScript async/await patterns for API calls",
                "Database optimization techniques for PostgreSQL"
            ]
            
            for content in memories:
                self.client.post("/api/memory/memories", json={
                    "user_id": USER_ID,
                    "content": content,
                    "type": "technical",
                    "project": "knowledgehub-test"
                })
                
            time.sleep(2)  # Wait for embeddings
            
            # Search for related content
            response = self.client.get("/api/memory/search", params={
                "user_id": USER_ID,
                "query": "Python error handling techniques",
                "limit": 5
            })
            
            if response.status_code == 200:
                results = response.json()
                if len(results) > 0:
                    # Check if the most relevant result is about Python
                    if "Python" in results[0].get("content", ""):
                        return "✅", 1.0, f"Vector search working - found {len(results)} relevant results"
                    else:
                        return "⚠️", 0.5, "Search returns results but relevance is questionable"
                else:
                    return "❌", 0.0, "Search returns no results"
            else:
                return "❌", 0.0, f"Search failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_3_session_management(self) -> Tuple[str, float, str]:
        """Test 3: Session Management and Linking"""
        self.log("Testing Session Management...")
        try:
            # Initialize a session
            response = self.client.post("/api/claude-auto/init-session", json={
                "user_id": USER_ID,
                "metadata": {
                    "source": "functionality_test",
                    "purpose": "testing session continuity"
                }
            })
            
            if response.status_code == 200:
                session_data = response.json()
                self.test_data["session_id"] = session_data.get("session_id")
                
                # Create a memory linked to this session
                memory_response = self.client.post("/api/memory/memories", json={
                    "user_id": USER_ID,
                    "content": "Session-linked memory for testing",
                    "type": "session",
                    "session_id": self.test_data["session_id"],
                    "project": "knowledgehub-test"
                })
                
                if memory_response.status_code == 200:
                    # Verify session memories can be retrieved
                    session_memories = self.client.get(f"/api/memory/session/{self.test_data['session_id']}")
                    if session_memories.status_code == 200:
                        memories = session_memories.json()
                        if len(memories) > 0:
                            return "✅", 1.0, "Session management working - memories linked to sessions"
                        else:
                            return "⚠️", 0.5, "Session created but memory linking not working"
                    else:
                        return "⚠️", 0.5, "Session created but retrieval failed"
                else:
                    return "⚠️", 0.5, "Session created but cannot link memories"
            else:
                return "❌", 0.0, f"Failed to create session: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_4_error_learning(self) -> Tuple[str, float, str]:
        """Test 4: Error Learning System"""
        self.log("Testing Error Learning...")
        try:
            # Record an error
            response = self.client.post("/api/mistake-learning/errors", json={
                "user_id": USER_ID,
                "error_type": "ImportError",
                "error_message": "No module named 'missing_module'",
                "context": {
                    "file": "test.py",
                    "line": 10,
                    "code": "import missing_module"
                },
                "solution": "Install the module with pip install missing_module",
                "tags": ["python", "import", "dependency"]
            })
            
            if response.status_code == 200:
                error_data = response.json()
                self.test_data["error_id"] = error_data.get("id")
                
                # Search for similar errors
                search_response = self.client.get("/api/mistake-learning/search", params={
                    "query": "ImportError module not found",
                    "user_id": USER_ID
                })
                
                if search_response.status_code == 200:
                    results = search_response.json()
                    if len(results) > 0 and results[0].get("solution"):
                        return "✅", 1.0, "Error learning working - errors tracked and solutions searchable"
                    else:
                        return "⚠️", 0.5, "Errors tracked but search not returning solutions"
                else:
                    return "⚠️", 0.5, "Error recorded but search failed"
            else:
                return "❌", 0.0, f"Failed to record error: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_5_decision_recording(self) -> Tuple[str, float, str]:
        """Test 5: Decision Recording and Tracking"""
        self.log("Testing Decision Recording...")
        try:
            # Record a decision
            response = self.client.post("/api/decisions", json={
                "user_id": USER_ID,
                "decision": "Use PostgreSQL for main database",
                "reasoning": "Better support for complex queries and JSONB data type",
                "alternatives": ["MongoDB", "MySQL", "SQLite"],
                "context": {
                    "project": "knowledgehub-test",
                    "requirements": ["ACID compliance", "JSON support", "Full-text search"]
                },
                "confidence": 0.9,
                "tags": ["database", "architecture", "postgresql"]
            })
            
            if response.status_code == 200:
                decision_data = response.json()
                self.test_data["decision_id"] = decision_data.get("id")
                
                # Search for decisions
                search_response = self.client.get("/api/decisions/search", params={
                    "query": "database selection",
                    "user_id": USER_ID
                })
                
                if search_response.status_code == 200:
                    results = search_response.json()
                    if len(results) > 0:
                        # Check if we can retrieve the full decision
                        if results[0].get("reasoning") and results[0].get("alternatives"):
                            return "✅", 1.0, "Decision recording working - decisions tracked with reasoning"
                        else:
                            return "⚠️", 0.5, "Decisions recorded but missing details"
                    else:
                        return "⚠️", 0.5, "Decision recorded but not searchable"
                else:
                    return "⚠️", 0.5, "Decision recorded but search failed"
            else:
                return "❌", 0.0, f"Failed to record decision: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_6_code_evolution(self) -> Tuple[str, float, str]:
        """Test 6: Code Evolution Tracking"""
        self.log("Testing Code Evolution...")
        try:
            # Track a code change
            response = self.client.post("/api/code-evolution/changes", json={
                "user_id": USER_ID,
                "file_path": "/test/example.py",
                "change_type": "refactor",
                "description": "Refactored database connection to use connection pooling",
                "before_snippet": "conn = psycopg2.connect(DATABASE_URL)",
                "after_snippet": "conn = connection_pool.getconn()",
                "impact": "Improved performance by 40%",
                "tags": ["performance", "database", "refactoring"]
            })
            
            if response.status_code == 200:
                change_data = response.json()
                self.test_data["code_change_id"] = change_data.get("id")
                
                # Get evolution history
                history_response = self.client.get("/api/code-evolution/history", params={
                    "file_path": "/test/example.py",
                    "user_id": USER_ID
                })
                
                if history_response.status_code == 200:
                    history = history_response.json()
                    if len(history) > 0 and history[0].get("description"):
                        return "✅", 1.0, "Code evolution tracking working - changes tracked with impact"
                    else:
                        return "⚠️", 0.5, "Changes tracked but history incomplete"
                else:
                    return "⚠️", 0.5, "Change recorded but history retrieval failed"
            else:
                return "❌", 0.0, f"Failed to track code change: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_7_proactive_predictions(self) -> Tuple[str, float, str]:
        """Test 7: Proactive AI Predictions"""
        self.log("Testing Proactive Predictions...")
        try:
            # Get task predictions
            response = self.client.get("/api/proactive/predict-tasks", params={
                "user_id": USER_ID,
                "context": json.dumps({
                    "current_file": "test.py",
                    "recent_actions": ["created function", "added imports"],
                    "project": "knowledgehub-test"
                })
            })
            
            if response.status_code == 200:
                predictions = response.json()
                if predictions.get("tasks") and len(predictions["tasks"]) > 0:
                    # Check if predictions are contextual
                    if any("test" in task.lower() or "function" in task.lower() 
                          for task in predictions["tasks"]):
                        return "✅", 1.0, "Proactive predictions working - contextual suggestions provided"
                    else:
                        return "⚠️", 0.5, "Predictions returned but not contextual"
                else:
                    return "⚠️", 0.5, "Predictions endpoint works but no tasks suggested"
            else:
                return "❌", 0.0, f"Failed to get predictions: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_8_web_ui(self) -> Tuple[str, float, str]:
        """Test 8: Web UI Real Data Display"""
        self.log("Testing Web UI...")
        try:
            # Check if UI is accessible
            ui_response = self.client.get("http://192.168.1.25:3100/", follow_redirects=True)
            
            if ui_response.status_code == 200:
                # Check API endpoints that UI uses
                endpoints = [
                    "/api/memory/stats",
                    "/api/claude-auto/sessions/recent",
                    "/api/mistake-learning/recent-errors",
                    "/api/decisions/recent"
                ]
                
                working_endpoints = 0
                for endpoint in endpoints:
                    try:
                        resp = self.client.get(endpoint, params={"user_id": USER_ID})
                        if resp.status_code == 200:
                            data = resp.json()
                            # Check if data is real (not empty or mock)
                            if data and not (isinstance(data, dict) and data.get("mock")):
                                working_endpoints += 1
                    except:
                        pass
                        
                if working_endpoints == len(endpoints):
                    return "✅", 1.0, "Web UI working - displaying real data from all endpoints"
                elif working_endpoints > 0:
                    return "⚠️", 0.5, f"Web UI partially working - {working_endpoints}/{len(endpoints)} endpoints"
                else:
                    return "❌", 0.0, "Web UI accessible but showing no real data"
            else:
                return "❌", 0.0, f"Web UI not accessible: {ui_response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_9_websocket_events(self) -> Tuple[str, float, str]:
        """Test 9: WebSocket Real-time Events"""
        self.log("Testing WebSocket Events...")
        try:
            events_received = []
            
            async def listen_for_events():
                try:
                    async with websockets.connect(WS_URL) as websocket:
                        # Subscribe to events
                        await websocket.send(json.dumps({
                            "type": "subscribe",
                            "user_id": USER_ID,
                            "events": ["memory_created", "error_tracked", "decision_made"]
                        }))
                        
                        # Wait for events
                        while len(events_received) < 3 and len(events_received) < 10:
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                                event = json.loads(message)
                                if event.get("type") != "subscription_confirmed":
                                    events_received.append(event)
                            except asyncio.TimeoutError:
                                break
                except Exception as e:
                    pass
                    
            # Start listening
            listen_task = asyncio.create_task(listen_for_events())
            
            # Trigger some events
            await asyncio.sleep(1)
            
            # Create a memory (should trigger event)
            self.client.post("/api/memory/memories", json={
                "user_id": USER_ID,
                "content": "WebSocket test memory",
                "type": "test"
            })
            
            # Record an error (should trigger event)
            self.client.post("/api/mistake-learning/errors", json={
                "user_id": USER_ID,
                "error_type": "TestError",
                "error_message": "WebSocket test error"
            })
            
            # Wait for events
            await asyncio.sleep(3)
            listen_task.cancel()
            
            if len(events_received) >= 2:
                return "✅", 1.0, f"WebSocket working - received {len(events_received)} real-time events"
            elif len(events_received) > 0:
                return "⚠️", 0.5, f"WebSocket partially working - only {len(events_received)} events"
            else:
                return "❌", 0.0, "WebSocket connects but no events received"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    async def test_10_cross_session_continuity(self) -> Tuple[str, float, str]:
        """Test 10: Cross-session Context Continuity"""
        self.log("Testing Cross-session Continuity...")
        try:
            # Create first session with context
            session1_response = self.client.post("/api/claude-auto/init-session", json={
                "user_id": USER_ID,
                "metadata": {
                    "project": "test-project",
                    "task": "implementing authentication"
                }
            })
            
            if session1_response.status_code == 200:
                session1_id = session1_response.json().get("session_id")
                
                # Add some context to session 1
                self.client.post("/api/memory/memories", json={
                    "user_id": USER_ID,
                    "content": "Decided to use JWT for authentication",
                    "type": "decision",
                    "session_id": session1_id
                })
                
                # Create second session
                session2_response = self.client.post("/api/claude-auto/init-session", json={
                    "user_id": USER_ID,
                    "metadata": {
                        "project": "test-project",
                        "restore_context": True
                    }
                })
                
                if session2_response.status_code == 200:
                    session2_data = session2_response.json()
                    
                    # Check if context was restored
                    context_response = self.client.get("/api/memory/context/quick/" + USER_ID)
                    
                    if context_response.status_code == 200:
                        context = context_response.json()
                        # Check if previous session's context is available
                        if any("JWT" in str(item) for item in context.values()):
                            return "✅", 1.0, "Cross-session continuity working - context preserved"
                        else:
                            return "⚠️", 0.5, "Sessions created but context not fully restored"
                    else:
                        return "⚠️", 0.5, "Sessions work but context retrieval failed"
                else:
                    return "❌", 0.0, "Failed to create second session"
            else:
                return "❌", 0.0, "Failed to create first session"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
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
        print("KNOWLEDGEHUB ACTUAL FUNCTIONALITY TEST")
        print("="*80 + "\n")
        
        for name, test_func in tests:
            status, score, details = await test_func()
            total_score += score
            self.results[name] = {
                "status": status,
                "score": score,
                "details": details
            }
            
            print(f"{status} {name}: {details}")
            detailed_results.append(f"{name}: {score*100:.0f}%")
            
        overall_percentage = (total_score / len(tests)) * 100
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print("\nDetailed Scores:")
        for result in detailed_results:
            print(f"  - {result}")
            
        print(f"\nOVERALL FUNCTIONALITY: {overall_percentage:.1f}%")
        print("="*80)
        
        # Save detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_percentage": overall_percentage,
            "test_results": self.results,
            "test_data": self.test_data
        }
        
        with open("knowledgehub_functionality_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        return overall_percentage

async def main():
    tester = KnowledgeHubTester()
    percentage = await tester.run_all_tests()
    return percentage

if __name__ == "__main__":
    result = asyncio.run(main())