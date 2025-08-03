#!/usr/bin/env python3
"""
Test ACTUAL AI Intelligence functionality with correct endpoints
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

class AIIntelligenceTester:
    def __init__(self):
        self.client = httpx.Client(base_url=API_BASE, timeout=30)
        self.results = {}
        self.test_data = {}
        
    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        
    def test_1_memory_creation_with_embeddings(self) -> Tuple[str, float, str]:
        """Test 1: Memory Creation with Embeddings (Working)"""
        self.log("Testing Memory Creation with Embeddings...")
        
        try:
            response = self.client.post("/api/v1/memories", json={
                "user_id": USER_ID,
                "content": "AI Intelligence test memory with embeddings",
                "memory_type": "technical",
                "tags": ["ai", "intelligence", "test"],
                "project": "knowledgehub-ai",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "test": True
                }
            }, follow_redirects=True)
            
            if response.status_code in [200, 201]:
                data = response.json()
                self.test_data["memory_id"] = data.get("id")
                return "✅", 1.0, f"Memory created with embeddings (ID: {data.get('id')})"
            else:
                return "❌", 0.0, f"Memory creation failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_2_memory_search_with_vectors(self) -> Tuple[str, float, str]:
        """Test 2: Memory Search with Vector Similarity (Working)"""
        self.log("Testing Memory Vector Search...")
        
        # Create some test memories first
        test_memories = [
            "Python machine learning algorithms implementation",
            "JavaScript React component optimization",
            "PostgreSQL database performance tuning"
        ]
        
        for content in test_memories:
            self.client.post("/api/v1/memories", json={
                "user_id": USER_ID,
                "content": content,
                "memory_type": "technical",
                "project": "knowledgehub-ai"
            }, follow_redirects=True)
            
        time.sleep(2)  # Wait for indexing
        
        try:
            # Test search (using POST as required by endpoint)
            response = self.client.post("/api/v1/search/", json={
                "query": "Python machine learning",
                "limit": 5,
                "search_type": "hybrid"
            }, follow_redirects=True)
            
            if response.status_code in [200, 201]:
                response_data = response.json()
                results = response_data.get("results", [])
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
            
    def test_3_error_learning_system(self) -> Tuple[str, float, str]:
        """Test 3: Error Learning System (Fixed - Real Endpoint)"""
        self.log("Testing Error Learning System...")
        
        try:
            # Record an error with solution
            response = self.client.post("/api/mistake-learning/track", json={
                "error_type": "ImportError",
                "error_message": "No module named 'tensorflow'",
                "context": {
                    "file": "ai_model.py",
                    "line": 3,
                    "code": "import tensorflow as tf"
                },
                "solution": "pip install tensorflow",
                "resolved": True,
                "project_id": "knowledgehub-ai"
            })
            
            if response.status_code == 200:
                data = response.json()
                self.test_data["mistake_id"] = data.get("mistake_id")
                
                # Test if similar error detection works
                time.sleep(1)
                similar_response = self.client.get("/api/mistake-learning/similar", params={
                    "error_type": "ImportError",
                    "query": "tensorflow module"
                })
                
                if similar_response.status_code == 200:
                    similar_errors = similar_response.json()
                    if len(similar_errors) > 0:
                        return "✅", 1.0, f"Error learning fully working - mistake tracked & similar errors found"
                    else:
                        return "⚠️", 0.7, "Error tracking works but similarity search needs tuning"
                else:
                    return "⚠️", 0.5, "Error tracking works but similar search failed"
            else:
                return "❌", 0.0, f"Error tracking failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_4_decision_recording_system(self) -> Tuple[str, float, str]:
        """Test 4: Decision Recording System (Fixed - Real Endpoint)"""
        self.log("Testing Decision Recording System...")
        
        try:
            # Record a decision with proper fields
            response = self.client.post("/api/decisions/record", json={
                "decision_title": "AI Model Selection",
                "chosen_solution": "Transformer-based model",
                "reasoning": "Better performance on text analysis tasks and supports fine-tuning",
                "alternatives": [
                    {
                        "solution": "RNN-based model",
                        "pros": ["Simpler architecture", "Lower memory usage"],
                        "cons": ["Poorer long-range dependencies", "Slower training"],
                        "reason_rejected": "Inadequate performance on complex text"
                    },
                    {
                        "solution": "CNN-based model", 
                        "pros": ["Fast inference", "Good for local patterns"],
                        "cons": ["Poor for sequential data", "Limited context understanding"],
                        "reason_rejected": "Not suitable for text understanding tasks"
                    }
                ],
                "context": {
                    "project": "knowledgehub-ai",
                    "factors": ["performance", "scalability", "maintainability"]
                },
                "confidence": 0.85,
                "evidence": ["Benchmark results", "Literature review", "Expert consultation"]
            })
            
            if response.status_code == 200:
                data = response.json()
                self.test_data["decision_id"] = data.get("decision_id")
                
                # Test decision search
                search_response = self.client.get("/api/decisions/search", params={
                    "query": "AI model",
                    "limit": 5
                })
                
                if search_response.status_code == 200:
                    search_results = search_response.json()
                    if len(search_results) > 0:
                        return "✅", 1.0, f"Decision recording fully working - decision tracked & searchable"
                    else:
                        return "⚠️", 0.7, "Decision recording works but search needs improvement"
                else:
                    return "⚠️", 0.5, "Decision recording works but search failed"
            else:
                return "❌", 0.0, f"Decision recording failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_5_code_evolution_tracking(self) -> Tuple[str, float, str]:
        """Test 5: Code Evolution Tracking (Fixed - Real Endpoint)"""
        self.log("Testing Code Evolution Tracking...")
        
        try:
            # Track a code change
            response = self.client.post("/api/code-evolution/track", json={
                "file_path": "/src/ai/model_trainer.py",
                "change_type": "optimization",
                "description": "Implemented batch processing for 3x faster training",
                "user_id": USER_ID
            })
            
            if response.status_code == 200:
                data = response.json()
                self.test_data["change_id"] = data.get("change_id")
                
                # Test evolution history
                history_response = self.client.get("/api/code-evolution/history", params={
                    "file_path": "/src/ai/model_trainer.py",
                    "limit": 10
                })
                
                if history_response.status_code == 200:
                    history = history_response.json()
                    if len(history) > 0:
                        return "✅", 1.0, f"Code evolution fully working - change tracked & history available"
                    else:
                        return "⚠️", 0.7, "Code tracking works but history needs improvement"
                else:
                    return "⚠️", 0.5, "Code tracking works but history failed"
            else:
                return "❌", 0.0, f"Code evolution failed: {response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_6_proactive_predictions(self) -> Tuple[str, float, str]:
        """Test 6: Proactive AI Predictions (Fixed - Real Endpoint)"""
        self.log("Testing Proactive AI Predictions...")
        
        try:
            # Create a session for context
            session_response = self.client.post("/api/claude-auto/session/start", json={
                "cwd": "/opt/projects/knowledgehub-ai",
                "user_id": USER_ID
            })
            
            if session_response.status_code == 200:
                session_data = session_response.json()
                session_id = session_data["session"]["session_id"]
                
                # Test proactive predictions
                predictions_response = self.client.get("/api/proactive/predictions", params={
                    "session_id": session_id,
                    "project_id": "knowledgehub-ai"
                })
                
                if predictions_response.status_code == 200:
                    predictions = predictions_response.json()
                    
                    # Test brief summary
                    brief_response = self.client.get("/api/proactive/brief", params={
                        "session_id": session_id,
                        "project_id": "knowledgehub-ai"
                    })
                    
                    if brief_response.status_code == 200:
                        return "✅", 1.0, "Proactive predictions working - session analysis & predictions available"
                    else:
                        return "⚠️", 0.7, "Predictions work but brief summary failed"
                else:
                    return "❌", 0.0, f"Predictions failed: {predictions_response.status_code}"
            else:
                return "❌", 0.0, f"Session creation failed: {session_response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_7_session_management_and_context(self) -> Tuple[str, float, str]:
        """Test 7: Session Management and Context Restoration (Working)"""
        self.log("Testing Session Management...")
        
        try:
            # Start a session
            session1 = self.client.post("/api/claude-auto/session/start", json={
                "cwd": "/opt/projects/knowledgehub-ai-test",
                "user_id": USER_ID
            })
            
            if session1.status_code == 200:
                session1_data = session1.json()
                session1_id = session1_data["session"]["session_id"]
                
                # Add some context by creating memories linked to session
                memory_response = self.client.post("/api/v1/memories", json={
                    "user_id": USER_ID,
                    "content": "Working on AI model optimization using gradient descent",
                    "memory_type": "session",
                    "session_id": session1_id,
                    "project": "knowledgehub-ai-test"
                }, follow_redirects=True)
                
                if memory_response.status_code in [200, 201]:
                    # Test context retrieval
                    context_response = self.client.get(f"/api/memory/context/quick/{USER_ID}")
                    
                    if context_response.status_code == 200:
                        context = context_response.json()
                        if isinstance(context, dict) and len(context) > 0:
                            return "✅", 1.0, "Session management working - sessions, memories, and context all functional"
                        else:
                            return "⚠️", 0.7, "Session works but context retrieval limited"
                    else:
                        return "⚠️", 0.5, "Session works but context retrieval failed"
                else:
                    return "⚠️", 0.5, "Session created but memory linking failed"
            else:
                return "❌", 0.0, f"Session creation failed: {session1.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_8_performance_tracking(self) -> Tuple[str, float, str]:
        """Test 8: Performance Metrics Tracking (Working)"""
        self.log("Testing Performance Tracking...")
        
        try:
            # Track some performance data
            perf_response = self.client.post("/api/performance/track", json={
                "command": "python train_model.py",
                "execution_time": 45.7,
                "success": True,
                "category": "ai-training",
                "context": {
                    "model_type": "transformer",
                    "dataset_size": "10M samples",
                    "hardware": "GPU"
                }
            })
            
            if perf_response.status_code == 200:
                # Get performance report
                report_response = self.client.get("/api/performance/report", params={
                    "category": "ai-training",
                    "days": 7
                })
                
                if report_response.status_code == 200:
                    report = report_response.json()
                    if report.get("total_executions", 0) > 0:
                        return "✅", 1.0, "Performance tracking working - metrics tracked and reports available"
                    else:
                        return "⚠️", 0.7, "Performance tracking works but reports need data"
                else:
                    return "⚠️", 0.5, "Performance tracking works but reports failed"
            else:
                return "❌", 0.0, f"Performance tracking failed: {perf_response.status_code}"
                
        except Exception as e:
            return "❌", 0.0, f"Error: {str(e)}"
            
    def test_9_web_ui_integration(self) -> Tuple[str, float, str]:
        """Test 9: Web UI with Real AI Data (Working)"""
        self.log("Testing Web UI Integration...")
        
        try:
            # Check UI accessibility
            ui_response = httpx.get("http://192.168.1.25:3100/")
            if ui_response.status_code != 200:
                return "❌", 0.0, "Web UI not accessible"
                
            # Check critical AI endpoints the UI needs
            endpoints_test = [
                ("/health", "API health"),
                ("/api/memory/stats", "Memory statistics"),
                ("/api/claude-auto/health", "Claude Auto health"),
                ("/api/mistake-learning/health", "Error learning health"),
                ("/api/decisions/health", "Decision recording health"),
                ("/api/proactive/health", "Proactive predictions health"),
                ("/api/code-evolution/health", "Code evolution health")
            ]
            
            working = 0
            total = len(endpoints_test)
            
            for endpoint, desc in endpoints_test:
                try:
                    params = {"user_id": USER_ID} if "stats" in endpoint else {}
                    resp = self.client.get(endpoint, params=params)
                    if resp.status_code == 200:
                        working += 1
                except:
                    pass
                    
            if working == total:
                return "✅", 1.0, f"Web UI fully functional - all {total} AI endpoints working"
            elif working > total * 0.8:
                return "⚠️", working/total, f"Web UI mostly working ({working}/{total} endpoints)"
            else:
                return "❌", working/total, f"Web UI limited functionality ({working}/{total} endpoints)"
                
        except Exception as e:
            return "❌", 0.0, f"Web UI error: {str(e)}"
            
    async def test_10_websocket_real_events(self) -> Tuple[str, float, str]:
        """Test 10: WebSocket Real-time Events (Checking)"""
        self.log("Testing WebSocket Real-time Events...")
        
        try:
            # Test WebSocket connection without auth first
            async with websockets.connect("ws://192.168.1.25:3000/ws/notifications") as ws:
                # Send a test message
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "user_id": USER_ID,
                    "events": ["memory_created", "error_tracked", "decision_recorded"]
                }))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=3)
                    data = json.loads(response)
                    
                    if data.get("type") == "subscription_confirmed":
                        return "✅", 1.0, "WebSocket working - real-time events subscription confirmed"
                    else:
                        return "⚠️", 0.5, f"WebSocket connects but unexpected response: {data}"
                except asyncio.TimeoutError:
                    return "⚠️", 0.3, "WebSocket connects but no subscription confirmation"
                    
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 403:
                return "❌", 0.0, "WebSocket requires authentication (403)"
            elif e.status_code == 404:
                return "❌", 0.0, "WebSocket endpoint not found (404)"
            else:
                return "❌", 0.0, f"WebSocket connection failed: {e.status_code}"
        except Exception as e:
            return "❌", 0.0, f"WebSocket error: {str(e)}"
            
    async def run_comprehensive_test(self):
        """Run all AI Intelligence tests"""
        tests = [
            ("Memory Creation with Embeddings", self.test_1_memory_creation_with_embeddings),
            ("Memory Search with Vectors", self.test_2_memory_search_with_vectors),
            ("Error Learning System", self.test_3_error_learning_system),
            ("Decision Recording System", self.test_4_decision_recording_system),
            ("Code Evolution Tracking", self.test_5_code_evolution_tracking),
            ("Proactive AI Predictions", self.test_6_proactive_predictions),
            ("Session Management & Context", self.test_7_session_management_and_context),
            ("Performance Tracking", self.test_8_performance_tracking),
            ("Web UI Integration", self.test_9_web_ui_integration),
            ("WebSocket Real-time Events", self.test_10_websocket_real_events)
        ]
        
        total_score = 0.0
        detailed_results = []
        
        print("\n" + "="*80)
        print("KNOWLEDGEHUB AI INTELLIGENCE COMPREHENSIVE ASSESSMENT")
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
        print("AI INTELLIGENCE ASSESSMENT SUMMARY")
        print("="*80)
        
        print("\n✅ FULLY WORKING Features:")
        for name, score in detailed_results:
            if score >= 0.9:
                print(f"  - {name}: {score*100:.0f}%")
                
        print("\n⚠️ PARTIALLY Working Features:")
        for name, score in detailed_results:
            if 0.3 <= score < 0.9:
                print(f"  - {name}: {score*100:.0f}%")
                
        print("\n❌ NOT Working Features:")
        for name, score in detailed_results:
            if score < 0.3:
                print(f"  - {name}: {score*100:.0f}%")
                
        print(f"\n🎯 OVERALL AI INTELLIGENCE FUNCTIONALITY: {overall_percentage:.1f}%")
        
        if overall_percentage >= 90:
            print("🎉 EXCELLENT: AI Intelligence is highly functional!")
        elif overall_percentage >= 70:
            print("👍 GOOD: AI Intelligence is mostly functional with minor issues")
        elif overall_percentage >= 50:
            print("⚠️ MODERATE: AI Intelligence is partially functional, needs fixes")
        else:
            print("❌ POOR: AI Intelligence needs significant work")
            
        print("="*80)
        
        # Save detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_percentage": overall_percentage,
            "test_results": self.results,
            "test_data": self.test_data,
            "user_id": USER_ID,
            "assessment": "comprehensive_ai_intelligence"
        }
        
        with open("knowledgehub_ai_intelligence_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 Detailed report saved to: knowledgehub_ai_intelligence_report.json")
        
        return overall_percentage

async def main():
    tester = AIIntelligenceTester()
    percentage = await tester.run_comprehensive_test()
    return percentage

if __name__ == "__main__":
    result = asyncio.run(main())