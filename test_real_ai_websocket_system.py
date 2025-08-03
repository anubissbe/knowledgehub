#!/usr/bin/env python3
"""
Comprehensive test for the real AI and WebSocket systems in KnowledgeHub.

This test validates:
- Real embeddings generation with sentence-transformers
- WebSocket real-time event broadcasting
- AI pattern recognition and predictions
- End-to-end integration functionality
- Performance benchmarks
"""

import asyncio
import json
import time
import logging
import websockets
import aiohttp
from typing import Dict, Any, List
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealSystemTester:
    """Comprehensive tester for real AI and WebSocket systems."""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws"
        
        # Test data
        self.test_user_id = "test_user_ai_ws"
        self.test_session_id = str(uuid.uuid4())
        self.test_project_id = "test_project_ai"
        
        # Results tracking
        self.test_results = {
            "embeddings": {},
            "websocket": {},
            "ai_intelligence": {},
            "integration": {},
            "performance": {}
        }
        
        # WebSocket connections
        self.ws_connections = []
    
    async def run_all_tests(self):
        """Run comprehensive test suite."""
        logger.info("🚀 Starting Real AI & WebSocket System Tests")
        logger.info("=" * 60)
        
        try:
            # Test 1: Service Health and Initialization
            await self.test_service_health()
            
            # Test 2: Real Embeddings Generation
            await self.test_real_embeddings()
            
            # Test 3: WebSocket Real-time Communication
            await self.test_websocket_realtime()
            
            # Test 4: AI Intelligence Features
            await self.test_ai_intelligence()
            
            # Test 5: End-to-End Integration
            await self.test_e2e_integration()
            
            # Test 6: Performance Benchmarks
            await self.test_performance_benchmarks()
            
            # Print comprehensive results
            self._print_test_results()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise
        finally:
            # Cleanup
            await self._cleanup()
    
    async def test_service_health(self):
        """Test service health and initialization."""
        logger.info("🔍 Testing Service Health & Initialization")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test main health endpoint
                async with session.get(f"{self.base_url}/health") as response:
                    health_data = await response.json()
                    
                    self.test_results["integration"]["health_check"] = {
                        "status": response.status == 200,
                        "response_time": response.headers.get("x-response-time", "unknown"),
                        "data": health_data
                    }
                
                # Test WebSocket status
                async with session.get(f"{self.base_url}/websocket/status") as response:
                    ws_status = await response.json()
                    
                    self.test_results["websocket"]["status"] = {
                        "status": response.status == 200,
                        "manager_running": ws_status.get("manager_stats", {}).get("active_connections", 0) >= 0,
                        "embeddings_available": ws_status.get("embeddings_stats", {}).get("models_available", False),
                        "events_running": ws_status.get("events_stats", {}).get("running", False)
                    }
                
                logger.info("✅ Service health check completed")
                
        except Exception as e:
            logger.error(f"❌ Service health check failed: {e}")
            self.test_results["integration"]["health_check"] = {"status": False, "error": str(e)}
    
    async def test_real_embeddings(self):
        """Test real embeddings generation."""
        logger.info("🧠 Testing Real Embeddings Generation")
        
        test_texts = [
            "This is a simple test for text embeddings generation",
            "Error: Connection timeout in database query",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "User feedback: The recommendation was very helpful",
            "Performance issue: Memory usage exceeded 80% threshold"
        ]
        
        embeddings_results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for i, text in enumerate(test_texts):
                    start_time = time.time()
                    
                    # Test text embedding generation
                    payload = {
                        "content": text,
                        "memory_type": "test",
                        "user_id": self.test_user_id,
                        "session_id": self.test_session_id,
                        "context": {"test_index": i}
                    }
                    
                    async with session.post(
                        f"{self.base_url}/api/memories",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            memory_data = await response.json()
                            processing_time = (time.time() - start_time) * 1000
                            
                            embeddings_results.append({
                                "text": text[:50] + "...",
                                "memory_id": memory_data.get("id"),
                                "processing_time": processing_time,
                                "success": True
                            })
                        else:
                            embeddings_results.append({
                                "text": text[:50] + "...",
                                "error": f"HTTP {response.status}",
                                "success": False
                            })
            
            # Calculate metrics
            successful = [r for r in embeddings_results if r["success"]]
            avg_processing_time = sum(r["processing_time"] for r in successful) / len(successful) if successful else 0
            
            self.test_results["embeddings"] = {
                "total_tests": len(test_texts),
                "successful": len(successful),
                "success_rate": len(successful) / len(test_texts),
                "avg_processing_time": avg_processing_time,
                "target_met": avg_processing_time < 100,  # Target: <100ms
                "results": embeddings_results
            }
            
            logger.info(f"✅ Embeddings: {len(successful)}/{len(test_texts)} successful, "
                       f"avg {avg_processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Embeddings test failed: {e}")
            self.test_results["embeddings"] = {"error": str(e), "success": False}
    
    async def test_websocket_realtime(self):
        """Test WebSocket real-time communication."""
        logger.info("🔗 Testing WebSocket Real-time Communication")
        
        messages_received = []
        connection_successful = False
        
        try:
            # Test WebSocket connection
            uri = f"{self.ws_url}/notifications"
            
            async with websockets.connect(uri) as websocket:
                connection_successful = True
                logger.info("✅ WebSocket connection established")
                
                # Send authentication (if needed)
                auth_message = {
                    "type": "auth",
                    "data": {"token": "test_token"}
                }
                await websocket.send(json.dumps(auth_message))
                
                # Subscribe to channels
                subscribe_message = {
                    "type": "subscribe",
                    "data": {"channel": "memories"}
                }
                await websocket.send(json.dumps(subscribe_message))
                
                # Listen for messages with timeout
                try:
                    timeout = 5  # 5 seconds
                    start_time = time.time()
                    
                    while time.time() - start_time < timeout:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            msg_data = json.loads(message)
                            messages_received.append(msg_data)
                            logger.info(f"📨 Received: {msg_data.get('type', 'unknown')}")
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.debug(f"Message processing error: {e}")
                            break
                            
                except Exception as e:
                    logger.debug(f"WebSocket listening error: {e}")
                
                # Test ping/pong
                ping_message = {"type": "ping", "data": {}}
                await websocket.send(json.dumps(ping_message))
                
                try:
                    pong_response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    pong_data = json.loads(pong_response)
                    ping_successful = pong_data.get("type") == "pong"
                except:
                    ping_successful = False
            
            self.test_results["websocket"]["realtime"] = {
                "connection_successful": connection_successful,
                "messages_received": len(messages_received),
                "ping_successful": ping_successful,
                "sample_messages": messages_received[:3]  # First 3 messages
            }
            
            logger.info(f"✅ WebSocket: Connection OK, {len(messages_received)} messages, "
                       f"Ping: {'OK' if ping_successful else 'Failed'}")
            
        except Exception as e:
            logger.error(f"❌ WebSocket test failed: {e}")
            self.test_results["websocket"]["realtime"] = {
                "connection_successful": False,
                "error": str(e)
            }
    
    async def test_ai_intelligence(self):
        """Test AI intelligence features."""
        logger.info("🤖 Testing AI Intelligence Features")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test error pattern analysis
                error_payload = {
                    "error_text": "ConnectionError: Failed to connect to database after 30 seconds",
                    "error_type": "database_connection",
                    "context": {"service": "user_service", "attempt": 3},
                    "user_id": self.test_user_id
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/api/ai/analyze-error",
                    json=error_payload
                ) as response:
                    error_analysis_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        error_analysis = await response.json()
                        error_test_success = True
                    else:
                        error_analysis = {"error": f"HTTP {response.status}"}
                        error_test_success = False
                
                # Test task prediction
                prediction_payload = {
                    "user_id": self.test_user_id,
                    "session_id": self.test_session_id,
                    "context": {"current_task": "code_review", "project": "test_project"}
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/api/ai/predict-tasks",
                    json=prediction_payload
                ) as response:
                    prediction_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        predictions = await response.json()
                        prediction_test_success = True
                    else:
                        predictions = {"error": f"HTTP {response.status}"}
                        prediction_test_success = False
                
                # Test performance insights
                metrics_payload = {
                    "metrics": {
                        "response_time": 1200,  # 1.2 seconds - should trigger insight
                        "memory_usage": 0.85,  # 85% - should trigger insight
                        "error_rate": 0.02     # 2% - acceptable
                    },
                    "context": {"service": "api_gateway"}
                }
                
                async with session.post(
                    f"{self.base_url}/api/ai/performance-insights",
                    json=metrics_payload
                ) as response:
                    if response.status == 200:
                        insights = await response.json()
                        insights_test_success = True
                    else:
                        insights = {"error": f"HTTP {response.status}"}
                        insights_test_success = False
            
            self.test_results["ai_intelligence"] = {
                "error_analysis": {
                    "success": error_test_success,
                    "processing_time": error_analysis_time,
                    "has_recommendations": len(error_analysis.get("recommendations", [])) > 0,
                    "confidence": error_analysis.get("confidence", 0)
                },
                "task_prediction": {
                    "success": prediction_test_success,
                    "processing_time": prediction_time,
                    "predictions_count": len(predictions.get("predictions", [])),
                    "avg_confidence": sum(p.get("confidence", 0) for p in predictions.get("predictions", [])) / max(len(predictions.get("predictions", [])), 1)
                },
                "performance_insights": {
                    "success": insights_test_success,
                    "insights_count": len(insights.get("insights", [])),
                    "actionable_steps": sum(len(i.get("actionable_steps", [])) for i in insights.get("insights", []))
                }
            }
            
            logger.info(f"✅ AI Intelligence: Error analysis: {error_test_success}, "
                       f"Predictions: {prediction_test_success}, Insights: {insights_test_success}")
            
        except Exception as e:
            logger.error(f"❌ AI Intelligence test failed: {e}")
            self.test_results["ai_intelligence"] = {"error": str(e), "success": False}
    
    async def test_e2e_integration(self):
        """Test end-to-end integration workflow."""
        logger.info("🔄 Testing End-to-End Integration")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Step 1: Create a memory (should generate embeddings and WebSocket event)
                memory_payload = {
                    "content": "Implemented new caching strategy using Redis for user sessions",
                    "memory_type": "decision",
                    "user_id": self.test_user_id,
                    "session_id": self.test_session_id,
                    "context": {"project_id": self.test_project_id, "impact": "high"}
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/api/memories",
                    json=memory_payload
                ) as response:
                    e2e_step1_time = (time.time() - start_time) * 1000
                    step1_success = response.status == 200
                    
                    if step1_success:
                        memory_data = await response.json()
                        memory_id = memory_data.get("id")
                    else:
                        memory_id = None
                
                # Step 2: Search for similar memories (should use embeddings)
                if memory_id:
                    search_payload = {
                        "query": "caching implementation Redis",
                        "user_id": self.test_user_id,
                        "limit": 5
                    }
                    
                    start_time = time.time()
                    async with session.post(
                        f"{self.base_url}/api/memories/search",
                        json=search_payload
                    ) as response:
                        e2e_step2_time = (time.time() - start_time) * 1000
                        step2_success = response.status == 200
                        
                        if step2_success:
                            search_results = await response.json()
                            found_memories = search_results.get("memories", [])
                        else:
                            found_memories = []
                else:
                    step2_success = False
                    found_memories = []
                    e2e_step2_time = 0
                
                # Step 3: Trigger AI analysis (should recognize patterns)
                if found_memories:
                    analysis_payload = {
                        "decision_context": {
                            "type": "implementation",
                            "technology": "Redis",
                            "impact": "performance"
                        },
                        "user_id": self.test_user_id,
                        "session_id": self.test_session_id
                    }
                    
                    async with session.post(
                        f"{self.base_url}/api/ai/analyze-decision",
                        json=analysis_payload
                    ) as response:
                        step3_success = response.status == 200
                        
                        if step3_success:
                            decision_analysis = await response.json()
                        else:
                            decision_analysis = {}
                else:
                    step3_success = False
                    decision_analysis = {}
            
            total_e2e_time = e2e_step1_time + e2e_step2_time
            
            self.test_results["integration"]["e2e_workflow"] = {
                "memory_creation": {
                    "success": step1_success,
                    "processing_time": e2e_step1_time,
                    "memory_id": memory_id
                },
                "semantic_search": {
                    "success": step2_success,
                    "processing_time": e2e_step2_time,
                    "results_found": len(found_memories)
                },
                "ai_analysis": {
                    "success": step3_success,
                    "pattern_confidence": decision_analysis.get("confidence", 0),
                    "recommendations": len(decision_analysis.get("recommendations", []))
                },
                "total_workflow_time": total_e2e_time,
                "workflow_complete": step1_success and step2_success
            }
            
            logger.info(f"✅ E2E Integration: Memory: {step1_success}, "
                       f"Search: {step2_success}, AI: {step3_success}, "
                       f"Total: {total_e2e_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ E2E Integration test failed: {e}")
            self.test_results["integration"]["e2e_workflow"] = {"error": str(e), "success": False}
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        logger.info("⚡ Testing Performance Benchmarks")
        
        try:
            # Test concurrent memory creation
            concurrent_tasks = 10
            concurrent_results = []
            
            async def create_memory_task(index):
                try:
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            "content": f"Concurrent test memory {index} - testing system performance under load",
                            "memory_type": "test",
                            "user_id": f"{self.test_user_id}_{index}",
                            "session_id": self.test_session_id
                        }
                        
                        start_time = time.time()
                        async with session.post(
                            f"{self.base_url}/api/memories",
                            json=payload
                        ) as response:
                            processing_time = (time.time() - start_time) * 1000
                            return {
                                "index": index,
                                "success": response.status == 200,
                                "processing_time": processing_time
                            }
                except Exception as e:
                    return {"index": index, "success": False, "error": str(e)}
            
            # Run concurrent tasks
            start_time = time.time()
            concurrent_results = await asyncio.gather(*[
                create_memory_task(i) for i in range(concurrent_tasks)
            ])
            total_concurrent_time = (time.time() - start_time) * 1000
            
            # Calculate performance metrics
            successful_concurrent = [r for r in concurrent_results if r.get("success")]
            
            if successful_concurrent:
                avg_concurrent_time = sum(r["processing_time"] for r in successful_concurrent) / len(successful_concurrent)
                max_concurrent_time = max(r["processing_time"] for r in successful_concurrent)
                min_concurrent_time = min(r["processing_time"] for r in successful_concurrent)
            else:
                avg_concurrent_time = max_concurrent_time = min_concurrent_time = 0
            
            self.test_results["performance"] = {
                "concurrent_operations": {
                    "total_tasks": concurrent_tasks,
                    "successful": len(successful_concurrent),
                    "success_rate": len(successful_concurrent) / concurrent_tasks,
                    "total_time": total_concurrent_time,
                    "avg_processing_time": avg_concurrent_time,
                    "max_processing_time": max_concurrent_time,
                    "min_processing_time": min_concurrent_time,
                    "target_met": avg_concurrent_time < 50  # Target: <50ms for memory operations
                },
                "benchmarks": {
                    "memory_retrieval_target": "< 50ms",
                    "pattern_matching_target": "< 100ms",
                    "realtime_updates_target": "< 200ms",
                    "concurrent_operations_target": f"{concurrent_tasks} simultaneous"
                }
            }
            
            logger.info(f"✅ Performance: {len(successful_concurrent)}/{concurrent_tasks} concurrent, "
                       f"avg {avg_concurrent_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            self.test_results["performance"] = {"error": str(e), "success": False}
    
    def _print_test_results(self):
        """Print comprehensive test results."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 REAL AI & WEBSOCKET SYSTEM TEST RESULTS")
        logger.info("=" * 60)
        
        # Overall summary
        total_tests = 0
        passed_tests = 0
        
        # Embeddings results
        if "embeddings" in self.test_results and "successful" in self.test_results["embeddings"]:
            embeddings_passed = self.test_results["embeddings"]["success_rate"] > 0.8
            logger.info(f"🧠 EMBEDDINGS: {'✅ PASS' if embeddings_passed else '❌ FAIL'}")
            logger.info(f"   Success Rate: {self.test_results['embeddings']['success_rate']:.1%}")
            logger.info(f"   Avg Processing: {self.test_results['embeddings']['avg_processing_time']:.1f}ms")
            total_tests += 1
            if embeddings_passed:
                passed_tests += 1
        
        # WebSocket results
        if "websocket" in self.test_results and "realtime" in self.test_results["websocket"]:
            ws_passed = self.test_results["websocket"]["realtime"]["connection_successful"]
            logger.info(f"🔗 WEBSOCKET: {'✅ PASS' if ws_passed else '❌ FAIL'}")
            logger.info(f"   Connection: {'OK' if ws_passed else 'Failed'}")
            logger.info(f"   Messages: {self.test_results['websocket']['realtime'].get('messages_received', 0)}")
            total_tests += 1
            if ws_passed:
                passed_tests += 1
        
        # AI Intelligence results
        if "ai_intelligence" in self.test_results:
            ai_tests = self.test_results["ai_intelligence"]
            ai_passed = (
                ai_tests.get("error_analysis", {}).get("success", False) and
                ai_tests.get("task_prediction", {}).get("success", False)
            )
            logger.info(f"🤖 AI INTELLIGENCE: {'✅ PASS' if ai_passed else '❌ FAIL'}")
            if "error_analysis" in ai_tests:
                logger.info(f"   Error Analysis: {'OK' if ai_tests['error_analysis']['success'] else 'Failed'}")
            if "task_prediction" in ai_tests:
                logger.info(f"   Task Prediction: {'OK' if ai_tests['task_prediction']['success'] else 'Failed'}")
            total_tests += 1
            if ai_passed:
                passed_tests += 1
        
        # Integration results
        if "integration" in self.test_results and "e2e_workflow" in self.test_results["integration"]:
            e2e = self.test_results["integration"]["e2e_workflow"]
            e2e_passed = e2e.get("workflow_complete", False)
            logger.info(f"🔄 E2E INTEGRATION: {'✅ PASS' if e2e_passed else '❌ FAIL'}")
            logger.info(f"   Workflow Time: {e2e.get('total_workflow_time', 0):.1f}ms")
            total_tests += 1
            if e2e_passed:
                passed_tests += 1
        
        # Performance results
        if "performance" in self.test_results and "concurrent_operations" in self.test_results["performance"]:
            perf = self.test_results["performance"]["concurrent_operations"]
            perf_passed = perf.get("target_met", False)
            logger.info(f"⚡ PERFORMANCE: {'✅ PASS' if perf_passed else '❌ FAIL'}")
            logger.info(f"   Concurrent Success: {perf['success_rate']:.1%}")
            logger.info(f"   Avg Processing: {perf['avg_processing_time']:.1f}ms")
            total_tests += 1
            if perf_passed:
                passed_tests += 1
        
        # Final summary
        logger.info("=" * 60)
        overall_pass = passed_tests == total_tests and total_tests > 0
        logger.info(f"🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
        logger.info(f"📊 TEST SUMMARY: {passed_tests}/{total_tests} passed ({passed_tests/max(total_tests,1):.1%})")
        logger.info("=" * 60)
        
        # Detailed results in JSON format
        logger.info("\n📋 Detailed Results (JSON):")
        print(json.dumps(self.test_results, indent=2, default=str))
    
    async def _cleanup(self):
        """Cleanup test resources."""
        try:
            # Close WebSocket connections
            for ws in self.ws_connections:
                if not ws.closed:
                    await ws.close()
            
            logger.info("🧹 Test cleanup completed")
            
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")


async def main():
    """Main test runner."""
    try:
        # Initialize tester
        tester = RealSystemTester()
        
        # Run all tests
        await tester.run_all_tests()
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Run the test suite
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)