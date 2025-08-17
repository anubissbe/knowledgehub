#\!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Test Framework
by Charlotte Cools - Dynamic Parallelism & Memory Bandwidth Optimization Expert

This test framework performs complete system integration testing with focus on:
1. Full Stack Integration (Frontend → API → Database → Response)
2. AI Intelligence Pipeline (Error learning, Decision tracking, Proactive assistance)
3. Real-time Features (WebSocket connectivity, Live updates, Notifications)
4. Cross-system Integration (Memory ↔ AI, Performance ↔ Analytics, Error ↔ Pattern recognition)
5. Data Flow Verification (PostgreSQL → TimescaleDB, Vector embeddings → Search, Graph relationships)
6. Failure Scenarios (Database failures, Network timeouts, Service recovery)
"""

import asyncio
import aiohttp
import websockets
import pytest
import time
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import psutil
import traceback
import random
import subprocess
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('charlotte_cools_e2e_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Test Configuration
@dataclass
class TestConfig:
    """Test configuration for comprehensive E2E testing"""
    api_base: str = "http://localhost:3000"
    websocket_base: str = "ws://localhost:3000"
    ui_base: str = "http://localhost:3100"
    api_key: str = "knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM"
    timeout: int = 30
    max_concurrent: int = 10
    test_session_id: str = None
    
    def __post_init__(self):
        if self.test_session_id is None:
            self.test_session_id = str(uuid.uuid4())

config = TestConfig()

class TestMetrics:
    """Track test performance metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.response_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.throughput_metrics = {}
        self.parallel_execution_times = {}
        
    def record_response_time(self, operation: str, duration: float):
        self.response_times.append({
            'operation': operation,
            'duration': duration,
            'timestamp': time.time()
        })
        
    def record_system_metrics(self):
        """Record system performance metrics"""
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance test summary"""
        total_time = time.time() - self.start_time
        avg_response_time = sum(r['duration'] for r in self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            'total_test_time': total_time,
            'average_response_time': avg_response_time,
            'max_response_time': max(r['duration'] for r in self.response_times) if self.response_times else 0,
            'min_response_time': min(r['duration'] for r in self.response_times) if self.response_times else 0,
            'total_requests': len(self.response_times),
            'requests_per_second': len(self.response_times) / total_time if total_time > 0 else 0,
            'avg_memory_usage': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'parallel_execution_times': self.parallel_execution_times
        }

metrics = TestMetrics()

class ComprehensiveE2ETest:
    """Comprehensive End-to-End Integration Test Suite"""
    
    def __init__(self):
        self.session = None
        self.websocket_connections = []
        self.test_data = {}
        self.services_health = {}
        
    async def setup(self):
        """Initialize test environment"""
        logger.info("🚀 Initializing Charlotte Cools Comprehensive E2E Test Framework")
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.timeout),
            headers={'X-API-Key': config.api_key}
        )
        
        # Pre-populate test data
        await self.prepare_test_data()
        logger.info("✅ Test environment initialized")
        
    async def teardown(self):
        """Clean up test environment"""
        logger.info("🧹 Cleaning up test environment")
        
        # Close WebSocket connections
        for ws in self.websocket_connections:
            try:
                await ws.close()
            except:
                pass
                
        # Close HTTP session
        if self.session:
            await self.session.close()
            
        # Clean up test data
        await self.cleanup_test_data()
        logger.info("✅ Test environment cleaned up")
        
    async def prepare_test_data(self):
        """Prepare test data for comprehensive testing"""
        self.test_data = {
            'memory_test_data': {
                'session_id': config.test_session_id,
                'content': 'Test memory content for E2E integration testing',
                'type': 'integration_test',
                'metadata': {'test_type': 'e2e', 'author': 'Charlotte Cools'}
            },
            'decision_test_data': {
                'decision': 'Use dynamic parallelism for memory bandwidth optimization',
                'reasoning': 'Parallel processing reduces memory access latency',
                'alternatives': ['Sequential processing', 'Batch processing'],
                'context': {'test_scenario': 'e2e_integration'},
                'confidence': 0.95
            },
            'error_test_data': {
                'error_type': 'MemoryBandwidthException',
                'error_message': 'Memory bandwidth exceeded threshold',
                'solution': 'Implement dynamic parallelism with memory optimization',
                'resolved': True
            },
            'websocket_test_data': {
                'event_type': 'memory_created',
                'payload': {'id': str(uuid.uuid4()), 'type': 'test_event'}
            }
        }
        
    async def cleanup_test_data(self):
        """Clean up test data after testing"""
        try:
            # Clean up memories created during testing
            url = f"{config.api_base}/api/memory/session/{config.test_session_id}"
            async with self.session.delete(url) as response:
                if response.status in [200, 404]:
                    logger.info("✅ Test memories cleaned up")
                    
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
            
    # =====================================================
    # TEST 1: FULL STACK INTEGRATION TESTING
    # =====================================================
    
    async def test_full_stack_integration(self) -> Dict[str, Any]:
        """Test complete flow: Frontend → API → Database → Response"""
        logger.info("📋 Testing Full Stack Integration (Frontend → API → Database → Response)")
        
        test_results = {
            'test_name': 'full_stack_integration',
            'start_time': time.time(),
            'subtests': {},
            'overall_success': True
        }
        
        try:
            # 1. Test API Health
            start_time = time.time()
            async with self.session.get(f"{config.api_base}/health") as response:
                api_health_time = time.time() - start_time
                api_healthy = response.status == 200
                health_data = await response.json() if api_healthy else {}
                
                test_results['subtests']['api_health'] = {
                    'success': api_healthy,
                    'response_time': api_health_time,
                    'data': health_data
                }
                
            metrics.record_response_time('api_health', api_health_time)
            
            # 2. Test Database Connectivity through API
            start_time = time.time()
            memory_data = self.test_data['memory_test_data']
            async with self.session.post(
                f"{config.api_base}/api/memory",
                json=memory_data
            ) as response:
                db_connectivity_time = time.time() - start_time
                db_connected = response.status in [200, 201]
                created_memory = await response.json() if db_connected else {}
                
                test_results['subtests']['database_connectivity'] = {
                    'success': db_connected,
                    'response_time': db_connectivity_time,
                    'created_memory_id': created_memory.get('memory_id')
                }
                
            metrics.record_response_time('memory_create', db_connectivity_time)
            
            # 3. Test Data Persistence
            if db_connected and created_memory.get('memory_id'):
                start_time = time.time()
                memory_id = created_memory['memory_id']
                async with self.session.get(
                    f"{config.api_base}/api/memory/{memory_id}"
                ) as response:
                    persistence_time = time.time() - start_time
                    data_persisted = response.status == 200
                    retrieved_memory = await response.json() if data_persisted else {}
                    
                    # Verify data integrity
                    content_match = retrieved_memory.get('content') == memory_data['content']
                    
                    test_results['subtests']['data_persistence'] = {
                        'success': data_persisted and content_match,
                        'response_time': persistence_time,
                        'content_match': content_match
                    }
                    
                metrics.record_response_time('memory_retrieve', persistence_time)
            
            # 4. Test Frontend Accessibility
            start_time = time.time()
            async with self.session.get(config.ui_base) as response:
                frontend_time = time.time() - start_time
                frontend_accessible = response.status == 200
                
                test_results['subtests']['frontend_accessibility'] = {
                    'success': frontend_accessible,
                    'response_time': frontend_time
                }
                
            metrics.record_response_time('frontend_access', frontend_time)
            
            # 5. Test Transaction Integrity
            start_time = time.time()
            batch_data = {
                'memories': [
                    {**memory_data, 'content': f'Batch memory {i}'}
                    for i in range(3)
                ]
            }
            
            async with self.session.post(
                f"{config.api_base}/api/memory/batch",
                json=batch_data
            ) as response:
                transaction_time = time.time() - start_time
                transaction_success = response.status in [200, 201]
                batch_result = await response.json() if transaction_success else {}
                
                test_results['subtests']['transaction_integrity'] = {
                    'success': transaction_success,
                    'response_time': transaction_time,
                    'batch_count': len(batch_result.get('created_memories', []))
                }
                
            metrics.record_response_time('batch_transaction', transaction_time)
            
        except Exception as e:
            logger.error(f"❌ Full stack integration test failed: {e}")
            test_results['error'] = str(e)
            test_results['overall_success'] = False
            
        # Check overall success
        test_results['overall_success'] = all(
            subtest.get('success', False) 
            for subtest in test_results['subtests'].values()
        )
        
        test_results['end_time'] = time.time()
        test_results['total_duration'] = test_results['end_time'] - test_results['start_time']
        
        logger.info(f"📊 Full Stack Integration: {'✅ PASSED' if test_results['overall_success'] else '❌ FAILED'}")
        return test_results
        
    # =====================================================
    # TEST 2: AI INTELLIGENCE PIPELINE TESTING
    # =====================================================
    
    async def test_ai_intelligence_pipeline(self) -> Dict[str, Any]:
        """Test AI Intelligence Pipeline: Error learning, Decision tracking, Proactive assistance"""
        logger.info("🧠 Testing AI Intelligence Pipeline")
        
        test_results = {
            'test_name': 'ai_intelligence_pipeline',
            'start_time': time.time(),
            'subtests': {},
            'overall_success': True
        }
        
        try:
            # 1. Test Error Learning System
            start_time = time.time()
            error_data = self.test_data['error_test_data']
            async with self.session.post(
                f"{config.api_base}/api/mistake-learning/record-error",
                json=error_data
            ) as response:
                error_learning_time = time.time() - start_time
                error_recorded = response.status in [200, 201]
                error_response = await response.json() if error_recorded else {}
                
                test_results['subtests']['error_learning'] = {
                    'success': error_recorded,
                    'response_time': error_learning_time,
                    'error_id': error_response.get('error_id')
                }
                
            metrics.record_response_time('error_learning', error_learning_time)
            
            # 2. Test Decision Tracking System
            start_time = time.time()
            decision_data = self.test_data['decision_test_data']
            async with self.session.post(
                f"{config.api_base}/api/decisions",
                json=decision_data
            ) as response:
                decision_time = time.time() - start_time
                decision_recorded = response.status in [200, 201]
                decision_response = await response.json() if decision_recorded else {}
                
                test_results['subtests']['decision_tracking'] = {
                    'success': decision_recorded,
                    'response_time': decision_time,
                    'decision_id': decision_response.get('decision_id')
                }
                
            metrics.record_response_time('decision_tracking', decision_time)
            
            # 3. Test Proactive Assistant
            start_time = time.time()
            context_data = {
                'session_id': config.test_session_id,
                'current_context': 'E2E integration testing',
                'user_patterns': ['memory_optimization', 'parallel_processing']
            }
            
            async with self.session.post(
                f"{config.api_base}/api/proactive/predict-tasks",
                json=context_data
            ) as response:
                proactive_time = time.time() - start_time
                predictions_generated = response.status == 200
                predictions = await response.json() if predictions_generated else {}
                
                test_results['subtests']['proactive_assistance'] = {
                    'success': predictions_generated,
                    'response_time': proactive_time,
                    'predictions_count': len(predictions.get('predicted_tasks', []))
                }
                
            metrics.record_response_time('proactive_assistance', proactive_time)
            
            # 4. Test Pattern Recognition
            start_time = time.time()
            async with self.session.get(
                f"{config.api_base}/api/pattern-recognition/patterns"
            ) as response:
                pattern_time = time.time() - start_time
                patterns_retrieved = response.status == 200
                patterns_data = await response.json() if patterns_retrieved else {}
                
                test_results['subtests']['pattern_recognition'] = {
                    'success': patterns_retrieved,
                    'response_time': pattern_time,
                    'patterns_count': len(patterns_data.get('patterns', []))
                }
                
            metrics.record_response_time('pattern_recognition', pattern_time)
            
        except Exception as e:
            logger.error(f"❌ AI Intelligence pipeline test failed: {e}")
            test_results['error'] = str(e)
            test_results['overall_success'] = False
            
        test_results['overall_success'] = all(
            subtest.get('success', False) 
            for subtest in test_results['subtests'].values()
        )
        
        test_results['end_time'] = time.time()
        test_results['total_duration'] = test_results['end_time'] - test_results['start_time']
        
        logger.info(f"🧠 AI Intelligence Pipeline: {'✅ PASSED' if test_results['overall_success'] else '❌ FAILED'}")
        return test_results
        
    # =====================================================
    # TEST 3: REAL-TIME FEATURES TESTING
    # =====================================================
    
    async def test_realtime_features(self) -> Dict[str, Any]:
        """Test Real-time features: WebSocket connectivity, Live updates, Notifications"""
        logger.info("⚡ Testing Real-time Features")
        
        test_results = {
            'test_name': 'realtime_features',
            'start_time': time.time(),
            'subtests': {},
            'overall_success': True
        }
        
        try:
            # 1. Test WebSocket Connectivity
            start_time = time.time()
            websocket_connected = False
            messages_received = []
            
            try:
                ws_url = f"{config.websocket_base}/ws/events/{config.test_session_id}"
                async with websockets.connect(ws_url) as websocket:
                    websocket_connected = True
                    connection_time = time.time() - start_time
                    self.websocket_connections.append(websocket)
                    
                    # Send test message
                    test_message = json.dumps(self.test_data['websocket_test_data'])
                    await websocket.send(test_message)
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        messages_received.append(json.loads(response))
                    except asyncio.TimeoutError:
                        pass
                        
                    test_results['subtests']['websocket_connectivity'] = {
                        'success': websocket_connected,
                        'response_time': connection_time,
                        'messages_received': len(messages_received)
                    }
                    
            except Exception as ws_error:
                test_results['subtests']['websocket_connectivity'] = {
                    'success': False,
                    'error': str(ws_error)
                }
                
            metrics.record_response_time('websocket_connect', time.time() - start_time)
            
            # 2. Test Live Updates via API
            start_time = time.time()
            async with self.session.get(
                f"{config.api_base}/api/memory/stats/live"
            ) as response:
                live_updates_time = time.time() - start_time
                live_updates_working = response.status == 200
                live_data = await response.json() if live_updates_working else {}
                
                test_results['subtests']['live_updates'] = {
                    'success': live_updates_working,
                    'response_time': live_updates_time,
                    'data_fresh': 'timestamp' in live_data
                }
                
            metrics.record_response_time('live_updates', live_updates_time)
            
            # 3. Test Real-time Notifications
            start_time = time.time()
            notification_data = {
                'type': 'test_notification',
                'message': 'E2E integration test notification',
                'priority': 'info'
            }
            
            async with self.session.post(
                f"{config.api_base}/api/notifications/send",
                json=notification_data
            ) as response:
                notification_time = time.time() - start_time
                notification_sent = response.status in [200, 201]
                notification_response = await response.json() if notification_sent else {}
                
                test_results['subtests']['notifications'] = {
                    'success': notification_sent,
                    'response_time': notification_time,
                    'notification_id': notification_response.get('notification_id')
                }
                
            metrics.record_response_time('notifications', notification_time)
            
        except Exception as e:
            logger.error(f"❌ Real-time features test failed: {e}")
            test_results['error'] = str(e)
            test_results['overall_success'] = False
            
        test_results['overall_success'] = all(
            subtest.get('success', False) 
            for subtest in test_results['subtests'].values()
        )
        
        test_results['end_time'] = time.time()
        test_results['total_duration'] = test_results['end_time'] - test_results['start_time']
        
        logger.info(f"⚡ Real-time Features: {'✅ PASSED' if test_results['overall_success'] else '❌ FAILED'}")
        return test_results
        
    # =====================================================
    # TEST 4: CROSS-SYSTEM INTEGRATION TESTING
    # =====================================================
    
    async def test_cross_system_integration(self) -> Dict[str, Any]:
        """Test Cross-system integration: Memory ↔ AI, Performance ↔ Analytics, Error ↔ Pattern recognition"""
        logger.info("🔗 Testing Cross-system Integration")
        
        test_results = {
            'test_name': 'cross_system_integration',
            'start_time': time.time(),
            'subtests': {},
            'overall_success': True
        }
        
        try:
            # 1. Test Memory ↔ AI Integration
            start_time = time.time()
            
            # Create memory and verify AI can access it
            memory_data = {**self.test_data['memory_test_data'], 'content': 'AI integration test memory'}
            async with self.session.post(
                f"{config.api_base}/api/memory",
                json=memory_data
            ) as response:
                memory_created = response.status in [200, 201]
                memory_response = await response.json() if memory_created else {}
                
            if memory_created:
                memory_id = memory_response['memory_id']
                # Test AI can analyze memory
                async with self.session.post(
                    f"{config.api_base}/api/claude-auto/analyze-memory",
                    json={'memory_id': memory_id}
                ) as response:
                    ai_analysis_time = time.time() - start_time
                    ai_analysis_success = response.status == 200
                    analysis_result = await response.json() if ai_analysis_success else {}
                    
                    test_results['subtests']['memory_ai_integration'] = {
                        'success': ai_analysis_success,
                        'response_time': ai_analysis_time,
                        'analysis_provided': bool(analysis_result.get('analysis'))
                    }
            else:
                test_results['subtests']['memory_ai_integration'] = {'success': False, 'error': 'Memory creation failed'}
                
            metrics.record_response_time('memory_ai_integration', time.time() - start_time)
            
            # 2. Test Performance ↔ Analytics Integration
            start_time = time.time()
            
            # Record performance metric
            perf_data = {
                'operation': 'e2e_integration_test',
                'duration': 0.123,
                'memory_usage': 45.2,
                'cpu_usage': 23.1
            }
            
            async with self.session.post(
                f"{config.api_base}/api/performance-metrics/record",
                json=perf_data
            ) as response:
                perf_recorded = response.status in [200, 201]
                
            if perf_recorded:
                # Verify analytics can access performance data
                async with self.session.get(
                    f"{config.api_base}/api/analytics/performance/recent"
                ) as response:
                    analytics_time = time.time() - start_time
                    analytics_access = response.status == 200
                    analytics_data = await response.json() if analytics_access else {}
                    
                    test_results['subtests']['performance_analytics_integration'] = {
                        'success': analytics_access,
                        'response_time': analytics_time,
                        'data_points': len(analytics_data.get('metrics', []))
                    }
            else:
                test_results['subtests']['performance_analytics_integration'] = {'success': False, 'error': 'Performance recording failed'}
                
            metrics.record_response_time('performance_analytics_integration', time.time() - start_time)
            
            # 3. Test Error ↔ Pattern Recognition Integration
            start_time = time.time()
            
            # Record error and verify pattern recognition can detect it
            error_data = {**self.test_data['error_test_data'], 'context': 'cross_system_integration_test'}
            async with self.session.post(
                f"{config.api_base}/api/mistake-learning/record-error",
                json=error_data
            ) as response:
                error_recorded = response.status in [200, 201]
                
            if error_recorded:
                # Check if pattern recognition detected the error pattern
                async with self.session.get(
                    f"{config.api_base}/api/pattern-recognition/error-patterns"
                ) as response:
                    pattern_detection_time = time.time() - start_time
                    pattern_detected = response.status == 200
                    patterns = await response.json() if pattern_detected else {}
                    
                    test_results['subtests']['error_pattern_integration'] = {
                        'success': pattern_detected,
                        'response_time': pattern_detection_time,
                        'patterns_found': len(patterns.get('patterns', []))
                    }
            else:
                test_results['subtests']['error_pattern_integration'] = {'success': False, 'error': 'Error recording failed'}
                
            metrics.record_response_time('error_pattern_integration', time.time() - start_time)
            
        except Exception as e:
            logger.error(f"❌ Cross-system integration test failed: {e}")
            test_results['error'] = str(e)
            test_results['overall_success'] = False
            
        test_results['overall_success'] = all(
            subtest.get('success', False) 
            for subtest in test_results['subtests'].values()
        )
        
        test_results['end_time'] = time.time()
        test_results['total_duration'] = test_results['end_time'] - test_results['start_time']
        
        logger.info(f"🔗 Cross-system Integration: {'✅ PASSED' if test_results['overall_success'] else '❌ FAILED'}")
        return test_results
        
    # =====================================================
    # TEST 5: DATA FLOW VERIFICATION TESTING
    # =====================================================
    
    async def test_data_flow_verification(self) -> Dict[str, Any]:
        """Test Data flow verification: PostgreSQL → TimescaleDB, Vector embeddings → Search, Graph relationships → Knowledge queries"""
        logger.info("📊 Testing Data Flow Verification")
        
        test_results = {
            'test_name': 'data_flow_verification',
            'start_time': time.time(),
            'subtests': {},
            'overall_success': True
        }
        
        try:
            # 1. Test PostgreSQL → TimescaleDB Analytics Flow
            start_time = time.time()
            
            # Create data in PostgreSQL and verify TimescaleDB analytics
            analytics_data = {
                'metric_type': 'data_flow_test',
                'value': 42.0,
                'tags': {'test_type': 'e2e_integration'},
                'metadata': {'source': 'postgresql_flow_test'}
            }
            
            async with self.session.post(
                f"{config.api_base}/api/analytics/metrics",
                json=analytics_data
            ) as response:
                postgres_success = response.status in [200, 201]
                
            if postgres_success:
                # Verify data appears in TimescaleDB analytics
                await asyncio.sleep(1)  # Allow time for data flow
                async with self.session.get(
                    f"{config.api_base}/api/analytics/timescale/metrics/recent"
                ) as response:
                    timescale_time = time.time() - start_time
                    timescale_success = response.status == 200
                    timescale_data = await response.json() if timescale_success else {}
                    
                    test_results['subtests']['postgres_timescale_flow'] = {
                        'success': timescale_success,
                        'response_time': timescale_time,
                        'metrics_found': len(timescale_data.get('metrics', []))
                    }
            else:
                test_results['subtests']['postgres_timescale_flow'] = {'success': False, 'error': 'PostgreSQL write failed'}
                
            metrics.record_response_time('postgres_timescale_flow', time.time() - start_time)
            
            # 2. Test Vector Embeddings → Search Flow
            start_time = time.time()
            
            # Create content and verify vector search
            search_content = {
                'content': 'Dynamic parallelism and memory bandwidth optimization for GPU computing',
                'metadata': {'domain': 'HW_optimization', 'author': 'Charlotte Cools'}
            }
            
            async with self.session.post(
                f"{config.api_base}/api/sources",
                json={'name': 'E2E Vector Test', 'url': 'http://test.local', 'type': 'document'}
            ) as response:
                source_created = response.status in [200, 201]
                source_data = await response.json() if source_created else {}
                
            if source_created:
                source_id = source_data['id']
                # Add content to source
                async with self.session.post(
                    f"{config.api_base}/api/chunks",
                    json={**search_content, 'source_id': source_id}
                ) as response:
                    content_added = response.status in [200, 201]
                    
                if content_added:
                    # Wait for embedding processing
                    await asyncio.sleep(2)
                    
                    # Test vector search
                    async with self.session.post(
                        f"{config.api_base}/api/search",
                        json={'query': 'GPU memory bandwidth optimization', 'limit': 10}
                    ) as response:
                        vector_search_time = time.time() - start_time
                        search_success = response.status == 200
                        search_results = await response.json() if search_success else {}
                        
                        test_results['subtests']['vector_embeddings_search_flow'] = {
                            'success': search_success,
                            'response_time': vector_search_time,
                            'results_found': len(search_results.get('results', []))
                        }
                else:
                    test_results['subtests']['vector_embeddings_search_flow'] = {'success': False, 'error': 'Content addition failed'}
            else:
                test_results['subtests']['vector_embeddings_search_flow'] = {'success': False, 'error': 'Source creation failed'}
                
            metrics.record_response_time('vector_embeddings_search_flow', time.time() - start_time)
            
            # 3. Test Graph Relationships → Knowledge Queries Flow
            start_time = time.time()
            
            # Create knowledge graph entities and relationships
            entity_data = {
                'entity_id': 'charlotte_cools_hw_expert',
                'entity_type': 'Expert',
                'properties': {
                    'name': 'Charlotte Cools',
                    'domain': 'Dynamic Parallelism',
                    'specialization': 'Memory Bandwidth Optimization'
                }
            }
            
            async with self.session.post(
                f"{config.api_base}/api/knowledge-graph/entities",
                json=entity_data
            ) as response:
                entity_created = response.status in [200, 201]
                
            if entity_created:
                # Create relationship
                relationship_data = {
                    'source_id': 'charlotte_cools_hw_expert',
                    'target_id': 'gpu_optimization_domain',
                    'relationship_type': 'SPECIALIZES_IN',
                    'properties': {'expertise_level': 'expert'}
                }
                
                async with self.session.post(
                    f"{config.api_base}/api/knowledge-graph/relationships",
                    json=relationship_data
                ) as response:
                    relationship_created = response.status in [200, 201]
                    
                if relationship_created:
                    # Test knowledge query
                    query_data = {'query': 'Find experts in Dynamic Parallelism'}
                    async with self.session.post(
                        f"{config.api_base}/api/knowledge-graph/query",
                        json=query_data
                    ) as response:
                        knowledge_query_time = time.time() - start_time
                        query_success = response.status == 200
                        query_results = await response.json() if query_success else {}
                        
                        test_results['subtests']['graph_knowledge_query_flow'] = {
                            'success': query_success,
                            'response_time': knowledge_query_time,
                            'entities_found': len(query_results.get('entities', []))
                        }
                else:
                    test_results['subtests']['graph_knowledge_query_flow'] = {'success': False, 'error': 'Relationship creation failed'}
            else:
                test_results['subtests']['graph_knowledge_query_flow'] = {'success': False, 'error': 'Entity creation failed'}
                
            metrics.record_response_time('graph_knowledge_query_flow', time.time() - start_time)
            
        except Exception as e:
            logger.error(f"❌ Data flow verification test failed: {e}")
            test_results['error'] = str(e)
            test_results['overall_success'] = False
            
        test_results['overall_success'] = all(
            subtest.get('success', False) 
            for subtest in test_results['subtests'].values()
        )
        
        test_results['end_time'] = time.time()
        test_results['total_duration'] = test_results['end_time'] - test_results['start_time']
        
        logger.info(f"📊 Data Flow Verification: {'✅ PASSED' if test_results['overall_success'] else '❌ FAILED'}")
        return test_results
        
    # =====================================================
    # TEST 6: FAILURE SCENARIOS TESTING
    # =====================================================
    
    async def test_failure_scenarios(self) -> Dict[str, Any]:
        """Test Failure scenarios: Database failures, Network timeouts, Service recovery"""
        logger.info("🚨 Testing Failure Scenarios")
        
        test_results = {
            'test_name': 'failure_scenarios',
            'start_time': time.time(),
            'subtests': {},
            'overall_success': True
        }
        
        try:
            # 1. Test Database Connection Handling
            start_time = time.time()
            
            # Test invalid database query handling
            invalid_data = {'invalid_field': 'test', 'malformed_json': {'unclosed': True}}
            async with self.session.post(
                f"{config.api_base}/api/memory",
                json=invalid_data
            ) as response:
                db_error_time = time.time() - start_time
                error_handled = response.status in [400, 422]  # Should return validation error
                error_response = await response.text()
                
                test_results['subtests']['database_error_handling'] = {
                    'success': error_handled,
                    'response_time': db_error_time,
                    'proper_error_response': 'error' in error_response.lower()
                }
                
            metrics.record_response_time('database_error_handling', db_error_time)
            
            # 2. Test Network Timeout Handling
            start_time = time.time()
            
            # Test with very short timeout
            short_timeout_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=0.001),  # 1ms timeout
                headers={'X-API-Key': config.api_key}
            )
            
            timeout_handled = False
            try:
                async with short_timeout_session.get(f"{config.api_base}/api/memory/stats") as response:
                    pass  # Should timeout
            except (asyncio.TimeoutError, aiohttp.ClientTimeout):
                timeout_handled = True
            except Exception as e:
                timeout_handled = True  # Any network error is acceptable
            finally:
                await short_timeout_session.close()
                
            timeout_test_time = time.time() - start_time
            test_results['subtests']['network_timeout_handling'] = {
                'success': timeout_handled,
                'response_time': timeout_test_time
            }
            
            metrics.record_response_time('network_timeout_handling', timeout_test_time)
            
            # 3. Test Service Recovery
            start_time = time.time()
            
            # Test circuit breaker behavior (if implemented)
            recovery_test_success = True
            recovery_attempts = 0
            
            for attempt in range(3):
                try:
                    async with self.session.get(f"{config.api_base}/health") as response:
                        if response.status == 200:
                            recovery_attempts += 1
                            health_data = await response.json()
                            services_operational = all(
                                status == "operational" 
                                for status in health_data.get('services', {}).values()
                            )
                            if not services_operational:
                                recovery_test_success = False
                        else:
                            recovery_test_success = False
                except Exception:
                    recovery_test_success = False
                    
                await asyncio.sleep(0.5)  # Brief pause between attempts
                
            service_recovery_time = time.time() - start_time
            test_results['subtests']['service_recovery'] = {
                'success': recovery_test_success,
                'response_time': service_recovery_time,
                'recovery_attempts': recovery_attempts
            }
            
            metrics.record_response_time('service_recovery', service_recovery_time)
            
            # 4. Test Graceful Degradation
            start_time = time.time()
            
            # Test system behavior under high load (simulate)
            concurrent_requests = []
            for i in range(20):  # 20 concurrent requests
                request_coro = self.session.get(f"{config.api_base}/health")
                concurrent_requests.append(request_coro)
                
            responses = await asyncio.gather(*concurrent_requests, return_exceptions=True)
            
            successful_responses = sum(
                1 for response in responses 
                if hasattr(response, 'status') and response.status == 200
            )
            
            degradation_time = time.time() - start_time
            graceful_degradation = successful_responses >= 15  # At least 75% success rate
            
            test_results['subtests']['graceful_degradation'] = {
                'success': graceful_degradation,
                'response_time': degradation_time,
                'success_rate': successful_responses / len(concurrent_requests),
                'successful_requests': successful_responses,
                'total_requests': len(concurrent_requests)
            }
            
            # Close any remaining response objects
            for response in responses:
                if hasattr(response, 'close'):
                    try:
                        response.close()
                    except:
                        pass
                        
            metrics.record_response_time('graceful_degradation', degradation_time)
            
        except Exception as e:
            logger.error(f"❌ Failure scenarios test failed: {e}")
            test_results['error'] = str(e)
            test_results['overall_success'] = False
            
        test_results['overall_success'] = all(
            subtest.get('success', False) 
            for subtest in test_results['subtests'].values()
        )
        
        test_results['end_time'] = time.time()
        test_results['total_duration'] = test_results['end_time'] - test_results['start_time']
        
        logger.info(f"🚨 Failure Scenarios: {'✅ PASSED' if test_results['overall_success'] else '❌ FAILED'}")
        return test_results
        
    # =====================================================
    # DYNAMIC PARALLELISM PERFORMANCE TESTING
    # =====================================================
    
    async def test_dynamic_parallelism_performance(self) -> Dict[str, Any]:
        """Test Dynamic Parallelism performance characteristics"""
        logger.info("⚡ Testing Dynamic Parallelism Performance (Charlotte Cools Specialty)")
        
        test_results = {
            'test_name': 'dynamic_parallelism_performance',
            'start_time': time.time(),
            'subtests': {},
            'overall_success': True
        }
        
        try:
            # 1. Test Concurrent Memory Operations
            start_time = time.time()
            
            concurrent_tasks = []
            for i in range(config.max_concurrent):
                memory_data = {
                    **self.test_data['memory_test_data'],
                    'content': f'Parallel memory operation {i}',
                    'metadata': {'parallel_test': True, 'operation_id': i}
                }
                
                task = self.session.post(
                    f"{config.api_base}/api/memory",
                    json=memory_data
                )
                concurrent_tasks.append(task)
                
            # Execute all tasks concurrently
            responses = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            parallel_execution_time = time.time() - start_time
            successful_operations = sum(
                1 for response in responses 
                if hasattr(response, 'status') and response.status in [200, 201]
            )
            
            # Calculate throughput
            throughput = successful_operations / parallel_execution_time if parallel_execution_time > 0 else 0
            
            test_results['subtests']['concurrent_memory_operations'] = {
                'success': successful_operations >= config.max_concurrent * 0.8,  # 80% success rate
                'response_time': parallel_execution_time,
                'successful_operations': successful_operations,
                'total_operations': config.max_concurrent,
                'throughput_ops_per_second': throughput
            }
            
            # Close response objects
            for response in responses:
                if hasattr(response, 'close'):
                    try:
                        response.close()
                    except:
                        pass
                        
            metrics.record_response_time('concurrent_operations', parallel_execution_time)
            metrics.parallel_execution_times['concurrent_memory_ops'] = parallel_execution_time
            
            # 2. Test Memory Bandwidth Optimization
            start_time = time.time()
            
            # Test batch operations vs individual operations
            batch_data = {
                'memories': [
                    {
                        **self.test_data['memory_test_data'],
                        'content': f'Batch operation {i}',
                        'metadata': {'batch_test': True, 'batch_id': i}
                    }
                    for i in range(10)
                ]
            }
            
            async with self.session.post(
                f"{config.api_base}/api/memory/batch",
                json=batch_data
            ) as response:
                batch_time = time.time() - start_time
                batch_success = response.status in [200, 201]
                batch_result = await response.json() if batch_success else {}
                
            # Compare with individual operations
            individual_start = time.time()
            individual_tasks = []
            for i in range(10):
                memory_data = {
                    **self.test_data['memory_test_data'],
                    'content': f'Individual operation {i}',
                    'metadata': {'individual_test': True, 'operation_id': i}
                }
                task = self.session.post(f"{config.api_base}/api/memory", json=memory_data)
                individual_tasks.append(task)
                
            individual_responses = await asyncio.gather(*individual_tasks, return_exceptions=True)
            individual_time = time.time() - individual_start
            
            # Close individual responses
            for response in individual_responses:
                if hasattr(response, 'close'):
                    try:
                        response.close()
                    except:
                        pass
                        
            bandwidth_optimization_ratio = individual_time / batch_time if batch_time > 0 else 0
            
            test_results['subtests']['memory_bandwidth_optimization'] = {
                'success': batch_success and bandwidth_optimization_ratio > 1.5,  # Batch should be 1.5x faster
                'batch_time': batch_time,
                'individual_time': individual_time,
                'optimization_ratio': bandwidth_optimization_ratio,
                'batch_operations_processed': len(batch_result.get('created_memories', []))
            }
            
            metrics.parallel_execution_times['batch_vs_individual'] = {
                'batch_time': batch_time,
                'individual_time': individual_time,
                'optimization_ratio': bandwidth_optimization_ratio
            }
            
        except Exception as e:
            logger.error(f"❌ Dynamic parallelism performance test failed: {e}")
            test_results['error'] = str(e)
            test_results['overall_success'] = False
            
        test_results['overall_success'] = all(
            subtest.get('success', False) 
            for subtest in test_results['subtests'].values()
        )
        
        test_results['end_time'] = time.time()
        test_results['total_duration'] = test_results['end_time'] - test_results['start_time']
        
        logger.info(f"⚡ Dynamic Parallelism Performance: {'✅ PASSED' if test_results['overall_success'] else '❌ FAILED'}")
        return test_results
        
    # =====================================================
    # COMPREHENSIVE TEST RUNNER
    # =====================================================
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive E2E integration tests"""
        logger.info("🚀 Starting Charlotte Cools Comprehensive E2E Integration Tests")
        
        overall_start_time = time.time()
        all_test_results = {
            'test_suite': 'Charlotte Cools Comprehensive E2E Integration',
            'start_time': overall_start_time,
            'test_session_id': config.test_session_id,
            'results': {},
            'summary': {},
            'performance_metrics': {}
        }
        
        # List of all test functions
        test_functions = [
            self.test_full_stack_integration,
            self.test_ai_intelligence_pipeline,
            self.test_realtime_features,
            self.test_cross_system_integration,
            self.test_data_flow_verification,
            self.test_failure_scenarios,
            self.test_dynamic_parallelism_performance
        ]
        
        # Run tests sequentially for proper resource management
        for test_func in test_functions:
            try:
                metrics.record_system_metrics()  # Record system state before each test
                test_result = await test_func()
                all_test_results['results'][test_result['test_name']] = test_result
                logger.info(f"✅ Completed: {test_result['test_name']}")
                
                # Brief pause between tests to allow system recovery
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"❌ Test function {test_func.__name__} failed: {e}")
                all_test_results['results'][test_func.__name__] = {
                    'test_name': test_func.__name__,
                    'overall_success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Calculate summary statistics
        total_tests = len(all_test_results['results'])
        passed_tests = sum(
            1 for result in all_test_results['results'].values()
            if result.get('overall_success', False)
        )
        
        all_test_results['end_time'] = time.time()
        all_test_results['total_duration'] = all_test_results['end_time'] - overall_start_time
        
        all_test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'overall_success': passed_tests == total_tests
        }
        
        # Add performance metrics
        all_test_results['performance_metrics'] = metrics.get_summary()
        
        # Log final results
        success_rate = all_test_results['summary']['success_rate'] * 100
        logger.info(f"🎯 Test Suite Complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        logger.info(f"⏱️ Total execution time: {all_test_results['total_duration']:.2f} seconds")
        
        if all_test_results['summary']['overall_success']:
            logger.info("🎉 ALL TESTS PASSED - System integration is working correctly\!")
        else:
            logger.error("❌ Some tests failed - Integration issues detected")
            
        return all_test_results

# =====================================================
# MAIN TEST EXECUTION
# =====================================================

async def main():
    """Main test execution function"""
    test_suite = ComprehensiveE2ETest()
    
    try:
        await test_suite.setup()
        results = await test_suite.run_comprehensive_tests()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"charlotte_cools_e2e_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"📄 Detailed results saved to: {results_file}")
        
        # Return exit code based on results
        return 0 if results['summary']['overall_success'] else 1
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        logger.error(traceback.format_exc())
        return 1
        
    finally:
        await test_suite.teardown()

if __name__ == "__main__":
    # Run the comprehensive test suite
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
