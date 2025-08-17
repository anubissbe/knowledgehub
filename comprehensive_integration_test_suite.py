#!/usr/bin/env python3
"""
KnowledgeHub Hybrid RAG System - Comprehensive Integration Testing Suite
=========================================================================

This test suite validates the complete transformation of the KnowledgeHub system
including:
- New hybrid RAG backend with LangGraph orchestration
- Database migration with enhanced schemas  
- New web UI with agent workflow visualization
- Integrated services (Zep, Firecrawl, Graphiti, Phoenix, Qdrant)
- DevOps infrastructure with monitoring and deployment

Test Categories:
1. System Integration Testing
2. Performance Validation
3. Feature Testing
4. Migration Validation
5. Security and Reliability Testing
"""

import asyncio
import time
import json
import requests
import psycopg2
import redis
import pytest
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import sys
import os
from urllib.parse import urljoin
import subprocess
import tempfile
import yaml

# Test configuration
TEST_CONFIG = {
    'api_base_url': 'http://localhost:3000',
    'webui_base_url': 'http://localhost:3100',
    'postgres_url': 'postgresql://knowledgehub:knowledgehub123@localhost:5433/knowledgehub',
    'redis_url': 'redis://localhost:6381/0',
    'weaviate_url': 'http://localhost:8090',
    'qdrant_url': 'http://localhost:6333',
    'neo4j_url': 'bolt://localhost:7687',
    'zep_url': 'http://localhost:8100',
    'ai_service_url': 'http://localhost:8002',
    'minio_url': 'http://localhost:9010',
    'timescale_url': 'postgresql://knowledgehub:knowledgehub123@localhost:5434/knowledgehub_analytics',
    'timeout': 30,
    'max_concurrent_requests': 10,
    'performance_thresholds': {
        'api_response_time': 2000,  # ms
        'search_response_time': 5000,  # ms
        'ui_load_time': 3000,  # ms
        'memory_usage_mb': 512,
        'cpu_usage_percent': 80
    }
}

@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    category: str
    status: str  # 'passed', 'failed', 'skipped'
    duration_ms: int
    details: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class IntegrationTestSuite:
    """Comprehensive integration test suite for KnowledgeHub hybrid RAG system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.results: List[TestResult] = []
        self.session = requests.Session()
        self.session.timeout = TEST_CONFIG['timeout']
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _time_function(self, func, *args, **kwargs) -> Tuple[Any, int]:
        """Time function execution and return result with duration"""
        start_time = time.time()
        result = func(*args, **kwargs)
        duration_ms = int((time.time() - start_time) * 1000)
        return result, duration_ms
    
    def _record_result(self, name: str, category: str, status: str, 
                      duration_ms: int, details: Dict[str, Any], 
                      error: str = None):
        """Record test result"""
        result = TestResult(
            name=name,
            category=category,
            status=status,
            duration_ms=duration_ms,
            details=details,
            error=error
        )
        self.results.append(result)
        
        # Log result
        status_emoji = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
        self.logger.info(f"{status_emoji} {name} ({duration_ms}ms) - {status}")
        if error:
            self.logger.error(f"Error details: {error}")
    
    # ===========================================
    # 1. SYSTEM INTEGRATION TESTING
    # ===========================================
    
    def test_service_health_checks(self):
        """Test health of all system services"""
        services = [
            ('API Gateway', f"{TEST_CONFIG['api_base_url']}/health"),
            ('Web UI', f"{TEST_CONFIG['webui_base_url']}/"),
            ('AI Service', f"{TEST_CONFIG['ai_service_url']}/health"),
            ('Weaviate', f"{TEST_CONFIG['weaviate_url']}/v1/.well-known/ready"),
            ('Qdrant', f"{TEST_CONFIG['qdrant_url']}/health"),
            ('Zep Memory', f"{TEST_CONFIG['zep_url']}/health"),
            ('MinIO', f"{TEST_CONFIG['minio_url']}/minio/health/live"),
        ]
        
        for service_name, url in services:
            try:
                response, duration = self._time_function(
                    self.session.get, url, timeout=10
                )
                
                if response.status_code == 200:
                    self._record_result(
                        f"Health Check: {service_name}",
                        "System Integration",
                        "passed",
                        duration,
                        {"status_code": response.status_code, "url": url}
                    )
                else:
                    self._record_result(
                        f"Health Check: {service_name}",
                        "System Integration", 
                        "failed",
                        duration,
                        {"status_code": response.status_code, "url": url},
                        f"Unexpected status code: {response.status_code}"
                    )
            except Exception as e:
                self._record_result(
                    f"Health Check: {service_name}",
                    "System Integration",
                    "failed",
                    0,
                    {"url": url},
                    str(e)
                )
    
    def test_database_connections(self):
        """Test database connectivity and schema validation"""
        databases = [
            ('PostgreSQL', TEST_CONFIG['postgres_url']),
            ('TimescaleDB', TEST_CONFIG['timescale_url']),
        ]
        
        for db_name, connection_url in databases:
            try:
                def test_db_connection():
                    conn = psycopg2.connect(connection_url)
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    cursor.close()
                    conn.close()
                    return result
                
                result, duration = self._time_function(test_db_connection)
                
                self._record_result(
                    f"Database Connection: {db_name}",
                    "System Integration",
                    "passed",
                    duration,
                    {"connection_url": connection_url.split('@')[1] if '@' in connection_url else connection_url}
                )
            except Exception as e:
                self._record_result(
                    f"Database Connection: {db_name}",
                    "System Integration", 
                    "failed",
                    0,
                    {"connection_url": connection_url.split('@')[1] if '@' in connection_url else connection_url},
                    str(e)
                )
    
    def test_redis_connectivity(self):
        """Test Redis cache connectivity"""
        try:
            def test_redis():
                r = redis.from_url(TEST_CONFIG['redis_url'])
                r.ping()
                # Test set/get
                r.set('test_key', 'test_value', ex=10)
                value = r.get('test_key')
                r.delete('test_key')
                return value.decode() if value else None
            
            result, duration = self._time_function(test_redis)
            
            if result == 'test_value':
                self._record_result(
                    "Redis Connectivity",
                    "System Integration",
                    "passed",
                    duration,
                    {"operation": "ping, set, get, delete"}
                )
            else:
                self._record_result(
                    "Redis Connectivity",
                    "System Integration",
                    "failed", 
                    duration,
                    {"operation": "ping, set, get, delete"},
                    f"Unexpected value: {result}"
                )
        except Exception as e:
            self._record_result(
                "Redis Connectivity",
                "System Integration",
                "failed",
                0,
                {},
                str(e)
            )
    
    def test_migration_validation(self):
        """Validate database migrations were applied correctly"""
        expected_tables = [
            'ai_memories', 'memory_clusters', 'memory_associations', 'memory_access_logs',
            'agent_definitions', 'workflow_definitions', 'workflow_executions', 'agent_tasks',
            'rag_configurations', 'rag_query_logs', 'enhanced_chunks', 'document_ingestion_logs',
            'search_result_cache', 'zep_session_mapping', 'firecrawl_jobs',
            'service_health_logs', 'performance_monitoring'
        ]
        
        try:
            def check_migrations():
                conn = psycopg2.connect(TEST_CONFIG['postgres_url'])
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """)
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                # Check migration log
                cursor.execute("SELECT migration_name, completed_at FROM migration_log ORDER BY completed_at DESC LIMIT 5")
                migrations = cursor.fetchall()
                
                cursor.close()
                conn.close()
                
                return existing_tables, migrations
            
            (existing_tables, migrations), duration = self._time_function(check_migrations)
            
            missing_tables = [table for table in expected_tables if table not in existing_tables]
            
            if not missing_tables:
                self._record_result(
                    "Database Migration Validation",
                    "Migration Validation",
                    "passed",
                    duration,
                    {
                        "tables_found": len(existing_tables),
                        "migrations_applied": len(migrations),
                        "latest_migration": migrations[0] if migrations else None
                    }
                )
            else:
                self._record_result(
                    "Database Migration Validation",
                    "Migration Validation",
                    "failed",
                    duration,
                    {
                        "missing_tables": missing_tables,
                        "tables_found": len(existing_tables)
                    },
                    f"Missing tables: {missing_tables}"
                )
        except Exception as e:
            self._record_result(
                "Database Migration Validation",
                "Migration Validation",
                "failed",
                0,
                {},
                str(e)
            )
    
    # ===========================================
    # 2. PERFORMANCE VALIDATION
    # ===========================================
    
    def test_api_performance(self):
        """Test API endpoint performance under load"""
        endpoints = [
            '/health',
            '/api/memory/session/test_user/summary',
            '/api/ai-features/status',
            '/api/rag/search',
        ]
        
        for endpoint in endpoints:
            try:
                url = urljoin(TEST_CONFIG['api_base_url'], endpoint)
                
                # Test single request performance
                if endpoint == '/api/rag/search':
                    data = {"query": "test query", "limit": 10}
                    response, duration = self._time_function(
                        self.session.post, url, json=data
                    )
                else:
                    response, duration = self._time_function(
                        self.session.get, url
                    )
                
                threshold = TEST_CONFIG['performance_thresholds']['api_response_time']
                status = "passed" if duration <= threshold else "failed"
                error = f"Response time {duration}ms exceeds threshold {threshold}ms" if status == "failed" else None
                
                self._record_result(
                    f"API Performance: {endpoint}",
                    "Performance Validation",
                    status,
                    duration,
                    {
                        "status_code": response.status_code,
                        "threshold_ms": threshold,
                        "endpoint": endpoint
                    },
                    error
                )
            except Exception as e:
                self._record_result(
                    f"API Performance: {endpoint}",
                    "Performance Validation",
                    "failed",
                    0,
                    {"endpoint": endpoint},
                    str(e)
                )
    
    def test_concurrent_requests(self):
        """Test system under concurrent load"""
        def make_request(request_id):
            try:
                url = f"{TEST_CONFIG['api_base_url']}/health"
                start_time = time.time()
                response = self.session.get(url)
                duration = int((time.time() - start_time) * 1000)
                return {
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'duration_ms': duration,
                    'success': response.status_code == 200
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'error': str(e),
                    'success': False
                }
        
        try:
            num_requests = TEST_CONFIG['max_concurrent_requests']
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_requests)]
                results = [future.result() for future in as_completed(futures)]
            total_duration = int((time.time() - start_time) * 1000)
            
            successful_requests = [r for r in results if r.get('success', False)]
            success_rate = len(successful_requests) / len(results) * 100
            avg_response_time = sum(r.get('duration_ms', 0) for r in successful_requests) / len(successful_requests) if successful_requests else 0
            
            status = "passed" if success_rate >= 95 and avg_response_time <= TEST_CONFIG['performance_thresholds']['api_response_time'] else "failed"
            
            self._record_result(
                "Concurrent Load Test",
                "Performance Validation",
                status,
                total_duration,
                {
                    "concurrent_requests": num_requests,
                    "success_rate": success_rate,
                    "avg_response_time_ms": avg_response_time,
                    "total_duration_ms": total_duration
                }
            )
        except Exception as e:
            self._record_result(
                "Concurrent Load Test",
                "Performance Validation",
                "failed",
                0,
                {},
                str(e)
            )
    
    def test_memory_usage(self):
        """Test system memory usage"""
        try:
            def check_memory():
                # Use docker stats to check memory usage
                result = subprocess.run(
                    ['docker', 'stats', '--no-stream', '--format', 'table {{.Container}}\t{{.MemUsage}}'],
                    capture_output=True, text=True, timeout=10
                )
                return result.stdout
            
            output, duration = self._time_function(check_memory)
            
            # Parse memory usage
            lines = output.strip().split('\n')[1:]  # Skip header
            containers = {}
            for line in lines:
                if '\t' in line:
                    container, mem_usage = line.split('\t')
                    if 'knowledgehub' in container:
                        containers[container] = mem_usage
            
            self._record_result(
                "Memory Usage Check",
                "Performance Validation",
                "passed",
                duration,
                {"containers": containers}
            )
        except Exception as e:
            self._record_result(
                "Memory Usage Check",
                "Performance Validation",
                "failed",
                0,
                {},
                str(e)
            )
    
    # ===========================================
    # 3. FEATURE TESTING
    # ===========================================
    
    def test_hybrid_rag_functionality(self):
        """Test hybrid RAG search functionality"""
        try:
            # Test search query
            search_data = {
                "query": "What is artificial intelligence?",
                "retrieval_mode": "hybrid",
                "limit": 10
            }
            
            url = f"{TEST_CONFIG['api_base_url']}/api/rag/search"
            response, duration = self._time_function(
                self.session.post, url, json=search_data
            )
            
            if response.status_code == 200:
                data = response.json()
                threshold = TEST_CONFIG['performance_thresholds']['search_response_time']
                status = "passed" if duration <= threshold else "failed"
                
                self._record_result(
                    "Hybrid RAG Search",
                    "Feature Testing",
                    status,
                    duration,
                    {
                        "results_count": len(data.get('results', [])),
                        "retrieval_mode": data.get('metadata', {}).get('retrieval_mode'),
                        "query": search_data['query']
                    }
                )
            else:
                self._record_result(
                    "Hybrid RAG Search",
                    "Feature Testing",
                    "failed",
                    duration,
                    {"status_code": response.status_code},
                    f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            self._record_result(
                "Hybrid RAG Search",
                "Feature Testing",
                "failed",
                0,
                {},
                str(e)
            )
    
    def test_agent_workflow_execution(self):
        """Test agent workflow execution"""
        try:
            # Test workflow execution
            workflow_data = {
                "workflow_name": "simple_qa",
                "input_data": {
                    "query": "What are the benefits of machine learning?",
                    "context": "technology discussion"
                }
            }
            
            url = f"{TEST_CONFIG['api_base_url']}/api/agent-workflows/execute"
            response, duration = self._time_function(
                self.session.post, url, json=workflow_data
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                self._record_result(
                    "Agent Workflow Execution",
                    "Feature Testing",
                    "passed",
                    duration,
                    {
                        "workflow_id": data.get('workflow_execution_id'),
                        "status": data.get('status'),
                        "workflow_name": workflow_data['workflow_name']
                    }
                )
            else:
                self._record_result(
                    "Agent Workflow Execution",
                    "Feature Testing",
                    "failed",
                    duration,
                    {"status_code": response.status_code},
                    f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            self._record_result(
                "Agent Workflow Execution",
                "Feature Testing",
                "failed",
                0,
                {},
                str(e)
            )
    
    def test_memory_system_functionality(self):
        """Test enhanced memory system"""
        try:
            # Test memory creation
            memory_data = {
                "content": "Test memory for integration testing",
                "memory_type": "test",
                "context": {"test": True},
                "importance": "high"
            }
            
            url = f"{TEST_CONFIG['api_base_url']}/api/memory"
            response, duration = self._time_function(
                self.session.post, url, json=memory_data
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                memory_id = data.get('id')
                
                # Test memory retrieval
                search_url = f"{TEST_CONFIG['api_base_url']}/api/memory/search"
                search_data = {"query": "integration testing", "limit": 5}
                search_response, search_duration = self._time_function(
                    self.session.post, search_url, json=search_data
                )
                
                if search_response.status_code == 200:
                    search_results = search_response.json()
                    
                    self._record_result(
                        "Memory System Functionality",
                        "Feature Testing",
                        "passed",
                        duration + search_duration,
                        {
                            "memory_created": memory_id,
                            "search_results": len(search_results.get('memories', [])),
                            "total_operations": 2
                        }
                    )
                else:
                    self._record_result(
                        "Memory System Functionality",
                        "Feature Testing",
                        "failed",
                        duration + search_duration,
                        {"memory_created": memory_id},
                        f"Search failed: HTTP {search_response.status_code}"
                    )
            else:
                self._record_result(
                    "Memory System Functionality",
                    "Feature Testing",
                    "failed",
                    duration,
                    {"status_code": response.status_code},
                    f"Memory creation failed: HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            self._record_result(
                "Memory System Functionality",
                "Feature Testing",
                "failed",
                0,
                {},
                str(e)
            )
    
    def test_web_ui_functionality(self):
        """Test web UI functionality and pages"""
        pages = [
            '/',
            '/ai-intelligence',
            '/memory-system',
            '/search-knowledge',
            '/sources',
            '/agent-workflows',
            '/hybrid-rag-dashboard'
        ]
        
        for page in pages:
            try:
                url = urljoin(TEST_CONFIG['webui_base_url'], page)
                response, duration = self._time_function(
                    self.session.get, url
                )
                
                threshold = TEST_CONFIG['performance_thresholds']['ui_load_time']
                
                if response.status_code == 200:
                    status = "passed" if duration <= threshold else "failed"
                    error = f"Load time {duration}ms exceeds threshold {threshold}ms" if status == "failed" else None
                    
                    self._record_result(
                        f"Web UI Page: {page}",
                        "Feature Testing",
                        status,
                        duration,
                        {
                            "status_code": response.status_code,
                            "content_length": len(response.content),
                            "threshold_ms": threshold
                        },
                        error
                    )
                else:
                    self._record_result(
                        f"Web UI Page: {page}",
                        "Feature Testing",
                        "failed",
                        duration,
                        {"status_code": response.status_code},
                        f"HTTP {response.status_code}"
                    )
            except Exception as e:
                self._record_result(
                    f"Web UI Page: {page}",
                    "Feature Testing",
                    "failed",
                    0,
                    {"page": page},
                    str(e)
                )
    
    # ===========================================
    # 4. SECURITY AND RELIABILITY TESTING
    # ===========================================
    
    def test_error_handling(self):
        """Test system error handling and recovery"""
        error_scenarios = [
            ("Invalid API endpoint", "/api/nonexistent", "GET", None),
            ("Invalid JSON payload", "/api/memory", "POST", "invalid json"),
            ("Large payload", "/api/memory", "POST", {"content": "x" * 10000}),
            ("Missing required fields", "/api/rag/search", "POST", {}),
        ]
        
        for scenario_name, endpoint, method, payload in error_scenarios:
            try:
                url = urljoin(TEST_CONFIG['api_base_url'], endpoint)
                
                if method == "GET":
                    response, duration = self._time_function(
                        self.session.get, url
                    )
                elif method == "POST":
                    if payload == "invalid json":
                        # Send invalid JSON
                        response, duration = self._time_function(
                            self.session.post, url, data="invalid json", 
                            headers={'Content-Type': 'application/json'}
                        )
                    else:
                        response, duration = self._time_function(
                            self.session.post, url, json=payload
                        )
                
                # Error handling is considered good if we get 4xx status codes for invalid requests
                expected_error = response.status_code >= 400 and response.status_code < 500
                status = "passed" if expected_error else "failed"
                
                self._record_result(
                    f"Error Handling: {scenario_name}",
                    "Security and Reliability",
                    status,
                    duration,
                    {
                        "status_code": response.status_code,
                        "endpoint": endpoint,
                        "method": method
                    }
                )
            except Exception as e:
                # For some scenarios, exceptions are expected (like invalid JSON)
                if "invalid json" in scenario_name.lower():
                    self._record_result(
                        f"Error Handling: {scenario_name}",
                        "Security and Reliability",
                        "passed",
                        0,
                        {"expected_exception": True},
                        f"Expected exception: {str(e)}"
                    )
                else:
                    self._record_result(
                        f"Error Handling: {scenario_name}",
                        "Security and Reliability",
                        "failed",
                        0,
                        {"endpoint": endpoint},
                        str(e)
                    )
    
    def test_data_integrity(self):
        """Test data integrity across services"""
        try:
            # Create test data and verify it's consistent across services
            test_data = {
                "content": f"Data integrity test {datetime.now().isoformat()}",
                "memory_type": "integration_test",
                "metadata": {"test_id": "data_integrity_test"}
            }
            
            # Create memory
            url = f"{TEST_CONFIG['api_base_url']}/api/memory"
            response, duration = self._time_function(
                self.session.post, url, json=test_data
            )
            
            if response.status_code in [200, 201]:
                memory_data = response.json()
                memory_id = memory_data.get('id')
                
                # Verify in database
                def check_database():
                    conn = psycopg2.connect(TEST_CONFIG['postgres_url'])
                    cursor = conn.cursor()
                    cursor.execute("SELECT content, memory_type FROM ai_memories WHERE id = %s", (memory_id,))
                    result = cursor.fetchone()
                    cursor.close()
                    conn.close()
                    return result
                
                db_result, db_duration = self._time_function(check_database)
                
                if db_result and db_result[0] == test_data['content']:
                    self._record_result(
                        "Data Integrity Check",
                        "Security and Reliability",
                        "passed",
                        duration + db_duration,
                        {
                            "memory_id": memory_id,
                            "verified_in_database": True,
                            "content_match": True
                        }
                    )
                else:
                    self._record_result(
                        "Data Integrity Check",
                        "Security and Reliability",
                        "failed",
                        duration + db_duration,
                        {
                            "memory_id": memory_id,
                            "db_result": db_result
                        },
                        "Data mismatch between API and database"
                    )
            else:
                self._record_result(
                    "Data Integrity Check",
                    "Security and Reliability",
                    "failed",
                    duration,
                    {"status_code": response.status_code},
                    f"Failed to create test data: HTTP {response.status_code}"
                )
        except Exception as e:
            self._record_result(
                "Data Integrity Check",
                "Security and Reliability",
                "failed",
                0,
                {},
                str(e)
            )
    
    # ===========================================
    # 5. REPORTING AND EXECUTION
    # ===========================================
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "passed"])
        failed_tests = len([r for r in self.results if r.status == "failed"])
        skipped_tests = len([r for r in self.results if r.status == "skipped"])
        
        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {
                    'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []
                }
            categories[result.category]['total'] += 1
            categories[result.category][result.status] += 1
            categories[result.category]['tests'].append({
                'name': result.name,
                'status': result.status,
                'duration_ms': result.duration_ms,
                'error': result.error
            })
        
        # Calculate performance metrics
        total_duration = sum(r.duration_ms for r in self.results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'skipped': skipped_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_duration_ms': total_duration,
                'avg_duration_ms': avg_duration,
                'timestamp': datetime.now().isoformat()
            },
            'categories': categories,
            'detailed_results': [
                {
                    'name': r.name,
                    'category': r.category,
                    'status': r.status,
                    'duration_ms': r.duration_ms,
                    'details': r.details,
                    'error': r.error,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ],
            'system_info': {
                'test_config': TEST_CONFIG,
                'test_timestamp': datetime.now().isoformat()
            }
        }
        
        return report
    
    def run_all_tests(self):
        """Execute all test categories"""
        self.logger.info("🚀 Starting KnowledgeHub Hybrid RAG System Integration Tests")
        
        # 1. System Integration Testing
        self.logger.info("📡 Running System Integration Tests...")
        self.test_service_health_checks()
        self.test_database_connections()
        self.test_redis_connectivity()
        
        # 2. Migration Validation
        self.logger.info("🔄 Running Migration Validation...")
        self.test_migration_validation()
        
        # 3. Performance Validation
        self.logger.info("⚡ Running Performance Validation...")
        self.test_api_performance()
        self.test_concurrent_requests()
        self.test_memory_usage()
        
        # 4. Feature Testing
        self.logger.info("🧪 Running Feature Tests...")
        self.test_hybrid_rag_functionality()
        self.test_agent_workflow_execution()
        self.test_memory_system_functionality()
        self.test_web_ui_functionality()
        
        # 5. Security and Reliability
        self.logger.info("🔒 Running Security and Reliability Tests...")
        self.test_error_handling()
        self.test_data_integrity()
        
        self.logger.info("✅ All tests completed!")
        return self.generate_report()

def main():
    """Main execution function"""
    try:
        # Create test suite
        test_suite = IntegrationTestSuite()
        
        # Run all tests
        report = test_suite.run_all_tests()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"integration_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report['summary']
        print(f"\n{'='*60}")
        print(f"KNOWLEDGEHUB INTEGRATION TEST REPORT")
        print(f"{'='*60}")
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏭️ Skipped: {summary['skipped']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️ Total Duration: {summary['total_duration_ms']}ms")
        print(f"📄 Report saved to: {report_file}")
        
        # Print category breakdown
        print(f"\n📂 CATEGORY BREAKDOWN:")
        for category, stats in report['categories'].items():
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Show failed tests
        failed_tests = [r for r in test_suite.results if r.status == "failed"]
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test.name}: {test.error}")
        
        print(f"\n{'='*60}")
        
        # Exit with error code if tests failed
        if summary['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()