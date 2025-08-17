#!/usr/bin/env python3
"""
KnowledgeHub Hybrid RAG System - Performance and Load Testing
============================================================

Comprehensive performance testing for the hybrid RAG transformation including:
- Load testing for concurrent users
- RAG performance benchmarking
- Memory system performance validation
- Agent workflow performance testing
- Database and vector store performance
- Real-time monitoring during tests
"""

import asyncio
import aiohttp
import time
import statistics
import threading
import psutil
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import random
import string
import subprocess
import sys

# Performance test configuration
PERF_CONFIG = {
    'api_base_url': 'http://localhost:3000',
    'concurrent_users': [1, 5, 10, 25, 50],
    'test_duration_seconds': 60,
    'ramp_up_seconds': 10,
    'think_time_seconds': 1,
    'requests_per_user': 100,
    'performance_thresholds': {
        'api_p95_ms': 2000,
        'rag_search_p95_ms': 5000,
        'memory_query_p95_ms': 1000,
        'workflow_p95_ms': 10000,
        'error_rate_percent': 5.0,
        'throughput_rps_min': 10,
        'cpu_usage_max_percent': 85,
        'memory_usage_max_mb': 2048
    },
    'sample_queries': [
        "What is machine learning?",
        "Explain artificial intelligence concepts",
        "How does neural network training work?",
        "What are the benefits of deep learning?",
        "Describe natural language processing",
        "What is computer vision?",
        "How do recommendation systems work?",
        "Explain reinforcement learning",
        "What is the difference between AI and ML?",
        "How do transformers work in NLP?"
    ]
}

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    timestamp: str
    test_name: str
    concurrent_users: int
    response_time_ms: float
    status_code: int
    success: bool
    error: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None

@dataclass 
class LoadTestResult:
    """Load test result summary"""
    test_name: str
    concurrent_users: int
    duration_seconds: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate_percent: float
    throughput_rps: float
    response_time_stats: Dict[str, float]
    resource_usage: Dict[str, float]
    timestamp: str

class PerformanceMonitor:
    """Real-time system performance monitor"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics
        
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Docker container metrics (if available)
                docker_stats = self._get_docker_stats()
                
                metric = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': memory.used / 1024 / 1024,
                    'disk_percent': disk.percent,
                    'docker_stats': docker_stats
                }
                self.metrics.append(metric)
                
            except Exception as e:
                logging.warning(f"Monitoring error: {e}")
            
            time.sleep(5)  # Monitor every 5 seconds
            
    def _get_docker_stats(self) -> Dict[str, Any]:
        """Get Docker container statistics"""
        try:
            result = subprocess.run(
                ['docker', 'stats', '--no-stream', '--format', 
                 'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                stats = {}
                for line in lines:
                    if 'knowledgehub' in line:
                        parts = line.split('\t')
                        if len(parts) >= 4:
                            container = parts[0]
                            stats[container] = {
                                'cpu_percent': parts[1],
                                'memory_usage': parts[2],
                                'memory_percent': parts[3]
                            }
                return stats
        except Exception as e:
            logging.warning(f"Docker stats error: {e}")
        
        return {}

class HybridRAGLoadTester:
    """Comprehensive load tester for hybrid RAG system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.metrics: List[PerformanceMetric] = []
        self.monitor = PerformanceMonitor()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _generate_test_user_id(self) -> str:
        """Generate random test user ID"""
        return f"test_user_{''.join(random.choices(string.ascii_lowercase, k=8))}"
    
    def _random_query(self) -> str:
        """Get random query from sample set"""
        return random.choice(PERF_CONFIG['sample_queries'])
    
    async def _make_request(self, session: aiohttp.ClientSession, url: str, 
                           method: str = 'GET', data: Dict = None, 
                           user_id: str = None) -> PerformanceMetric:
        """Make HTTP request and record performance metric"""
        start_time = time.time()
        endpoint = url.replace(PERF_CONFIG['api_base_url'], '')
        
        try:
            if method.upper() == 'POST':
                async with session.post(url, json=data) as response:
                    await response.text()  # Read response
                    response_time = (time.time() - start_time) * 1000
                    
                    return PerformanceMetric(
                        timestamp=datetime.now().isoformat(),
                        test_name="api_request",
                        concurrent_users=0,  # Will be set by caller
                        response_time_ms=response_time,
                        status_code=response.status,
                        success=response.status < 400,
                        user_id=user_id,
                        endpoint=endpoint
                    )
            else:
                async with session.get(url) as response:
                    await response.text()  # Read response
                    response_time = (time.time() - start_time) * 1000
                    
                    return PerformanceMetric(
                        timestamp=datetime.now().isoformat(),
                        test_name="api_request",
                        concurrent_users=0,  # Will be set by caller
                        response_time_ms=response_time,
                        status_code=response.status,
                        success=response.status < 400,
                        user_id=user_id,
                        endpoint=endpoint
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                test_name="api_request",
                concurrent_users=0,
                response_time_ms=response_time,
                status_code=0,
                success=False,
                error=str(e),
                user_id=user_id,
                endpoint=endpoint
            )
    
    async def _user_simulation(self, user_id: str, session: aiohttp.ClientSession, 
                              duration_seconds: int) -> List[PerformanceMetric]:
        """Simulate single user behavior"""
        metrics = []
        start_time = time.time()
        
        while (time.time() - start_time) < duration_seconds:
            # Simulate different user actions
            actions = [
                self._simulate_rag_search,
                self._simulate_memory_query,
                self._simulate_workflow_execution,
                self._simulate_health_check
            ]
            
            # Weight actions (RAG search is most common)
            weights = [0.4, 0.3, 0.2, 0.1]
            action = random.choices(actions, weights=weights)[0]
            
            try:
                metric = await action(session, user_id)
                metrics.append(metric)
                
                # Think time between requests
                await asyncio.sleep(PERF_CONFIG['think_time_seconds'])
                
            except Exception as e:
                self.logger.error(f"User {user_id} action failed: {e}")
        
        return metrics
    
    async def _simulate_rag_search(self, session: aiohttp.ClientSession, 
                                  user_id: str) -> PerformanceMetric:
        """Simulate RAG search request"""
        url = f"{PERF_CONFIG['api_base_url']}/api/rag/search"
        data = {
            "query": self._random_query(),
            "retrieval_mode": random.choice(["dense", "sparse", "hybrid"]),
            "limit": random.randint(5, 20)
        }
        
        metric = await self._make_request(session, url, 'POST', data, user_id)
        metric.test_name = "rag_search"
        return metric
    
    async def _simulate_memory_query(self, session: aiohttp.ClientSession, 
                                    user_id: str) -> PerformanceMetric:
        """Simulate memory search request"""
        url = f"{PERF_CONFIG['api_base_url']}/api/memory/search"
        data = {
            "query": self._random_query(),
            "limit": random.randint(5, 15)
        }
        
        metric = await self._make_request(session, url, 'POST', data, user_id)
        metric.test_name = "memory_search"
        return metric
    
    async def _simulate_workflow_execution(self, session: aiohttp.ClientSession, 
                                          user_id: str) -> PerformanceMetric:
        """Simulate agent workflow execution"""
        url = f"{PERF_CONFIG['api_base_url']}/api/agent-workflows/execute"
        data = {
            "workflow_name": "simple_qa",
            "input_data": {
                "query": self._random_query(),
                "context": "performance test"
            }
        }
        
        metric = await self._make_request(session, url, 'POST', data, user_id)
        metric.test_name = "workflow_execution"
        return metric
    
    async def _simulate_health_check(self, session: aiohttp.ClientSession, 
                                    user_id: str) -> PerformanceMetric:
        """Simulate health check request"""
        url = f"{PERF_CONFIG['api_base_url']}/health"
        
        metric = await self._make_request(session, url, 'GET', None, user_id)
        metric.test_name = "health_check"
        return metric
    
    async def _run_load_test(self, concurrent_users: int, 
                            duration_seconds: int) -> LoadTestResult:
        """Run load test with specified concurrent users"""
        self.logger.info(f"🚀 Starting load test: {concurrent_users} users for {duration_seconds}s")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Configure session
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        start_time = time.time()
        user_metrics = []
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create user tasks
            tasks = []
            for i in range(concurrent_users):
                user_id = self._generate_test_user_id()
                task = asyncio.create_task(
                    self._user_simulation(user_id, session, duration_seconds)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect metrics
                for result in results:
                    if isinstance(result, list):
                        user_metrics.extend(result)
                    elif isinstance(result, Exception):
                        self.logger.error(f"Task failed: {result}")
                        
            except Exception as e:
                self.logger.error(f"Load test failed: {e}")
        
        # Stop monitoring
        resource_metrics = self.monitor.stop_monitoring()
        
        # Calculate statistics
        total_duration = time.time() - start_time
        
        # Set concurrent_users for all metrics
        for metric in user_metrics:
            metric.concurrent_users = concurrent_users
        
        return self._calculate_load_test_stats(
            "load_test", concurrent_users, total_duration, 
            user_metrics, resource_metrics
        )
    
    def _calculate_load_test_stats(self, test_name: str, concurrent_users: int,
                                  duration_seconds: float, metrics: List[PerformanceMetric],
                                  resource_metrics: List[Dict]) -> LoadTestResult:
        """Calculate load test statistics"""
        
        if not metrics:
            return LoadTestResult(
                test_name=test_name,
                concurrent_users=concurrent_users,
                duration_seconds=int(duration_seconds),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate_percent=100.0,
                throughput_rps=0.0,
                response_time_stats={},
                resource_usage={},
                timestamp=datetime.now().isoformat()
            )
        
        # Request statistics
        total_requests = len(metrics)
        successful_requests = len([m for m in metrics if m.success])
        failed_requests = total_requests - successful_requests
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
        throughput = successful_requests / duration_seconds if duration_seconds > 0 else 0
        
        # Response time statistics
        response_times = [m.response_time_ms for m in metrics if m.success]
        
        if response_times:
            response_time_stats = {
                'min': min(response_times),
                'max': max(response_times),
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'p95': self._percentile(response_times, 95),
                'p99': self._percentile(response_times, 99)
            }
        else:
            response_time_stats = {}
        
        # Resource usage statistics
        resource_usage = {}
        if resource_metrics:
            cpu_values = [m['cpu_percent'] for m in resource_metrics if 'cpu_percent' in m]
            memory_values = [m['memory_used_mb'] for m in resource_metrics if 'memory_used_mb' in m]
            
            if cpu_values:
                resource_usage['cpu_percent_avg'] = statistics.mean(cpu_values)
                resource_usage['cpu_percent_max'] = max(cpu_values)
            
            if memory_values:
                resource_usage['memory_mb_avg'] = statistics.mean(memory_values)
                resource_usage['memory_mb_max'] = max(memory_values)
        
        return LoadTestResult(
            test_name=test_name,
            concurrent_users=concurrent_users,
            duration_seconds=int(duration_seconds),
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate_percent=error_rate,
            throughput_rps=throughput,
            response_time_stats=response_time_stats,
            resource_usage=resource_usage,
            timestamp=datetime.now().isoformat()
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        self.logger.info("🎯 Starting KnowledgeHub Performance Benchmark")
        
        results = []
        
        # Test different concurrent user loads
        for concurrent_users in PERF_CONFIG['concurrent_users']:
            try:
                result = await self._run_load_test(
                    concurrent_users, 
                    PERF_CONFIG['test_duration_seconds']
                )
                results.append(result)
                
                # Brief pause between tests
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Load test failed for {concurrent_users} users: {e}")
        
        # Analyze results
        analysis = self._analyze_performance_results(results)
        
        return {
            'benchmark_timestamp': datetime.now().isoformat(),
            'configuration': PERF_CONFIG,
            'results': [asdict(r) for r in results],
            'analysis': analysis
        }
    
    def _analyze_performance_results(self, results: List[LoadTestResult]) -> Dict[str, Any]:
        """Analyze performance test results"""
        thresholds = PERF_CONFIG['performance_thresholds']
        
        analysis = {
            'overall_status': 'PASSED',
            'threshold_violations': [],
            'performance_summary': {},
            'scalability_analysis': {},
            'recommendations': []
        }
        
        if not results:
            analysis['overall_status'] = 'FAILED'
            analysis['threshold_violations'].append("No test results available")
            return analysis
        
        # Check thresholds
        for result in results:
            concurrent_users = result.concurrent_users
            
            # Error rate check
            if result.error_rate_percent > thresholds['error_rate_percent']:
                analysis['threshold_violations'].append(
                    f"Error rate {result.error_rate_percent:.1f}% exceeds threshold {thresholds['error_rate_percent']}% "
                    f"at {concurrent_users} users"
                )
            
            # Throughput check
            if result.throughput_rps < thresholds['throughput_rps_min']:
                analysis['threshold_violations'].append(
                    f"Throughput {result.throughput_rps:.1f} RPS below minimum {thresholds['throughput_rps_min']} "
                    f"at {concurrent_users} users"
                )
            
            # Response time checks
            if result.response_time_stats and 'p95' in result.response_time_stats:
                p95_time = result.response_time_stats['p95']
                if p95_time > thresholds['api_p95_ms']:
                    analysis['threshold_violations'].append(
                        f"P95 response time {p95_time:.0f}ms exceeds threshold {thresholds['api_p95_ms']}ms "
                        f"at {concurrent_users} users"
                    )
            
            # Resource usage checks
            if result.resource_usage:
                if 'cpu_percent_max' in result.resource_usage:
                    cpu_max = result.resource_usage['cpu_percent_max']
                    if cpu_max > thresholds['cpu_usage_max_percent']:
                        analysis['threshold_violations'].append(
                            f"CPU usage {cpu_max:.1f}% exceeds threshold {thresholds['cpu_usage_max_percent']}% "
                            f"at {concurrent_users} users"
                        )
                
                if 'memory_mb_max' in result.resource_usage:
                    memory_max = result.resource_usage['memory_mb_max']
                    if memory_max > thresholds['memory_usage_max_mb']:
                        analysis['threshold_violations'].append(
                            f"Memory usage {memory_max:.0f}MB exceeds threshold {thresholds['memory_usage_max_mb']}MB "
                            f"at {concurrent_users} users"
                        )
        
        # Set overall status
        if analysis['threshold_violations']:
            analysis['overall_status'] = 'FAILED'
        
        # Performance summary
        max_throughput = max((r.throughput_rps for r in results), default=0)
        min_error_rate = min((r.error_rate_percent for r in results), default=100)
        
        analysis['performance_summary'] = {
            'max_throughput_rps': max_throughput,
            'min_error_rate_percent': min_error_rate,
            'max_concurrent_users_tested': max((r.concurrent_users for r in results), default=0)
        }
        
        # Scalability analysis
        if len(results) >= 2:
            # Compare first and last results
            first = results[0]
            last = results[-1]
            
            throughput_degradation = ((first.throughput_rps - last.throughput_rps) / first.throughput_rps * 100) if first.throughput_rps > 0 else 0
            
            analysis['scalability_analysis'] = {
                'throughput_degradation_percent': throughput_degradation,
                'scalability_rating': 'Good' if throughput_degradation < 20 else 'Poor'
            }
        
        # Recommendations
        if analysis['threshold_violations']:
            analysis['recommendations'].append("Performance optimization required")
            analysis['recommendations'].append("Review system resource allocation")
            analysis['recommendations'].append("Consider horizontal scaling")
        else:
            analysis['recommendations'].append("Performance meets requirements")
            analysis['recommendations'].append("System ready for production load")
        
        return analysis

def main():
    """Main execution function"""
    async def run_tests():
        try:
            # Create load tester
            tester = HybridRAGLoadTester()
            
            # Run performance benchmark
            results = await tester.run_performance_benchmark()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"performance_test_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print summary
            analysis = results['analysis']
            print(f"\n{'='*70}")
            print(f"KNOWLEDGEHUB PERFORMANCE TEST RESULTS")
            print(f"{'='*70}")
            print(f"🎯 Overall Status: {analysis['overall_status']}")
            print(f"📊 Max Throughput: {analysis['performance_summary'].get('max_throughput_rps', 0):.1f} RPS")
            print(f"❌ Min Error Rate: {analysis['performance_summary'].get('min_error_rate_percent', 0):.1f}%")
            print(f"👥 Max Concurrent Users: {analysis['performance_summary'].get('max_concurrent_users_tested', 0)}")
            
            # Scalability analysis
            if 'scalability_analysis' in analysis:
                print(f"📈 Scalability: {analysis['scalability_analysis'].get('scalability_rating', 'Unknown')}")
            
            # Threshold violations
            if analysis['threshold_violations']:
                print(f"\n⚠️ THRESHOLD VIOLATIONS:")
                for violation in analysis['threshold_violations']:
                    print(f"  - {violation}")
            
            # Recommendations
            if analysis['recommendations']:
                print(f"\n💡 RECOMMENDATIONS:")
                for rec in analysis['recommendations']:
                    print(f"  - {rec}")
            
            print(f"\n📄 Detailed results saved to: {results_file}")
            print(f"{'='*70}")
            
            # Exit with appropriate code
            if analysis['overall_status'] == 'FAILED':
                sys.exit(1)
            else:
                sys.exit(0)
                
        except Exception as e:
            print(f"\n💥 Performance test failed: {e}")
            sys.exit(1)
    
    # Run async tests
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n⏹️ Performance tests interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()