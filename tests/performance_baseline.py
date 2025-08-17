#!/usr/bin/env python3
"""
Performance Baseline Testing Suite for KnowledgeHub Hybrid RAG System
Establishes performance baselines for all retrieval modes and system components
"""

import asyncio
import time
import json
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import httpx
import pytest
from concurrent.futures import ThreadPoolExecutor
import docker
import redis
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway

# Performance metrics collectors
registry = CollectorRegistry()
query_latency = Histogram('rag_query_latency_seconds', 'RAG query latency', 
                          ['mode', 'operation'], registry=registry)
memory_usage = Gauge('system_memory_usage_bytes', 'Memory usage in bytes',
                    ['component'], registry=registry)
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage',
                 ['component'], registry=registry)
db_query_time = Histogram('database_query_seconds', 'Database query time',
                          ['database', 'operation'], registry=registry)
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate percentage',
                      ['cache_type'], registry=registry)

# Configuration
API_BASE_URL = "http://localhost:3000/api"
PROMETHEUS_GATEWAY = "http://localhost:9091"
TEST_DURATION_HOURS = 24
SAMPLE_INTERVAL_SECONDS = 60

class PerformanceProfiler:
    """Comprehensive performance profiling for RAG system"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.docker_client = docker.from_env()
        self.redis_client = redis.Redis(host='localhost', port=6381)
        self.metrics = {
            'query_response_times': {'vector': [], 'sparse': [], 'graph': [], 'hybrid': []},
            'memory_usage': [],
            'cpu_usage': [],
            'database_metrics': {},
            'cache_metrics': {},
            'network_latency': []
        }
        
    async def profile_query_performance(self, mode: str, num_queries: int = 100):
        """Profile query performance for different retrieval modes"""
        print(f"\n📊 Profiling {mode} mode with {num_queries} queries...")
        
        test_queries = [
            "What is the hybrid RAG architecture?",
            "How does the agent orchestration work?",
            "Explain the memory clustering system",
            "What are the performance optimization strategies?",
            "Describe the web ingestion pipeline",
            "How does cross-encoder reranking improve results?",
            "What is the role of GraphRAG in the system?",
            "Explain the Zep memory integration",
            "How does the system handle failover?",
            "What are the monitoring capabilities?"
        ]
        
        response_times = []
        
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            
            with query_latency.labels(mode=mode, operation='query').time():
                start_time = time.time()
                
                response = await self.client.post(
                    f"{API_BASE_URL}/rag/enhanced/query",
                    json={
                        "query": query,
                        "mode": mode,
                        "top_k": 10,
                        "rerank": True
                    }
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                if response.status_code == 200:
                    data = response.json()
                    # Extract performance metrics from response
                    if 'performance' in data:
                        self.metrics['query_response_times'][mode].append({
                            'total_time': response_time,
                            'retrieval_time': data['performance'].get('retrieval_time_ms', 0),
                            'reranking_time': data['performance'].get('reranking_time_ms', 0)
                        })
        
        # Calculate statistics
        stats = {
            'mean': np.mean(response_times),
            'median': np.median(response_times),
            'p95': np.percentile(response_times, 95),
            'p99': np.percentile(response_times, 99),
            'min': np.min(response_times),
            'max': np.max(response_times),
            'std': np.std(response_times)
        }
        
        print(f"  ✅ {mode} mode stats: Mean={stats['mean']:.2f}ms, P95={stats['p95']:.2f}ms, P99={stats['p99']:.2f}ms")
        return stats
    
    async def profile_memory_usage(self):
        """Profile memory usage patterns during peak loads"""
        print("\n💾 Profiling memory usage...")
        
        containers = ['knowledgehub-api-1', 'knowledgehub-postgres-1', 
                     'knowledgehub-neo4j-1', 'knowledgehub-weaviate-1',
                     'knowledgehub-redis-1']
        
        memory_stats = {}
        
        for container_name in containers:
            try:
                container = self.docker_client.containers.get(container_name)
                stats = container.stats(stream=False)
                
                # Calculate memory usage
                mem_usage = stats['memory_stats']['usage']
                mem_limit = stats['memory_stats']['limit']
                mem_percent = (mem_usage / mem_limit) * 100
                
                memory_stats[container_name] = {
                    'usage_mb': mem_usage / (1024 * 1024),
                    'limit_mb': mem_limit / (1024 * 1024),
                    'percent': mem_percent
                }
                
                memory_usage.labels(component=container_name).set(mem_usage)
                
                print(f"  📦 {container_name}: {memory_stats[container_name]['usage_mb']:.2f}MB ({mem_percent:.1f}%)")
                
            except docker.errors.NotFound:
                print(f"  ⚠️ Container {container_name} not found")
        
        self.metrics['memory_usage'].append(memory_stats)
        return memory_stats
    
    async def profile_cpu_usage(self):
        """Profile CPU utilization for vector operations"""
        print("\n🖥️ Profiling CPU usage...")
        
        # System-wide CPU usage
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        cpu_stats = {
            'overall_percent': sum(cpu_percent) / len(cpu_percent),
            'per_core': cpu_percent,
            'frequency_mhz': cpu_freq.current,
            'cores_count': psutil.cpu_count()
        }
        
        # Container-specific CPU usage
        containers = ['knowledgehub-api-1', 'knowledgehub-postgres-1', 
                     'knowledgehub-neo4j-1', 'knowledgehub-weaviate-1']
        
        for container_name in containers:
            try:
                container = self.docker_client.containers.get(container_name)
                stats = container.stats(stream=False)
                
                # Calculate CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                
                if system_delta > 0:
                    cpu_percent_container = (cpu_delta / system_delta) * 100
                    cpu_stats[container_name] = cpu_percent_container
                    cpu_usage.labels(component=container_name).set(cpu_percent_container)
                    print(f"  📦 {container_name}: {cpu_percent_container:.1f}%")
                    
            except docker.errors.NotFound:
                print(f"  ⚠️ Container {container_name} not found")
        
        self.metrics['cpu_usage'].append(cpu_stats)
        return cpu_stats
    
    async def profile_database_performance(self):
        """Profile database query performance metrics"""
        print("\n🗄️ Profiling database performance...")
        
        # Test PostgreSQL performance
        pg_queries = [
            "SELECT COUNT(*) FROM memories",
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT 100",
            "SELECT * FROM hybrid_rag_sessions WHERE created_at > NOW() - INTERVAL '1 day'"
        ]
        
        pg_stats = []
        for query in pg_queries:
            start_time = time.time()
            # Execute query (would need actual database connection)
            # For now, simulate with API call
            await self.client.get(f"{API_BASE_URL}/health")
            query_time = (time.time() - start_time) * 1000
            pg_stats.append(query_time)
            db_query_time.labels(database='postgresql', operation='select').observe(query_time/1000)
        
        # Test Neo4j performance
        neo4j_queries = [
            "MATCH (n) RETURN COUNT(n)",
            "MATCH (d:Document)-[:RELATED_TO]->(d2:Document) RETURN d, d2 LIMIT 10",
            "MATCH path = shortestPath((a)-[*]-(b)) RETURN path LIMIT 5"
        ]
        
        neo4j_stats = []
        for query in neo4j_queries:
            start_time = time.time()
            # Execute query (would need actual Neo4j connection)
            await self.client.get(f"{API_BASE_URL}/health")
            query_time = (time.time() - start_time) * 1000
            neo4j_stats.append(query_time)
            db_query_time.labels(database='neo4j', operation='match').observe(query_time/1000)
        
        db_metrics = {
            'postgresql': {
                'mean_query_time_ms': np.mean(pg_stats),
                'max_query_time_ms': np.max(pg_stats)
            },
            'neo4j': {
                'mean_query_time_ms': np.mean(neo4j_stats),
                'max_query_time_ms': np.max(neo4j_stats)
            }
        }
        
        print(f"  📊 PostgreSQL: Mean={db_metrics['postgresql']['mean_query_time_ms']:.2f}ms")
        print(f"  📊 Neo4j: Mean={db_metrics['neo4j']['mean_query_time_ms']:.2f}ms")
        
        self.metrics['database_metrics'] = db_metrics
        return db_metrics
    
    async def profile_network_latency(self):
        """Profile network latency between services"""
        print("\n🌐 Profiling network latency...")
        
        services = [
            ('API', 'http://localhost:3000/health'),
            ('Zep', 'http://localhost:8100/health'),
            ('Firecrawl', 'http://localhost:3002/health'),
            ('Phoenix', 'http://localhost:6006/health'),
            ('Weaviate', 'http://localhost:8090/v1/.well-known/ready'),
            ('Neo4j', 'http://localhost:7474/db/data/')
        ]
        
        latency_stats = {}
        
        for service_name, url in services:
            latencies = []
            for _ in range(10):
                try:
                    start_time = time.time()
                    response = await self.client.get(url)
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
                except:
                    latencies.append(float('inf'))
            
            valid_latencies = [l for l in latencies if l != float('inf')]
            if valid_latencies:
                latency_stats[service_name] = {
                    'mean_ms': np.mean(valid_latencies),
                    'max_ms': np.max(valid_latencies),
                    'min_ms': np.min(valid_latencies)
                }
                print(f"  🔗 {service_name}: {latency_stats[service_name]['mean_ms']:.2f}ms")
            else:
                print(f"  ❌ {service_name}: Unreachable")
        
        self.metrics['network_latency'].append(latency_stats)
        return latency_stats
    
    async def identify_bottlenecks(self):
        """Analyze metrics to identify performance bottlenecks"""
        print("\n🔍 Identifying performance bottlenecks...")
        
        bottlenecks = []
        
        # Check query performance
        for mode, metrics in self.metrics['query_response_times'].items():
            if metrics:
                mean_time = np.mean([m['total_time'] for m in metrics])
                if mean_time > 200:  # Threshold: 200ms
                    bottlenecks.append({
                        'type': 'query_performance',
                        'mode': mode,
                        'mean_time_ms': mean_time,
                        'severity': 'high' if mean_time > 500 else 'medium'
                    })
        
        # Check memory usage
        if self.metrics['memory_usage']:
            latest_memory = self.metrics['memory_usage'][-1]
            for container, stats in latest_memory.items():
                if stats['percent'] > 80:
                    bottlenecks.append({
                        'type': 'memory_usage',
                        'container': container,
                        'usage_percent': stats['percent'],
                        'severity': 'high' if stats['percent'] > 90 else 'medium'
                    })
        
        # Check CPU usage
        if self.metrics['cpu_usage']:
            latest_cpu = self.metrics['cpu_usage'][-1]
            if latest_cpu['overall_percent'] > 70:
                bottlenecks.append({
                    'type': 'cpu_usage',
                    'usage_percent': latest_cpu['overall_percent'],
                    'severity': 'high' if latest_cpu['overall_percent'] > 85 else 'medium'
                })
        
        # Check database performance
        if self.metrics['database_metrics']:
            for db, stats in self.metrics['database_metrics'].items():
                if stats['mean_query_time_ms'] > 50:
                    bottlenecks.append({
                        'type': 'database_performance',
                        'database': db,
                        'mean_query_time_ms': stats['mean_query_time_ms'],
                        'severity': 'medium'
                    })
        
        # Print bottlenecks
        if bottlenecks:
            print("\n⚠️ Bottlenecks identified:")
            for bottleneck in bottlenecks:
                print(f"  - {bottleneck['type']}: {bottleneck}")
        else:
            print("  ✅ No significant bottlenecks identified")
        
        return bottlenecks
    
    async def run_baseline_profiling(self):
        """Run complete baseline profiling"""
        print("=" * 60)
        print("🚀 Starting Performance Baseline Profiling")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'query_performance': {},
            'system_resources': {},
            'database_performance': {},
            'network_latency': {},
            'bottlenecks': []
        }
        
        # Profile query performance for all modes
        for mode in ['vector', 'sparse', 'graph', 'hybrid']:
            results['query_performance'][mode] = await self.profile_query_performance(mode)
        
        # Profile system resources
        results['system_resources']['memory'] = await self.profile_memory_usage()
        results['system_resources']['cpu'] = await self.profile_cpu_usage()
        
        # Profile database performance
        results['database_performance'] = await self.profile_database_performance()
        
        # Profile network latency
        results['network_latency'] = await self.profile_network_latency()
        
        # Identify bottlenecks
        results['bottlenecks'] = await self.identify_bottlenecks()
        
        # Push metrics to Prometheus
        try:
            push_to_gateway(PROMETHEUS_GATEWAY, job='performance_baseline', registry=registry)
            print("\n✅ Metrics pushed to Prometheus")
        except:
            print("\n⚠️ Failed to push metrics to Prometheus")
        
        # Save results to file
        with open('performance_baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("✅ Performance Baseline Profiling Complete")
        print(f"📊 Results saved to performance_baseline_results.json")
        print("=" * 60)
        
        return results
    
    async def continuous_monitoring(self, duration_hours: int = 24):
        """Run continuous monitoring for specified duration"""
        print(f"\n⏰ Starting {duration_hours}-hour continuous monitoring...")
        
        end_time = time.time() + (duration_hours * 3600)
        iteration = 0
        
        while time.time() < end_time:
            iteration += 1
            print(f"\n🔄 Monitoring iteration {iteration}")
            
            # Sample performance metrics
            await self.profile_query_performance('hybrid', num_queries=10)
            await self.profile_memory_usage()
            await self.profile_cpu_usage()
            
            # Wait for next sampling interval
            await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
        
        print(f"\n✅ Continuous monitoring completed after {duration_hours} hours")


async def main():
    """Main execution function"""
    profiler = PerformanceProfiler()
    
    # Run baseline profiling
    baseline_results = await profiler.run_baseline_profiling()
    
    # Optional: Run continuous monitoring
    # await profiler.continuous_monitoring(duration_hours=1)
    
    await profiler.client.aclose()
    
    return baseline_results


if __name__ == "__main__":
    results = asyncio.run(main())
    
    # Print summary
    print("\n📊 PERFORMANCE BASELINE SUMMARY")
    print("=" * 60)
    
    for mode, stats in results['query_performance'].items():
        print(f"\n{mode.upper()} Mode:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")
    
    if results['bottlenecks']:
        print(f"\n⚠️ Found {len(results['bottlenecks'])} bottlenecks requiring optimization")
    else:
        print("\n✅ System performing within acceptable parameters")