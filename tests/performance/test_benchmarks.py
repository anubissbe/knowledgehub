"""
Performance benchmarks for KnowledgeHub.

Measures performance of critical operations and ensures they meet
acceptable thresholds under various load conditions.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from datetime import datetime
from statistics import mean, median
from unittest.mock import MagicMock
from uuid import uuid4

from api.services.memory_service import MemoryService
from api.services.session_service import SessionService
from api.services.ai_service import AIService
from api.services.pattern_service import PatternService


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest_asyncio.fixture
    async def benchmark_services(self, db_session):
        """Create services with performance optimizations for benchmarking."""
        # Mock external services for consistent benchmarking
        ai_service = AIService()
        ai_service.openai_client = MagicMock()
        ai_service.weaviate_client = MagicMock()
        
        # Configure mock responses for consistent timing
        ai_service.openai_client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]
        
        ai_service.weaviate_client.query.get.return_value.with_near_text.return_value.with_limit.return_value.do.return_value = {
            "data": {"Get": {"Memory": []}}
        }
        
        return {
            "memory": MemoryService(),
            "session": SessionService(),
            "ai": ai_service,
            "pattern": PatternService()
        }
    
    @pytest.mark.asyncio
    async def test_memory_creation_performance(self, benchmark_services, performance_timer):
        """Benchmark memory creation performance."""
        memory_service = benchmark_services["memory"]
        
        # Test parameters
        num_memories = 100
        batch_sizes = [1, 10, 50]
        
        results = {}
        
        for batch_size in batch_sizes:
            times = []
            
            for batch_start in range(0, num_memories, batch_size):
                batch_end = min(batch_start + batch_size, num_memories)
                batch_data = [
                    {
                        "content": f"Performance test memory {i}",
                        "memory_type": "performance_test",
                        "user_id": str(uuid4()),
                        "tags": ["performance", "benchmark"]
                    }
                    for i in range(batch_start, batch_end)
                ]
                
                with performance_timer as timer:
                    if batch_size == 1:
                        # Single creation
                        await memory_service.create_memory(**batch_data[0])
                    else:
                        # Bulk creation
                        await memory_service.bulk_create_memories(batch_data)
                
                times.append(timer.elapsed)
            
            results[f"batch_size_{batch_size}"] = {
                "total_time": sum(times),
                "avg_time_per_operation": mean(times) / batch_size,
                "median_time": median(times),
                "operations_per_second": batch_size / mean(times)
            }
        
        # Assert performance thresholds
        single_ops_per_sec = results["batch_size_1"]["operations_per_second"]
        batch_10_ops_per_sec = results["batch_size_10"]["operations_per_second"]
        batch_50_ops_per_sec = results["batch_size_50"]["operations_per_second"]
        
        # Single operations should be at least 10/sec
        assert single_ops_per_sec >= 10, f"Single creation too slow: {single_ops_per_sec} ops/sec"
        
        # Batch operations should be significantly faster
        assert batch_10_ops_per_sec >= single_ops_per_sec * 2, "Batch-10 not providing expected speedup"
        assert batch_50_ops_per_sec >= single_ops_per_sec * 5, "Batch-50 not providing expected speedup"
        
        print(f"\nMemory Creation Performance:")
        for batch_size, stats in results.items():
            print(f"  {batch_size}: {stats['operations_per_second']:.1f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_memory_search_performance(self, benchmark_services, create_test_data, performance_timer):
        """Benchmark memory search performance with different dataset sizes."""
        memory_service = benchmark_services["memory"]
        
        # Create test datasets of different sizes
        dataset_sizes = [100, 500, 1000]
        search_queries = [
            "python function",
            "database query",
            "user interface",
            "error handling",
            "performance optimization"
        ]
        
        results = {}
        
        for size in dataset_sizes:
            # Create test dataset
            test_data = [
                {
                    "content": f"Test memory {i} about programming concepts",
                    "memory_type": "test",
                    "user_id": str(uuid4()),
                    "tags": ["test", f"dataset_{size}"]
                }
                for i in range(size)
            ]
            
            await memory_service.bulk_create_memories(test_data)
            
            # Benchmark search performance
            search_times = []
            
            for query in search_queries:
                with performance_timer as timer:
                    results_found = await memory_service.search_memories(
                        query=query,
                        limit=10,
                        similarity_threshold=0.1
                    )
                
                search_times.append(timer.elapsed)
            
            results[f"dataset_{size}"] = {
                "avg_search_time": mean(search_times),
                "median_search_time": median(search_times),
                "searches_per_second": 1 / mean(search_times)
            }
        
        # Assert performance thresholds
        for size, stats in results.items():
            searches_per_sec = stats["searches_per_second"]
            avg_time = stats["avg_search_time"]
            
            # All searches should complete within 2 seconds
            assert avg_time <= 2.0, f"Search too slow for {size}: {avg_time:.2f}s"
            
            # Should handle at least 1 search per second
            assert searches_per_sec >= 0.5, f"Search throughput too low for {size}: {searches_per_sec:.1f}/sec"
        
        print(f"\nMemory Search Performance:")
        for dataset, stats in results.items():
            print(f"  {dataset}: {stats['searches_per_second']:.1f} searches/sec, {stats['avg_search_time']:.3f}s avg")
    
    @pytest.mark.asyncio
    async def test_session_operations_performance(self, benchmark_services, performance_timer):
        """Benchmark session operation performance."""
        session_service = benchmark_services["session"]
        
        # Test parameters
        num_sessions = 50
        num_updates_per_session = 10
        
        # Benchmark session creation
        session_creation_times = []
        sessions = []
        
        for i in range(num_sessions):
            with performance_timer as timer:
                session = await session_service.create_session(
                    user_id=str(uuid4()),
                    session_type="performance_test",
                    context_data={"test": f"session_{i}"}
                )
            
            session_creation_times.append(timer.elapsed)
            sessions.append(session)
        
        # Benchmark session updates
        update_times = []
        
        for session in sessions[:10]:  # Test subset for updates
            for j in range(num_updates_per_session):
                with performance_timer as timer:
                    await session_service.update_session_context(
                        session_id=session.session_id,
                        context_data={"update": j, "timestamp": datetime.utcnow().isoformat()},
                        merge=True
                    )
                
                update_times.append(timer.elapsed)
        
        # Benchmark session retrieval
        retrieval_times = []
        
        for session in sessions:
            with performance_timer as timer:
                retrieved = await session_service.get_session(session.session_id)
            
            retrieval_times.append(timer.elapsed)
            assert retrieved is not None
        
        # Calculate performance metrics
        creation_ops_per_sec = 1 / mean(session_creation_times)
        update_ops_per_sec = 1 / mean(update_times)
        retrieval_ops_per_sec = 1 / mean(retrieval_times)
        
        # Assert performance thresholds
        assert creation_ops_per_sec >= 20, f"Session creation too slow: {creation_ops_per_sec:.1f} ops/sec"
        assert update_ops_per_sec >= 50, f"Session updates too slow: {update_ops_per_sec:.1f} ops/sec"
        assert retrieval_ops_per_sec >= 100, f"Session retrieval too slow: {retrieval_ops_per_sec:.1f} ops/sec"
        
        print(f"\nSession Operations Performance:")
        print(f"  Creation: {creation_ops_per_sec:.1f} ops/sec")
        print(f"  Updates: {update_ops_per_sec:.1f} ops/sec")
        print(f"  Retrieval: {retrieval_ops_per_sec:.1f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, benchmark_services, performance_timer):
        """Benchmark performance under concurrent load."""
        memory_service = benchmark_services["memory"]
        session_service = benchmark_services["session"]
        
        # Test parameters
        num_concurrent_users = 10
        operations_per_user = 20
        
        async def user_workload(user_id: str, operation_count: int):
            """Simulate a user's workload."""
            timings = {
                "session_ops": [],
                "memory_ops": [],
                "search_ops": []
            }
            
            # Create session
            start_time = time.time()
            session = await session_service.create_session(
                user_id=user_id,
                session_type="concurrent_test"
            )
            timings["session_ops"].append(time.time() - start_time)
            
            # Perform mixed operations
            for i in range(operation_count):
                if i % 3 == 0:
                    # Create memory
                    start_time = time.time()
                    await memory_service.create_memory(
                        content=f"Concurrent test memory {i}",
                        memory_type="concurrent_test",
                        user_id=user_id
                    )
                    timings["memory_ops"].append(time.time() - start_time)
                    
                elif i % 3 == 1:
                    # Update session
                    start_time = time.time()
                    await session_service.update_session_context(
                        session_id=session.session_id,
                        context_data={"operation": i},
                        merge=True
                    )
                    timings["session_ops"].append(time.time() - start_time)
                    
                else:
                    # Search memories
                    start_time = time.time()
                    await memory_service.search_memories(
                        query="concurrent test",
                        user_id=user_id,
                        limit=5
                    )
                    timings["search_ops"].append(time.time() - start_time)
            
            return timings
        
        # Execute concurrent workloads
        with performance_timer as total_timer:
            tasks = [
                user_workload(f"user_{i}", operations_per_user)
                for i in range(num_concurrent_users)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        total_time = total_timer.elapsed
        exceptions = [r for r in results if isinstance(r, Exception)]
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # Assert no exceptions occurred
        assert len(exceptions) == 0, f"Exceptions during concurrent operations: {exceptions}"
        
        # Calculate aggregate performance metrics
        all_session_times = []
        all_memory_times = []
        all_search_times = []
        
        for result in successful_results:
            all_session_times.extend(result["session_ops"])
            all_memory_times.extend(result["memory_ops"])
            all_search_times.extend(result["search_ops"])
        
        total_operations = len(all_session_times) + len(all_memory_times) + len(all_search_times)
        overall_ops_per_sec = total_operations / total_time
        
        # Performance assertions
        assert overall_ops_per_sec >= 50, f"Overall concurrent performance too low: {overall_ops_per_sec:.1f} ops/sec"
        
        if all_session_times:
            avg_session_time = mean(all_session_times)
            assert avg_session_time <= 0.1, f"Session operations too slow under load: {avg_session_time:.3f}s"
        
        if all_memory_times:
            avg_memory_time = mean(all_memory_times)
            assert avg_memory_time <= 0.2, f"Memory operations too slow under load: {avg_memory_time:.3f}s"
        
        if all_search_times:
            avg_search_time = mean(all_search_times)
            assert avg_search_time <= 0.5, f"Search operations too slow under load: {avg_search_time:.3f}s"
        
        print(f"\nConcurrent Operations Performance:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Overall throughput: {overall_ops_per_sec:.1f} ops/sec")
        print(f"  Concurrent users: {num_concurrent_users}")
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, benchmark_services, performance_timer):
        """Test memory usage optimization under large data loads."""
        import psutil
        import os
        
        memory_service = benchmark_services["memory"]
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        large_content_size = 10000  # 10KB per memory
        num_large_memories = 100
        
        large_memories = []
        
        with performance_timer as timer:
            for i in range(num_large_memories):
                content = "x" * large_content_size  # Large content
                memory = await memory_service.create_memory(
                    content=content,
                    memory_type="large_test",
                    user_id=str(uuid4()),
                    tags=["large", "memory_test"]
                )
                large_memories.append(memory)
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Performance assertions
        creation_time = timer.elapsed
        avg_time_per_large_memory = creation_time / num_large_memories
        
        assert avg_time_per_large_memory <= 0.5, f"Large memory creation too slow: {avg_time_per_large_memory:.3f}s"
        assert memory_increase <= 200, f"Memory usage increase too high: {memory_increase:.1f}MB"
        
        # Test search performance with large content
        search_times = []
        for i in range(10):
            with performance_timer as search_timer:
                results = await memory_service.search_memories(
                    query="large memory test",
                    memory_type="large_test",
                    limit=5
                )
            search_times.append(search_timer.elapsed)
        
        avg_search_time = mean(search_times)
        assert avg_search_time <= 1.0, f"Search with large content too slow: {avg_search_time:.3f}s"
        
        print(f"\nMemory Usage Optimization:")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        print(f"  Large memory creation: {avg_time_per_large_memory:.3f}s avg")
        print(f"  Search with large content: {avg_search_time:.3f}s avg")
    
    @pytest.mark.asyncio
    async def test_ai_service_performance(self, benchmark_services, performance_timer):
        """Benchmark AI service performance."""
        ai_service = benchmark_services["ai"]
        
        # Test embedding generation performance
        texts = [f"Performance test text {i} for embedding generation" for i in range(50)]
        
        # Single embedding generation
        single_times = []
        for text in texts[:10]:
            with performance_timer as timer:
                embedding = await ai_service.generate_embedding(text)
            single_times.append(timer.elapsed)
            assert len(embedding) == 1536
        
        # Batch embedding generation
        batch_sizes = [5, 10, 25]
        batch_results = {}
        
        for batch_size in batch_sizes:
            batch_texts = texts[:batch_size]
            
            with performance_timer as timer:
                embeddings = await ai_service.batch_generate_embeddings(batch_texts)
            
            batch_time = timer.elapsed
            batch_results[batch_size] = {
                "total_time": batch_time,
                "time_per_embedding": batch_time / batch_size,
                "embeddings_per_second": batch_size / batch_time
            }
            
            assert len(embeddings) == batch_size
        
        # Performance assertions
        avg_single_time = mean(single_times)
        single_eps = 1 / avg_single_time  # embeddings per second
        
        assert avg_single_time <= 2.0, f"Single embedding too slow: {avg_single_time:.3f}s"
        assert single_eps >= 0.5, f"Single embedding throughput too low: {single_eps:.1f}/sec"
        
        # Batch operations should be more efficient
        for batch_size, stats in batch_results.items():
            time_per_embedding = stats["time_per_embedding"]
            assert time_per_embedding <= avg_single_time, f"Batch-{batch_size} not more efficient than single"
        
        print(f"\nAI Service Performance:")
        print(f"  Single embedding: {single_eps:.1f}/sec ({avg_single_time:.3f}s avg)")
        for batch_size, stats in batch_results.items():
            print(f"  Batch-{batch_size}: {stats['embeddings_per_second']:.1f}/sec ({stats['time_per_embedding']:.3f}s per embedding)")
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_performance(self, benchmark_services, create_test_data, performance_timer):
        """Benchmark pattern analysis performance."""
        pattern_service = benchmark_services["pattern"]
        memory_service = benchmark_services["memory"]
        
        # Create diverse code samples for analysis
        code_samples = [
            {
                "content": f"def function_{i}():\n    return {i * 2}",
                "memory_type": "code",
                "user_id": str(uuid4()),
                "tags": ["pattern_test", "function"],
                "metadata": {"language": "python", "complexity": "simple"}
            }
            for i in range(20)
        ] + [
            {
                "content": f"class Class_{i}:\n    def __init__(self):\n        self.value = {i}",
                "memory_type": "code",
                "user_id": str(uuid4()),
                "tags": ["pattern_test", "class"],
                "metadata": {"language": "python", "complexity": "medium"}
            }
            for i in range(15)
        ]
        
        # Create test data
        await memory_service.bulk_create_memories(code_samples)
        
        # Benchmark pattern analysis
        user_id = code_samples[0]["user_id"]
        
        with performance_timer as timer:
            analysis_result = await pattern_service.analyze_code_patterns(
                user_id=user_id,
                force_refresh=True
            )
        
        analysis_time = timer.elapsed
        
        # Performance assertions
        assert analysis_time <= 5.0, f"Pattern analysis too slow: {analysis_time:.2f}s"
        assert "patterns" in analysis_result
        
        # Test pattern trend analysis
        with performance_timer as trend_timer:
            trend_analysis = await pattern_service.analyze_pattern_trends(
                time_window_days=1
            )
        
        trend_time = trend_timer.elapsed
        assert trend_time <= 3.0, f"Pattern trend analysis too slow: {trend_time:.2f}s"
        
        print(f"\nPattern Analysis Performance:")
        print(f"  Code pattern analysis: {analysis_time:.2f}s")
        print(f"  Pattern trend analysis: {trend_time:.2f}s")
        print(f"  Code samples analyzed: {len(code_samples)}")


@pytest.mark.load
class TestLoadTesting:
    """Load testing for KnowledgeHub under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, benchmark_services, performance_timer):
        """Test system performance under sustained load."""
        memory_service = benchmark_services["memory"]
        session_service = benchmark_services["session"]
        
        # Load test parameters
        duration_seconds = 60  # 1 minute sustained load
        concurrent_users = 20
        operations_per_user_per_second = 2
        
        async def sustained_user_load(user_id: str, duration: int):
            """Generate sustained load for a user."""
            operations_completed = 0
            start_time = time.time()
            
            session = await session_service.create_session(
                user_id=user_id,
                session_type="load_test"
            )
            
            while time.time() - start_time < duration:
                try:
                    # Perform mixed operations
                    operation_type = operations_completed % 3
                    
                    if operation_type == 0:
                        await memory_service.create_memory(
                            content=f"Load test memory {operations_completed}",
                            memory_type="load_test",
                            user_id=user_id
                        )
                    elif operation_type == 1:
                        await memory_service.search_memories(
                            query="load test",
                            user_id=user_id,
                            limit=5
                        )
                    else:
                        await session_service.update_session_context(
                            session_id=session.session_id,
                            context_data={"operations": operations_completed},
                            merge=True
                        )
                    
                    operations_completed += 1
                    
                    # Rate limiting
                    await asyncio.sleep(1 / operations_per_user_per_second)
                    
                except Exception as e:
                    print(f"Error in sustained load for user {user_id}: {e}")
                    break
            
            return operations_completed
        
        # Execute sustained load test
        with performance_timer as timer:
            tasks = [
                sustained_user_load(f"load_user_{i}", duration_seconds)
                for i in range(concurrent_users)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        total_time = timer.elapsed
        exceptions = [r for r in results if isinstance(r, Exception)]
        successful_operations = [r for r in results if not isinstance(r, Exception)]
        
        total_operations = sum(successful_operations)
        overall_ops_per_sec = total_operations / total_time
        
        # Load test assertions
        assert len(exceptions) <= concurrent_users * 0.1, f"Too many exceptions: {len(exceptions)}"
        assert total_operations >= duration_seconds * concurrent_users * operations_per_user_per_second * 0.8, "Insufficient operations completed"
        assert overall_ops_per_sec >= concurrent_users * operations_per_user_per_second * 0.8, f"Throughput too low: {overall_ops_per_sec:.1f} ops/sec"
        
        print(f"\nSustained Load Test Results:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Concurrent users: {concurrent_users}")
        print(f"  Total operations: {total_operations}")
        print(f"  Operations per second: {overall_ops_per_sec:.1f}")
        print(f"  Exceptions: {len(exceptions)}")
    
    @pytest.mark.asyncio
    async def test_spike_load(self, benchmark_services, performance_timer):
        """Test system response to sudden load spikes."""
        memory_service = benchmark_services["memory"]
        
        # Spike test parameters
        spike_operations = 200
        spike_duration = 10  # seconds
        
        async def spike_operation(operation_id: int):
            """Single operation in the spike."""
            return await memory_service.create_memory(
                content=f"Spike test memory {operation_id}",
                memory_type="spike_test",
                user_id=str(uuid4()),
                tags=["spike", "load_test"]
            )
        
        # Execute spike test
        with performance_timer as timer:
            tasks = [spike_operation(i) for i in range(spike_operations)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        spike_time = timer.elapsed
        exceptions = [r for r in results if isinstance(r, Exception)]
        successful_operations = spike_operations - len(exceptions)
        
        operations_per_second = successful_operations / spike_time
        
        # Spike test assertions
        assert len(exceptions) <= spike_operations * 0.05, f"Too many exceptions during spike: {len(exceptions)}"
        assert spike_time <= spike_duration * 2, f"Spike took too long: {spike_time:.1f}s"
        assert operations_per_second >= 10, f"Spike throughput too low: {operations_per_second:.1f} ops/sec"
        
        print(f"\nSpike Load Test Results:")
        print(f"  Spike operations: {spike_operations}")
        print(f"  Spike duration: {spike_time:.1f}s")
        print(f"  Operations per second: {operations_per_second:.1f}")
        print(f"  Success rate: {(successful_operations/spike_operations)*100:.1f}%")