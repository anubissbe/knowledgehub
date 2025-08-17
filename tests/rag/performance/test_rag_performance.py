"""
Performance and Load Tests for RAG System.

Comprehensive performance testing including:
- Throughput benchmarks under load
- Memory usage profiling and leak detection
- Concurrent user simulation  
- Scalability testing with large datasets
- Performance regression detection

Author: Peter Verschuere - Test-Driven Development Expert
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock
import asyncio
import time
import psutil
import gc
from typing import List, Dict, Any
import numpy as np

from api.services.rag_pipeline import RAGPipeline, RAGConfig, Document, Chunk, ChunkingStrategy, RetrievalStrategy


@pytest.mark.performance
@pytest.mark.rag
@pytest.mark.slow
class TestRAGPerformanceBenchmarks:
    """Performance benchmarks for RAG system components."""
    
    @pytest.mark.asyncio
    async def test_chunking_performance_benchmark(self, performance_metrics, quality_gates):
        """Benchmark chunking performance across strategies and document sizes."""
        thresholds = quality_gates["performance_thresholds"]
        
        # Test different document sizes
        document_sizes = [1000, 5000, 10000, 50000]  # 1KB to 50KB
        strategies = [ChunkingStrategy.SLIDING, ChunkingStrategy.SEMANTIC, ChunkingStrategy.HIERARCHICAL]
        
        results = {}
        
        for strategy in strategies:
            results[strategy.value] = {}
            
            config = RAGConfig(
                chunking_strategy=strategy,
                chunk_size=512,
                chunk_overlap=128
            )
            
            from api.services.rag_pipeline import AdvancedChunker
            chunker = AdvancedChunker(config)
            
            for size in document_sizes:
                # Generate document of specified size
                content = "This is performance test content for chunking benchmarks. " * (size // 60)
                document = Document(id=f"perf_doc_{size}", content=content)
                
                # Benchmark chunking
                performance_metrics.start_timer(f"chunk_{strategy.value}_{size}")
                
                if strategy == ChunkingStrategy.SEMANTIC:
                    with patch('nltk.download'):
                        chunks = chunker.chunk_document(document)
                else:
                    chunks = chunker.chunk_document(document)
                
                elapsed = performance_metrics.end_timer(f"chunk_{strategy.value}_{size}")
                
                results[strategy.value][size] = {
                    "time_ms": elapsed * 1000,
                    "chunks_created": len(chunks),
                    "throughput_kb_per_sec": (size / 1000) / elapsed if elapsed > 0 else float('inf')
                }
                
                # Verify performance threshold for 10KB documents
                if size == 10000:
                    performance_metrics.assert_performance(
                        f"chunk_{strategy.value}_{size}", 
                        thresholds["chunking_time_ms"] / 1000
                    )
        
        # Log performance results
        for strategy, size_results in results.items():
            print(f"\n{strategy} Chunking Performance:")
            for size, metrics in size_results.items():
                print(f"  {size//1000}KB: {metrics['time_ms']:.2f}ms, "
                      f"{metrics['chunks_created']} chunks, "
                      f"{metrics['throughput_kb_per_sec']:.2f} KB/s")
    
    @pytest.mark.asyncio
    async def test_retrieval_performance_benchmark(self, mock_db_session, performance_metrics, quality_gates):
        """Benchmark retrieval performance across strategies."""
        thresholds = quality_gates["performance_thresholds"]
        strategies = [RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID, RetrievalStrategy.ENSEMBLE]
        
        # Simulate different result set sizes
        result_sizes = [5, 10, 25, 50]
        
        results = {}
        
        for strategy in strategies:
            results[strategy.value] = {}
            
            config = RAGConfig(retrieval_strategy=strategy, top_k=50)
            from api.services.rag_pipeline import HybridRetriever
            
            with patch.object(HybridRetriever, '_get_embedding') as mock_embed:
                mock_embed.return_value = np.random.rand(1536)
                retriever = HybridRetriever(config, mock_db_session)
                
                for result_size in result_sizes:
                    # Mock database results
                    mock_results = [
                        MagicMock(id=f"chunk_{i}", content=f"Content {i}", 
                                document_id=f"doc_{i}", distance=0.1 + i*0.01)
                        for i in range(result_size)
                    ]
                    mock_db_session.execute.return_value.fetchall.return_value = mock_results
                    
                    query = f"Performance test query for {result_size} results"
                    
                    # Benchmark retrieval
                    performance_metrics.start_timer(f"retrieve_{strategy.value}_{result_size}")
                    
                    if strategy == RetrievalStrategy.VECTOR:
                        chunks = await retriever._vector_retrieval(query)
                    elif strategy == RetrievalStrategy.HYBRID:
                        with patch.object(retriever, '_vector_retrieval') as mock_vector, \
                             patch.object(retriever, '_merge_and_rerank') as mock_merge:
                            mock_vector.return_value = [
                                Chunk(id=f"v_{i}", content=f"Vector {i}", document_id=f"doc_{i}", position=i)
                                for i in range(result_size//2)
                            ]
                            mock_merge.return_value = mock_vector.return_value
                            chunks = await retriever._hybrid_retrieval(query)
                    else:  # ENSEMBLE
                        with patch.object(retriever, '_vector_retrieval') as mock_vector, \
                             patch.object(retriever, '_bm25_retrieval') as mock_bm25, \
                             patch.object(retriever, '_semantic_retrieval') as mock_semantic:
                            
                            mock_chunks = [
                                Chunk(id=f"e_{i}", content=f"Ensemble {i}", document_id=f"doc_{i}", position=i)
                                for i in range(result_size//3)
                            ]
                            mock_vector.return_value = mock_chunks
                            mock_bm25.return_value = mock_chunks[:result_size//4]
                            mock_semantic.return_value = mock_chunks[:result_size//5]
                            
                            chunks = await retriever._ensemble_retrieval(query)
                    
                    elapsed = performance_metrics.end_timer(f"retrieve_{strategy.value}_{result_size}")
                    
                    results[strategy.value][result_size] = {
                        "time_ms": elapsed * 1000,
                        "chunks_retrieved": len(chunks),
                        "throughput_chunks_per_sec": len(chunks) / elapsed if elapsed > 0 else float('inf')
                    }
                    
                    # Verify performance threshold for standard retrieval
                    if result_size == 10:
                        performance_metrics.assert_performance(
                            f"retrieve_{strategy.value}_{result_size}",
                            thresholds["retrieval_time_ms"] / 1000
                        )
        
        # Log retrieval performance results
        for strategy, size_results in results.items():
            print(f"\n{strategy} Retrieval Performance:")
            for size, metrics in size_results.items():
                print(f"  {size} results: {metrics['time_ms']:.2f}ms, "
                      f"{metrics['throughput_chunks_per_sec']:.2f} chunks/s")
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance_benchmark(self, rag_pipeline_basic, performance_metrics, quality_gates):
        """Benchmark complete end-to-end RAG pipeline performance."""
        thresholds = quality_gates["performance_thresholds"]
        
        # Test queries of different complexities
        query_types = {
            "simple": "What is Python?",
            "medium": "How does machine learning work in practice?",
            "complex": "Explain the detailed process of implementing a neural network for classification tasks with performance optimization."
        }
        
        results = {}
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            # Mock different response complexities
            mock_responses = {
                "simple": "Python is a programming language.",
                "medium": "Machine learning works by training algorithms on data to make predictions.",
                "complex": "Neural network implementation involves multiple steps: data preprocessing, architecture design, forward propagation, backpropagation, and optimization techniques."
            }
            
            for complexity, query in query_types.items():
                mock_retrieve.return_value = [
                    Chunk(id=f"{complexity}_{i}", content=f"{complexity} content {i}", 
                          document_id=f"doc_{i}", position=i)
                    for i in range(5)  # 5 chunks per query
                ]
                mock_generate.return_value = mock_responses[complexity]
                
                # Benchmark end-to-end processing
                performance_metrics.start_timer(f"e2e_{complexity}")
                
                result = await rag_pipeline_basic.process_query(query)
                
                elapsed = performance_metrics.end_timer(f"e2e_{complexity}")
                
                results[complexity] = {
                    "time_ms": elapsed * 1000,
                    "response_length": len(result['response']),
                    "chunks_used": result['chunks_used'],
                    "throughput_queries_per_sec": 1 / elapsed if elapsed > 0 else float('inf')
                }
                
                # Verify performance threshold
                performance_metrics.assert_performance(
                    f"e2e_{complexity}",
                    thresholds["end_to_end_time_ms"] / 1000
                )
        
        # Log end-to-end performance
        print("\nEnd-to-End Pipeline Performance:")
        for complexity, metrics in results.items():
            print(f"  {complexity}: {metrics['time_ms']:.2f}ms, "
                  f"{metrics['throughput_queries_per_sec']:.2f} queries/s")


@pytest.mark.load
@pytest.mark.rag
@pytest.mark.slow
class TestRAGLoadTesting:
    """Load testing for RAG system under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self, rag_pipeline_basic, performance_metrics):
        """Simulate concurrent users querying the RAG system."""
        concurrent_users = [5, 10, 20, 50]
        queries_per_user = 5
        
        results = {}
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [
                Chunk(id="load_test", content="Load test content", document_id="doc1", position=0)
            ]
            mock_generate.return_value = "Load test response"
            
            for user_count in concurrent_users:
                print(f"\nTesting {user_count} concurrent users...")
                
                async def simulate_user(user_id: int) -> List[Dict]:
                    user_results = []
                    for query_num in range(queries_per_user):
                        query = f"User {user_id} query {query_num}"
                        start_time = time.time()
                        try:
                            result = await rag_pipeline_basic.process_query(query)
                            end_time = time.time()
                            user_results.append({
                                "user_id": user_id,
                                "query_num": query_num,
                                "success": True,
                                "response_time": end_time - start_time,
                                "response_length": len(result['response'])
                            })
                        except Exception as e:
                            end_time = time.time()
                            user_results.append({
                                "user_id": user_id,
                                "query_num": query_num,
                                "success": False,
                                "response_time": end_time - start_time,
                                "error": str(e)
                            })
                    return user_results
                
                # Run concurrent user simulation
                performance_metrics.start_timer(f"load_test_{user_count}_users")
                
                tasks = [simulate_user(user_id) for user_id in range(user_count)]
                all_results = await asyncio.gather(*tasks)
                
                elapsed = performance_metrics.end_timer(f"load_test_{user_count}_users")
                
                # Analyze results
                flat_results = [result for user_results in all_results for result in user_results]
                successful_queries = [r for r in flat_results if r['success']]
                failed_queries = [r for r in flat_results if not r['success']]
                
                if successful_queries:
                    avg_response_time = sum(r['response_time'] for r in successful_queries) / len(successful_queries)
                    max_response_time = max(r['response_time'] for r in successful_queries)
                    min_response_time = min(r['response_time'] for r in successful_queries)
                else:
                    avg_response_time = max_response_time = min_response_time = 0
                
                total_queries = len(flat_results)
                success_rate = len(successful_queries) / total_queries if total_queries > 0 else 0
                queries_per_second = total_queries / elapsed if elapsed > 0 else 0
                
                results[user_count] = {
                    "total_queries": total_queries,
                    "successful_queries": len(successful_queries),
                    "failed_queries": len(failed_queries),
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time,
                    "min_response_time": min_response_time,
                    "max_response_time": max_response_time,
                    "queries_per_second": queries_per_second,
                    "total_time": elapsed
                }
                
                # Assertions for load testing
                assert success_rate >= 0.95, f"Success rate {success_rate:.2%} too low for {user_count} users"
                assert avg_response_time < 2.0, f"Avg response time {avg_response_time:.2f}s too high"
                assert max_response_time < 5.0, f"Max response time {max_response_time:.2f}s too high"
        
        # Log load test results
        print("\nLoad Test Summary:")
        print("Users\tQueries\tSuccess%\tAvg Time\tMax Time\tQPS")
        for user_count, metrics in results.items():
            print(f"{user_count}\t{metrics['total_queries']}\t{metrics['success_rate']:.1%}\t"
                  f"{metrics['avg_response_time']:.3f}s\t{metrics['max_response_time']:.3f}s\t"
                  f"{metrics['queries_per_second']:.1f}")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, rag_pipeline_basic, memory_profiler):
        """Test for memory leaks during extended operation."""
        import gc
        
        # Baseline memory measurement
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_profiler.start("memory_leak_test")
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [
                Chunk(id="mem_test", content="Memory test content", document_id="doc1", position=0)
            ]
            mock_generate.return_value = "Memory test response"
            
            # Run many operations to detect leaks
            memory_measurements = []
            
            for i in range(100):  # 100 operations
                await rag_pipeline_basic.process_query(f"Memory test query {i}")
                
                if i % 10 == 0:  # Measure every 10 operations
                    gc.collect()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_measurements.append({
                        "operation": i,
                        "memory_mb": current_memory,
                        "memory_increase": current_memory - initial_memory
                    })
                    memory_profiler.checkpoint(f"after_{i}_operations")
        
        memory_profiler.stop("memory_leak_test")
        
        # Analyze memory usage
        final_memory = memory_measurements[-1]["memory_mb"]
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage Analysis:")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Check for excessive memory growth
        assert memory_increase < 50, f"Memory increase {memory_increase:.2f}MB suggests potential leak"
        
        # Check for steady memory growth (potential leak indicator)
        if len(memory_measurements) >= 5:
            recent_measurements = memory_measurements[-5:]
            memory_trend = [m["memory_mb"] for m in recent_measurements]
            
            # Simple trend analysis - memory shouldn't consistently increase
            increasing_count = sum(1 for i in range(1, len(memory_trend)) 
                                 if memory_trend[i] > memory_trend[i-1])
            
            # Allow some variance, but not consistent growth
            assert increasing_count < len(memory_trend) - 1, "Consistent memory growth indicates leak"
    
    @pytest.mark.asyncio
    async def test_stress_testing_large_documents(self, mock_db_session, performance_metrics):
        """Stress test with large documents and many chunks."""
        # Create very large document
        large_content = """
        This is a comprehensive stress testing document for the RAG system.
        It contains extensive information about machine learning, artificial intelligence,
        natural language processing, and various technical topics that would be found
        in a real-world knowledge base. The document includes detailed explanations,
        code examples, mathematical formulas, and references to research papers.
        """ * 1000  # Approximately 500KB document
        
        config = RAGConfig(
            chunk_size=512,
            chunk_overlap=128,
            chunking_strategy=ChunkingStrategy.SLIDING,
            retrieval_strategy=RetrievalStrategy.VECTOR,
            top_k=20
        )
        
        with patch('api.services.rag_pipeline.HybridRetriever._get_embedding') as mock_embed:
            mock_embed.return_value = np.random.rand(1536)
            pipeline = RAGPipeline(config, mock_db_session)
            
            # Test document ingestion performance
            performance_metrics.start_timer("large_document_ingestion")
            
            with patch.object(pipeline, '_store_document_and_chunks'):
                document = await pipeline.ingest_document(large_content, {"type": "stress_test"})
            
            ingestion_time = performance_metrics.end_timer("large_document_ingestion")
            
            print(f"Large document ingestion:")
            print(f"  Content size: {len(large_content):,} characters")
            print(f"  Chunks created: {len(document.chunks):,}")
            print(f"  Ingestion time: {ingestion_time:.2f}s")
            print(f"  Throughput: {len(large_content)/1000/ingestion_time:.2f} KB/s")
            
            # Should complete within reasonable time
            assert ingestion_time < 10.0, f"Large document ingestion too slow: {ingestion_time:.2f}s"
            assert len(document.chunks) > 100, "Should create many chunks from large document"
            assert len(document.chunks) < 2000, "Shouldn't create excessive chunks"
    
    @pytest.mark.asyncio
    async def test_database_performance_under_load(self, populated_test_db, performance_metrics):
        """Test database performance under heavy RAG operations."""
        config = RAGConfig(top_k=10, retrieval_strategy=RetrievalStrategy.VECTOR)
        
        with patch('api.services.rag_pipeline.HybridRetriever._get_embedding') as mock_embed:
            mock_embed.return_value = np.random.rand(1536)
            
            from api.services.rag_pipeline import HybridRetriever
            retriever = HybridRetriever(config, populated_test_db)
            
            # Simulate heavy database load
            queries = [f"Database performance test query {i}" for i in range(50)]
            
            performance_metrics.start_timer("database_load_test")
            
            # Execute queries sequentially to test database performance
            for query in queries:
                try:
                    chunks = await retriever._vector_retrieval(query)
                    assert isinstance(chunks, list)  # Basic validation
                except Exception as e:
                    pytest.fail(f"Database query failed: {e}")
            
            db_load_time = performance_metrics.end_timer("database_load_test")
            
            print(f"Database load test:")
            print(f"  Queries executed: {len(queries)}")
            print(f"  Total time: {db_load_time:.2f}s")
            print(f"  Average time per query: {db_load_time/len(queries)*1000:.2f}ms")
            print(f"  Queries per second: {len(queries)/db_load_time:.2f}")
            
            # Database should handle load efficiently
            avg_query_time = db_load_time / len(queries)
            assert avg_query_time < 0.5, f"Database queries too slow: {avg_query_time:.3f}s avg"


@pytest.mark.performance
@pytest.mark.rag
class TestRAGPerformanceRegression:
    """Performance regression testing to catch performance degradations."""
    
    async def test_performance_baseline_establishment(self, performance_metrics, quality_gates):
        """Establish performance baselines for regression testing."""
        baselines = quality_gates["performance_thresholds"]
        
        # Document expected performance characteristics
        expected_performance = {
            "chunking_1kb_doc": {"max_time_ms": baselines["chunking_time_ms"], "target_chunks": [3, 8]},
            "retrieval_10_results": {"max_time_ms": baselines["retrieval_time_ms"], "min_precision": 0.8},
            "e2e_simple_query": {"max_time_ms": baselines["end_to_end_time_ms"], "min_relevance": 0.7},
            "memory_usage_100_ops": {"max_memory_mb": baselines["memory_usage_mb"], "max_increase_mb": 50}
        }
        
        # Store baselines for future regression testing
        import json
        baseline_file = "/tmp/rag_performance_baselines.json"
        with open(baseline_file, "w") as f:
            json.dump(expected_performance, f, indent=2)
        
        print(f"Performance baselines established in {baseline_file}")
        
        # Validate baseline reasonableness
        for operation, limits in expected_performance.items():
            if "max_time_ms" in limits:
                assert limits["max_time_ms"] > 0, f"Invalid time baseline for {operation}"
                assert limits["max_time_ms"] < 10000, f"Baseline too high for {operation}"
    
    @pytest.mark.asyncio
    async def test_chunking_performance_regression(self, performance_metrics):
        """Detect chunking performance regressions."""
        # This would compare against stored baselines in real implementation
        from api.services.rag_pipeline import AdvancedChunker, RAGConfig, Document, ChunkingStrategy
        
        config = RAGConfig(chunking_strategy=ChunkingStrategy.SLIDING, chunk_size=512)
        chunker = AdvancedChunker(config)
        
        # Standard test document (1KB)
        test_content = "Performance regression test document content. " * 50
        document = Document(id="regression_test", content=test_content)
        
        # Benchmark current performance
        performance_metrics.start_timer("regression_chunking")
        chunks = chunker.chunk_document(document)
        current_time = performance_metrics.end_timer("regression_chunking")
        
        # Compare against baseline (would be loaded from file in real implementation)
        baseline_time = 0.1  # 100ms baseline
        regression_threshold = 1.5  # 50% performance regression threshold
        
        performance_ratio = current_time / baseline_time
        
        print(f"Chunking Performance Regression Test:")
        print(f"  Current time: {current_time*1000:.2f}ms")
        print(f"  Baseline time: {baseline_time*1000:.2f}ms")
        print(f"  Performance ratio: {performance_ratio:.2f}x")
        
        assert performance_ratio < regression_threshold, \
            f"Chunking performance regression detected: {performance_ratio:.2f}x slower than baseline"
        
        # Validate output quality hasn't regressed
        assert len(chunks) > 0, "Chunking should produce results"
        assert len(chunks) < 20, "Chunking shouldn't create excessive chunks for 1KB document"
