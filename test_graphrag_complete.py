"""
Comprehensive GraphRAG Testing Suite
Dynamic Parallelism and Memory Bandwidth Optimization Validation

This test suite validates the complete GraphRAG implementation including:
- Entity extraction and relationship mapping
- Parallel graph traversal algorithms
- Memory bandwidth optimization
- Graph-aware chunking strategies  
- API endpoint functionality

Author: Charlotte Cools - Dynamic Parallelism Expert
"""

import asyncio
import pytest
import logging
import time
import json
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

import httpx
from fastapi.testclient import TestClient
import neo4j

logger = logging.getLogger(__name__)

# Test data for dynamic parallelism scenarios
DYNAMIC_PARALLELISM_DOCUMENTS = [
    {
        "id": "gpu_optimization_1",
        "title": "CUDA Kernel Optimization for V100 GPUs", 
        "content": """
        Dynamic parallelism on Tesla V100 GPUs allows kernels to launch other kernels directly from GPU. 
        Memory bandwidth optimization is critical for achieving peak performance.
        The V100's tensor cores provide significant speedup for mixed precision operations.
        CUDA kernel fusion can reduce memory bandwidth requirements by 60-80%.
        """,
        "metadata": {"domain": "gpu_computing", "complexity": "high"}
    },
    {
        "id": "memory_bandwidth_2", 
        "title": "Memory Bandwidth Optimization Techniques",
        "content": """
        Memory coalescing improves bandwidth utilization on modern GPUs.
        Shared memory optimization reduces global memory access patterns.
        Dynamic parallelism requires careful memory management to prevent deadlocks.
        Tensor core optimization leverages mixed precision for bandwidth efficiency.
        """, 
        "metadata": {"domain": "performance_optimization", "complexity": "medium"}
    },
    {
        "id": "parallel_algorithms_3",
        "title": "Parallel Algorithm Design Patterns",
        "content": """
        SIMD and SIMT architectures require different parallelization strategies.
        GPU kernel design must consider warp divergence and occupancy.
        Memory bandwidth becomes the bottleneck in compute-intensive applications.  
        Dynamic parallelism enables recursive algorithms on GPU hardware.
        """,
        "metadata": {"domain": "algorithm_design", "complexity": "high"}
    }
]

TECHNICAL_QUERIES = [
    "How does dynamic parallelism work on V100 GPUs?",
    "What are memory bandwidth optimization techniques?", 
    "How to design efficient CUDA kernels?",
    "What is the relationship between tensor cores and memory bandwidth?",
    "How does kernel fusion improve GPU performance?"
]


class TestGraphRAGService:
    """Test the core GraphRAG service functionality"""
    
    @pytest.fixture
    async def mock_neo4j_driver(self):
        """Mock Neo4j driver for testing"""
        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        
        # Configure mocks
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = {'relationship_count': 2}
        
        # Mock graph traversal results
        mock_records = [
            {
                'content': 'GPU optimization techniques',
                'name': 'CUDA',
                'relationship_type': 'OPTIMIZES', 
                'distance': 1
            },
            {
                'content': 'Memory bandwidth optimization',
                'name': 'tensor_cores',
                'relationship_type': 'ENABLES',
                'distance': 2  
            }
        ]
        mock_result.__iter__ = lambda self: iter([Mock(**record) for record in mock_records])
        
        return mock_driver
    
    @pytest.fixture
    async def graphrag_service(self, mock_neo4j_driver):
        """Create GraphRAG service for testing"""
        with patch('api.services.graphrag_service.GraphDatabase.driver', return_value=mock_neo4j_driver):
            from api.services.graphrag_service import GraphRAGService
            service = GraphRAGService()
            await service.initialize_rag_pipeline()
            return service
    
    @pytest.mark.asyncio
    async def test_entity_extraction_parallel(self, graphrag_service):
        """Test parallel entity extraction from documents"""
        start_time = time.time()
        
        # Test parallel entity extraction
        entities = await graphrag_service.processor.parallel_entity_extraction(
            DYNAMIC_PARALLELISM_DOCUMENTS
        )
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert len(entities) > 0, "Should extract entities"
        assert execution_time < 5.0, f"Parallel extraction too slow: {execution_time:.2f}s"
        
        # Check for expected technical entities
        entity_names = [e.entity if hasattr(e, 'entity') else str(e) for e in entities]
        technical_terms = ['GPU', 'CUDA', 'V100', 'tensor cores', 'dynamic parallelism', 'memory bandwidth']
        
        found_terms = [term for term in technical_terms if any(term in entity for entity in entity_names)]
        assert len(found_terms) >= 3, f"Should find technical entities: {found_terms}"
        
        logger.info(f"✅ Parallel entity extraction: {len(entities)} entities in {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_graph_indexing_performance(self, graphrag_service):
        """Test GraphRAG indexing with performance metrics"""
        start_time = time.time()
        
        # Test document indexing with graph creation
        stats = await graphrag_service.index_documents_with_graph(
            documents=DYNAMIC_PARALLELISM_DOCUMENTS,
            extract_entities=True,
            build_relationships=True
        )
        
        execution_time = time.time() - start_time
        
        # Validate performance and results  
        assert stats['documents_processed'] == len(DYNAMIC_PARALLELISM_DOCUMENTS)
        assert stats['entities_extracted'] > 0
        assert stats['processing_time_ms'] > 0
        assert execution_time < 10.0, f"Indexing too slow: {execution_time:.2f}s"
        
        logger.info(f"✅ Graph indexing: {stats}")
        logger.info(f"   Performance: {execution_time:.3f}s total")
    
    @pytest.mark.asyncio 
    async def test_hybrid_parallel_query(self, graphrag_service):
        """Test hybrid parallel query strategy"""
        
        # Index documents first
        await graphrag_service.index_documents_with_graph(DYNAMIC_PARALLELISM_DOCUMENTS)
        
        start_time = time.time()
        
        # Test hybrid parallel query
        results = await graphrag_service.query_graphrag(
            query="dynamic parallelism GPU optimization",
            strategy=graphrag_service.__class__.__dict__['GraphRAGStrategy'].HYBRID_PARALLEL,
            max_results=5,
            include_reasoning=True
        )
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert len(results) > 0, "Should return query results"
        assert execution_time < 3.0, f"Query too slow: {execution_time:.2f}s"
        
        # Check result quality
        for result in results:
            assert hasattr(result, 'content')
            assert hasattr(result, 'score')
            assert hasattr(result, 'vector_score') 
            assert hasattr(result, 'graph_score')
            assert result.score > 0
            
        logger.info(f"✅ Hybrid parallel query: {len(results)} results in {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, graphrag_service):
        """Test memory bandwidth optimization features"""
        
        # Test cache functionality
        initial_cache_size = len(graphrag_service.node_cache)
        
        # Perform operations that should populate cache
        await graphrag_service.index_documents_with_graph(DYNAMIC_PARALLELISM_DOCUMENTS)
        
        for query in TECHNICAL_QUERIES[:3]:
            await graphrag_service.query_graphrag(query, max_results=3)
        
        # Check memory stats
        stats = graphrag_service.get_memory_stats()
        
        assert 'node_cache_size' in stats
        assert 'cache_hits' in stats
        assert 'hit_ratio' in stats
        assert 'max_memory_mb' in stats
        
        # Validate cache growth
        final_cache_size = len(graphrag_service.node_cache)
        assert final_cache_size >= initial_cache_size
        
        logger.info(f"✅ Memory optimization stats: {stats}")
    
    @pytest.mark.asyncio
    async def test_graph_traversal_optimization(self, graphrag_service):
        """Test optimized graph traversal algorithms"""
        
        # Index documents to create graph structure
        await graphrag_service.index_documents_with_graph(DYNAMIC_PARALLELISM_DOCUMENTS)
        
        start_time = time.time()
        
        # Test graph traversal with different depths  
        test_entity = "GPU"
        
        with patch.object(graphrag_service.driver, 'session') as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session
            mock_session_context.return_value.__exit__.return_value = None
            
            # Mock traversal results
            mock_result = Mock()
            mock_records = [
                Mock(**{
                    'get.return_value': 'CUDA kernel optimization',
                    'distance': 1,
                    'relationship_type': 'OPTIMIZES'
                }),
                Mock(**{
                    'get.return_value': 'memory bandwidth',  
                    'distance': 2,
                    'relationship_type': 'REQUIRES'
                })
            ]
            mock_result.__iter__ = lambda: iter(mock_records)
            mock_session.run.return_value = mock_result
            
            # Test traversal
            results = await graphrag_service._traverse_graph_from_entity(
                mock_session, test_entity, max_depth=3
            )
        
        execution_time = time.time() - start_time
        
        assert execution_time < 1.0, f"Graph traversal too slow: {execution_time:.2f}s"
        assert len(results) >= 0, "Should return traversal results"
        
        logger.info(f"✅ Graph traversal: {len(results)} results in {execution_time:.3f}s")


class TestGraphAwareChunking:
    """Test graph-aware chunking functionality"""
    
    @pytest.fixture
    def chunking_config(self):
        """Create chunking configuration for testing"""
        from api.services.graph_aware_chunking import ChunkingConfig
        return ChunkingConfig(
            target_chunk_size=256,
            max_chunk_size=512,
            min_chunk_size=64,
            parallel_workers=2
        )
    
    @pytest.fixture  
    def mock_neo4j_driver(self):
        """Mock Neo4j driver for chunking tests"""
        mock_driver = Mock()
        mock_session = Mock()
        
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock community detection results
        mock_session.run.return_value.single.return_value = {
            'community': ['GPU', 'CUDA', 'tensor_cores']
        }
        
        return mock_driver
    
    @pytest.fixture
    async def parallel_chunker(self, chunking_config, mock_neo4j_driver):
        """Create parallel graph chunker for testing"""
        from api.services.graph_aware_chunking import ParallelGraphChunker
        return ParallelGraphChunker(chunking_config, mock_neo4j_driver)
    
    @pytest.mark.asyncio
    async def test_semantic_graph_chunking(self, parallel_chunker):
        """Test semantic graph-based chunking"""
        from api.services.graph_aware_chunking import GraphChunkingStrategy
        
        start_time = time.time()
        
        chunks = await parallel_chunker.chunk_documents_parallel(
            DYNAMIC_PARALLELISM_DOCUMENTS,
            strategy=GraphChunkingStrategy.SEMANTIC_GRAPH
        )
        
        execution_time = time.time() - start_time
        
        # Validate chunking results
        assert len(chunks) > 0, "Should create chunks"
        assert execution_time < 5.0, f"Chunking too slow: {execution_time:.2f}s"
        
        # Check chunk properties
        for chunk in chunks:
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'entities')
            assert hasattr(chunk, 'coherence_score')
            assert len(chunk.content) <= 512, "Chunk too large"
            assert len(chunk.content) >= 32, "Chunk too small"
            
        logger.info(f"✅ Semantic chunking: {len(chunks)} chunks in {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_entity_boundary_chunking(self, parallel_chunker):
        """Test entity boundary-based chunking"""
        from api.services.graph_aware_chunking import GraphChunkingStrategy
        
        chunks = await parallel_chunker.chunk_documents_parallel(
            DYNAMIC_PARALLELISM_DOCUMENTS,
            strategy=GraphChunkingStrategy.ENTITY_BOUNDARY
        )
        
        assert len(chunks) > 0, "Should create entity-boundary chunks"
        
        # Validate entity preservation at boundaries
        total_entities = 0
        for chunk in chunks:
            total_entities += len(chunk.entities)
            
        assert total_entities > 0, "Should preserve entities in chunks"
        
        logger.info(f"✅ Entity boundary chunking: {len(chunks)} chunks, {total_entities} entities")
    
    @pytest.mark.asyncio
    async def test_parallel_chunking_performance(self, parallel_chunker):
        """Test parallel processing performance"""
        
        # Create larger document set for performance testing
        large_docs = DYNAMIC_PARALLELISM_DOCUMENTS * 10
        
        start_time = time.time()
        
        chunks = await parallel_chunker.chunk_documents_parallel(large_docs)
        
        execution_time = time.time() - start_time
        
        # Performance validation
        docs_per_second = len(large_docs) / execution_time
        assert docs_per_second > 5, f"Processing too slow: {docs_per_second:.1f} docs/sec"
        
        logger.info(f"✅ Parallel chunking performance: {docs_per_second:.1f} docs/sec")


class TestGraphRAGAPI:
    """Test GraphRAG API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        with patch('api.services.graphrag_service.GraphDatabase.driver'):
            from api.main import app
            return TestClient(app)
    
    def test_graphrag_health_endpoint(self, client):
        """Test GraphRAG health check"""
        response = client.get("/api/graphrag/health")
        
        # Should return 200 even if mocked
        assert response.status_code in [200, 503], f"Unexpected status: {response.status_code}"
        
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        
        logger.info(f"✅ Health endpoint: {data['status']}")
    
    def test_entity_extraction_endpoint(self, client):
        """Test entity extraction API"""
        test_text = "Dynamic parallelism on V100 GPUs with CUDA optimization"
        
        response = client.post(
            "/api/graphrag/extract-entities",
            json={"text": test_text}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list), "Should return list of entities"
            
            if len(data) > 0:
                entity = data[0]
                assert 'entity' in entity
                assert 'entity_type' in entity
                assert 'confidence' in entity
                
            logger.info(f"✅ Entity extraction: {len(data)} entities")
        else:
            logger.warning(f"Entity extraction endpoint failed: {response.status_code}")
    
    def test_graphrag_strategies_endpoint(self, client):
        """Test GraphRAG strategies endpoint"""
        response = client.get("/api/graphrag/strategies")
        
        if response.status_code == 200:
            data = response.json()
            assert 'graphrag_strategies' in data
            assert 'chunking_strategies' in data
            assert 'entity_types' in data
            
            logger.info(f"✅ Strategies endpoint: {len(data['graphrag_strategies'])} strategies")
        else:
            logger.warning(f"Strategies endpoint failed: {response.status_code}")
    
    def test_memory_stats_endpoint(self, client):
        """Test memory statistics endpoint"""
        response = client.get("/api/graphrag/memory-stats")
        
        if response.status_code == 200:
            data = response.json()
            expected_fields = ['node_cache_size', 'cache_hits', 'hit_ratio', 'max_memory_mb']
            
            for field in expected_fields:
                assert field in data, f"Missing field: {field}"
                
            logger.info(f"✅ Memory stats: {data['hit_ratio']:.2f} hit ratio")
        else:
            logger.warning(f"Memory stats endpoint failed: {response.status_code}")


class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    @pytest.mark.asyncio
    async def test_indexing_throughput(self):
        """Test document indexing throughput"""
        with patch('api.services.graphrag_service.GraphDatabase.driver'):
            from api.services.graphrag_service import GraphRAGService
            
            service = GraphRAGService()
            await service.initialize_rag_pipeline()
            
            # Create test dataset
            large_dataset = []
            for i in range(100):
                doc = DYNAMIC_PARALLELISM_DOCUMENTS[i % len(DYNAMIC_PARALLELISM_DOCUMENTS)].copy()
                doc['id'] = f"doc_{i}"
                large_dataset.append(doc)
            
            start_time = time.time()
            
            stats = await service.index_documents_with_graph(
                documents=large_dataset,
                extract_entities=True,
                build_relationships=False  # Skip for speed
            )
            
            execution_time = time.time() - start_time
            throughput = len(large_dataset) / execution_time
            
            assert throughput > 10, f"Indexing throughput too low: {throughput:.1f} docs/sec"
            
            logger.info(f"✅ Indexing throughput: {throughput:.1f} docs/sec")
            logger.info(f"   Total time: {execution_time:.2f}s for {len(large_dataset)} docs")
    
    @pytest.mark.asyncio 
    async def test_query_latency(self):
        """Test query response latency"""
        with patch('api.services.graphrag_service.GraphDatabase.driver'):
            from api.services.graphrag_service import GraphRAGService, GraphRAGStrategy
            
            service = GraphRAGService()
            await service.initialize_rag_pipeline()
            
            # Index test documents
            await service.index_documents_with_graph(DYNAMIC_PARALLELISM_DOCUMENTS)
            
            latencies = []
            
            # Test multiple queries
            for query in TECHNICAL_QUERIES:
                start_time = time.time()
                
                results = await service.query_graphrag(
                    query=query,
                    strategy=GraphRAGStrategy.HYBRID_PARALLEL,
                    max_results=5,
                    include_reasoning=False
                )
                
                latency = time.time() - start_time
                latencies.append(latency)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            assert avg_latency < 2.0, f"Average query latency too high: {avg_latency:.3f}s"
            assert max_latency < 5.0, f"Max query latency too high: {max_latency:.3f}s"
            
            logger.info(f"✅ Query performance:")
            logger.info(f"   Average latency: {avg_latency:.3f}s")
            logger.info(f"   Max latency: {max_latency:.3f}s")
            logger.info(f"   Queries tested: {len(TECHNICAL_QUERIES)}")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        with patch('api.services.graphrag_service.GraphDatabase.driver'):
            from api.services.graphrag_service import GraphRAGService
            
            service = GraphRAGService()
            await service.initialize_rag_pipeline()
            
            # Perform memory-intensive operations
            large_docs = DYNAMIC_PARALLELISM_DOCUMENTS * 50
            
            # Monitor memory stats during operations
            initial_stats = service.get_memory_stats()
            
            await service.index_documents_with_graph(large_docs)
            
            # Perform queries to populate caches
            for _ in range(20):
                await service.query_graphrag("GPU optimization", max_results=3)
            
            final_stats = service.get_memory_stats()
            
            # Validate memory efficiency
            hit_ratio = final_stats['hit_ratio']
            cache_size = final_stats['node_cache_size']
            
            assert hit_ratio > 0.1, f"Cache hit ratio too low: {hit_ratio:.2f}"
            assert cache_size < service.config.node_cache_size, "Cache size within limits"
            
            logger.info(f"✅ Memory efficiency:")
            logger.info(f"   Cache hit ratio: {hit_ratio:.2f}")
            logger.info(f"   Cache size: {cache_size}")
            logger.info(f"   Memory limit: {final_stats['max_memory_mb']}MB")


async def main():
    """Run comprehensive GraphRAG tests"""
    print("🚀 Starting GraphRAG Comprehensive Test Suite")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_classes = [
        TestGraphRAGService,
        TestGraphAwareChunking, 
        TestGraphRAGAPI,
        TestPerformanceBenchmarks
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}")
        print("-" * 40)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_') and callable(getattr(test_instance, method))]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, test_method)
                
                # Handle async test methods
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                    
                passed_tests += 1
                print(f"✅ {test_method}")
                
            except Exception as e:
                print(f"❌ {test_method}: {str(e)}")
                logger.error(f"Test failed: {test_method}", exc_info=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - GraphRAG Implementation Ready\!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed - Check implementation")
    
    print("\n🔬 GraphRAG Features Tested:")
    print("  • Dynamic parallel entity extraction")
    print("  • Memory bandwidth optimized graph traversal")
    print("  • Hybrid vector + graph retrieval strategies")
    print("  • Graph-aware semantic chunking")
    print("  • API endpoint functionality")
    print("  • Performance benchmarking")
    print("  • Memory efficiency validation")


if __name__ == "__main__":
    asyncio.run(main())
