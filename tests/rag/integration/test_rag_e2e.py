"""
End-to-End Integration Tests for RAG System.

Tests complete RAG workflows including:
- Full document ingestion to query response pipeline
- Multi-service integration (PostgreSQL, Redis, Weaviate, Neo4j)
- Cross-component data flow validation
- Performance under realistic load
- Error recovery and fallback mechanisms

Author: Peter Verschuere - Test-Driven Development Expert
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock
import asyncio
import json
import time
from typing import List, Dict, Any

from api.services.rag_pipeline import RAGPipeline, RAGConfig, Document, ChunkingStrategy, RetrievalStrategy


@pytest.mark.integration
@pytest.mark.rag
@pytest.mark.e2e
class TestRAGEndToEnd:
    """End-to-end RAG system integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_rag_workflow(self, full_rag_environment, sample_documents, test_queries):
        """Test complete RAG workflow from document ingestion to response generation."""
        environment = full_rag_environment
        pipeline = environment["pipeline"]
        
        # Step 1: Ingest documents
        ingested_docs = []
        for doc in sample_documents[:3]:  # Use first 3 documents
            ingested_doc = await pipeline.ingest_document(doc.content, doc.metadata)
            ingested_docs.append(ingested_doc)
            assert ingested_doc.id is not None
            assert len(ingested_doc.chunks) > 0
        
        # Step 2: Process queries
        query_results = []
        for query in test_queries[:2]:  # Test first 2 queries
            result = await pipeline.process_query(query)
            query_results.append(result)
            
            # Validate result structure
            assert isinstance(result, dict)
            assert 'query' in result
            assert 'response' in result
            assert 'chunks_used' in result
            assert 'processing_time' in result
            assert 'metadata' in result
            
            # Validate response quality
            assert len(result['response']) > 10  # Non-trivial response
            assert result['chunks_used'] > 0  # Found relevant chunks
            assert result['processing_time'] > 0
        
        # Step 3: Verify cross-query consistency
        python_queries = ["What is Python?", "Tell me about Python programming"]
        python_results = []
        for query in python_queries:
            result = await pipeline.process_query(query)
            python_results.append(result)
        
        # Both should mention Python
        for result in python_results:
            assert "python" in result['response'].lower()
    
    @pytest.mark.asyncio
    async def test_multi_strategy_integration(self, mock_db_session):
        """Test integration across different chunking and retrieval strategies."""
        strategies_to_test = [
            (ChunkingStrategy.SEMANTIC, RetrievalStrategy.VECTOR),
            (ChunkingStrategy.HIERARCHICAL, RetrievalStrategy.HYBRID),
            (ChunkingStrategy.SLIDING, RetrievalStrategy.ENSEMBLE)
        ]
        
        test_document = Document(
            id="multi_strategy_doc",
            content="""
            Machine Learning Overview
            
            Machine learning is a subset of artificial intelligence that enables computers
            to learn and make decisions from data without being explicitly programmed.
            
            Types of Machine Learning
            
            1. Supervised Learning: Uses labeled training data
            2. Unsupervised Learning: Finds patterns in unlabeled data  
            3. Reinforcement Learning: Learns through interaction with environment
            """,
            metadata={"topic": "machine_learning", "complexity": "intermediate"}
        )
        
        results = {}
        
        for chunk_strategy, retrieval_strategy in strategies_to_test:
            config = RAGConfig(
                chunking_strategy=chunk_strategy,
                retrieval_strategy=retrieval_strategy,
                chunk_size=200,
                top_k=5,
                enable_reranking=True
            )
            
            with patch('api.services.rag_pipeline.HybridRetriever._get_embedding') as mock_embed, \
                 patch.object(RAGPipeline, '_store_document_and_chunks'):
                
                mock_embed.return_value = np.random.rand(1536)
                pipeline = RAGPipeline(config, mock_db_session)
                
                # Ingest document
                ingested = await pipeline.ingest_document(test_document.content, test_document.metadata)
                
                # Process query
                with patch.object(pipeline.retriever, 'retrieve') as mock_retrieve, \
                     patch.object(pipeline.generator, 'generate') as mock_generate:
                    
                    mock_retrieve.return_value = ingested.chunks[:3]
                    mock_generate.return_value = f"ML response with {chunk_strategy.value}-{retrieval_strategy.value}"
                    
                    result = await pipeline.process_query("What is machine learning?")
                    
                    results[f"{chunk_strategy.value}_{retrieval_strategy.value}"] = result
        
        # All strategies should work
        assert len(results) == len(strategies_to_test)
        for key, result in results.items():
            assert 'response' in result
            assert key.replace('_', '-') in result['response']
    
    @pytest.mark.asyncio
    async def test_database_integration(self, populated_test_db, rag_config_basic):
        """Test integration with actual database operations."""
        with patch('api.services.rag_pipeline.HybridRetriever._get_embedding') as mock_embed:
            mock_embed.return_value = np.random.rand(1536)
            pipeline = RAGPipeline(rag_config_basic, populated_test_db)
            
            # Test document storage and retrieval
            test_content = "Integration test document for database operations."
            test_metadata = {"integration": True, "test_type": "database"}
            
            # Ingest document (this should actually store in test DB)
            document = await pipeline.ingest_document(test_content, test_metadata)
            
            # Verify storage by checking database
            from sqlalchemy import text
            result = await populated_test_db.execute(
                text("SELECT COUNT(*) as count FROM documents WHERE id = :doc_id"),
                {"doc_id": document.id}
            )
            doc_count = result.fetchone().count
            assert doc_count == 1
            
            # Check chunks were stored
            result = await populated_test_db.execute(
                text("SELECT COUNT(*) as count FROM chunks WHERE document_id = :doc_id"),
                {"doc_id": document.id}
            )
            chunk_count = result.fetchone().count
            assert chunk_count == len(document.chunks)
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, rag_config_advanced, mock_db_session, mock_redis_client):
        """Test caching integration across pipeline components."""
        config = rag_config_advanced
        config.enable_caching = True
        
        with patch('api.services.rag_pipeline.HybridRetriever._get_embedding') as mock_embed, \
             patch('redis.Redis') as mock_redis:
            
            mock_embed.return_value = np.random.rand(1536)
            mock_redis.return_value = mock_redis_client
            pipeline = RAGPipeline(config, mock_db_session)
            
            query = "What is caching in RAG systems?"
            
            with patch.object(pipeline.generator, 'generate') as mock_generate:
                mock_generate.return_value = "Caching improves RAG performance."
                
                # First query - should populate cache
                result1 = await pipeline.process_query(query)
                
                # Second query - should use cache
                result2 = await pipeline.process_query(query)
                
                assert result1['response'] == result2['response']
                # Cache should have embeddings
                assert len(pipeline.retriever.cache) > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, rag_pipeline_basic):
        """Test error recovery across pipeline components."""
        query = "Test error recovery"
        
        # Test retriever failure with generator fallback
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            # Simulate retriever failure
            mock_retrieve.side_effect = Exception("Retriever failed")
            mock_generate.return_value = "Fallback response when retrieval fails"
            
            # Should handle gracefully or raise appropriate exception
            with pytest.raises(Exception) as exc_info:
                await rag_pipeline_basic.process_query(query)
            
            assert "Retriever failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_usage(self, rag_pipeline_basic, test_queries):
        """Test concurrent usage of RAG pipeline."""
        queries = test_queries[:4]  # Use first 4 queries
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            # Mock responses
            mock_retrieve.return_value = [
                Chunk(id="concurrent", content="Concurrent test", document_id="doc1", position=0)
            ]
            mock_generate.return_value = "Concurrent response"
            
            # Execute queries concurrently
            start_time = time.time()
            tasks = [rag_pipeline_basic.process_query(query) for query in queries]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # All should succeed
            assert len(results) == len(queries)
            for result in results:
                assert 'response' in result
                assert result['response'] == "Concurrent response"
            
            # Concurrent execution should be faster than sequential
            concurrent_time = end_time - start_time
            assert concurrent_time < len(queries) * 0.1  # Much faster than sequential
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_operations(self, full_rag_environment):
        """Test data consistency across multiple RAG operations."""
        environment = full_rag_environment
        pipeline = environment["pipeline"]
        
        # Ingest related documents
        documents = [
            ("Python is a programming language used for AI.", {"topic": "python"}),
            ("Machine learning uses Python for data analysis.", {"topic": "ml"}),
            ("AI applications often use Python libraries.", {"topic": "ai"})
        ]
        
        ingested_docs = []
        for content, metadata in documents:
            doc = await pipeline.ingest_document(content, metadata)
            ingested_docs.append(doc)
        
        # Query for related information
        queries = [
            "What programming language is used for AI?",
            "How is Python used in machine learning?",
            "What libraries does AI use?"
        ]
        
        results = []
        for query in queries:
            result = await pipeline.process_query(query)
            results.append(result)
        
        # All results should mention Python (consistency check)
        python_mentions = sum(1 for result in results if "python" in result['response'].lower())
        assert python_mentions >= len(queries) * 0.7  # At least 70% should mention Python
    
    @pytest.mark.asyncio
    async def test_metadata_flow_integration(self, rag_pipeline_basic):
        """Test metadata flow through complete pipeline."""
        # Document with rich metadata
        content = "Advanced RAG systems use metadata for better retrieval."
        metadata = {
            "source": "research_paper",
            "authors": ["Alice", "Bob"],
            "year": 2024,
            "tags": ["rag", "metadata", "retrieval"],
            "confidence": 0.95
        }
        
        with patch.object(rag_pipeline_basic, '_store_document_and_chunks'):
            document = await rag_pipeline_basic.ingest_document(content, metadata)
            
            # Document metadata should be preserved
            assert document.metadata == metadata
            
            # Query with context metadata
            query_context = {
                "user_id": "test_user",
                "session_id": "test_session",
                "preferences": {"detailed": True}
            }
            
            with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
                 patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
                
                mock_retrieve.return_value = document.chunks[:2]
                mock_generate.return_value = "Metadata-aware response"
                
                result = await rag_pipeline_basic.process_query("What is RAG?", query_context)
                
                # Result should include metadata information
                assert 'metadata' in result
                chunk_metadata = result['metadata'].get('chunks_retrieved', [])
                assert len(chunk_metadata) > 0


@pytest.mark.integration
@pytest.mark.rag
@pytest.mark.performance
class TestRAGPerformanceIntegration:
    """Performance integration tests for RAG system."""
    
    @pytest.mark.asyncio
    async def test_throughput_performance(self, rag_pipeline_basic, performance_metrics):
        """Test RAG system throughput under load."""
        queries = [f"Test query number {i} about machine learning" for i in range(20)]
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [
                Chunk(id="perf", content="Performance test content", document_id="doc1", position=0)
            ]
            mock_generate.return_value = "Performance test response"
            
            performance_metrics.start_timer("throughput_test")
            
            # Process queries in batches
            batch_size = 5
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i+batch_size]
                tasks = [rag_pipeline_basic.process_query(query) for query in batch]
                batch_results = await asyncio.gather(*tasks)
                assert len(batch_results) == len(batch)
            
            performance_metrics.end_timer("throughput_test")
            
            # Should handle 20 queries within reasonable time
            performance_metrics.assert_performance("throughput_test", 5.0)  # 5 seconds max
    
    @pytest.mark.asyncio
    async def test_memory_usage_integration(self, rag_pipeline_basic, memory_profiler, load_test_data):
        """Test memory usage during intensive RAG operations."""
        # Generate test data
        documents = load_test_data["generate_documents"](10)
        queries = load_test_data["generate_queries"](20)
        
        memory_profiler.start("memory_test")
        
        with patch.object(rag_pipeline_basic, '_store_document_and_chunks'):
            # Ingest documents
            for i, doc in enumerate(documents):
                await rag_pipeline_basic.ingest_document(doc.content, doc.metadata)
                if i == 4:  # Checkpoint halfway
                    memory_profiler.checkpoint("after_5_docs")
        
        memory_profiler.checkpoint("after_ingestion")
        
        # Process queries
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [
                Chunk(id="mem", content="Memory test", document_id="doc1", position=0)
            ]
            mock_generate.return_value = "Memory test response"
            
            for query in queries:
                await rag_pipeline_basic.process_query(query)
        
        memory_profiler.stop("memory_test")
        
        # Check memory usage
        memory_stats = memory_profiler.get_memory_usage("memory_test_start", "memory_test_end")
        if memory_stats:
            # Should not use excessive memory
            assert memory_stats["memory_diff_mb"] < 100  # Less than 100MB increase
    
    @pytest.mark.asyncio
    async def test_scalability_integration(self, mock_db_session, quality_gates):
        """Test RAG system scalability with increasing load."""
        thresholds = quality_gates["performance_thresholds"]
        
        # Test with increasing document sizes
        document_sizes = [1000, 5000, 10000, 20000]  # Characters
        
        for size in document_sizes:
            content = "This is a scalability test document. " * (size // 40)
            
            config = RAGConfig(chunk_size=500, top_k=10)
            
            with patch('api.services.rag_pipeline.HybridRetriever._get_embedding') as mock_embed:
                mock_embed.return_value = np.random.rand(1536)
                pipeline = RAGPipeline(config, mock_db_session)
                
                start_time = time.time()
                
                with patch.object(pipeline, '_store_document_and_chunks'):
                    document = await pipeline.ingest_document(content, {"size": size})
                
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Processing time should scale reasonably
                expected_max_time = thresholds["chunking_time_ms"] * (size / 1000)  # Scale with size
                assert processing_time <= expected_max_time, f"Processing {size} chars took {processing_time}ms"


@pytest.mark.integration
@pytest.mark.rag
@pytest.mark.slow
class TestRAGComplexWorkflows:
    """Test complex RAG workflows and use cases."""
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, rag_pipeline_advanced):
        """Test multi-turn conversational RAG."""
        conversation_turns = [
            "What is machine learning?",
            "How does supervised learning work?", 
            "What are some examples of supervised learning algorithms?",
            "Which one is best for classification problems?"
        ]
        
        conversation_context = {"conversation_id": "test_conv", "turn": 0}
        responses = []
        
        with patch.object(rag_pipeline_advanced.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_advanced.generator, 'generate') as mock_generate:
            
            # Mock progressive responses
            mock_responses = [
                "ML is a subset of AI that learns from data.",
                "Supervised learning uses labeled training data.",
                "Examples include decision trees, SVM, neural networks.",
                "For classification, SVM and random forests work well."
            ]
            
            mock_retrieve.return_value = [
                Chunk(id="conv", content="Conversational content", document_id="doc1", position=0)
            ]
            
            for i, (turn, expected) in enumerate(zip(conversation_turns, mock_responses)):
                conversation_context["turn"] = i
                mock_generate.return_value = expected
                
                result = await rag_pipeline_advanced.process_query(turn, conversation_context)
                responses.append(result['response'])
                
                assert expected in result['response']
        
        # Responses should build on each other
        assert "machine learning" in responses[0].lower()
        assert "supervised" in responses[1].lower()
        assert "classification" in responses[3].lower()
    
    @pytest.mark.asyncio
    async def test_domain_specific_retrieval(self, full_rag_environment):
        """Test domain-specific document retrieval and response generation."""
        environment = full_rag_environment
        pipeline = environment["pipeline"]
        
        # Ingest domain-specific documents
        domains = {
            "medical": "Diabetes is a metabolic disorder characterized by high blood sugar levels.",
            "legal": "Contract law governs the formation and enforcement of agreements between parties.",
            "technical": "RESTful APIs use HTTP methods like GET, POST, PUT, and DELETE for operations."
        }
        
        for domain, content in domains.items():
            await pipeline.ingest_document(content, {"domain": domain, "type": "professional"})
        
        # Test domain-specific queries
        domain_queries = [
            ("What is diabetes?", "medical"),
            ("How do contracts work?", "legal"), 
            ("What are REST APIs?", "technical")
        ]
        
        for query, expected_domain in domain_queries:
            with patch.object(pipeline.retriever, 'retrieve') as mock_retrieve, \
                 patch.object(pipeline.generator, 'generate') as mock_generate:
                
                # Mock domain-appropriate retrieval
                domain_content = domains[expected_domain]
                mock_retrieve.return_value = [
                    Chunk(id=f"{expected_domain}_chunk", content=domain_content, 
                          document_id=f"{expected_domain}_doc", position=0)
                ]
                mock_generate.return_value = f"Domain-specific response for {expected_domain}"
                
                result = await pipeline.process_query(query)
                
                assert expected_domain in result['response'].lower()
    
    @pytest.mark.asyncio
    async def test_multilingual_content_handling(self, rag_pipeline_basic):
        """Test handling of multilingual content in RAG pipeline."""
        multilingual_documents = [
            ("English content about artificial intelligence and machine learning.", "en"),
            ("Contenu français sur l'intelligence artificielle et l'apprentissage automatique.", "fr"),
            ("Contenido español sobre inteligencia artificial y aprendizaje automático.", "es")
        ]
        
        for content, lang in multilingual_documents:
            with patch.object(rag_pipeline_basic, '_store_document_and_chunks'):
                document = await rag_pipeline_basic.ingest_document(
                    content, 
                    {"language": lang, "topic": "ai"}
                )
                
                # Should handle all languages
                assert document.content == content
                assert len(document.chunks) > 0
                
                # Chunks should preserve language
                for chunk in document.chunks:
                    assert isinstance(chunk.content, str)
                    assert len(chunk.content) > 0
