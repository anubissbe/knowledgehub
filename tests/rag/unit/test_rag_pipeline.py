"""
Unit tests for complete RAG pipeline integration.

Tests the end-to-end RAG pipeline including:
- Document ingestion and chunking
- Query processing and retrieval
- Context construction and response generation
- Performance optimization features
- Error handling and edge cases

Author: Peter Verschuere - Test-Driven Development Expert
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
import json
from datetime import datetime

from api.services.rag_pipeline import (
    RAGPipeline, RAGConfig, Document, Chunk, ChunkingStrategy, RetrievalStrategy
)


@pytest.mark.unit
@pytest.mark.rag
class TestRAGPipeline:
    """Test suite for complete RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, rag_config_basic, mock_db_session):
        """Test RAG pipeline initialization."""
        with patch('api.services.rag_pipeline.HybridRetriever._get_embedding') as mock_embed:
            mock_embed.return_value = np.random.rand(1536)
            pipeline = RAGPipeline(rag_config_basic, mock_db_session)
            
            assert pipeline.config == rag_config_basic
            assert pipeline.db == mock_db_session
            assert hasattr(pipeline, 'chunker')
            assert hasattr(pipeline, 'retriever')
            assert hasattr(pipeline, 'reranker')
            assert hasattr(pipeline, 'generator')
    
    @pytest.mark.asyncio
    async def test_document_ingestion(self, rag_pipeline_basic, sample_documents):
        """Test document ingestion process."""
        document = sample_documents[0]  # Python programming document
        
        with patch.object(rag_pipeline_basic, '_store_document_and_chunks') as mock_store:
            ingested_doc = await rag_pipeline_basic.ingest_document(
                document.content, 
                document.metadata
            )
            
            assert isinstance(ingested_doc, Document)
            assert ingested_doc.content == document.content
            assert ingested_doc.metadata == document.metadata
            assert len(ingested_doc.chunks) > 0
            
            # Should have called storage
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_processing_basic(self, rag_pipeline_basic, test_queries):
        """Test basic query processing."""
        query = test_queries[0]  # "What is Python programming language?"
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            # Mock retrieval results
            mock_chunks = [
                Chunk(id="c1", content="Python is a programming language", 
                      document_id="doc1", position=0)
            ]
            mock_retrieve.return_value = mock_chunks
            mock_generate.return_value = "Python is a high-level programming language."
            
            result = await rag_pipeline_basic.process_query(query)
            
            assert isinstance(result, dict)
            assert 'query' in result
            assert 'response' in result
            assert 'chunks_used' in result
            assert 'processing_time' in result
            assert 'metadata' in result
            
            assert result['query'] == query
            assert result['chunks_used'] == len(mock_chunks)
            assert isinstance(result['processing_time'], float)
    
    @pytest.mark.asyncio
    async def test_query_processing_advanced(self, rag_pipeline_advanced, test_queries):
        """Test advanced query processing with all features enabled."""
        query = test_queries[1]  # "How does machine learning work?"
        
        with patch.object(rag_pipeline_advanced.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_advanced.reranker, 'rerank') as mock_rerank, \
             patch.object(rag_pipeline_advanced.generator, 'generate') as mock_generate, \
             patch.object(rag_pipeline_advanced, '_self_correct') as mock_correct, \
             patch.object(rag_pipeline_advanced, '_preprocess_query') as mock_preprocess:
            
            # Mock preprocessing (HyDE)
            mock_preprocess.return_value = f"{query} enhanced with hypothetical document"
            
            # Mock retrieval results
            mock_chunks = [
                Chunk(id="c1", content="Machine learning uses algorithms", 
                      document_id="doc1", position=0),
                Chunk(id="c2", content="ML learns from data patterns", 
                      document_id="doc2", position=1)
            ]
            mock_retrieve.return_value = mock_chunks
            
            # Mock reranking
            mock_rerank.return_value = mock_chunks[:1]  # Return top result
            
            # Mock generation and correction
            mock_response = "Machine learning works by training algorithms on data."
            mock_generate.return_value = mock_response
            mock_correct.return_value = mock_response + " (verified)"
            
            result = await rag_pipeline_advanced.process_query(query)
            
            # Verify all advanced features were used
            mock_preprocess.assert_called_once()
            mock_retrieve.assert_called_once()
            mock_rerank.assert_called_once()
            mock_generate.assert_called_once()
            mock_correct.assert_called_once()
            
            assert result['response'].endswith("(verified)")
    
    @pytest.mark.asyncio
    async def test_context_construction(self, rag_pipeline_basic):
        """Test context construction from retrieved chunks."""
        chunks = [
            Chunk(id="c1", content="Python is a programming language", 
                  document_id="doc1", position=0,
                  context_before="", context_after="used for AI"),
            Chunk(id="c2", content="Machine learning uses Python", 
                  document_id="doc2", position=0,
                  context_before="Many developers", context_after="for data science")
        ]
        
        context = rag_pipeline_basic._construct_context(chunks)
        
        assert isinstance(context, str)
        assert "Python is a programming language" in context
        assert "Machine learning uses Python" in context
        assert "---" in context  # Chunk separator
        assert "[Context:" in context  # Context markers
    
    @pytest.mark.asyncio
    async def test_context_length_limits(self, rag_config_basic, mock_db_session):
        """Test context length limiting."""
        config = rag_config_basic
        config.max_context_length = 100  # Small limit
        
        with patch('api.services.rag_pipeline.HybridRetriever._get_embedding'):
            pipeline = RAGPipeline(config, mock_db_session)
            
            # Create chunks that exceed limit
            large_chunks = [
                Chunk(id=f"c{i}", content="This is a chunk with content. " * 10, 
                      document_id="doc1", position=i)
                for i in range(5)
            ]
            
            context = pipeline._construct_context(large_chunks)
            
            # Should respect token limit
            token_count = len(pipeline.chunker.tokenizer.encode(context))
            assert token_count <= config.max_context_length + 50  # Some tolerance
    
    @pytest.mark.asyncio
    async def test_hyde_preprocessing(self, rag_pipeline_advanced):
        """Test HyDE (Hypothetical Document Embedding) preprocessing."""
        query = "What is deep learning?"
        
        with patch.object(rag_pipeline_advanced, '_generate_hypothetical_document') as mock_hyde:
            mock_hyde.return_value = "Deep learning uses neural networks with multiple layers"
            
            processed_query = await rag_pipeline_advanced._preprocess_query(query)
            
            mock_hyde.assert_called_once_with(query)
            assert query in processed_query
            assert "Deep learning uses neural networks" in processed_query
    
    @pytest.mark.asyncio
    async def test_self_correction(self, rag_pipeline_advanced):
        """Test self-correction mechanism."""
        response = "This is the initial response with potential errors."
        chunks = [
            Chunk(id="c1", content="Factual information", document_id="doc1", position=0)
        ]
        
        # Mock self-correction (simplified implementation)
        corrected = await rag_pipeline_advanced._self_correct(response, chunks)
        
        # Current implementation is pass-through
        assert corrected == response
    
    @pytest.mark.asyncio
    async def test_error_handling(self, rag_pipeline_basic):
        """Test error handling in pipeline operations."""
        query = "test query"
        
        # Test retrieval failure
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve:
            mock_retrieve.side_effect = Exception("Retrieval error")
            
            try:
                await rag_pipeline_basic.process_query(query)
                pytest.fail("Should have raised exception")
            except Exception as e:
                assert "Retrieval error" in str(e)
    
    @pytest.mark.asyncio
    async def test_empty_retrieval_results(self, rag_pipeline_basic):
        """Test handling of empty retrieval results."""
        query = "nonexistent information"
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            # Mock empty results
            mock_retrieve.return_value = []
            mock_generate.return_value = "No relevant information found."
            
            result = await rag_pipeline_basic.process_query(query)
            
            assert result['chunks_used'] == 0
            assert "No relevant information found" in result['response']
    
    @pytest.mark.asyncio
    async def test_pipeline_performance(self, rag_pipeline_basic, performance_metrics):
        """Test end-to-end pipeline performance."""
        query = "What is Python?"
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [
                Chunk(id="c1", content="Python info", document_id="doc1", position=0)
            ]
            mock_generate.return_value = "Python is a language."
            
            performance_metrics.start_timer("end_to_end")
            result = await rag_pipeline_basic.process_query(query)
            performance_metrics.end_timer("end_to_end")
            
            # Should complete within 500ms for simple queries
            performance_metrics.assert_performance("end_to_end", 0.5)
            
            # Should also record internal timing
            assert result['processing_time'] > 0
    
    @pytest.mark.parametrize("chunking_strategy", list(ChunkingStrategy))
    @pytest.mark.parametrize("retrieval_strategy", [RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID])
    @pytest.mark.asyncio
    async def test_strategy_combinations(self, mock_db_session, chunking_strategy, retrieval_strategy):
        """Test different combinations of chunking and retrieval strategies."""
        config = RAGConfig(
            chunking_strategy=chunking_strategy,
            retrieval_strategy=retrieval_strategy,
            chunk_size=256,
            top_k=5,
            enable_reranking=False,
            enable_caching=False
        )
        
        with patch('api.services.rag_pipeline.HybridRetriever._get_embedding'):
            pipeline = RAGPipeline(config, mock_db_session)
            
            # Test document ingestion
            with patch.object(pipeline, '_store_document_and_chunks'):
                document = await pipeline.ingest_document(
                    "Test content for strategy testing.",
                    {"test": True}
                )
                
                assert isinstance(document, Document)
                assert len(document.chunks) > 0
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, rag_pipeline_basic):
        """Test metadata preservation through pipeline."""
        query = "test query"
        context = {"user_id": "test_user", "session_id": "test_session"}
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_chunks = [
                Chunk(id="c1", content="Test content", document_id="doc1", position=0,
                      metadata={"source": "test", "confidence": 0.9})
            ]
            mock_retrieve.return_value = mock_chunks
            mock_generate.return_value = "Test response"
            
            result = await rag_pipeline_basic.process_query(query, context)
            
            # Check metadata in result
            assert 'metadata' in result
            assert 'chunks_retrieved' in result['metadata']
            assert len(result['metadata']['chunks_retrieved']) > 0
            
            chunk_info = result['metadata']['chunks_retrieved'][0]
            assert 'id' in chunk_info
            assert 'score' in chunk_info
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, rag_pipeline_basic):
        """Test handling of concurrent queries."""
        queries = ["Query 1", "Query 2", "Query 3"]
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_retrieve.return_value = [
                Chunk(id="c1", content="Content", document_id="doc1", position=0)
            ]
            mock_generate.return_value = "Response"
            
            # Process queries concurrently
            import asyncio
            tasks = [rag_pipeline_basic.process_query(query) for query in queries]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(queries)
            assert all('response' in result for result in results)
    
    @pytest.mark.asyncio
    async def test_chunking_configuration_impact(self, mock_db_session, sample_documents):
        """Test impact of different chunking configurations."""
        document_content = sample_documents[0].content
        
        configs = [
            RAGConfig(chunk_size=100, chunking_strategy=ChunkingStrategy.SLIDING),
            RAGConfig(chunk_size=500, chunking_strategy=ChunkingStrategy.SEMANTIC),
            RAGConfig(chunk_size=300, chunking_strategy=ChunkingStrategy.HIERARCHICAL)
        ]
        
        chunk_counts = []
        
        for config in configs:
            with patch('api.services.rag_pipeline.HybridRetriever._get_embedding'), \
                 patch('api.services.rag_pipeline.RAGPipeline._store_document_and_chunks'):
                
                pipeline = RAGPipeline(config, mock_db_session)
                document = await pipeline.ingest_document(document_content, {})
                chunk_counts.append(len(document.chunks))
        
        # Different configurations should produce different chunk counts
        assert len(set(chunk_counts)) > 1, "Different configs should produce different chunking results"


@pytest.mark.unit
@pytest.mark.rag  
class TestRAGPipelineEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_very_long_query(self, rag_pipeline_basic):
        """Test handling of very long queries."""
        long_query = "What is machine learning? " * 100  # Very long query
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_retrieve.return_value = []
            mock_generate.return_value = "Response to long query"
            
            result = await rag_pipeline_basic.process_query(long_query)
            
            assert 'response' in result
            assert result['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_special_characters_query(self, rag_pipeline_basic):
        """Test queries with special characters."""
        special_query = "What is C++ & Python? <script>alert('test')</script>"
        
        with patch.object(rag_pipeline_basic.retriever, 'retrieve') as mock_retrieve, \
             patch.object(rag_pipeline_basic.generator, 'generate') as mock_generate:
            
            mock_retrieve.return_value = []
            mock_generate.return_value = "Safe response"
            
            result = await rag_pipeline_basic.process_query(special_query)
            
            assert 'response' in result
            # Should not contain the script tag in response
            assert '<script>' not in result['response']
    
    @pytest.mark.asyncio
    async def test_unicode_content(self, rag_pipeline_basic):
        """Test handling of Unicode content."""
        unicode_content = "Python支持Unicode字符。機械学習は人工知能の一分野です。"
        
        with patch.object(rag_pipeline_basic, '_store_document_and_chunks'):
            document = await rag_pipeline_basic.ingest_document(unicode_content, {})
            
            assert document.content == unicode_content
            assert len(document.chunks) > 0
            
            # Chunks should preserve Unicode
            for chunk in document.chunks:
                # Should not have encoding errors
                assert isinstance(chunk.content, str)
    
    @pytest.mark.asyncio
    async def test_large_document_ingestion(self, rag_pipeline_basic, performance_metrics):
        """Test ingestion of large documents."""
        # Create 50KB document
        large_content = "This is a large document for testing. " * 2000
        
        performance_metrics.start_timer("large_ingestion")
        
        with patch.object(rag_pipeline_basic, '_store_document_and_chunks'):
            document = await rag_pipeline_basic.ingest_document(large_content, {})
        
        performance_metrics.end_timer("large_ingestion")
        
        # Should complete within reasonable time
        performance_metrics.assert_performance("large_ingestion", 2.0)  # 2 seconds max
        
        # Should produce reasonable number of chunks
        assert len(document.chunks) > 10
        assert len(document.chunks) < 1000
