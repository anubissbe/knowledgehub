"""
Unit tests for RAG retrieval strategies.

Tests all 6 retrieval strategies with comprehensive coverage:
- Vector similarity search
- Hybrid vector + keyword search  
- Ensemble multiple methods
- Iterative progressive refinement
- Graph-enhanced retrieval
- Adaptive query-dependent strategy

Author: Peter Verschuere - Test-Driven Development Expert
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
import asyncio

from api.services.rag_pipeline import (
    HybridRetriever, RetrievalStrategy, RAGConfig, Chunk
)


@pytest.mark.unit
@pytest.mark.rag
class TestHybridRetriever:
    """Test suite for hybrid retrieval strategies."""
    
    @pytest_asyncio.fixture
    async def mock_retriever(self, rag_config_basic, mock_db_session):
        """Create retriever with mocked dependencies."""
        retriever = HybridRetriever(rag_config_basic, mock_db_session)
        
        # Mock embedding generation
        async def mock_embedding(text: str) -> np.ndarray:
            # Return deterministic embedding
            np.random.seed(hash(text) % 2**32)
            return np.random.rand(1536)
        
        retriever._get_embedding = mock_embedding
        return retriever
    
    async def test_retriever_initialization(self, rag_config_basic, mock_db_session):
        """Test retriever initialization."""
        retriever = HybridRetriever(rag_config_basic, mock_db_session)
        
        assert retriever.config == rag_config_basic
        assert retriever.db == mock_db_session
        assert isinstance(retriever.cache, dict)
    
    @pytest.mark.asyncio
    async def test_vector_retrieval(self, mock_retriever):
        """Test pure vector similarity search."""
        query = "What is machine learning?"
        
        chunks = await mock_retriever._vector_retrieval(query)
        
        assert isinstance(chunks, list)
        # Mock should return results
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'metadata')
            assert 'distance' in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, mock_retriever):
        """Test hybrid vector + keyword retrieval."""
        query = "Python programming language"
        
        # Mock both vector and keyword results
        with patch.object(mock_retriever, '_vector_retrieval') as mock_vector, \
             patch.object(mock_retriever, '_merge_and_rerank') as mock_merge:
            
            mock_vector_chunks = [
                Chunk(id="v1", content="Python is great", document_id="doc1", position=0)
            ]
            mock_vector.return_value = mock_vector_chunks
            mock_merge.return_value = mock_vector_chunks
            
            chunks = await mock_retriever._hybrid_retrieval(query)
            
            mock_vector.assert_called_once_with(query)
            mock_merge.assert_called_once()
            assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_ensemble_retrieval(self, mock_retriever):
        """Test ensemble retrieval with voting."""
        query = "FastAPI framework"
        
        # Mock individual retrieval methods
        with patch.object(mock_retriever, '_vector_retrieval') as mock_vector, \
             patch.object(mock_retriever, '_bm25_retrieval') as mock_bm25, \
             patch.object(mock_retriever, '_semantic_retrieval') as mock_semantic:
            
            # Mock results from each method
            chunk1 = Chunk(id="c1", content="FastAPI is fast", document_id="doc1", position=0)
            chunk2 = Chunk(id="c2", content="FastAPI uses Python", document_id="doc2", position=1)
            chunk3 = Chunk(id="c1", content="FastAPI is fast", document_id="doc1", position=0)  # Duplicate
            
            mock_vector.return_value = [chunk1, chunk2]
            mock_bm25.return_value = [chunk3]  # Same as chunk1
            mock_semantic.return_value = [chunk2]
            
            chunks = await mock_retriever._ensemble_retrieval(query)
            
            # Should use reciprocal rank fusion
            assert len(chunks) > 0
            # Chunk1 should rank higher due to appearing in multiple results
            chunk_ids = [c.id for c in chunks]
            assert "c1" in chunk_ids
    
    @pytest.mark.asyncio
    async def test_graph_enhanced_retrieval(self, mock_retriever):
        """Test graph-enhanced retrieval."""
        query = "neural networks"
        
        with patch.object(mock_retriever, '_vector_retrieval') as mock_vector:
            initial_chunks = [
                Chunk(id="g1", content="Neural networks learn", document_id="doc1", position=0)
            ]
            mock_vector.return_value = initial_chunks
            
            chunks = await mock_retriever._graph_enhanced_retrieval(query)
            
            assert len(chunks) > 0
            # Should add graph enhancement metadata
            for chunk in chunks:
                assert chunk.metadata.get("graph_enhanced") is True
    
    @pytest.mark.asyncio
    async def test_iterative_retrieval(self, mock_retriever):
        """Test iterative retrieval with query refinement."""
        query = "machine learning algorithms"
        
        with patch.object(mock_retriever, '_vector_retrieval') as mock_vector, \
             patch.object(mock_retriever, '_refine_query') as mock_refine:
            
            # Mock iterative results
            iter1_chunks = [Chunk(id="i1", content="ML algorithms", document_id="doc1", position=0)]
            iter2_chunks = [Chunk(id="i2", content="Deep learning", document_id="doc2", position=0)]
            
            mock_vector.side_effect = [iter1_chunks, iter2_chunks, []]  # Third iteration empty
            mock_refine.side_effect = ["machine learning algorithms supervised", "machine learning deep"]
            
            chunks = await mock_retriever._iterative_retrieval(query)
            
            assert len(chunks) >= 2
            assert mock_vector.call_count >= 2
            assert mock_refine.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_adaptive_retrieval(self, mock_retriever):
        """Test adaptive retrieval strategy selection."""
        # Test different query types
        test_cases = [
            ("What?", "hybrid"),  # Short question -> hybrid
            ("API endpoint authentication security", "ensemble"),  # Technical terms -> ensemble  
            ("Explain the detailed process of machine learning model training", "iterative"),  # Long -> iterative
            ("Python", "vector")  # Default -> vector
        ]
        
        for query, expected_strategy in test_cases:
            with patch.object(mock_retriever, f'_{expected_strategy}_retrieval') as mock_method:
                mock_method.return_value = [
                    Chunk(id="a1", content="Adaptive result", document_id="doc1", position=0)
                ]
                
                chunks = await mock_retriever._adaptive_retrieval(query)
                
                mock_method.assert_called_once_with(query)
                assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_bm25_retrieval(self, mock_retriever):
        """Test BM25 keyword-based retrieval."""
        query = "Python programming"
        
        chunks = await mock_retriever._bm25_retrieval(query)
        
        assert isinstance(chunks, list)
        # Mock DB should return results
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert 'bm25_score' in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_semantic_retrieval(self, mock_retriever):
        """Test semantic retrieval (fallback to vector)."""
        query = "artificial intelligence"
        
        with patch.object(mock_retriever, '_vector_retrieval') as mock_vector:
            mock_vector.return_value = [
                Chunk(id="s1", content="AI is powerful", document_id="doc1", position=0)
            ]
            
            chunks = await mock_retriever._semantic_retrieval(query)
            
            mock_vector.assert_called_once_with(query)
            assert len(chunks) > 0
    
    @pytest.mark.parametrize("strategy", list(RetrievalStrategy))
    @pytest.mark.asyncio
    async def test_all_retrieval_strategies(self, mock_retriever, strategy):
        """Test all retrieval strategies work without errors."""
        query = "test query for all strategies"
        
        # Mock the appropriate method for each strategy
        strategy_map = {
            RetrievalStrategy.VECTOR: '_vector_retrieval',
            RetrievalStrategy.HYBRID: '_hybrid_retrieval', 
            RetrievalStrategy.ENSEMBLE: '_ensemble_retrieval',
            RetrievalStrategy.GRAPH: '_graph_enhanced_retrieval',
            RetrievalStrategy.ITERATIVE: '_iterative_retrieval',
            RetrievalStrategy.ADAPTIVE: '_adaptive_retrieval'
        }
        
        method_name = strategy_map[strategy]
        with patch.object(mock_retriever, method_name) as mock_method:
            mock_method.return_value = [
                Chunk(id="test", content="Test result", document_id="doc1", position=0)
            ]
            
            chunks = await mock_retriever.retrieve(query, strategy)
            
            mock_method.assert_called_once()
            assert len(chunks) > 0
    
    async def test_merge_and_rerank(self, mock_retriever):
        """Test merging and reranking multiple chunk lists."""
        # Create test chunks with different scores
        chunks1 = [
            Chunk(id="c1", content="First result", document_id="doc1", position=0,
                  metadata={"distance": 0.1}),
            Chunk(id="c2", content="Second result", document_id="doc2", position=1,
                  metadata={"distance": 0.3})
        ]
        
        chunks2 = [
            Chunk(id="c1", content="First result", document_id="doc1", position=0,
                  metadata={"rank": 0.9}),  # Same chunk, different score
            Chunk(id="c3", content="Third result", document_id="doc3", position=0,
                  metadata={"rank": 0.5})
        ]
        
        merged = mock_retriever._merge_and_rerank(chunks1, chunks2)
        
        assert len(merged) > 0
        # c1 should rank highest due to appearing in both lists
        assert merged[0].id == "c1"
    
    async def test_query_refinement(self, mock_retriever):
        """Test query refinement logic."""
        original_query = "machine learning"
        chunks = [
            Chunk(id="r1", content="Machine learning uses algorithms and data science techniques", 
                  document_id="doc1", position=0),
            Chunk(id="r2", content="Supervised learning requires labeled datasets", 
                  document_id="doc2", position=1)
        ]
        
        refined_query = mock_retriever._refine_query(original_query, chunks)
        
        # Should add relevant terms from chunks
        assert refined_query != original_query
        assert "machine learning" in refined_query.lower()
    
    async def test_technical_term_detection(self, mock_retriever):
        """Test technical term detection."""
        technical_queries = [
            "API endpoint authentication",
            "database optimization algorithm", 
            "function implementation details"
        ]
        
        non_technical_queries = [
            "what is the weather like",
            "how are you doing today",
            "simple question here"
        ]
        
        for query in technical_queries:
            assert mock_retriever._detect_technical_terms(query) is True
        
        for query in non_technical_queries:
            assert mock_retriever._detect_technical_terms(query) is False
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, mock_retriever):
        """Test embedding caching functionality."""
        text = "test text for caching"
        
        # Enable caching
        mock_retriever.config.enable_caching = True
        
        # First call should generate embedding
        embedding1 = await mock_retriever._get_embedding(text)
        
        # Second call should use cache
        embedding2 = await mock_retriever._get_embedding(text)
        
        # Should be identical (cached)
        np.testing.assert_array_equal(embedding1, embedding2)
        
        # Cache should contain the embedding
        assert len(mock_retriever.cache) > 0
    
    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, mock_retriever):
        """Test similarity threshold filtering in vector retrieval."""
        query = "test query"
        
        # Mock database results with varying distances
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(id="close", content="Close match", document_id="doc1", distance=0.1),
            MagicMock(id="far", content="Far match", document_id="doc2", distance=0.9)
        ]
        mock_retriever.db.execute.return_value = mock_result
        
        # Set high threshold
        mock_retriever.config.similarity_threshold = 0.8
        
        chunks = await mock_retriever._vector_retrieval(query)
        
        # Only close matches should be returned
        assert len(chunks) == 1
        assert chunks[0].id == "close"
    
    @pytest.mark.asyncio
    async def test_retrieval_performance(self, mock_retriever, performance_metrics):
        """Test retrieval performance meets requirements."""
        query = "performance test query"
        
        performance_metrics.start_timer("retrieval")
        chunks = await mock_retriever._vector_retrieval(query)
        performance_metrics.end_timer("retrieval")
        
        # Should complete retrieval within 200ms
        performance_metrics.assert_performance("retrieval", 0.2)
        
        # Should return reasonable number of results
        assert len(chunks) <= mock_retriever.config.top_k


@pytest.mark.unit 
@pytest.mark.rag
class TestRetrievalHelpers:
    """Test helper methods in retrieval implementation."""
    
    async def test_config_validation(self, mock_db_session):
        """Test retrieval configuration validation."""
        # Valid config
        valid_config = RAGConfig(
            top_k=10,
            similarity_threshold=0.7,
            retrieval_strategy=RetrievalStrategy.HYBRID
        )
        
        retriever = HybridRetriever(valid_config, mock_db_session)
        assert retriever.config.top_k == 10
        assert retriever.config.similarity_threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, mock_retriever):
        """Test handling of empty or invalid queries."""
        # Empty query
        chunks = await mock_retriever._vector_retrieval("")
        assert isinstance(chunks, list)
        
        # Whitespace-only query  
        chunks = await mock_retriever._vector_retrieval("   ")
        assert isinstance(chunks, list)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_retriever):
        """Test error handling in retrieval methods."""
        # Mock database error
        mock_retriever.db.execute.side_effect = Exception("Database error")
        
        # Should handle gracefully
        try:
            chunks = await mock_retriever._vector_retrieval("test query")
        except Exception:
            pytest.fail("Retrieval should handle database errors gracefully")
    
    @pytest.mark.asyncio
    async def test_concurrent_retrieval(self, mock_retriever):
        """Test concurrent retrieval operations."""
        queries = ["query1", "query2", "query3"]
        
        # Mock successful retrievals
        with patch.object(mock_retriever, '_vector_retrieval') as mock_vector:
            mock_vector.return_value = [
                Chunk(id="test", content="Test", document_id="doc1", position=0)
            ]
            
            # Run concurrent retrievals
            tasks = [mock_retriever.retrieve(query) for query in queries]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(queries)
            assert all(len(result) > 0 for result in results)
