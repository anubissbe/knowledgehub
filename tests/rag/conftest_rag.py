"""
RAG-Specific Test Fixtures for KnowledgeHub

Comprehensive fixtures for testing all RAG system components including:
- Advanced RAG Pipeline with 6 chunking + 6 retrieval strategies
- LlamaIndex integration with mathematical optimizations  
- GraphRAG with Neo4j knowledge graph
- Performance optimizations and intelligent caching

Author: Peter Verschuere - Test-Driven Development Expert
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import numpy as np

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# RAG System Imports
from api.services.rag_pipeline import (
    RAGPipeline, RAGConfig, AdvancedChunker, HybridRetriever,
    ChunkingStrategy, RetrievalStrategy, Document, Chunk,
    AdvancedReranker, ResponseGenerator
)

logger = logging.getLogger(__name__)

# ============================================================================
# RAG CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def rag_config_basic():
    """Basic RAG configuration for simple tests."""
    return RAGConfig(
        chunk_size=256,
        chunk_overlap=64,
        chunking_strategy=ChunkingStrategy.SLIDING,
        top_k=5,
        retrieval_strategy=RetrievalStrategy.VECTOR,
        similarity_threshold=0.7,
        enable_reranking=False,
        enable_caching=False,
        enable_hyde=False,
        enable_graph_rag=False,
        enable_self_correction=False
    )

@pytest.fixture  
def rag_config_advanced():
    """Advanced RAG configuration for comprehensive testing."""
    return RAGConfig(
        chunk_size=512,
        chunk_overlap=128,
        chunking_strategy=ChunkingStrategy.HIERARCHICAL,
        top_k=10,
        retrieval_strategy=RetrievalStrategy.ENSEMBLE,
        similarity_threshold=0.75,
        enable_reranking=True,
        rerank_top_k=5,
        enable_caching=True,
        cache_ttl=3600,
        enable_compression=True,
        enable_hyde=True,
        enable_graph_rag=True,
        enable_self_correction=True,
        max_context_length=4096,
        temperature=0.1
    )

@pytest.fixture
def all_chunking_strategies():
    """All available chunking strategies for comprehensive testing."""
    return list(ChunkingStrategy)

@pytest.fixture
def all_retrieval_strategies():
    """All available retrieval strategies for comprehensive testing."""
    return list(RetrievalStrategy)

# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        Document(
            id="doc1",
            content="""
            Python is a high-level programming language known for its simplicity and readability.
            It was created by Guido van Rossum and first released in 1991. Python supports
            multiple programming paradigms including procedural, object-oriented, and functional
            programming. The language emphasizes code readability with its notable use of
            significant whitespace.
            """,
            metadata={"type": "programming_language", "category": "technical", "language": "en"}
        ),
        Document(
            id="doc2", 
            content="""
            Machine learning is a subset of artificial intelligence that focuses on the
            development of algorithms that can learn and make predictions or decisions
            based on data. Common types include supervised learning, unsupervised learning, and
            reinforcement learning.
            """,
            metadata={"type": "machine_learning", "category": "ai", "language": "en"}
        ),
        Document(
            id="doc3",
            content="""
            FastAPI is a modern, fast web framework for building APIs with Python 3.6+
            based on standard Python type hints. It provides automatic API documentation,
            data validation, serialization, and authentication.
            """,
            metadata={"type": "web_framework", "category": "technical", "language": "en"}
        )
    ]

@pytest.fixture
def sample_chunks():
    """Pre-chunked sample data for testing retrieval."""
    return [
        Chunk(
            id="chunk1",
            content="Python is a high-level programming language known for its simplicity.",
            document_id="doc1",
            position=0,
            metadata={"strategy": "sliding_window"},
            embedding=np.random.rand(1536).tolist(),
            importance_score=0.9
        ),
        Chunk(
            id="chunk2", 
            content="Machine learning focuses on algorithms that learn from data.",
            document_id="doc2",
            position=0,
            metadata={"strategy": "semantic"},
            embedding=np.random.rand(1536).tolist(),
            importance_score=0.8
        )
    ]

@pytest.fixture
def test_queries():
    """Sample queries for RAG testing."""
    return [
        "What is Python programming language?",
        "How does machine learning work?", 
        "What are the benefits of FastAPI?",
        "Explain test-driven development"
    ]

# ============================================================================
# MOCK SERVICE FIXTURES  
# ============================================================================

@pytest.fixture
def mock_embeddings_service():
    """Mock embeddings service for testing."""
    mock_service = AsyncMock()
    
    async def mock_generate_embedding(text: str) -> np.ndarray:
        # Return deterministic embedding based on text hash
        import hashlib
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(text_hash % 2**32)
        embedding = np.random.rand(1536)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    mock_service.generate_embedding = mock_generate_embedding
    return mock_service

@pytest_asyncio.fixture
async def mock_db_session():
    """Mock database session for RAG testing."""
    mock_session = MagicMock()
    
    # Mock query results for vector search
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        MagicMock(
            id="chunk1",
            content="Python is a programming language", 
            document_id="doc1",
            distance=0.1
        ),
        MagicMock(
            id="chunk2",
            content="Machine learning uses data",
            document_id="doc2", 
            distance=0.2
        )
    ]
    
    mock_session.execute.return_value = mock_result
    mock_session.commit.return_value = None
    
    return mock_session

# ============================================================================
# RAG COMPONENT FIXTURES
# ============================================================================

@pytest_asyncio.fixture
async def rag_chunker(rag_config_basic):
    """RAG chunker instance for testing."""
    return AdvancedChunker(rag_config_basic)

@pytest_asyncio.fixture  
async def rag_retriever(rag_config_basic, mock_db_session):
    """RAG retriever instance with mocked dependencies."""
    retriever = HybridRetriever(rag_config_basic, mock_db_session)
    # Mock the embedding service
    with patch.object(retriever, '_get_embedding') as mock_embed:
        mock_embed.return_value = np.random.rand(1536)
        yield retriever

@pytest_asyncio.fixture
async def rag_pipeline_basic(rag_config_basic, mock_db_session):
    """Basic RAG pipeline for testing."""
    with patch('api.services.rag_pipeline.HybridRetriever._get_embedding') as mock_embed:
        mock_embed.return_value = np.random.rand(1536)
        pipeline = RAGPipeline(rag_config_basic, mock_db_session)
        yield pipeline

# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================

@pytest.fixture
def performance_metrics():
    """Performance metrics tracker for RAG testing."""
    class MetricsTracker:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
        
        def start_timer(self, operation: str):
            self.start_times[operation] = time.perf_counter()
        
        def end_timer(self, operation: str):
            if operation in self.start_times:
                elapsed = time.perf_counter() - self.start_times[operation]
                self.metrics[operation] = elapsed
                return elapsed
            return None
        
        def get_metrics(self) -> Dict[str, float]:
            return self.metrics.copy()
        
        def assert_performance(self, operation: str, max_time: float):
            assert operation in self.metrics, f"No metrics recorded for {operation}"
            actual_time = self.metrics[operation]
            assert actual_time <= max_time, f"{operation} took {actual_time:.3f}s, expected <{max_time}s"
    
    return MetricsTracker()

# ============================================================================
# QUALITY GATES FIXTURES
# ============================================================================

@pytest.fixture
def quality_gates():
    """Quality gates for RAG testing."""
    return {
        "performance_thresholds": {
            "chunking_time_ms": 100,
            "retrieval_time_ms": 200, 
            "end_to_end_time_ms": 500,
            "memory_usage_mb": 1000
        },
        "accuracy_thresholds": {
            "retrieval_precision": 0.8,
            "retrieval_recall": 0.7,
            "answer_relevance": 0.75
        },
        "coverage_requirements": {
            "code_coverage": 80,
            "critical_path_coverage": 90
        }
    }
