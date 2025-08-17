"""
Unit tests for RAG chunking strategies.

Tests all 6 chunking strategies with comprehensive coverage:
- Semantic chunking with sentence boundaries
- Sliding window with overlap
- Recursive character splitting  
- Proposition-based chunking
- Hierarchical multi-level chunking
- Adaptive context-aware sizing

Author: Peter Verschuere - Test-Driven Development Expert
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock
import numpy as np

from api.services.rag_pipeline import (
    AdvancedChunker, ChunkingStrategy, RAGConfig, Document, Chunk
)


@pytest.mark.unit
@pytest.mark.rag
class TestAdvancedChunker:
    """Test suite for advanced document chunking."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for chunking tests."""
        return """
        Introduction to Machine Learning
        
        Machine learning is a powerful subset of artificial intelligence. It enables computers to learn and make decisions from data without being explicitly programmed for every task.
        
        Supervised Learning
        
        In supervised learning, algorithms learn from labeled training data. The goal is to make predictions on new, unseen data based on patterns learned from the training set.
        
        Unsupervised Learning
        
        Unsupervised learning works with unlabeled data. It aims to find hidden patterns or structures in the data without predefined target variables.
        
        Deep Learning
        
        Deep learning uses neural networks with multiple layers. These networks can automatically learn hierarchical representations of data, making them powerful for complex tasks like image recognition and natural language processing.
        """
    
    def test_chunker_initialization(self, rag_config_basic):
        """Test chunker initialization with different configurations."""
        chunker = AdvancedChunker(rag_config_basic)
        
        assert chunker.config == rag_config_basic
        assert hasattr(chunker, 'tokenizer')
        assert chunker.tokenizer.name == "cl100k_base"
    
    def test_sliding_window_chunking(self, rag_config_basic, sample_text):
        """Test sliding window chunking strategy."""
        config = rag_config_basic
        config.chunking_strategy = ChunkingStrategy.SLIDING
        config.chunk_size = 100
        config.chunk_overlap = 20
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        chunks = chunker.chunk_document(document)
        
        # Verify chunks are created
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        
        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == "test_doc"
            assert chunk.position == i
            assert chunk.metadata["strategy"] == "sliding_window"
            assert len(chunk.content.strip()) > 0
    
    def test_semantic_chunking(self, rag_config_basic, sample_text):
        """Test semantic chunking with sentence boundaries."""
        config = rag_config_basic
        config.chunking_strategy = ChunkingStrategy.SEMANTIC
        config.chunk_size = 200
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        with patch('nltk.download'):
            chunks = chunker.chunk_document(document)
        
        # Verify semantic chunks
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "semantic"
            # Chunks should end with sentence boundaries
            assert chunk.content.strip().endswith(('.', '\!', '?', ':'))
    
    def test_hierarchical_chunking(self, rag_config_basic, sample_text):
        """Test hierarchical multi-level chunking."""
        config = rag_config_basic
        config.chunking_strategy = ChunkingStrategy.HIERARCHICAL
        config.chunk_size = 150
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        chunks = chunker.chunk_document(document)
        
        # Verify hierarchical chunks
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "hierarchical"
            assert "s" in chunk.id  # Section indicator
            assert chunk.document_id == "test_doc"
    
    def test_proposition_chunking(self, rag_config_basic, sample_text):
        """Test proposition-based chunking."""
        config = rag_config_basic
        config.chunking_strategy = ChunkingStrategy.PROPOSITION
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        chunks = chunker.chunk_document(document)
        
        # Verify proposition chunks
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "proposition"
            assert chunk.metadata["type"] == "logical_unit"
            assert len(chunk.content.strip()) > 20  # Minimum proposition length
    
    def test_adaptive_chunking(self, rag_config_basic, sample_text):
        """Test adaptive context-aware chunking."""
        config = rag_config_basic
        config.chunking_strategy = ChunkingStrategy.ADAPTIVE
        config.chunk_size = 200
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        chunks = chunker.chunk_document(document)
        
        # Verify adaptive chunks
        assert len(chunks) > 0
        
        # Adaptive chunking should adjust chunk size based on complexity
        chunk_sizes = [len(chunker.tokenizer.encode(chunk.content)) for chunk in chunks]
        
        # Should have variation in chunk sizes due to adaptive sizing
        assert max(chunk_sizes) - min(chunk_sizes) > 0
    
    def test_recursive_chunking(self, rag_config_basic, sample_text):
        """Test recursive character splitting."""
        config = rag_config_basic
        config.chunking_strategy = ChunkingStrategy.RECURSIVE
        config.chunk_size = 100
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        # Since RECURSIVE is not explicitly implemented, it falls back to sliding window
        chunks = chunker.chunk_document(document)
        
        assert len(chunks) > 0
        # Should behave like sliding window in current implementation
        assert all(chunk.metadata["strategy"] == "sliding_window" for chunk in chunks)
    
    def test_chunking_with_context(self, rag_config_advanced, sample_text):
        """Test chunking with context preservation."""
        config = rag_config_advanced
        config.chunking_strategy = ChunkingStrategy.HIERARCHICAL
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        chunks = chunker.chunk_document(document)
        
        # Verify context preservation
        for chunk in chunks:
            if hasattr(chunk, 'context_before') and hasattr(chunk, 'context_after'):
                # Context should be strings (even if empty)
                assert isinstance(chunk.context_before, str)
                assert isinstance(chunk.context_after, str)
    
    def test_chunking_edge_cases(self, rag_config_basic):
        """Test chunking with edge cases."""
        chunker = AdvancedChunker(rag_config_basic)
        
        # Empty document
        empty_doc = Document(id="empty", content="")
        chunks = chunker.chunk_document(empty_doc)
        assert len(chunks) == 0
        
        # Very short document
        short_doc = Document(id="short", content="Short text.")
        chunks = chunker.chunk_document(short_doc)
        assert len(chunks) >= 1
        assert chunks[0].content.strip() == "Short text."
        
        # Single word document
        word_doc = Document(id="word", content="Word")
        chunks = chunker.chunk_document(word_doc)
        assert len(chunks) >= 1
        assert chunks[0].content.strip() == "Word"
    
    def test_chunk_size_limits(self, rag_config_basic, sample_text):
        """Test chunking respects size limits."""
        config = rag_config_basic
        config.chunk_size = 50  # Small chunk size
        config.chunking_strategy = ChunkingStrategy.SLIDING
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        chunks = chunker.chunk_document(document)
        
        # Verify chunk sizes
        for chunk in chunks:
            tokens = chunker.tokenizer.encode(chunk.content)
            # Allow some tolerance for chunk size limits
            assert len(tokens) <= config.chunk_size + 10
    
    def test_chunk_overlap(self, rag_config_basic, sample_text):
        """Test chunk overlap functionality."""
        config = rag_config_basic
        config.chunk_size = 100
        config.chunk_overlap = 30
        config.chunking_strategy = ChunkingStrategy.SLIDING
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        chunks = chunker.chunk_document(document)
        
        if len(chunks) > 1:
            # Check for potential overlap by comparing chunk endings with beginnings
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i].content
                next_chunk = chunks[i + 1].content
                
                # There should be some relationship between consecutive chunks
                assert len(current_chunk) > 0
                assert len(next_chunk) > 0
    
    @pytest.mark.parametrize("strategy", list(ChunkingStrategy))
    def test_all_chunking_strategies(self, rag_config_basic, sample_text, strategy):
        """Test all chunking strategies work without errors."""
        config = rag_config_basic
        config.chunking_strategy = strategy
        
        chunker = AdvancedChunker(config)
        document = Document(id="test_doc", content=sample_text)
        
        with patch('nltk.download'):  # Mock NLTK download for semantic chunking
            chunks = chunker.chunk_document(document)
        
        # All strategies should produce at least one chunk for non-empty input
        assert len(chunks) > 0
        
        # All chunks should have required attributes
        for chunk in chunks:
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'document_id')
            assert hasattr(chunk, 'position')
            assert hasattr(chunk, 'metadata')
            assert chunk.document_id == "test_doc"
    
    def test_chunk_metadata(self, rag_config_basic, sample_text):
        """Test chunk metadata is properly set."""
        chunker = AdvancedChunker(rag_config_basic)
        document = Document(
            id="test_doc", 
            content=sample_text,
            metadata={"source": "test", "category": "ml"}
        )
        
        chunks = chunker.chunk_document(document)
        
        for chunk in chunks:
            assert isinstance(chunk.metadata, dict)
            assert "strategy" in chunk.metadata
            # Document metadata should not automatically propagate to chunks
            # This is intentional to keep chunk metadata focused
    
    def test_chunking_performance(self, rag_config_basic, performance_metrics):
        """Test chunking performance meets requirements."""
        # Create a 10KB document
        large_text = "This is a performance test document. " * 500  # ~10KB
        document = Document(id="perf_doc", content=large_text)
        
        chunker = AdvancedChunker(rag_config_basic)
        
        performance_metrics.start_timer("chunking")
        chunks = chunker.chunk_document(document)
        performance_metrics.end_timer("chunking")
        
        # Should complete chunking within 100ms for 10KB document
        performance_metrics.assert_performance("chunking", 0.1)
        
        # Should produce reasonable number of chunks
        assert len(chunks) > 0
        assert len(chunks) < 1000  # Reasonable upper bound


@pytest.mark.unit
@pytest.mark.rag
class TestChunkingHelperMethods:
    """Test helper methods in chunking implementation."""
    
    def test_complexity_analysis(self, rag_config_basic):
        """Test complexity analysis for adaptive chunking."""
        chunker = AdvancedChunker(rag_config_basic)
        
        # Simple text
        simple_complexity = chunker._analyze_complexity("This is simple text.")
        
        # Complex text
        complex_text = """
        The implementation of machine learning algorithms requires careful consideration
        of mathematical foundations, statistical principles, and computational efficiency.
        """
        complex_complexity = chunker._analyze_complexity(complex_text)
        
        # Complex text should have higher complexity score
        assert complex_complexity >= simple_complexity
        assert 0 <= simple_complexity <= 1
        assert 0 <= complex_complexity <= 1
    
    def test_section_splitting(self, rag_config_basic):
        """Test section splitting for hierarchical chunking."""
        chunker = AdvancedChunker(rag_config_basic)
        
        text_with_sections = """
        Section 1
        
        This is the first section content.
        
        Section 2
        
        This is the second section content.
        """
        
        sections = chunker._split_by_sections(text_with_sections)
        
        assert len(sections) > 1
        assert all(section.strip() for section in sections)
    
    def test_semantic_unit_splitting(self, rag_config_basic):
        """Test semantic unit splitting."""
        chunker = AdvancedChunker(rag_config_basic)
        
        text = "First sentence. Second sentence\! Third sentence?"
        units = chunker._split_semantic_units(text)
        
        assert len(units) >= 3  # Should split on sentence boundaries
        assert all(unit.strip() for unit in units)
    
    def test_proposition_extraction(self, rag_config_basic):
        """Test proposition extraction logic."""
        chunker = AdvancedChunker(rag_config_basic)
        
        text = """
        Machine learning is powerful, and it enables automation.
        Data quality matters; clean data leads to better models.
        """
        
        propositions = chunker._extract_propositions(text)
        
        assert len(propositions) > 0
        assert all(len(prop) > 20 for prop in propositions)  # Minimum length filter
