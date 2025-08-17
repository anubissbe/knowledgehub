"""
Hybrid RAG System Data Models.

This module defines the data models for the hybrid retrieval-augmented generation
system that combines dense, sparse, and graph-based retrieval methods.
"""

from sqlalchemy import Column, String, Text, JSON, DateTime, Float, Integer, Boolean, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timezone
from enum import Enum
import uuid
import hashlib

from .base import Base


class RetrievalMode(str, Enum):
    """Retrieval modes for hybrid search."""
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    GRAPH_ONLY = "graph_only"
    DENSE_SPARSE = "dense_sparse"
    DENSE_GRAPH = "dense_graph"
    SPARSE_GRAPH = "sparse_graph"
    HYBRID_ALL = "hybrid_all"


class IngestionMethod(str, Enum):
    """Methods for document ingestion."""
    MANUAL = "manual"
    API = "api"
    FIRECRAWL = "firecrawl"
    BULK_UPLOAD = "bulk_upload"
    SCHEDULED = "scheduled"


class IngestionStatus(str, Enum):
    """Status of document ingestion process."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RAGConfiguration(Base):
    """RAG system configurations for different use cases."""
    __tablename__ = "rag_configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    
    # Retrieval Weights
    dense_weight = Column(Float, default=0.4)
    sparse_weight = Column(Float, default=0.3)
    graph_weight = Column(Float, default=0.2)
    rerank_weight = Column(Float, default=0.1)
    
    # Model Configuration
    dense_model = Column(String(200), default='BAAI/bge-base-en-v1.5')
    rerank_model = Column(String(200), default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Performance Tuning
    dense_top_k = Column(Integer, default=50)
    sparse_top_k = Column(Integer, default=50)
    graph_top_k = Column(Integer, default=30)
    rerank_top_k = Column(Integer, default=20)
    
    # Quality Thresholds
    min_dense_score = Column(Float, default=0.5)
    min_sparse_score = Column(Float, default=0.1)
    min_graph_score = Column(Float, default=0.3)
    min_final_score = Column(Float, default=0.2)
    
    # Caching and Performance
    enable_caching = Column(Boolean, default=True)
    cache_ttl_seconds = Column(Integer, default=300)
    
    # Metadata
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    query_logs = relationship("RAGQueryLog", back_populates="config")
    
    __table_args__ = (
        Index('idx_rag_configurations_default', 'is_default'),
        Index('idx_rag_configurations_name', 'name'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "graph_weight": self.graph_weight,
            "rerank_weight": self.rerank_weight,
            "dense_model": self.dense_model,
            "rerank_model": self.rerank_model,
            "dense_top_k": self.dense_top_k,
            "sparse_top_k": self.sparse_top_k,
            "graph_top_k": self.graph_top_k,
            "rerank_top_k": self.rerank_top_k,
            "min_dense_score": self.min_dense_score,
            "min_sparse_score": self.min_sparse_score,
            "min_graph_score": self.min_graph_score,
            "min_final_score": self.min_final_score,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class RAGQueryLog(Base):
    """Logs of RAG queries for performance analysis."""
    __tablename__ = "rag_query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False)
    session_id = Column(String(255))
    
    # Query Details
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False)
    retrieval_mode = Column(String(50), nullable=False)
    config_id = Column(UUID(as_uuid=True), ForeignKey('rag_configurations.id'))
    
    # Results Metadata
    total_results = Column(Integer, default=0)
    dense_results_count = Column(Integer, default=0)
    sparse_results_count = Column(Integer, default=0)
    graph_results_count = Column(Integer, default=0)
    final_results_count = Column(Integer, default=0)
    
    # Performance Metrics
    total_time_ms = Column(Integer, nullable=False)
    dense_retrieval_time_ms = Column(Integer, default=0)
    sparse_retrieval_time_ms = Column(Integer, default=0)
    graph_retrieval_time_ms = Column(Integer, default=0)
    rerank_time_ms = Column(Integer, default=0)
    
    # Quality Metrics
    avg_relevance_score = Column(Float)
    cache_hit = Column(Boolean, default=False)
    user_feedback_score = Column(Integer)  # 1-5 rating if provided
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    config = relationship("RAGConfiguration", back_populates="query_logs")
    
    __table_args__ = (
        Index('idx_rag_query_logs_user_id', 'user_id'),
        Index('idx_rag_query_logs_query_hash', 'query_hash'),
        Index('idx_rag_query_logs_created_at', 'created_at'),
        Index('idx_rag_query_logs_performance', 'total_time_ms', 'total_results'),
        Index('idx_rag_query_logs_session', 'session_id'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query_text": self.query_text,
            "query_hash": self.query_hash,
            "retrieval_mode": self.retrieval_mode,
            "config_id": str(self.config_id) if self.config_id else None,
            "total_results": self.total_results,
            "dense_results_count": self.dense_results_count,
            "sparse_results_count": self.sparse_results_count,
            "graph_results_count": self.graph_results_count,
            "final_results_count": self.final_results_count,
            "total_time_ms": self.total_time_ms,
            "dense_retrieval_time_ms": self.dense_retrieval_time_ms,
            "sparse_retrieval_time_ms": self.sparse_retrieval_time_ms,
            "graph_retrieval_time_ms": self.graph_retrieval_time_ms,
            "rerank_time_ms": self.rerank_time_ms,
            "avg_relevance_score": self.avg_relevance_score,
            "cache_hit": self.cache_hit,
            "user_feedback_score": self.user_feedback_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentIngestionLog(Base):
    """Tracks document ingestion process and results."""
    __tablename__ = "document_ingestion_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    source_url = Column(Text)
    source_type = Column(String(50))
    
    # Ingestion Process
    ingestion_method = Column(String(50), nullable=False)
    status = Column(String(50), default=IngestionStatus.PENDING.value)
    
    # Processing Results
    total_chunks = Column(Integer, default=0)
    successful_chunks = Column(Integer, default=0)
    failed_chunks = Column(Integer, default=0)
    
    # Content Analysis
    entities_extracted = Column(JSON, default=lambda: [])
    content_length = Column(Integer)
    language = Column(String(10))
    content_type = Column(String(100))
    
    # Performance
    processing_time_ms = Column(Integer)
    
    # Error Handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    __table_args__ = (
        Index('idx_document_ingestion_document_id', 'document_id'),
        Index('idx_document_ingestion_status', 'status'),
        Index('idx_document_ingestion_method', 'ingestion_method'),
        Index('idx_document_ingestion_started_at', 'started_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "source_url": self.source_url,
            "source_type": self.source_type,
            "ingestion_method": self.ingestion_method,
            "status": self.status,
            "total_chunks": self.total_chunks,
            "successful_chunks": self.successful_chunks,
            "failed_chunks": self.failed_chunks,
            "entities_extracted": self.entities_extracted,
            "content_length": self.content_length,
            "language": self.language,
            "content_type": self.content_type,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class EnhancedChunk(Base):
    """Enhanced chunks with multiple embeddings and metadata."""
    __tablename__ = "enhanced_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    content_length = Column(Integer, nullable=False)
    
    # Embeddings (multiple models support)
    embedding_primary = Column(ARRAY(Float))  # vector(768)
    embedding_secondary = Column(ARRAY(Float))  # vector(384) for smaller/faster models
    embedding_model = Column(String(200), default='BAAI/bge-base-en-v1.5')
    
    # Semantic Metadata
    entities = Column(JSON, default=lambda: [])
    keywords = Column(JSON, default=lambda: [])
    topics = Column(JSON, default=lambda: [])
    sentiment_score = Column(Float)
    complexity_score = Column(Float)
    
    # Graph Relationships
    related_chunks = Column(JSON, default=lambda: [])
    graph_node_id = Column(String(100))
    
    # Quality Metrics
    coherence_score = Column(Float)
    informativeness_score = Column(Float)
    
    # Chunk Hierarchy
    parent_chunk_id = Column(UUID(as_uuid=True), ForeignKey('enhanced_chunks.id'))
    child_chunks = Column(JSON, default=lambda: [])
    hierarchy_level = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    parent_chunk = relationship("EnhancedChunk", remote_side=[id])
    
    __table_args__ = (
        Index('idx_enhanced_chunks_document_id', 'document_id'),
        Index('idx_enhanced_chunks_content_hash', 'content_hash'),
        Index('idx_enhanced_chunks_parent', 'parent_chunk_id'),
        Index('idx_enhanced_chunks_hierarchy', 'hierarchy_level'),
        Index('idx_enhanced_chunks_chunk_index', 'chunk_index'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_index": self.chunk_index,
            "content": self.content,
            "content_hash": self.content_hash,
            "content_length": self.content_length,
            "embedding_model": self.embedding_model,
            "entities": self.entities,
            "keywords": self.keywords,
            "topics": self.topics,
            "sentiment_score": self.sentiment_score,
            "complexity_score": self.complexity_score,
            "related_chunks": self.related_chunks,
            "graph_node_id": self.graph_node_id,
            "coherence_score": self.coherence_score,
            "informativeness_score": self.informativeness_score,
            "parent_chunk_id": str(self.parent_chunk_id) if self.parent_chunk_id else None,
            "child_chunks": self.child_chunks,
            "hierarchy_level": self.hierarchy_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SearchResultCache(Base):
    """Cache for search results to improve performance."""
    __tablename__ = "search_result_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_hash = Column(String(64), nullable=False)
    config_hash = Column(String(64), nullable=False)
    user_context_hash = Column(String(64))
    
    # Cached Results
    results_data = Column(JSON, nullable=False)
    metadata = Column(JSON, default=lambda: {})
    
    # Cache Management
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_search_result_cache_query_hash', 'query_hash'),
        Index('idx_search_result_cache_expires_at', 'expires_at'),
        Index('idx_search_result_cache_hit_count', 'hit_count'),
        Index('idx_search_result_cache_composite', 'query_hash', 'config_hash', 'user_context_hash'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "query_hash": self.query_hash,
            "config_hash": self.config_hash,
            "user_context_hash": self.user_context_hash,
            "results_data": self.results_data,
            "metadata": self.metadata,
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def increment_hit_count(self):
        """Increment hit count and update last accessed time."""
        self.hit_count += 1
        self.last_accessed = datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now(timezone.utc) > self.expires_at.replace(tzinfo=timezone.utc)


class FirecrawlJob(Base):
    """Firecrawl ingestion job tracking."""
    __tablename__ = "firecrawl_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(255), nullable=False, unique=True)
    url = Column(Text, nullable=False)
    crawl_type = Column(String(50), nullable=False)  # 'single', 'map', 'crawl'
    
    # Job Configuration
    config = Column(JSON, nullable=False)
    
    # Status Tracking
    status = Column(String(50), default=IngestionStatus.PENDING.value)
    progress = Column(Float, default=0.0)
    
    # Results
    pages_found = Column(Integer, default=0)
    pages_processed = Column(Integer, default=0)
    documents_created = Column(Integer, default=0)
    
    # Error Handling
    error_message = Column(Text)
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Integration
    created_by_user_id = Column(String(255))
    
    __table_args__ = (
        Index('idx_firecrawl_jobs_job_id', 'job_id'),
        Index('idx_firecrawl_jobs_status', 'status'),
        Index('idx_firecrawl_jobs_started_at', 'started_at'),
        Index('idx_firecrawl_jobs_user', 'created_by_user_id'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "job_id": self.job_id,
            "url": self.url,
            "crawl_type": self.crawl_type,
            "config": self.config,
            "status": self.status,
            "progress": self.progress,
            "pages_found": self.pages_found,
            "pages_processed": self.pages_processed,
            "documents_created": self.documents_created,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by_user_id": self.created_by_user_id,
        }


# Pydantic Models for API

class RAGConfigurationCreate(BaseModel):
    """Schema for creating RAG configurations."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    dense_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    graph_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    rerank_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    dense_model: str = Field(default='BAAI/bge-base-en-v1.5')
    rerank_model: str = Field(default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    dense_top_k: int = Field(default=50, ge=1, le=200)
    sparse_top_k: int = Field(default=50, ge=1, le=200)
    graph_top_k: int = Field(default=30, ge=1, le=100)
    rerank_top_k: int = Field(default=20, ge=1, le=100)
    min_dense_score: float = Field(default=0.5, ge=0.0, le=1.0)
    min_sparse_score: float = Field(default=0.1, ge=0.0, le=1.0)
    min_graph_score: float = Field(default=0.3, ge=0.0, le=1.0)
    min_final_score: float = Field(default=0.2, ge=0.0, le=1.0)
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip().lower()
    
    @field_validator('dense_weight', 'sparse_weight', 'graph_weight', 'rerank_weight')
    def validate_weights_sum(cls, v, info):
        # This validator would need access to all weights to validate sum
        # For now, just ensure individual weights are valid
        return v


class RAGQueryRequest(BaseModel):
    """Schema for RAG query requests."""
    query: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID_ALL)
    config_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    include_reasoning: bool = Field(default=False)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class RAGQueryResponse(BaseModel):
    """Schema for RAG query responses."""
    response: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    reasoning_steps: Optional[List[str]] = None


class DocumentIngestionRequest(BaseModel):
    """Schema for document ingestion requests."""
    content: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    ingestion_method: IngestionMethod = Field(default=IngestionMethod.API)
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    
    @field_validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class FirecrawlJobRequest(BaseModel):
    """Schema for Firecrawl job requests."""
    url: str = Field(..., description="URL to crawl")
    crawl_type: str = Field(default="single", description="Type of crawl: single, map, or crawl")
    config: Dict[str, Any] = Field(default_factory=dict, description="Firecrawl configuration")
    
    @field_validator('url')
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        # Basic URL validation
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v.strip()


class SearchResult(BaseModel):
    """Schema for individual search results."""
    content: str
    doc_id: str
    chunk_id: Optional[str] = None
    dense_score: float = 0.0
    sparse_score: float = 0.0
    graph_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    entities: List[str] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_path: List[str] = Field(default_factory=list)


class RAGPerformanceMetrics(BaseModel):
    """Schema for RAG performance metrics."""
    total_queries: int
    avg_response_time: float
    cache_hit_rate: float
    avg_results_per_query: float
    queries_by_mode: Dict[str, int]
    performance_by_mode: Dict[str, Dict[str, float]]
    top_queries: List[Dict[str, Any]]


class RAGAnalytics(BaseModel):
    """Schema for RAG analytics data."""
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_processing_time: float
    ingestion_stats: Dict[str, int]
    performance_trends: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]


def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash for content deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def generate_query_hash(query: str, config: Dict[str, Any] = None) -> str:
    """Generate hash for query caching."""
    query_data = {
        "query": query.strip().lower(),
        "config": config or {}
    }
    query_str = str(sorted(query_data.items()))
    return hashlib.sha256(query_str.encode('utf-8')).hexdigest()[:16]