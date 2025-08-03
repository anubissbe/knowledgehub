"""
Enhanced Memory Data Models for AI-Powered Memory System.

This module defines the data models for the intelligent memory system that stores,
retrieves, and manages contextual memories with embeddings, clustering, and decay.
"""

from sqlalchemy import Column, String, Text, JSON, DateTime, Float, Integer, Boolean, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import uuid

from .base import Base


class MemoryType(str, Enum):
    """Types of memories in the system."""
    CONVERSATION = "conversation"
    GENERAL = "general"  # Add support for existing general memories
    DECISION = "decision"
    ERROR = "error"
    LEARNING = "learning"
    CONTEXT = "context"
    PATTERN = "pattern"
    PREFERENCE = "preference"
    FACT = "fact"
    PROCEDURE = "procedure"
    ASSOCIATION = "association"


class MemoryImportance(str, Enum):
    """Importance levels for memory prioritization."""
    CRITICAL = "critical"      # Never decay, always keep
    HIGH = "high"             # Slow decay
    MEDIUM = "medium"         # Normal decay
    LOW = "low"              # Fast decay
    EPHEMERAL = "ephemeral"   # Very fast decay


class MemoryCluster(Base):
    """Memory clusters for grouping related memories."""
    __tablename__ = "memory_clusters"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    cluster_type = Column(String(50), nullable=False)
    centroid_embedding = Column(ARRAY(Float), nullable=True)
    topic_keywords = Column(JSON, default=lambda: [])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Stats
    memory_count = Column(Integer, default=0)
    avg_relevance = Column(Float, default=0.0)
    last_accessed = Column(DateTime(timezone=True))
    
    # Relationships
    memories = relationship("Memory", back_populates="cluster")
    
    __table_args__ = (
        Index('idx_memory_clusters_type', 'cluster_type'),
        Index('idx_memory_clusters_updated', 'updated_at'),
    )


class Memory(Base):
    """Enhanced AI Memory model with embeddings, clustering, and decay."""
    __tablename__ = "ai_memories"
    
    # Core Identity
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    
    # Memory Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)  # SHA-256 for deduplication
    memory_type = Column(String(50), nullable=False, index=True)
    
    # Context and Metadata
    context = Column(JSON, default=lambda: {})
    meta_data = Column('metadata', JSON, default=lambda: {})
    tags = Column(ARRAY(String), default=lambda: [])
    
    # Embeddings and Clustering
    embeddings = Column(ARRAY(Float), nullable=True)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey('memory_clusters.id'), nullable=True)
    cluster_distance = Column(Float, nullable=True)
    
    # Relevance and Scoring
    relevance_score = Column(Float, default=1.0)
    importance = Column(String(20), default=MemoryImportance.MEDIUM.value)
    confidence_score = Column(Float, default=1.0)
    
    # Access and Usage Patterns
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    last_context_match = Column(DateTime(timezone=True))
    
    # Temporal Information
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Decay and Lifecycle
    decay_factor = Column(Float, default=1.0)
    is_archived = Column(Boolean, default=False)
    archive_reason = Column(String(100))
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships and Associations
    parent_memory_id = Column(UUID(as_uuid=True), ForeignKey('ai_memories.id'), nullable=True)
    related_memories = Column(ARRAY(UUID), default=lambda: [])
    
    # Knowledge Graph Integration
    knowledge_entities = Column(JSON, default=lambda: [])  # Extracted entities
    knowledge_relations = Column(JSON, default=lambda: [])  # Relations to other memories
    
    # Performance Tracking
    retrieval_latency = Column(Float)  # Last retrieval time in ms
    embedding_version = Column(String(20), default="v1.0")
    
    # Relationships
    cluster = relationship("MemoryCluster", back_populates="memories")
    child_memories = relationship("Memory", backref="parent_memory", remote_side=[id])
    
    __table_args__ = (
        # Core indexes for retrieval
        Index('idx_memories_user_session', 'user_id', 'session_id'),
        Index('idx_memories_type_relevance', 'memory_type', 'relevance_score'),
        Index('idx_memories_content_hash', 'content_hash'),
        
        # Temporal indexes
        Index('idx_memories_created_at', 'created_at'),
        Index('idx_memories_last_accessed', 'last_accessed'),
        Index('idx_memories_expires_at', 'expires_at'),
        
        # Performance indexes
        Index('idx_memories_importance_decay', 'importance', 'decay_factor'),
        Index('idx_memories_cluster_distance', 'cluster_id', 'cluster_distance'),
        Index('idx_memories_archived', 'is_archived'),
        
        # Compound indexes for common queries
        Index('idx_memories_user_type_relevance', 'user_id', 'memory_type', 'relevance_score'),
        Index('idx_memories_session_active', 'session_id', 'is_archived', 'expires_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "content": self.content,
            "content_hash": self.content_hash,
            "memory_type": self.memory_type,
            "context": self.context,
            "metadata": self.meta_data,
            "tags": self.tags,
            "relevance_score": self.relevance_score,
            "importance": self.importance,
            "confidence_score": self.confidence_score,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "cluster_id": str(self.cluster_id) if self.cluster_id else None,
            "decay_factor": self.decay_factor,
            "is_archived": self.is_archived,
        }

    def increment_access(self) -> None:
        """Increment access count and update accessed timestamp"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


class MemoryAssociation(Base):
    """Associations between memories for graph-like relationships."""
    __tablename__ = "memory_associations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_memory_id = Column(UUID(as_uuid=True), ForeignKey('ai_memories.id'), nullable=False)
    target_memory_id = Column(UUID(as_uuid=True), ForeignKey('ai_memories.id'), nullable=False)
    
    association_type = Column(String(50), nullable=False)  # 'similar', 'causal', 'temporal', etc.
    strength = Column(Float, default=1.0)  # Association strength 0-1
    confidence = Column(Float, default=1.0)  # Confidence in association
    
    context = Column(JSON, default=lambda: {})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_reinforced = Column(DateTime(timezone=True), server_default=func.now())
    reinforcement_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_associations_source', 'source_memory_id'),
        Index('idx_associations_target', 'target_memory_id'),
        Index('idx_associations_type_strength', 'association_type', 'strength'),
    )


class MemoryAccess(Base):
    """Track memory access patterns for analytics and optimization."""
    __tablename__ = "memory_access_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(UUID(as_uuid=True), ForeignKey('ai_memories.id'), nullable=False)
    user_id = Column(String(255), nullable=False)
    session_id = Column(String(255), nullable=False)
    
    access_type = Column(String(50), nullable=False)  # 'retrieval', 'update', 'association'
    context_similarity = Column(Float)  # How well context matched
    retrieval_method = Column(String(50))  # 'semantic', 'keyword', 'temporal'
    
    query_context = Column(JSON, default=lambda: {})
    response_time_ms = Column(Float)
    result_rank = Column(Integer)  # Position in search results
    
    accessed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_access_logs_memory', 'memory_id'),
        Index('idx_access_logs_user_session', 'user_id', 'session_id'),
        Index('idx_access_logs_accessed_at', 'accessed_at'),
    )


# Keep existing MemoryItem for backward compatibility
class MemoryItem(Base):
    """Model for storing memory items from MCP clients"""
    
    __tablename__ = "memory_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), unique=True)
    tags = Column(ARRAY(Text), default=[])
    meta_data = Column('metadata', JSON, default={})
    embedding_id = Column(String(255))
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    accessed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<MemoryItem(id={self.id}, tags={self.tags}, access_count={self.access_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "content": self.content,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "metadata": self.meta_data,
            "embedding_id": self.embedding_id,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
        }
    
    def increment_access(self) -> None:
        """Increment access count and update accessed timestamp"""
        self.access_count += 1
        self.accessed_at = datetime.now(timezone.utc)


# Pydantic Models for API

class MemoryCreate(BaseModel):
    """Schema for creating new memories."""
    user_id: str = Field(..., min_length=1, max_length=255)
    session_id: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    memory_type: MemoryType
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    importance: MemoryImportance = MemoryImportance.MEDIUM
    expires_at: Optional[datetime] = None
    
    @field_validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @field_validator('tags')
    def validate_tags(cls, v):
        return [tag.strip().lower() for tag in v if tag.strip()]


class MemoryUpdate(BaseModel):
    """Schema for updating existing memories."""
    content: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    importance: Optional[MemoryImportance] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    expires_at: Optional[datetime] = None


class MemoryRetrievalQuery(BaseModel):
    """Schema for memory retrieval queries."""
    query: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    memory_types: Optional[List[MemoryType]] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=100)
    min_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    include_archived: bool = False
    time_window_hours: Optional[int] = Field(None, ge=1)


class MemoryResponse(BaseModel):
    """Schema for memory responses."""
    id: str
    user_id: str
    session_id: str
    content: str
    memory_type: MemoryType
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    tags: List[str]
    relevance_score: float
    importance: MemoryImportance
    confidence_score: float
    access_count: int
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime
    cluster_id: Optional[str] = None
    
    model_config = ConfigDict(from_attributes = True)


class MemoryClusterResponse(BaseModel):
    """Schema for memory cluster responses."""
    id: str
    name: str
    description: Optional[str]
    cluster_type: str
    topic_keywords: List[str]
    memory_count: int
    avg_relevance: float
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes = True)


class MemoryAnalytics(BaseModel):
    """Schema for memory analytics data."""
    total_memories: int
    memories_by_type: Dict[str, int]
    memories_by_importance: Dict[str, int]
    avg_relevance_score: float
    most_accessed_memories: List[MemoryResponse]
    cluster_distribution: Dict[str, int]
    memory_growth_trend: List[Dict[str, Any]]
    decay_statistics: Dict[str, Any]