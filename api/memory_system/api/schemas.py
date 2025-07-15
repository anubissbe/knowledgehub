"""Pydantic schemas for memory system API"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, validator, ConfigDict


class MemoryTypeEnum(str, Enum):
    """Memory type enumeration"""
    fact = "fact"
    preference = "preference"
    code = "code"
    decision = "decision"
    error = "error"
    pattern = "pattern"
    entity = "entity"


# Session Schemas
class SessionBase(BaseModel):
    """Base session schema"""
    user_id: str = Field(..., description="Unique user identifier")
    project_id: Optional[UUID] = Field(None, description="Optional project association")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Session tags")


class SessionCreate(SessionBase):
    """Schema for creating a new session"""
    parent_session_id: Optional[UUID] = Field(None, description="Parent session for continuity")


class SessionUpdate(BaseModel):
    """Schema for updating session"""
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    ended_at: Optional[datetime] = None


class SessionResponse(SessionBase):
    """Schema for session response"""
    id: UUID
    started_at: datetime
    ended_at: Optional[datetime]
    parent_session_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    duration: Optional[float]
    is_active: bool
    memory_count: int
    
    model_config = ConfigDict(from_attributes=True)


class SessionSummary(BaseModel):
    """Lightweight session summary"""
    id: UUID
    user_id: str
    started_at: datetime
    duration: Optional[float]
    memory_count: int
    is_active: bool
    tags: List[str]


# Memory Schemas
class MemoryBase(BaseModel):
    """Base memory schema"""
    content: str = Field(..., description="Memory content")
    summary: Optional[str] = Field(None, description="Condensed summary")
    memory_type: MemoryTypeEnum = Field(..., description="Type of memory")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Extraction confidence")
    entities: Optional[List[str]] = Field(default_factory=list, description="Related entities")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class MemoryCreate(MemoryBase):
    """Schema for creating a new memory"""
    session_id: UUID = Field(..., description="Session this memory belongs to")
    related_memories: Optional[List[UUID]] = Field(default_factory=list, description="Related memory IDs")


class MemoryUpdate(BaseModel):
    """Schema for updating memory"""
    summary: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    entities: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(MemoryBase):
    """Schema for memory response"""
    id: UUID
    session_id: UUID
    related_memories: List[UUID]
    has_embedding: bool
    access_count: int
    last_accessed: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    age_days: float
    relevance_score: float
    is_recent: bool
    is_high_importance: bool
    
    model_config = ConfigDict(from_attributes=True)


# Context Schemas
class ContextRequest(BaseModel):
    """Schema for context loading request"""
    session_id: Optional[UUID] = Field(None, description="Current session ID")
    user_id: str = Field(..., description="User ID for context")
    project_id: Optional[UUID] = Field(None, description="Project context")
    max_tokens: int = Field(2000, gt=0, le=8000, description="Maximum tokens for context")
    include_types: Optional[List[MemoryTypeEnum]] = Field(None, description="Memory types to include")
    time_window_days: Optional[int] = Field(30, gt=0, description="Days of history to consider")


class ContextResponse(BaseModel):
    """Schema for context response"""
    session_id: Optional[UUID]
    recent_messages: List[Dict[str, Any]]
    relevant_memories: List[MemoryResponse]
    project_facts: List[str]
    user_preferences: List[str]
    total_tokens: int
    context_summary: str


# Search Schemas
class MemorySearchRequest(BaseModel):
    """Schema for memory search request"""
    query: Optional[str] = Field(None, description="Search query")
    user_id: Optional[str] = Field(None, description="Filter by user")
    project_id: Optional[UUID] = Field(None, description="Filter by project")
    memory_types: Optional[List[MemoryTypeEnum]] = Field(None, description="Filter by types")
    min_importance: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum importance")
    limit: int = Field(10, gt=0, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Pagination offset")
    use_vector_search: bool = Field(True, description="Enable vector similarity search")


class MemorySearchResponse(BaseModel):
    """Schema for memory search response"""
    results: List[MemoryResponse]
    total: int
    query: Optional[str]
    limit: int
    offset: int
    search_time_ms: Optional[int] = Field(None, description="Search execution time in milliseconds")


# Batch Operations
class MemoryBatchCreate(BaseModel):
    """Schema for batch memory creation"""
    session_id: UUID
    memories: List[MemoryBase]


class MemoryBatchResponse(BaseModel):
    """Schema for batch operation response"""
    created: int
    failed: int
    memories: List[MemoryResponse]
    errors: Optional[List[Dict[str, Any]]] = None


# Analytics Schemas
class MemoryStats(BaseModel):
    """Memory statistics"""
    total_memories: int
    by_type: Dict[str, int]
    average_importance: float
    high_importance_count: int
    recent_count: int
    total_sessions: int
    active_sessions: int


class UserMemoryProfile(BaseModel):
    """User memory profile"""
    user_id: str
    total_memories: int
    total_sessions: int
    favorite_topics: List[str]
    common_entities: List[str]
    memory_types_distribution: Dict[str, float]
    average_session_duration: float
    last_active: datetime