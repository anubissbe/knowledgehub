"""Context injection schemas for Claude-Code integration"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class ContextRelevanceEnum(str, Enum):
    """Context relevance levels"""
    critical = "critical"    # Must include in context
    high = "high"           # Very relevant
    medium = "medium"       # Somewhat relevant
    low = "low"            # Background context
    

class ContextTypeEnum(str, Enum):
    """Types of context to retrieve"""
    recent = "recent"              # Recent memories from session
    similar = "similar"            # Semantically similar memories
    entities = "entities"          # Entity-based relevant memories
    decisions = "decisions"        # Important decisions made
    errors = "errors"             # Errors and solutions
    patterns = "patterns"         # Recognized patterns
    preferences = "preferences"   # User preferences


class ContextRequest(BaseModel):
    """Request for context retrieval"""
    user_id: str = Field(..., description="User requesting context")
    session_id: Optional[UUID] = Field(None, description="Current session ID")
    query: Optional[str] = Field(None, description="Query for semantic search")
    
    # Context filtering
    context_types: List[ContextTypeEnum] = Field(
        default=[ContextTypeEnum.recent, ContextTypeEnum.similar],
        description="Types of context to retrieve"
    )
    max_memories: int = Field(20, ge=1, le=100, description="Maximum memories to retrieve")
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens for context")
    min_relevance: float = Field(0.3, ge=0.0, le=1.0, description="Minimum relevance score")
    
    # Time filtering
    time_window_hours: Optional[int] = Field(
        None, 
        description="Only include memories from last N hours"
    )
    
    # Memory type filtering  
    memory_types: Optional[List[str]] = Field(None, description="Filter by memory types")
    
    # Project/domain filtering
    project_id: Optional[UUID] = Field(None, description="Filter by project")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    
    model_config = ConfigDict(from_attributes=True)


class ContextMemory(BaseModel):
    """Memory item with context relevance"""
    id: UUID
    content: str
    summary: Optional[str]
    memory_type: str
    importance: float
    confidence: float
    entities: List[str]
    
    # Context-specific fields
    relevance_score: float = Field(..., description="Relevance to current context")
    relevance_reason: str = Field(..., description="Why this memory is relevant")
    context_type: ContextTypeEnum = Field(..., description="How this context was selected")
    
    # Metadata
    created_at: datetime
    session_id: UUID
    age_days: float
    
    model_config = ConfigDict(from_attributes=True)


class ContextSection(BaseModel):
    """A section of organized context"""
    title: str = Field(..., description="Section title")
    context_type: ContextTypeEnum = Field(..., description="Type of context")
    memories: List[ContextMemory] = Field(..., description="Memories in this section")
    token_count: int = Field(..., description="Estimated token count for this section")
    relevance_score: float = Field(..., description="Average relevance of section")


class ContextResponse(BaseModel):
    """Formatted context response for Claude-Code"""
    user_id: str
    session_id: Optional[UUID]
    query: Optional[str]
    
    # Context sections
    sections: List[ContextSection] = Field(..., description="Organized context sections")
    
    # Metadata
    total_memories: int = Field(..., description="Total memories retrieved")
    total_tokens: int = Field(..., description="Total estimated tokens")
    max_relevance: float = Field(..., description="Highest relevance score")
    retrieval_time_ms: float = Field(..., description="Time taken to retrieve context")
    
    # Formatted context
    formatted_context: str = Field(..., description="LLM-ready formatted context")
    context_summary: str = Field(..., description="Brief summary of context")
    
    model_config = ConfigDict(from_attributes=True)


class ContextStats(BaseModel):
    """Statistics about context retrieval"""
    total_memories_available: int
    memories_retrieved: int
    context_types_used: List[ContextTypeEnum]
    average_relevance: float
    token_efficiency: float  # memories per token
    
    # Performance metrics
    retrieval_time_ms: float
    formatting_time_ms: float
    total_time_ms: float


class ContextUpdateRequest(BaseModel):
    """Request to update context based on usage"""
    memory_ids: List[UUID] = Field(..., description="Memory IDs that were used")
    effectiveness_score: float = Field(
        0.5, ge=0.0, le=1.0, 
        description="How effective was this context"
    )
    feedback: Optional[str] = Field(None, description="Optional feedback")
    
    model_config = ConfigDict(from_attributes=True)