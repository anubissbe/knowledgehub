"""
Session Management Models for AI-Powered Session Continuity.

This module defines the data models for intelligent session management with
state preservation, context windows, and cross-session continuity.
"""

from sqlalchemy import Column, String, Text, JSON, DateTime, Float, Integer, Boolean, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid

from .base import Base


class SessionState(str, Enum):
    """Session state enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    TRANSFERRED = "transferred"
    ERROR = "error"


class SessionType(str, Enum):
    """Types of sessions."""
    INTERACTIVE = "interactive"      # Regular interactive session
    BACKGROUND = "background"        # Background processing
    BATCH = "batch"                 # Batch operations
    WORKFLOW = "workflow"           # Workflow execution
    LEARNING = "learning"           # Learning session
    RECOVERY = "recovery"           # Session recovery
    DEBUGGING = "debugging"         # Debug session


class HandoffReason(str, Enum):
    """Reasons for session handoff."""
    USER_REQUEST = "user_request"
    TIMEOUT = "timeout"
    ERROR_RECOVERY = "error_recovery"
    CONTEXT_LIMIT = "context_limit"
    PERFORMANCE = "performance"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class Session(Base):
    """Enhanced session model with state management and context preservation."""
    __tablename__ = "ai_sessions"
    
    # Core Identity
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    project_id = Column(String(255), nullable=True, index=True)
    
    # Session Metadata
    session_type = Column(String(50), nullable=False, default=SessionType.INTERACTIVE.value)
    state = Column(String(50), nullable=False, default=SessionState.ACTIVE.value)
    title = Column(String(500))
    description = Column(Text)
    
    # Context and State
    context_window = Column(JSON, default=lambda: [])  # List of memory IDs in context
    context_summary = Column(Text)  # Compressed context summary
    context_embeddings = Column(ARRAY(Float))  # Context window embeddings
    context_size = Column(Integer, default=0)
    max_context_size = Column(Integer, default=100)
    
    # Session State
    active_tasks = Column(JSON, default=lambda: [])  # Current active tasks
    task_queue = Column(JSON, default=lambda: [])    # Queued tasks
    session_variables = Column(JSON, default=lambda: {})  # Session variables
    preferences = Column(JSON, default=lambda: {})   # User preferences for this session
    
    # Performance and Quality
    interaction_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    success_rate = Column(Float, default=1.0)
    avg_response_time = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))
    total_duration = Column(Integer, default=0)  # Duration in seconds
    
    # Recovery and Continuity
    parent_session_id = Column(UUID(as_uuid=True), ForeignKey('ai_sessions.id'), nullable=True)
    recovery_data = Column(JSON, default=lambda: {})  # Data for session recovery
    checkpoint_data = Column(JSON, default=lambda: {})  # Session checkpoints
    last_checkpoint = Column(DateTime(timezone=True))
    
    # Cross-session Linking
    related_sessions = Column(ARRAY(UUID), default=lambda: [])
    session_chain = Column(ARRAY(UUID), default=lambda: [])  # Session continuation chain
    
    # Quality and Analytics
    user_satisfaction = Column(Float)  # User-reported satisfaction
    completion_status = Column(String(100))
    goals_achieved = Column(JSON, default=lambda: [])
    goals_pending = Column(JSON, default=lambda: [])
    
    # Technical Details
    client_info = Column(JSON, default=lambda: {})  # Client/browser info
    ip_address = Column(String(45))  # IPv4/IPv6
    user_agent = Column(String(500))
    api_version = Column(String(20), default="v1.0")
    
    # Relationships
    child_sessions = relationship("Session", backref="parent_session", remote_side=[id])
    handoffs = relationship("SessionHandoff", back_populates="source_session", foreign_keys="SessionHandoff.source_session_id")
    
    __table_args__ = (
        # Core performance indexes
        Index('idx_sessions_user_active', 'user_id', 'state', 'last_active'),
        Index('idx_sessions_project_state', 'project_id', 'state'),
        Index('idx_sessions_type_started', 'session_type', 'started_at'),
        
        # Timeline and analysis indexes
        Index('idx_sessions_duration', 'total_duration'),
        Index('idx_sessions_success_rate', 'success_rate'),
        Index('idx_sessions_last_active', 'last_active'),
        
        # Recovery and linking indexes
        Index('idx_sessions_parent_recovery', 'parent_session_id', 'state'),
        Index('idx_sessions_checkpoints', 'last_checkpoint'),
        
        # Cross-session analysis
        Index('idx_sessions_chain', 'session_chain'),
        Index('idx_sessions_user_project_time', 'user_id', 'project_id', 'started_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "project_id": self.project_id,
            "session_type": self.session_type,
            "state": self.state,
            "title": self.title,
            "description": self.description,
            "context_size": self.context_size,
            "max_context_size": self.max_context_size,
            "interaction_count": self.interaction_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_duration": self.total_duration,
            "parent_session_id": str(self.parent_session_id) if self.parent_session_id else None,
            "completion_status": self.completion_status,
            "user_satisfaction": self.user_satisfaction
        }

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_active = datetime.utcnow()

    def add_to_context(self, memory_id: str, max_size: Optional[int] = None):
        """Add memory to context window."""
        if not self.context_window:
            self.context_window = []
        
        # Remove if already exists (move to end)
        if memory_id in self.context_window:
            self.context_window.remove(memory_id)
        
        # Add to end
        self.context_window.append(memory_id)
        
        # Trim if necessary
        max_size = max_size or self.max_context_size
        if len(self.context_window) > max_size:
            self.context_window = self.context_window[-max_size:]
        
        self.context_size = len(self.context_window)
        self.update_activity()


class SessionHandoff(Base):
    """Session handoff for transferring context between sessions."""
    __tablename__ = "session_handoffs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_session_id = Column(UUID(as_uuid=True), ForeignKey('ai_sessions.id'), nullable=False)
    target_session_id = Column(UUID(as_uuid=True), ForeignKey('ai_sessions.id'), nullable=True)
    
    # Handoff Details
    reason = Column(String(50), nullable=False)
    handoff_message = Column(Text)
    context_data = Column(JSON, default=lambda: {})
    continuation_instructions = Column(Text)
    
    # Status
    status = Column(String(50), default="pending")  # pending, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Quality
    success_rate = Column(Float)
    user_feedback = Column(Text)
    
    # Relationships
    source_session = relationship("Session", back_populates="handoffs", foreign_keys=[source_session_id])
    
    __table_args__ = (
        Index('idx_handoffs_source_session', 'source_session_id'),
        Index('idx_handoffs_target_session', 'target_session_id'),
        Index('idx_handoffs_status_created', 'status', 'created_at'),
    )


class SessionCheckpoint(Base):
    """Session checkpoints for recovery and rollback."""
    __tablename__ = "session_checkpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('ai_sessions.id'), nullable=False)
    
    # Checkpoint Details
    checkpoint_name = Column(String(200), nullable=False)
    description = Column(Text)
    checkpoint_type = Column(String(50), default="manual")  # manual, auto, scheduled
    
    # State Snapshot
    session_state = Column(JSON, nullable=False)
    context_snapshot = Column(JSON, default=lambda: {})
    memory_ids = Column(ARRAY(UUID), default=lambda: [])
    variables_snapshot = Column(JSON, default=lambda: {})
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(255))  # user or system
    interaction_count = Column(Integer, default=0)
    
    # Recovery
    is_recovery_point = Column(Boolean, default=False)
    recovery_priority = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_checkpoints_session_created', 'session_id', 'created_at'),
        Index('idx_checkpoints_recovery', 'session_id', 'is_recovery_point', 'recovery_priority'),
        Index('idx_checkpoints_type', 'checkpoint_type', 'created_at'),
    )


class SessionMetrics(Base):
    """Detailed session performance metrics."""
    __tablename__ = "session_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('ai_sessions.id'), nullable=False)
    
    # Performance Metrics
    metric_type = Column(String(50), nullable=False)  # response_time, memory_usage, etc.
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    
    # Context
    interaction_number = Column(Integer)
    task_context = Column(String(100))
    
    # Timing
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_metrics_session_type', 'session_id', 'metric_type'),
        Index('idx_metrics_recorded', 'recorded_at'),
    )


# Pydantic Models for API

class SessionCreate(BaseModel):
    """Schema for creating new sessions."""
    user_id: str = Field(..., min_length=1, max_length=255)
    project_id: Optional[str] = Field(None, max_length=255)
    session_type: SessionType = SessionType.INTERACTIVE
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    max_context_size: int = Field(default=100, ge=10, le=1000)
    parent_session_id: Optional[str] = None
    
    @field_validator('title')
    def validate_title(cls, v):
        if v and not v.strip():
            return None
        return v.strip() if v else None


class SessionUpdate(BaseModel):
    """Schema for updating sessions."""
    title: Optional[str] = None
    description: Optional[str] = None
    state: Optional[SessionState] = None
    preferences: Optional[Dict[str, Any]] = None
    max_context_size: Optional[int] = Field(None, ge=10, le=1000)
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=1.0)
    completion_status: Optional[str] = None


class SessionHandoffCreate(BaseModel):
    """Schema for creating session handoffs."""
    source_session_id: str
    reason: HandoffReason
    handoff_message: str = Field(..., min_length=1)
    continuation_instructions: Optional[str] = None
    context_data: Dict[str, Any] = Field(default_factory=dict)
    target_session_id: Optional[str] = None


class SessionCheckpointCreate(BaseModel):
    """Schema for creating session checkpoints."""
    session_id: str
    checkpoint_name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    checkpoint_type: str = "manual"
    is_recovery_point: bool = False
    recovery_priority: int = Field(default=0, ge=0, le=10)


class SessionResponse(BaseModel):
    """Schema for session responses."""
    id: str
    user_id: str
    project_id: Optional[str]
    session_type: SessionType
    state: SessionState
    title: Optional[str]
    description: Optional[str]
    context_size: int
    max_context_size: int
    interaction_count: int
    error_count: int
    success_rate: float
    started_at: datetime
    last_active: datetime
    ended_at: Optional[datetime]
    total_duration: int
    parent_session_id: Optional[str]
    completion_status: Optional[str]
    user_satisfaction: Optional[float]
    
    model_config = ConfigDict(from_attributes = True)


class SessionContextResponse(BaseModel):
    """Schema for session context responses."""
    session_id: str
    context_window: List[str]  # Memory IDs
    context_summary: Optional[str]
    context_size: int
    max_context_size: int
    last_updated: datetime
    
    model_config = ConfigDict(from_attributes = True)


class SessionAnalytics(BaseModel):
    """Schema for session analytics."""
    total_sessions: int
    active_sessions: int
    avg_session_duration: float
    avg_success_rate: float
    sessions_by_type: Dict[str, int]
    sessions_by_state: Dict[str, int]
    user_satisfaction_avg: Optional[float]
    top_projects: List[Dict[str, Any]]
    session_trends: List[Dict[str, Any]]
    handoff_statistics: Dict[str, Any]


class SessionRecoveryInfo(BaseModel):
    """Schema for session recovery information."""
    session_id: str
    recoverable: bool
    last_checkpoint: Optional[datetime]
    recovery_options: List[Dict[str, Any]]
    estimated_data_loss: str
    recommended_action: str