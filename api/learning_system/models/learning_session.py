"""Learning Session Model

Tracks learning sessions that persist across conversation sessions,
enabling continuous learning and knowledge transfer.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, ForeignKey, Float, Integer, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from ...models.base import Base


class LearningSessionType(PyEnum):
    """Types of learning sessions"""
    USER_INTERACTION = "user_interaction"  # Learning from user interactions
    OUTCOME_ANALYSIS = "outcome_analysis"  # Learning from decision outcomes
    PATTERN_DISCOVERY = "pattern_discovery"  # Discovering new patterns
    KNOWLEDGE_CONSOLIDATION = "knowledge_consolidation"  # Consolidating knowledge
    CROSS_SESSION_TRANSFER = "cross_session_transfer"  # Transferring knowledge between sessions


class LearningSessionStatus(PyEnum):
    """Status of learning sessions"""
    ACTIVE = "active"        # Currently learning
    COMPLETED = "completed"  # Learning session completed
    PAUSED = "paused"       # Temporarily paused
    FAILED = "failed"       # Learning session failed


class LearningSession(Base):
    """Model for tracking learning sessions across conversation sessions"""
    
    __tablename__ = 'learning_sessions'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Session identification
    session_type = Column(
        String(50),  # Using String instead of Enum for flexibility
        nullable=False,
        index=True,
        comment='Type of learning session'
    )
    session_name = Column(
        String(255),
        nullable=True,
        comment='Human-readable name for the session'
    )
    
    # User and context tracking
    user_id = Column(
        String(255),
        nullable=False,
        index=True,
        comment='User identifier for this learning session'
    )
    conversation_session_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('memory_sessions.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        comment='Associated conversation session'
    )
    
    # Learning session lifecycle
    status = Column(
        String(20),  # Using String instead of Enum for flexibility
        nullable=False,
        default=LearningSessionStatus.ACTIVE.value,
        index=True,
        comment='Current status of the learning session'
    )
    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment='When the learning session started'
    )
    ended_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment='When the learning session ended'
    )
    last_activity_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment='Last activity in this learning session'
    )
    
    # Learning session data
    learning_objectives = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='What this learning session aims to achieve'
    )
    learning_context = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='Context and parameters for learning'
    )
    session_metadata = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Additional metadata for the session'
    )
    
    # Learning metrics
    patterns_learned = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Number of patterns learned in this session'
    )
    patterns_reinforced = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Number of patterns reinforced in this session'
    )
    knowledge_units_created = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Number of knowledge units created'
    )
    success_rate = Column(
        Float,
        nullable=True,
        comment='Success rate of learning in this session'
    )
    
    # Cross-session continuity
    parent_learning_session_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('learning_sessions.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        comment='Parent learning session for continuity'
    )
    transferred_knowledge_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Amount of knowledge transferred from previous sessions'
    )
    
    # Quality metrics
    learning_effectiveness = Column(
        Float,
        nullable=True,
        comment='Measured effectiveness of learning (0.0 to 1.0)'
    )
    knowledge_retention_score = Column(
        Float,
        nullable=True,
        comment='How well knowledge is retained (0.0 to 1.0)'
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )
    
    # Relationships
    conversation_session = relationship('MemorySession', backref='learning_sessions')
    parent_session = relationship('LearningSession', remote_side=[id], backref='child_sessions')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_learning_session_user_status', 'user_id', 'status'),
        Index('idx_learning_session_type_started', 'session_type', 'started_at'),
        Index('idx_learning_session_activity', 'last_activity_at'),
        Index('idx_learning_session_effectiveness', 'learning_effectiveness'),
    )
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity_at = datetime.now(timezone.utc)
    
    def complete_session(self, success_rate: Optional[float] = None):
        """Mark learning session as completed"""
        self.status = LearningSessionStatus.COMPLETED.value
        self.ended_at = datetime.now(timezone.utc)
        if success_rate is not None:
            self.success_rate = success_rate
        self.update_activity()
    
    def pause_session(self):
        """Pause the learning session"""
        self.status = LearningSessionStatus.PAUSED.value
        self.update_activity()
    
    def resume_session(self):
        """Resume the learning session"""
        self.status = LearningSessionStatus.ACTIVE.value
        self.update_activity()
    
    def fail_session(self, reason: str):
        """Mark learning session as failed"""
        self.status = LearningSessionStatus.FAILED.value
        self.ended_at = datetime.now(timezone.utc)
        if self.session_metadata is None:
            self.session_metadata = {}
        self.session_metadata['failure_reason'] = reason
        self.update_activity()
    
    def add_learning_objective(self, objective: str, priority: str = "medium"):
        """Add a learning objective to the session"""
        if self.learning_objectives is None:
            self.learning_objectives = {}
        if 'objectives' not in self.learning_objectives:
            self.learning_objectives['objectives'] = []
        
        self.learning_objectives['objectives'].append({
            'objective': objective,
            'priority': priority,
            'added_at': datetime.now(timezone.utc).isoformat(),
            'completed': False
        })
    
    def complete_objective(self, objective: str):
        """Mark a learning objective as completed"""
        if self.learning_objectives and 'objectives' in self.learning_objectives:
            for obj in self.learning_objectives['objectives']:
                if obj['objective'] == objective:
                    obj['completed'] = True
                    obj['completed_at'] = datetime.now(timezone.utc).isoformat()
                    break
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of learning progress"""
        duration = None
        if self.ended_at:
            duration = (self.ended_at - self.started_at).total_seconds()
        elif self.status == LearningSessionStatus.ACTIVE.value:
            duration = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        
        return {
            'session_id': str(self.id),
            'session_type': self.session_type,
            'session_name': self.session_name,
            'user_id': self.user_id,
            'status': self.status,
            'duration_seconds': duration,
            'patterns_learned': self.patterns_learned,
            'patterns_reinforced': self.patterns_reinforced,
            'knowledge_units_created': self.knowledge_units_created,
            'success_rate': self.success_rate,
            'learning_effectiveness': self.learning_effectiveness,
            'knowledge_retention_score': self.knowledge_retention_score,
            'transferred_knowledge_count': self.transferred_knowledge_count,
            'has_parent_session': self.parent_learning_session_id is not None,
            'child_sessions_count': len(self.child_sessions) if hasattr(self, 'child_sessions') else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'session_type': self.session_type,
            'session_name': self.session_name,
            'user_id': self.user_id,
            'conversation_session_id': str(self.conversation_session_id) if self.conversation_session_id else None,
            'status': self.status,
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'last_activity_at': self.last_activity_at.isoformat(),
            'learning_objectives': self.learning_objectives,
            'learning_context': self.learning_context,
            'session_metadata': self.session_metadata,
            'patterns_learned': self.patterns_learned,
            'patterns_reinforced': self.patterns_reinforced,
            'knowledge_units_created': self.knowledge_units_created,
            'success_rate': self.success_rate,
            'parent_learning_session_id': str(self.parent_learning_session_id) if self.parent_learning_session_id else None,
            'transferred_knowledge_count': self.transferred_knowledge_count,
            'learning_effectiveness': self.learning_effectiveness,
            'knowledge_retention_score': self.knowledge_retention_score,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self) -> str:
        return (
            f"<LearningSession(id={self.id}, type={self.session_type}, "
            f"user={self.user_id}, status={self.status})>"
        )