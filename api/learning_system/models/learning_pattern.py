"""Learning Pattern Model

Stores patterns learned from user interactions, code analysis, and outcomes.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, Text, Float, Integer, Enum, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB

from ...models.base import Base


class PatternType(PyEnum):
    """Types of patterns that can be learned"""
    CODE = "code"               # Code patterns and preferences
    PREFERENCE = "preference"   # User preferences
    WORKFLOW = "workflow"       # Workflow patterns
    SUCCESS = "success"         # Success patterns
    ERROR = "error"            # Error patterns
    DECISION = "decision"      # Decision patterns
    CORRECTION = "correction"  # Correction patterns from feedback


class LearningPattern(Base):
    """Model for storing learned patterns"""
    
    __tablename__ = 'learning_patterns'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Pattern identification
    pattern_type = Column(
        Enum(PatternType, name='pattern_type_enum', create_type=True),
        nullable=False,
        index=True,
        comment='Type of pattern'
    )
    pattern_hash = Column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        comment='Hash of pattern data for deduplication'
    )
    
    # Pattern data
    pattern_data = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='Structured pattern information'
    )
    
    # Pattern metadata
    confidence_score = Column(
        Float,
        nullable=False,
        default=0.5,
        index=True,
        comment='Confidence in this pattern (0.0 to 1.0)'
    )
    initial_confidence = Column(
        Float,
        nullable=True,
        comment='Initial confidence when pattern was created'
    )
    usage_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Number of times pattern has been used'
    )
    reinforcement_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Number of times pattern has been reinforced'
    )
    
    # Source tracking
    source = Column(
        String(50),
        nullable=False,
        comment='Source of the pattern (e.g., user_interaction, outcome_analysis)'
    )
    created_by_session = Column(
        PGUUID(as_uuid=True),
        nullable=True,
        comment='Session that created this pattern'
    )
    
    # Temporal data
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
    last_used = Column(
        DateTime(timezone=True),
        nullable=True,
        comment='Last time this pattern was applied'
    )
    
    # Performance tracking
    effectiveness_score = Column(
        Float,
        nullable=True,
        comment='Measured effectiveness of pattern (0.0 to 1.0)'
    )
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_pattern_type_confidence', 'pattern_type', 'confidence_score'),
        Index('idx_pattern_last_used', 'last_used'),
        Index('idx_pattern_source', 'source'),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.initial_confidence is None:
            self.initial_confidence = self.confidence_score
    
    def reinforce(self, success_score: float = 1.0):
        """Reinforce pattern with positive outcome"""
        self.reinforcement_count += 1
        # Weighted update of confidence
        weight = 1.0 / (self.usage_count + 1)
        self.confidence_score = (
            self.confidence_score * (1 - weight) + 
            success_score * weight
        )
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        self.last_used = datetime.now(timezone.utc)
        self.usage_count += 1
    
    def weaken(self, failure_score: float = 0.0):
        """Weaken pattern due to negative outcome"""
        weight = 1.0 / (self.usage_count + 1)
        self.confidence_score = (
            self.confidence_score * (1 - weight) + 
            failure_score * weight
        )
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        self.last_used = datetime.now(timezone.utc)
        self.usage_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'pattern_type': self.pattern_type.value,
            'pattern_hash': self.pattern_hash,
            'pattern_data': self.pattern_data,
            'confidence_score': self.confidence_score,
            'initial_confidence': self.initial_confidence,
            'usage_count': self.usage_count,
            'reinforcement_count': self.reinforcement_count,
            'source': self.source,
            'created_by_session': str(self.created_by_session) if self.created_by_session else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'effectiveness_score': self.effectiveness_score
        }
    
    def __repr__(self) -> str:
        return (
            f"<LearningPattern(id={self.id}, type={self.pattern_type.value}, "
            f"confidence={self.confidence_score:.2f}, usage={self.usage_count})>"
        )