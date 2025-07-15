"""User Feedback Model

Stores user feedback including corrections, ratings, and confirmations.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean, Enum, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship

from ...models.base import Base


class FeedbackType(PyEnum):
    """Types of user feedback"""
    CORRECTION = "correction"      # User corrected the output
    RATING = "rating"             # User rated the output
    CONFIRMATION = "confirmation"  # User confirmed correctness
    REJECTION = "rejection"       # User rejected the output


class UserFeedback(Base):
    """Model for storing user feedback"""
    
    __tablename__ = 'user_feedback'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Memory reference (optional)
    memory_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('memories.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        comment='Related memory if applicable'
    )
    
    # Feedback classification
    feedback_type = Column(
        Enum(FeedbackType, name='feedback_type_enum', create_type=True),
        nullable=False,
        comment='Type of feedback'
    )
    
    # Feedback content
    original_content = Column(
        Text,
        nullable=True,
        comment='Original content that was corrected'
    )
    corrected_content = Column(
        Text,
        nullable=True,
        comment='User-provided correction'
    )
    
    # Feedback data
    feedback_data = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='Additional feedback data (e.g., rating value)'
    )
    
    # Context
    context_data = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Context when feedback was provided'
    )
    
    # Processing
    applied = Column(
        Boolean,
        nullable=False,
        default=False,
        comment='Whether feedback has been applied'
    )
    applied_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment='When feedback was applied'
    )
    processing_result = Column(
        JSONB,
        nullable=True,
        comment='Result of processing this feedback'
    )
    
    # Session tracking
    session_id = Column(
        PGUUID(as_uuid=True),
        nullable=True,
        comment='Session that provided this feedback'
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    
    # Relationships
    memory = relationship('Memory', backref='feedback')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_feedback_type_created', 'feedback_type', 'created_at'),
        Index('idx_feedback_memory_type', 'memory_id', 'feedback_type'),
        Index('idx_feedback_applied', 'applied'),
    )
    
    def get_rating(self) -> Optional[int]:
        """Get rating value if this is rating feedback"""
        if self.feedback_type == FeedbackType.RATING:
            return self.feedback_data.get('rating')
        return None
    
    def get_correction_diff(self) -> Optional[Dict[str, str]]:
        """Get correction difference if this is correction feedback"""
        if self.feedback_type == FeedbackType.CORRECTION:
            return {
                'original': self.original_content,
                'corrected': self.corrected_content
            }
        return None
    
    def apply_feedback(self):
        """Mark feedback as applied"""
        self.applied = True
        self.applied_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'memory_id': str(self.memory_id) if self.memory_id else None,
            'feedback_type': self.feedback_type.value,
            'original_content': self.original_content,
            'corrected_content': self.corrected_content,
            'feedback_data': self.feedback_data,
            'context_data': self.context_data,
            'applied': self.applied,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'processing_result': self.processing_result,
            'session_id': str(self.session_id) if self.session_id else None,
            'created_at': self.created_at.isoformat(),
            'rating': self.get_rating(),
            'correction_diff': self.get_correction_diff()
        }
    
    def __repr__(self) -> str:
        return (
            f"<UserFeedback(id={self.id}, type={self.feedback_type.value}, "
            f"applied={self.applied})>"
        )