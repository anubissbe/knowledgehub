"""Decision Outcome Model

Tracks the outcomes of decisions made by the system to measure effectiveness.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, ForeignKey, Float, Enum, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship

from ...models.base import Base


class OutcomeType(PyEnum):
    """Types of decision outcomes"""
    SUCCESS = "success"      # Decision led to successful outcome
    FAILURE = "failure"      # Decision led to failure
    PARTIAL = "partial"      # Partially successful
    UNKNOWN = "unknown"      # Outcome unclear


class DecisionOutcome(Base):
    """Model for tracking decision outcomes"""
    
    __tablename__ = 'decision_outcomes'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Decision reference
    decision_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('memories.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment='Reference to the decision memory'
    )
    
    # Outcome classification
    outcome_type = Column(
        Enum(OutcomeType, name='outcome_type_enum', create_type=True),
        nullable=False,
        default=OutcomeType.UNKNOWN,
        comment='Type of outcome'
    )
    
    # Success measurement
    success_score = Column(
        Float,
        nullable=False,
        default=0.5,
        comment='Success score from 0.0 (failure) to 1.0 (success)'
    )
    
    # Outcome details
    outcome_data = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='Detailed outcome information'
    )
    
    # Impact measurement
    impact_data = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Data about the impact of the decision'
    )
    
    # User feedback if provided
    user_feedback_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('user_feedback.id', ondelete='SET NULL'),
        nullable=True,
        comment='Associated user feedback if any'
    )
    
    # Timing
    measured_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment='When the outcome was measured'
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    
    # Relationships
    decision = relationship('MemorySystemMemory', backref='outcomes')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_outcome_decision_measured', 'decision_id', 'measured_at'),
        Index('idx_outcome_type_score', 'outcome_type', 'success_score'),
    )
    
    def is_successful(self) -> bool:
        """Check if outcome was successful"""
        return self.outcome_type == OutcomeType.SUCCESS or self.success_score >= 0.7
    
    def is_failure(self) -> bool:
        """Check if outcome was a failure"""
        return self.outcome_type == OutcomeType.FAILURE or self.success_score <= 0.3
    
    def get_impact_metrics(self) -> Dict[str, Any]:
        """Get impact metrics from outcome"""
        metrics = {
            'success_score': self.success_score,
            'outcome_type': self.outcome_type.value
        }
        
        if self.impact_data:
            metrics.update({
                'performance': self.impact_data.get('performance_metrics', {}),
                'user_satisfaction': self.impact_data.get('user_satisfaction'),
                'errors': self.impact_data.get('errors_encountered', []),
                'time_to_complete': self.impact_data.get('time_to_complete')
            })
        
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'decision_id': str(self.decision_id),
            'outcome_type': self.outcome_type.value,
            'success_score': self.success_score,
            'outcome_data': self.outcome_data,
            'impact_data': self.impact_data,
            'user_feedback_id': str(self.user_feedback_id) if self.user_feedback_id else None,
            'measured_at': self.measured_at.isoformat(),
            'created_at': self.created_at.isoformat(),
            'is_successful': self.is_successful(),
            'is_failure': self.is_failure(),
            'impact_metrics': self.get_impact_metrics()
        }
    
    def __repr__(self) -> str:
        return (
            f"<DecisionOutcome(id={self.id}, decision={self.decision_id}, "
            f"type={self.outcome_type.value}, score={self.success_score:.2f})>"
        )