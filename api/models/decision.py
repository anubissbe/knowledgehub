"""
Decision Model for Recording Technical Decisions with Reasoning
"""

from sqlalchemy import Column, String, Text, DateTime, Float, JSON, Boolean, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from .base import Base


class Decision(Base):
    """Model for tracking technical decisions with full reasoning"""
    
    __tablename__ = "decisions"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    decision_id = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    chosen_solution = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=False)
    
    # Confidence and metrics
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    adjusted_confidence = Column(Float)
    impact_score = Column(Float)
    complexity_score = Column(Float)
    
    # Categories and context
    category = Column(String(50), index=True)
    context = Column(JSON, default={})
    evidence = Column(JSON, default=[])
    trade_offs = Column(JSON, default={})
    
    # Tracking fields
    user_id = Column(String(100), index=True)
    session_id = Column(String(100), index=True)
    project_id = Column(String(100), index=True)
    
    # Outcome tracking
    outcome = Column(String(50))  # success, failure, partial, unknown
    outcome_details = Column(JSON, default={})
    outcome_recorded_at = Column(DateTime(timezone=True))
    
    # Metadata
    tags = Column(JSON, default=[])
    extra_data = Column('metadata', JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    alternatives = relationship("DecisionAlternative", back_populates="decision", cascade="all, delete-orphan")
    
    # Create indexes for better query performance
    __table_args__ = (
        Index('idx_decision_category', 'category', 'created_at'),
        Index('idx_decision_user', 'user_id', 'created_at'),
        Index('idx_decision_project', 'project_id', 'created_at'),
        Index('idx_decision_confidence', 'confidence', 'adjusted_confidence'),
        Index('idx_decision_outcome', 'outcome', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Decision(id={self.id}, title={self.title}, confidence={self.confidence})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': str(self.id),
            'decision_id': self.decision_id,
            'title': self.title,
            'chosen_solution': self.chosen_solution,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'adjusted_confidence': self.adjusted_confidence,
            'impact_score': self.impact_score,
            'complexity_score': self.complexity_score,
            'category': self.category,
            'context': self.context,
            'evidence': self.evidence,
            'trade_offs': self.trade_offs,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'project_id': self.project_id,
            'outcome': self.outcome,
            'outcome_details': self.outcome_details,
            'outcome_recorded_at': self.outcome_recorded_at.isoformat() if self.outcome_recorded_at else None,
            'tags': self.tags,
            'metadata': self.extra_data,
            'alternatives': [alt.to_dict() for alt in self.alternatives],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class DecisionAlternative(Base):
    """Model for storing alternative solutions considered"""
    
    __tablename__ = "decision_alternatives"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    decision_id = Column(UUID(as_uuid=True), ForeignKey('decisions.id'), nullable=False)
    
    solution = Column(Text, nullable=False)
    pros = Column(JSON, default=[])
    cons = Column(JSON, default=[])
    risk_level = Column(String(20))  # low, medium, high, critical
    complexity = Column(String(20))  # simple, moderate, complex, very_complex
    estimated_effort = Column(String(50))  # time estimate
    
    # Scoring
    feasibility_score = Column(Float)
    risk_score = Column(Float)
    benefit_score = Column(Float)
    
    # Why not chosen
    rejection_reason = Column(Text)
    
    # Relationship
    decision = relationship("Decision", back_populates="alternatives")
    
    def __repr__(self):
        return f"<DecisionAlternative(solution={self.solution[:50]}...)>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': str(self.id),
            'solution': self.solution,
            'pros': self.pros,
            'cons': self.cons,
            'risk_level': self.risk_level,
            'complexity': self.complexity,
            'estimated_effort': self.estimated_effort,
            'feasibility_score': self.feasibility_score,
            'risk_score': self.risk_score,
            'benefit_score': self.benefit_score,
            'rejection_reason': self.rejection_reason
        }