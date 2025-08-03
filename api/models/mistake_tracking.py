"""
Mistake Tracking Model for Learning from Errors
"""

from sqlalchemy import Column, String, Text, DateTime, Float, JSON, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid

from .base import Base


class MistakeTracking(Base):
    """Model for tracking mistakes and their solutions for learning"""
    
    __tablename__ = "mistake_tracking"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    error_type = Column(String(100), nullable=False, index=True)
    error_message = Column(Text, nullable=False)
    error_context = Column(JSON, default={})
    
    # Solution fields
    solution = Column(Text)
    solution_steps = Column(JSON, default=[])
    resolved = Column(Boolean, default=False)
    resolution_time = Column(Float)  # Time in seconds to resolve
    
    # Learning fields
    pattern = Column(String(200), index=True)
    category = Column(String(50), index=True)
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    frequency = Column(Float, default=1.0)
    
    # User and session tracking
    user_id = Column(String(100), index=True)
    session_id = Column(String(100), index=True)
    project_id = Column(String(100), index=True)
    
    # Metadata
    tags = Column(JSON, default=[])
    extra_data = Column('metadata', JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime(timezone=True))
    
    # Create indexes for better query performance
    __table_args__ = (
        Index('idx_error_pattern', 'error_type', 'pattern'),
        Index('idx_user_errors', 'user_id', 'created_at'),
        Index('idx_session_errors', 'session_id', 'created_at'),
        Index('idx_project_errors', 'project_id', 'created_at'),
        Index('idx_unresolved_errors', 'resolved', 'created_at'),
    )
    
    def __repr__(self):
        return f"<MistakeTracking(id={self.id}, error_type={self.error_type}, resolved={self.resolved})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': str(self.id),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'error_context': self.error_context,
            'solution': self.solution,
            'solution_steps': self.solution_steps,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time,
            'pattern': self.pattern,
            'category': self.category,
            'severity': self.severity,
            'frequency': self.frequency,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'project_id': self.project_id,
            'tags': self.tags,
            'metadata': self.extra_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class ErrorPattern(Base):
    """Model for storing learned error patterns"""
    
    __tablename__ = "error_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern = Column(String(200), unique=True, nullable=False)
    error_type = Column(String(100), nullable=False, index=True)
    
    # Pattern statistics
    occurrence_count = Column(Float, default=1.0)
    success_rate = Column(Float, default=0.0)  # How often the solution works
    avg_resolution_time = Column(Float)  # Average time to resolve
    
    # Recommended solution
    recommended_solution = Column(Text)
    solution_confidence = Column(Float, default=0.5)
    
    # Learning metadata
    first_seen = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<ErrorPattern(pattern={self.pattern}, success_rate={self.success_rate})>"