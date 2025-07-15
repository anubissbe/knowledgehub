"""Knowledge Transfer Model

Tracks how knowledge and patterns are transferred between learning sessions,
enabling cross-session learning continuity.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, ForeignKey, Float, Integer, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from ...models.base import Base


class TransferType(PyEnum):
    """Types of knowledge transfer"""
    PATTERN_TRANSFER = "pattern_transfer"      # Transfer of learned patterns
    CONTEXT_TRANSFER = "context_transfer"      # Transfer of context information
    PREFERENCE_TRANSFER = "preference_transfer"  # Transfer of user preferences
    DECISION_TRANSFER = "decision_transfer"    # Transfer of decision patterns
    OUTCOME_TRANSFER = "outcome_transfer"      # Transfer of outcome knowledge


class TransferStatus(PyEnum):
    """Status of knowledge transfer"""
    PENDING = "pending"        # Transfer scheduled but not started
    IN_PROGRESS = "in_progress"  # Transfer in progress
    COMPLETED = "completed"    # Transfer completed successfully
    FAILED = "failed"         # Transfer failed
    PARTIAL = "partial"       # Transfer partially completed


class KnowledgeTransfer(Base):
    """Model for tracking knowledge transfers between learning sessions"""
    
    __tablename__ = 'knowledge_transfers'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Transfer identification
    transfer_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment='Type of knowledge being transferred'
    )
    transfer_name = Column(
        String(255),
        nullable=True,
        comment='Human-readable name for the transfer'
    )
    
    # Source and destination
    source_learning_session_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('learning_sessions.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment='Source learning session'
    )
    destination_learning_session_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('learning_sessions.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment='Destination learning session'
    )
    
    # Transfer details
    transferred_data = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='Data being transferred'
    )
    transfer_metadata = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Metadata about the transfer'
    )
    
    # Transfer status and metrics
    status = Column(
        String(20),
        nullable=False,
        default=TransferStatus.PENDING.value,
        index=True,
        comment='Current status of the transfer'
    )
    transfer_score = Column(
        Float,
        nullable=True,
        comment='Success score of the transfer (0.0 to 1.0)'
    )
    confidence_score = Column(
        Float,
        nullable=True,
        comment='Confidence in the transferred knowledge (0.0 to 1.0)'
    )
    
    # Transfer size and impact
    knowledge_units_transferred = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Number of knowledge units transferred'
    )
    patterns_transferred = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Number of patterns transferred'
    )
    estimated_impact = Column(
        Float,
        nullable=True,
        comment='Estimated impact of the transfer (0.0 to 1.0)'
    )
    actual_impact = Column(
        Float,
        nullable=True,
        comment='Measured actual impact of the transfer (0.0 to 1.0)'
    )
    
    # Timing information
    scheduled_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment='When the transfer was scheduled'
    )
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment='When the transfer started'
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment='When the transfer completed'
    )
    
    # Quality and validation
    validation_results = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Results of transfer validation'
    )
    error_details = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Error details if transfer failed'
    )
    
    # User and context
    user_id = Column(
        String(255),
        nullable=False,
        index=True,
        comment='User for whom the transfer is being made'
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
    source_session = relationship(
        'LearningSession',
        foreign_keys=[source_learning_session_id],
        backref='outgoing_transfers'
    )
    destination_session = relationship(
        'LearningSession',
        foreign_keys=[destination_learning_session_id],
        backref='incoming_transfers'
    )
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_transfer_source_destination', 'source_learning_session_id', 'destination_learning_session_id'),
        Index('idx_transfer_type_status', 'transfer_type', 'status'),
        Index('idx_transfer_user_scheduled', 'user_id', 'scheduled_at'),
        Index('idx_transfer_completed', 'completed_at'),
    )
    
    def start_transfer(self):
        """Mark transfer as started"""
        self.status = TransferStatus.IN_PROGRESS.value
        self.started_at = datetime.now(timezone.utc)
    
    def complete_transfer(self, success_score: float, actual_impact: Optional[float] = None):
        """Mark transfer as completed"""
        self.status = TransferStatus.COMPLETED.value
        self.completed_at = datetime.now(timezone.utc)
        self.transfer_score = success_score
        if actual_impact is not None:
            self.actual_impact = actual_impact
    
    def fail_transfer(self, error_message: str, error_details: Optional[Dict[str, Any]] = None):
        """Mark transfer as failed"""
        self.status = TransferStatus.FAILED.value
        self.completed_at = datetime.now(timezone.utc)
        
        if self.error_details is None:
            self.error_details = {}
        self.error_details['error_message'] = error_message
        self.error_details['failed_at'] = datetime.now(timezone.utc).isoformat()
        
        if error_details:
            self.error_details.update(error_details)
    
    def set_partial_completion(self, success_score: float, completed_units: int):
        """Mark transfer as partially completed"""
        self.status = TransferStatus.PARTIAL.value
        self.transfer_score = success_score
        self.knowledge_units_transferred = completed_units
    
    def add_validation_result(self, validation_type: str, result: Dict[str, Any]):
        """Add validation result"""
        if self.validation_results is None:
            self.validation_results = {}
        
        self.validation_results[validation_type] = {
            'result': result,
            'validated_at': datetime.now(timezone.utc).isoformat()
        }
    
    def get_transfer_duration(self) -> Optional[float]:
        """Get transfer duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return None
    
    def get_transfer_efficiency(self) -> Optional[float]:
        """Calculate transfer efficiency"""
        if not self.transfer_score:
            return None
        
        duration = self.get_transfer_duration()
        if not duration:
            return None
        
        # Efficiency = (success_score * knowledge_units) / duration
        # Normalized by expected baseline
        baseline_rate = 1.0  # 1 unit per second baseline
        efficiency = (self.transfer_score * self.knowledge_units_transferred) / (duration * baseline_rate)
        return min(1.0, efficiency)
    
    def get_impact_accuracy(self) -> Optional[float]:
        """Calculate how accurate the impact estimation was"""
        if self.estimated_impact is None or self.actual_impact is None:
            return None
        
        # Calculate accuracy as inverse of absolute difference
        difference = abs(self.estimated_impact - self.actual_impact)
        accuracy = 1.0 - difference
        return max(0.0, accuracy)
    
    def get_transfer_summary(self) -> Dict[str, Any]:
        """Get a summary of the transfer"""
        return {
            'transfer_id': str(self.id),
            'transfer_type': self.transfer_type,
            'transfer_name': self.transfer_name,
            'user_id': self.user_id,
            'status': self.status,
            'source_session_id': str(self.source_learning_session_id),
            'destination_session_id': str(self.destination_learning_session_id),
            'knowledge_units_transferred': self.knowledge_units_transferred,
            'patterns_transferred': self.patterns_transferred,
            'transfer_score': self.transfer_score,
            'confidence_score': self.confidence_score,
            'estimated_impact': self.estimated_impact,
            'actual_impact': self.actual_impact,
            'duration_seconds': self.get_transfer_duration(),
            'efficiency': self.get_transfer_efficiency(),
            'impact_accuracy': self.get_impact_accuracy(),
            'scheduled_at': self.scheduled_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'is_successful': self.status == TransferStatus.COMPLETED.value,
            'has_validation_results': bool(self.validation_results),
            'has_errors': bool(self.error_details)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'transfer_type': self.transfer_type,
            'transfer_name': self.transfer_name,
            'source_learning_session_id': str(self.source_learning_session_id),
            'destination_learning_session_id': str(self.destination_learning_session_id),
            'transferred_data': self.transferred_data,
            'transfer_metadata': self.transfer_metadata,
            'status': self.status,
            'transfer_score': self.transfer_score,
            'confidence_score': self.confidence_score,
            'knowledge_units_transferred': self.knowledge_units_transferred,
            'patterns_transferred': self.patterns_transferred,
            'estimated_impact': self.estimated_impact,
            'actual_impact': self.actual_impact,
            'scheduled_at': self.scheduled_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'validation_results': self.validation_results,
            'error_details': self.error_details,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self) -> str:
        return (
            f"<KnowledgeTransfer(id={self.id}, type={self.transfer_type}, "
            f"status={self.status}, score={self.transfer_score})>"
        )