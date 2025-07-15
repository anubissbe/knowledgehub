"""Pattern Evolution Model

Tracks how learning patterns evolve and change across sessions,
enabling understanding of pattern lifecycle and improvement.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, ForeignKey, Float, Integer, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from ...models.base import Base


class EvolutionType(PyEnum):
    """Types of pattern evolution"""
    CREATION = "creation"          # Pattern was created
    REINFORCEMENT = "reinforcement"  # Pattern was reinforced
    WEAKENING = "weakening"        # Pattern was weakened
    MODIFICATION = "modification"  # Pattern data was modified
    MERGER = "merger"             # Pattern was merged with another
    SPLIT = "split"               # Pattern was split into multiple
    DEPRECATION = "deprecation"    # Pattern was deprecated
    REACTIVATION = "reactivation"  # Pattern was reactivated


class EvolutionTrigger(PyEnum):
    """What triggered the pattern evolution"""
    USER_FEEDBACK = "user_feedback"      # User provided feedback
    OUTCOME_ANALYSIS = "outcome_analysis"  # Analysis of decision outcomes
    PATTERN_CONFLICT = "pattern_conflict"  # Conflict with other patterns
    AUTOMATIC_LEARNING = "automatic_learning"  # Automatic learning process
    MANUAL_ADJUSTMENT = "manual_adjustment"  # Manual adjustment by system
    CROSS_SESSION_TRANSFER = "cross_session_transfer"  # Transfer between sessions


class PatternEvolution(Base):
    """Model for tracking pattern evolution across sessions"""
    
    __tablename__ = 'pattern_evolutions'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Pattern tracking
    pattern_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('learning_patterns.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment='The pattern that evolved'
    )
    
    # Evolution details
    evolution_type = Column(
        String(20),
        nullable=False,
        index=True,
        comment='Type of evolution that occurred'
    )
    evolution_trigger = Column(
        String(30),
        nullable=False,
        index=True,
        comment='What triggered this evolution'
    )
    
    # Evolution data
    previous_state = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Previous state of the pattern'
    )
    new_state = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='New state of the pattern after evolution'
    )
    evolution_delta = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='What specifically changed'
    )
    
    # Evolution metrics
    previous_confidence = Column(
        Float,
        nullable=True,
        comment='Previous confidence score'
    )
    new_confidence = Column(
        Float,
        nullable=False,
        comment='New confidence score after evolution'
    )
    confidence_change = Column(
        Float,
        nullable=True,
        comment='Change in confidence (new - previous)'
    )
    
    # Context information
    learning_session_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('learning_sessions.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        comment='Learning session where evolution occurred'
    )
    conversation_session_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey('memory_sessions.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        comment='Conversation session where evolution occurred'
    )
    user_id = Column(
        String(255),
        nullable=False,
        index=True,
        comment='User associated with this evolution'
    )
    
    # Evolution quality
    evolution_quality = Column(
        Float,
        nullable=True,
        comment='Quality of the evolution (0.0 to 1.0)'
    )
    validation_score = Column(
        Float,
        nullable=True,
        comment='Validation score for the evolution (0.0 to 1.0)'
    )
    impact_score = Column(
        Float,
        nullable=True,
        comment='Measured impact of the evolution (0.0 to 1.0)'
    )
    
    # Related patterns (for mergers, splits, etc.)
    related_pattern_ids = Column(
        ARRAY(PGUUID(as_uuid=True)),
        nullable=True,
        default=list,
        comment='Other patterns involved in this evolution'
    )
    
    # Evolution evidence
    evidence_data = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Evidence supporting this evolution'
    )
    source_data = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Source data that triggered the evolution'
    )
    
    # Reversal information
    is_reversible = Column(
        Boolean,
        nullable=False,
        default=True,
        comment='Whether this evolution can be reversed'
    )
    reversed_by = Column(
        PGUUID(as_uuid=True),
        ForeignKey('pattern_evolutions.id', ondelete='SET NULL'),
        nullable=True,
        comment='Evolution that reversed this one'
    )
    reversal_reason = Column(
        Text,
        nullable=True,
        comment='Reason for reversal if applicable'
    )
    
    # Timestamps
    evolved_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment='When the evolution occurred'
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    
    # Relationships
    pattern = relationship('LearningPattern', backref='evolutions')
    learning_session = relationship('LearningSession', backref='pattern_evolutions')
    conversation_session = relationship('MemorySession', backref='pattern_evolutions')
    reversal_evolution = relationship('PatternEvolution', remote_side=[id])
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_evolution_pattern_evolved', 'pattern_id', 'evolved_at'),
        Index('idx_evolution_type_trigger', 'evolution_type', 'evolution_trigger'),
        Index('idx_evolution_user_session', 'user_id', 'learning_session_id'),
        Index('idx_evolution_quality', 'evolution_quality'),
        Index('idx_evolution_impact', 'impact_score'),
    )
    
    def calculate_confidence_change(self):
        """Calculate confidence change"""
        if self.previous_confidence is not None:
            self.confidence_change = self.new_confidence - self.previous_confidence
        else:
            self.confidence_change = None
    
    def add_evidence(self, evidence_type: str, evidence_data: Dict[str, Any]):
        """Add evidence supporting this evolution"""
        if self.evidence_data is None:
            self.evidence_data = {}
        
        self.evidence_data[evidence_type] = {
            'data': evidence_data,
            'added_at': datetime.now(timezone.utc).isoformat()
        }
    
    def add_related_pattern(self, pattern_id: UUID):
        """Add a related pattern to this evolution"""
        if self.related_pattern_ids is None:
            self.related_pattern_ids = []
        
        if pattern_id not in self.related_pattern_ids:
            self.related_pattern_ids.append(pattern_id)
    
    def reverse_evolution(self, reason: str) -> 'PatternEvolution':
        """Create a reversal evolution"""
        reversal = PatternEvolution(
            pattern_id=self.pattern_id,
            evolution_type=EvolutionType.MODIFICATION.value,
            evolution_trigger=EvolutionTrigger.MANUAL_ADJUSTMENT.value,
            previous_state=self.new_state,
            new_state=self.previous_state,
            previous_confidence=self.new_confidence,
            new_confidence=self.previous_confidence,
            learning_session_id=self.learning_session_id,
            conversation_session_id=self.conversation_session_id,
            user_id=self.user_id,
            reversal_reason=reason,
            is_reversible=False  # Reversals are not reversible
        )
        
        # Mark this evolution as reversed
        self.reversed_by = reversal.id
        
        return reversal
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution"""
        return {
            'evolution_id': str(self.id),
            'pattern_id': str(self.pattern_id),
            'evolution_type': self.evolution_type,
            'evolution_trigger': self.evolution_trigger,
            'confidence_change': self.confidence_change,
            'previous_confidence': self.previous_confidence,
            'new_confidence': self.new_confidence,
            'evolution_quality': self.evolution_quality,
            'validation_score': self.validation_score,
            'impact_score': self.impact_score,
            'user_id': self.user_id,
            'learning_session_id': str(self.learning_session_id) if self.learning_session_id else None,
            'conversation_session_id': str(self.conversation_session_id) if self.conversation_session_id else None,
            'evolved_at': self.evolved_at.isoformat(),
            'is_reversible': self.is_reversible,
            'was_reversed': self.reversed_by is not None,
            'has_related_patterns': bool(self.related_pattern_ids),
            'has_evidence': bool(self.evidence_data)
        }
    
    def get_evolution_impact(self) -> Dict[str, Any]:
        """Analyze the impact of this evolution"""
        impact_analysis = {
            'confidence_impact': 'neutral',
            'confidence_magnitude': 0.0,
            'overall_impact': 'unknown',
            'impact_categories': []
        }
        
        # Analyze confidence impact
        if self.confidence_change is not None:
            impact_analysis['confidence_magnitude'] = abs(self.confidence_change)
            
            if self.confidence_change > 0.1:
                impact_analysis['confidence_impact'] = 'positive'
                impact_analysis['impact_categories'].append('confidence_increase')
            elif self.confidence_change < -0.1:
                impact_analysis['confidence_impact'] = 'negative'
                impact_analysis['impact_categories'].append('confidence_decrease')
            else:
                impact_analysis['confidence_impact'] = 'neutral'
        
        # Analyze overall impact
        if self.impact_score is not None:
            if self.impact_score > 0.7:
                impact_analysis['overall_impact'] = 'high'
            elif self.impact_score > 0.4:
                impact_analysis['overall_impact'] = 'medium'
            else:
                impact_analysis['overall_impact'] = 'low'
        
        # Analyze evolution type impact
        if self.evolution_type == EvolutionType.REINFORCEMENT.value:
            impact_analysis['impact_categories'].append('pattern_strengthening')
        elif self.evolution_type == EvolutionType.WEAKENING.value:
            impact_analysis['impact_categories'].append('pattern_weakening')
        elif self.evolution_type == EvolutionType.CREATION.value:
            impact_analysis['impact_categories'].append('new_knowledge')
        elif self.evolution_type == EvolutionType.MERGER.value:
            impact_analysis['impact_categories'].append('knowledge_consolidation')
        
        return impact_analysis
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'pattern_id': str(self.pattern_id),
            'evolution_type': self.evolution_type,
            'evolution_trigger': self.evolution_trigger,
            'previous_state': self.previous_state,
            'new_state': self.new_state,
            'evolution_delta': self.evolution_delta,
            'previous_confidence': self.previous_confidence,
            'new_confidence': self.new_confidence,
            'confidence_change': self.confidence_change,
            'learning_session_id': str(self.learning_session_id) if self.learning_session_id else None,
            'conversation_session_id': str(self.conversation_session_id) if self.conversation_session_id else None,
            'user_id': self.user_id,
            'evolution_quality': self.evolution_quality,
            'validation_score': self.validation_score,
            'impact_score': self.impact_score,
            'related_pattern_ids': [str(pid) for pid in self.related_pattern_ids] if self.related_pattern_ids else [],
            'evidence_data': self.evidence_data,
            'source_data': self.source_data,
            'is_reversible': self.is_reversible,
            'reversed_by': str(self.reversed_by) if self.reversed_by else None,
            'reversal_reason': self.reversal_reason,
            'evolved_at': self.evolved_at.isoformat(),
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self) -> str:
        return (
            f"<PatternEvolution(id={self.id}, pattern={self.pattern_id}, "
            f"type={self.evolution_type}, trigger={self.evolution_trigger})>"
        )