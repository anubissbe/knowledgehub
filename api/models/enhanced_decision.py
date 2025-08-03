"""
Enhanced Decision Recording and Analysis Models.

This module provides comprehensive models for tracking decisions, alternatives,
reasoning, outcomes, and decision patterns for AI-powered recommendations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sqlalchemy import Column, String, Float, Text, DateTime, Boolean, ForeignKey, Integer, Index, JSON as SQLAlchemyJSON, text
from sqlalchemy.dialects.postgresql import UUID as PostgreUUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declared_attr

from .base import Base


# Enums
class DecisionType(str, Enum):
    """Types of decisions that can be tracked."""
    TECHNICAL = "technical"
    ARCHITECTURAL = "architectural"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    OPTIMIZATION = "optimization"
    REFACTORING = "refactoring"
    TOOLING = "tooling"
    PROCESS = "process"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    OTHER = "other"


class DecisionStatus(str, Enum):
    """Status of a decision."""
    PROPOSED = "proposed"
    EVALUATING = "evaluating"
    DECIDED = "decided"
    IMPLEMENTED = "implemented"
    VALIDATED = "validated"
    REVISED = "revised"
    DEPRECATED = "deprecated"


class OutcomeStatus(str, Enum):
    """Status of a decision outcome."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    FAILED = "failed"
    REVISED = "revised"
    UNKNOWN = "unknown"


class ImpactLevel(str, Enum):
    """Impact level of a decision."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


# SQLAlchemy Models
class EnhancedDecision(Base):
    """
    Enhanced decision model for comprehensive decision tracking and analysis.
    
    Tracks decisions made during development with alternatives, reasoning,
    context, outcomes, and confidence scores for learning and recommendations.
    """
    __tablename__ = "enhanced_decisions"
    
    # Core fields
    id = Column(PostgreUUID(as_uuid=True), primary_key=True, default=uuid4)
    decision_type = Column(String(50), nullable=False, index=True)
    category = Column(String(100), index=True)  # Fine-grained categorization
    title = Column(String(500), nullable=False)
    description = Column(Text)
    
    # Decision details
    chosen_option = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.5, nullable=False)
    
    # Context and metadata
    context = Column(JSONB, default={})  # Execution context when decision was made
    constraints = Column(JSONB, default={})  # Constraints that influenced decision
    assumptions = Column(JSONB, default={})  # Assumptions made
    risks = Column(JSONB, default={})  # Identified risks
    dependencies = Column(ARRAY(String))  # Dependencies on other decisions/systems
    
    # Impact analysis
    impact_analysis = Column(JSONB, default={})
    impact_level = Column(String(20), default=ImpactLevel.MEDIUM.value)
    affected_components = Column(ARRAY(String))
    estimated_effort = Column(Float)  # Estimated effort in hours
    actual_effort = Column(Float)  # Actual effort spent
    
    # Tracking fields
    status = Column(String(20), default=DecisionStatus.PROPOSED.value, index=True)
    user_id = Column(String(255), index=True)
    session_id = Column(PostgreUUID(as_uuid=True), index=True)
    project_id = Column(String(255), index=True)
    tags = Column(ARRAY(String))
    
    # Decision tree
    parent_decision_id = Column(PostgreUUID(as_uuid=True), ForeignKey('enhanced_decisions.id'))
    decision_path = Column(ARRAY(PostgreUUID(as_uuid=True)))  # Path from root decision
    
    # Learning and patterns
    pattern_hash = Column(String(64), index=True)  # Hash for similar decision detection
    embeddings = Column(ARRAY(Float))  # Vector embeddings for similarity search
    success_probability = Column(Float)  # Predicted success probability
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    decided_at = Column(DateTime, default=datetime.utcnow)
    implemented_at = Column(DateTime)
    validated_at = Column(DateTime)
    deprecated_at = Column(DateTime)
    
    # Relationships
    alternatives = relationship("EnhancedAlternative", back_populates="decision", cascade="all, delete-orphan")
    outcome = relationship("EnhancedDecisionOutcome", back_populates="decision", uselist=False, cascade="all, delete-orphan")
    feedback_items = relationship("EnhancedDecisionFeedback", back_populates="decision", cascade="all, delete-orphan")
    revisions = relationship("EnhancedDecisionRevision", back_populates="original_decision", 
                           foreign_keys="[EnhancedDecisionRevision.original_decision_id]",
                           cascade="all, delete-orphan")
    child_decisions = relationship("EnhancedDecision", backref="parent_decision", remote_side=[id])
    
    # Indexes
    __table_args__ = (
        Index('idx_enhanced_decisions_type_status', 'decision_type', 'status'),
        Index('idx_enhanced_decisions_user_project', 'user_id', 'project_id'),
        Index('idx_enhanced_decisions_confidence', 'confidence_score'),
        Index('idx_enhanced_decisions_impact', 'impact_level'),
        Index('idx_enhanced_decisions_pattern', 'pattern_hash'),
    )
    
    def update_confidence(self, new_confidence: float):
        """Update confidence score with bounds checking."""
        self.confidence_score = max(0.0, min(1.0, new_confidence))
        self.updated_at = datetime.utcnow()
    
    def calculate_success_metrics(self) -> Dict[str, Any]:
        """Calculate success metrics based on outcome and feedback."""
        if not self.outcome:
            return {"success": None, "metrics": {}}
        
        metrics = {
            "outcome_success": self.outcome.success_rating,
            "implementation_time": None,
            "feedback_score": 0.0,
            "revision_count": len(self.revisions)
        }
        
        if self.implemented_at and self.decided_at:
            metrics["implementation_time"] = (self.implemented_at - self.decided_at).total_seconds() / 3600
        
        if self.feedback_items:
            feedback_scores = [f.rating for f in self.feedback_items if f.rating]
            if feedback_scores:
                metrics["feedback_score"] = sum(feedback_scores) / len(feedback_scores)
        
        return {
            "success": self.outcome.status == OutcomeStatus.SUCCESSFUL.value,
            "metrics": metrics
        }


class EnhancedAlternative(Base):
    """Alternative options considered for a decision."""
    __tablename__ = "enhanced_decision_alternatives"
    
    id = Column(PostgreUUID(as_uuid=True), primary_key=True, default=uuid4)
    decision_id = Column(PostgreUUID(as_uuid=True), ForeignKey('enhanced_decisions.id'), nullable=False)
    
    # Alternative details
    option = Column(Text, nullable=False)
    description = Column(Text)
    pros = Column(ARRAY(String))
    cons = Column(ARRAY(String))
    
    # Evaluation
    evaluation_score = Column(Float)  # Score during evaluation
    rejection_reason = Column(Text)  # Why this wasn't chosen
    
    # Analysis
    feasibility_score = Column(Float)
    risk_score = Column(Float)
    cost_estimate = Column(Float)
    complexity_score = Column(Float)
    
    # Metadata
    metadata_field = Column('metadata', JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship
    decision = relationship("EnhancedDecision", back_populates="alternatives")
    
    # Indexes
    __table_args__ = (
        Index('idx_enhanced_alternatives_decision', 'decision_id'),
        Index('idx_enhanced_alternatives_scores', 'evaluation_score', 'feasibility_score'),
    )


class EnhancedDecisionOutcome(Base):
    """Outcome and results of a decision."""
    __tablename__ = "enhanced_decision_outcomes"
    
    id = Column(PostgreUUID(as_uuid=True), primary_key=True, default=uuid4)
    decision_id = Column(PostgreUUID(as_uuid=True), ForeignKey('enhanced_decisions.id'), unique=True, nullable=False)
    
    # Outcome details
    status = Column(String(20), default=OutcomeStatus.PENDING.value, index=True)
    success_rating = Column(Float)  # 0.0 to 1.0
    description = Column(Text)
    
    # Metrics
    performance_metrics = Column(JSONB, default={})
    quality_metrics = Column(JSONB, default={})
    business_metrics = Column(JSONB, default={})
    
    # Analysis
    lessons_learned = Column(Text)
    unexpected_consequences = Column(ARRAY(String))
    positive_impacts = Column(ARRAY(String))
    negative_impacts = Column(ARRAY(String))
    
    # Validation
    validation_method = Column(String(100))
    validation_data = Column(JSONB, default={})
    validated_by = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    measured_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    decision = relationship("EnhancedDecision", back_populates="outcome")
    
    # Indexes
    __table_args__ = (
        Index('idx_enhanced_outcomes_status', 'status'),
        Index('idx_enhanced_outcomes_success', 'success_rating'),
    )
    
    def calculate_overall_success(self) -> float:
        """Calculate overall success score from various metrics."""
        scores = []
        
        if self.success_rating is not None:
            scores.append(self.success_rating)
        
        # Add weighted scores from different metric categories
        for metrics in [self.performance_metrics, self.quality_metrics, self.business_metrics]:
            if metrics and 'score' in metrics:
                scores.append(metrics['score'])
        
        return sum(scores) / len(scores) if scores else 0.5


class EnhancedDecisionFeedback(Base):
    """Feedback on decisions from users and automated systems."""
    __tablename__ = "enhanced_decision_feedback"
    
    id = Column(PostgreUUID(as_uuid=True), primary_key=True, default=uuid4)
    decision_id = Column(PostgreUUID(as_uuid=True), ForeignKey('enhanced_decisions.id'), nullable=False)
    
    # Feedback details
    user_id = Column(String(255), index=True)
    feedback_type = Column(String(50))  # user, automated, peer_review
    rating = Column(Float)  # 0.0 to 1.0
    comment = Column(Text)
    
    # Specific feedback areas
    effectiveness_rating = Column(Float)
    implementation_rating = Column(Float)
    maintainability_rating = Column(Float)
    
    # Suggestions
    improvement_suggestions = Column(Text)
    alternative_approach = Column(Text)
    
    # Metadata
    context = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship
    decision = relationship("EnhancedDecision", back_populates="feedback_items")
    
    # Indexes
    __table_args__ = (
        Index('idx_enhanced_feedback_decision_user', 'decision_id', 'user_id'),
        Index('idx_enhanced_feedback_rating', 'rating'),
    )


class EnhancedDecisionRevision(Base):
    """Revisions made to decisions based on outcomes and learning."""
    __tablename__ = "enhanced_decision_revisions"
    
    id = Column(PostgreUUID(as_uuid=True), primary_key=True, default=uuid4)
    original_decision_id = Column(PostgreUUID(as_uuid=True), ForeignKey('enhanced_decisions.id'), nullable=False)
    revised_decision_id = Column(PostgreUUID(as_uuid=True), ForeignKey('enhanced_decisions.id'))
    
    # Revision details
    revision_reason = Column(Text, nullable=False)
    revision_type = Column(String(50))  # minor, major, complete_reversal
    
    # Changes made
    changes_summary = Column(Text)
    changed_fields = Column(JSONB, default={})
    
    # Impact
    impact_assessment = Column(Text)
    migration_required = Column(Boolean, default=False)
    migration_steps = Column(JSONB, default={})
    
    # Metadata
    revised_by = Column(String(255))
    approved_by = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    original_decision = relationship("EnhancedDecision", foreign_keys=[original_decision_id], back_populates="revisions")
    revised_decision = relationship("EnhancedDecision", foreign_keys=[revised_decision_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_enhanced_revisions_original', 'original_decision_id'),
        Index('idx_enhanced_revisions_type', 'revision_type'),
    )


class DecisionPattern(Base):
    """Learned patterns from decision history for recommendations."""
    __tablename__ = "decision_patterns"
    
    id = Column(PostgreUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Pattern identification
    pattern_type = Column(String(100), index=True)
    pattern_name = Column(String(255))
    description = Column(Text)
    
    # Pattern characteristics
    context_patterns = Column(JSONB, default={})
    decision_characteristics = Column(JSONB, default={})
    success_indicators = Column(JSONB, default={})
    failure_indicators = Column(JSONB, default={})
    
    # Statistics
    occurrence_count = Column(Integer, default=0)
    success_rate = Column(Float)
    avg_confidence = Column(Float)
    avg_implementation_time = Column(Float)  # hours
    
    # Embeddings for similarity
    pattern_embeddings = Column(ARRAY(Float))
    
    # Recommendations
    recommended_approach = Column(Text)
    best_practices = Column(ARRAY(String))
    common_pitfalls = Column(ARRAY(String))
    
    # Metadata
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    confidence_threshold = Column(Float, default=0.7)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_patterns_type', 'pattern_type'),
        Index('idx_patterns_success', 'success_rate'),
        Index('idx_patterns_occurrences', 'occurrence_count'),
    )
    
    def update_statistics(self, new_decision_success: bool, confidence: float, implementation_hours: float):
        """Update pattern statistics with new decision data."""
        self.occurrence_count += 1
        
        # Update success rate
        if self.success_rate is None:
            self.success_rate = 1.0 if new_decision_success else 0.0
        else:
            # Moving average
            alpha = 0.1  # Learning rate
            self.success_rate = alpha * (1.0 if new_decision_success else 0.0) + (1 - alpha) * self.success_rate
        
        # Update average confidence
        if self.avg_confidence is None:
            self.avg_confidence = confidence
        else:
            self.avg_confidence = alpha * confidence + (1 - alpha) * self.avg_confidence
        
        # Update average implementation time
        if implementation_hours and implementation_hours > 0:
            if self.avg_implementation_time is None:
                self.avg_implementation_time = implementation_hours
            else:
                self.avg_implementation_time = alpha * implementation_hours + (1 - alpha) * self.avg_implementation_time
        
        self.last_seen = datetime.utcnow()


# Pydantic Models for API
class AlternativeCreate(BaseModel):
    """Schema for creating a decision alternative."""
    option: str
    description: Optional[str] = None
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    evaluation_score: Optional[float] = None
    rejection_reason: Optional[str] = None
    feasibility_score: Optional[float] = None
    risk_score: Optional[float] = None
    cost_estimate: Optional[float] = None
    complexity_score: Optional[float] = None


class DecisionCreate(BaseModel):
    """Schema for creating a decision."""
    decision_type: DecisionType
    category: Optional[str] = None
    title: str
    description: Optional[str] = None
    chosen_option: str
    alternatives: List[AlternativeCreate] = Field(default_factory=list)
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    assumptions: Dict[str, Any] = Field(default_factory=dict)
    risks: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    impact_analysis: Dict[str, Any] = Field(default_factory=dict)
    impact_level: ImpactLevel = ImpactLevel.MEDIUM
    affected_components: List[str] = Field(default_factory=list)
    estimated_effort: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    parent_decision_id: Optional[str] = None


class OutcomeCreate(BaseModel):
    """Schema for creating a decision outcome."""
    decision_id: str
    status: OutcomeStatus
    success_rating: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = None
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    business_metrics: Dict[str, Any] = Field(default_factory=dict)
    lessons_learned: Optional[str] = None
    unexpected_consequences: List[str] = Field(default_factory=list)
    positive_impacts: List[str] = Field(default_factory=list)
    negative_impacts: List[str] = Field(default_factory=list)
    validation_method: Optional[str] = None
    validation_data: Dict[str, Any] = Field(default_factory=dict)


class FeedbackCreate(BaseModel):
    """Schema for creating decision feedback."""
    decision_id: str
    feedback_type: str = "user"
    rating: Optional[float] = Field(None, ge=0.0, le=1.0)
    comment: Optional[str] = None
    effectiveness_rating: Optional[float] = Field(None, ge=0.0, le=1.0)
    implementation_rating: Optional[float] = Field(None, ge=0.0, le=1.0)
    maintainability_rating: Optional[float] = Field(None, ge=0.0, le=1.0)
    improvement_suggestions: Optional[str] = None
    alternative_approach: Optional[str] = None


class DecisionResponse(BaseModel):
    """Schema for decision response."""
    id: str
    decision_type: str
    category: Optional[str]
    title: str
    description: Optional[str]
    chosen_option: str
    reasoning: str
    confidence_score: float
    status: str
    impact_level: str
    created_at: datetime
    decided_at: Optional[datetime]
    alternatives_count: int = 0
    has_outcome: bool = False
    success_metrics: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes = True)


class DecisionTreeNode(BaseModel):
    """Schema for decision tree visualization."""
    id: str
    title: str
    decision_type: str
    confidence_score: float
    status: str
    impact_level: str
    children: List['DecisionTreeNode'] = Field(default_factory=list)
    outcome_status: Optional[str] = None
    success_rating: Optional[float] = None


# Enable forward reference
DecisionTreeNode.model_rebuild()


class DecisionAnalytics(BaseModel):
    """Schema for decision analytics."""
    total_decisions: int
    decisions_by_type: Dict[str, int]
    decisions_by_status: Dict[str, int]
    avg_confidence_score: float
    success_rate: float
    avg_implementation_time_hours: float
    top_patterns: List[Dict[str, Any]]
    impact_distribution: Dict[str, int]
    recent_trends: Dict[str, Any]