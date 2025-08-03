"""
Error Pattern Recognition Models for AI-Powered Error Learning.

This module defines the data models for intelligent error tracking, pattern
recognition, and automated solution discovery.
"""

from sqlalchemy import Column, String, Text, JSON, DateTime, Float, Integer, Boolean, Index, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, ARRAY, TSVECTOR
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid

from .base import Base


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGIC = "logic"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class SolutionStatus(str, Enum):
    """Solution verification status."""
    VERIFIED = "verified"
    SUGGESTED = "suggested"
    EXPERIMENTAL = "experimental"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class EnhancedErrorPattern(Base):
    """Enhanced error pattern model with embeddings and learning capabilities."""
    __tablename__ = "enhanced_error_patterns"
    
    # Core Identity
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_hash = Column(String(64), unique=True, nullable=False, index=True)
    
    # Error Details
    error_type = Column(String(100), nullable=False, index=True)
    error_category = Column(String(50), nullable=False, default=ErrorCategory.UNKNOWN.value)
    error_message = Column(Text, nullable=False)
    error_code = Column(String(50), index=True)
    severity = Column(String(20), nullable=False, default=ErrorSeverity.MEDIUM.value)
    
    # Pattern Information
    stack_trace = Column(Text)
    pattern_regex = Column(Text)  # Regex to match similar errors
    key_indicators = Column(ARRAY(String))  # Key words/phrases that identify this pattern
    
    # Context and Environment
    context = Column(JSON, default=lambda: {})
    environment_factors = Column(JSON, default=lambda: {})  # OS, language version, etc.
    prerequisites = Column(JSON, default=lambda: {})  # Conditions that lead to error
    
    # Solution Information
    primary_solution = Column(Text)
    alternative_solutions = Column(JSON, default=lambda: [])
    solution_steps = Column(JSON, default=lambda: [])  # Step-by-step resolution
    solution_code = Column(Text)  # Code snippet for fix
    solution_explanation = Column(Text)
    
    # Learning Metrics
    occurrences = Column(Integer, default=1)
    success_rate = Column(Float, default=0.0)
    avg_resolution_time = Column(Float, default=0.0)  # In seconds
    false_positive_rate = Column(Float, default=0.0)
    
    # AI/ML Features
    embeddings = Column(ARRAY(Float))  # Error pattern embeddings
    solution_embeddings = Column(ARRAY(Float))  # Solution embeddings
    confidence_score = Column(Float, default=0.5)
    
    # Metadata
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(255))
    
    # Search Optimization
    search_vector = Column(TSVECTOR)
    
    # Relationships
    occurrences_list = relationship("ErrorOccurrence", back_populates="pattern", cascade="all, delete-orphan")
    solutions = relationship("ErrorSolution", back_populates="pattern", cascade="all, delete-orphan")
    feedback = relationship("ErrorFeedback", back_populates="pattern", cascade="all, delete-orphan")
    
    __table_args__ = (
        # Performance indexes
        Index('idx_enhanced_error_patterns_type_category', 'error_type', 'error_category'),
        Index('idx_enhanced_error_patterns_severity_occurrences', 'severity', 'occurrences'),
        Index('idx_enhanced_error_patterns_success_rate', 'success_rate'),
        Index('idx_enhanced_error_patterns_last_seen', 'last_seen'),
        
        # Full-text search
        Index('idx_enhanced_error_patterns_search', 'search_vector', postgresql_using='gin'),
        
        # Embedding similarity (for pgvector when available)
        # Index('idx_error_patterns_embeddings', 'embeddings', postgresql_using='ivfflat'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "pattern_hash": self.pattern_hash,
            "error_type": self.error_type,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "severity": self.severity,
            "primary_solution": self.primary_solution,
            "occurrences": self.occurrences,
            "success_rate": self.success_rate,
            "confidence_score": self.confidence_score,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None
        }

    def update_metrics(self, success: bool, resolution_time: float):
        """Update learning metrics based on feedback."""
        self.occurrences += 1
        self.last_seen = datetime.utcnow()
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        
        # Update average resolution time
        if success and resolution_time > 0:
            self.avg_resolution_time = alpha * resolution_time + (1 - alpha) * self.avg_resolution_time


class ErrorOccurrence(Base):
    """Individual error occurrence tracking."""
    __tablename__ = "error_occurrences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('enhanced_error_patterns.id'), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey('ai_sessions.id'))
    user_id = Column(String(255), nullable=False, index=True)
    
    # Occurrence Details
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    full_error_message = Column(Text, nullable=False)
    full_stack_trace = Column(Text)
    
    # Context at time of error
    execution_context = Column(JSON, default=lambda: {})
    system_state = Column(JSON, default=lambda: {})
    user_actions = Column(JSON, default=lambda: [])  # Actions leading to error
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolution_time = Column(Float)  # Seconds to resolve
    applied_solution_id = Column(UUID(as_uuid=True), ForeignKey('error_solutions.id'))
    resolution_notes = Column(Text)
    
    # Relationships
    pattern = relationship("EnhancedErrorPattern", back_populates="occurrences_list")
    applied_solution = relationship("ErrorSolution", foreign_keys=[applied_solution_id])
    
    __table_args__ = (
        Index('idx_error_occurrences_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_error_occurrences_pattern_resolved', 'pattern_id', 'resolved'),
    )


class ErrorSolution(Base):
    """Solution variations and their effectiveness."""
    __tablename__ = "error_solutions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('enhanced_error_patterns.id'), nullable=False)
    
    # Solution Details
    solution_text = Column(Text, nullable=False)
    solution_code = Column(Text)
    solution_steps = Column(JSON, default=lambda: [])
    prerequisites = Column(JSON, default=lambda: [])
    
    # Effectiveness Metrics
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    avg_resolution_time = Column(Float, default=0.0)
    effectiveness_score = Column(Float, default=0.5)
    
    # Status
    status = Column(String(50), default=SolutionStatus.SUGGESTED.value)
    verified_by = Column(String(255))
    verified_at = Column(DateTime(timezone=True))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(255))
    last_used = Column(DateTime(timezone=True))
    
    # Relationships
    pattern = relationship("EnhancedErrorPattern", back_populates="solutions")
    
    __table_args__ = (
        Index('idx_error_solutions_pattern_effectiveness', 'pattern_id', 'effectiveness_score'),
        Index('idx_error_solutions_status', 'status'),
    )

    def update_effectiveness(self, success: bool, resolution_time: float):
        """Update solution effectiveness based on usage."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        total = self.success_count + self.failure_count
        self.effectiveness_score = self.success_count / total if total > 0 else 0.5
        
        if success and resolution_time > 0:
            alpha = 0.1
            self.avg_resolution_time = alpha * resolution_time + (1 - alpha) * self.avg_resolution_time
        
        self.last_used = datetime.utcnow()


class ErrorFeedback(Base):
    """User feedback on error patterns and solutions."""
    __tablename__ = "error_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(UUID(as_uuid=True), ForeignKey('enhanced_error_patterns.id'), nullable=False)
    occurrence_id = Column(UUID(as_uuid=True), ForeignKey('error_occurrences.id'))
    solution_id = Column(UUID(as_uuid=True), ForeignKey('error_solutions.id'))
    
    # Feedback Details
    user_id = Column(String(255), nullable=False)
    helpful = Column(Boolean, nullable=False)
    feedback_text = Column(Text)
    suggested_improvement = Column(Text)
    
    # Resolution Details
    actual_solution = Column(Text)
    resolution_time = Column(Float)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    pattern = relationship("EnhancedErrorPattern", back_populates="feedback")
    
    __table_args__ = (
        Index('idx_error_feedback_pattern_helpful', 'pattern_id', 'helpful'),
        Index('idx_error_feedback_created', 'created_at'),
    )


class ErrorPrediction(Base):
    """Predictive error analysis and prevention."""
    __tablename__ = "error_predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('ai_sessions.id'))
    user_id = Column(String(255), nullable=False)
    
    # Prediction Details
    predicted_pattern_id = Column(UUID(as_uuid=True), ForeignKey('error_patterns.id'))
    prediction_confidence = Column(Float, nullable=False)
    risk_factors = Column(JSON, default=lambda: [])
    
    # Context Leading to Prediction
    context_snapshot = Column(JSON, default=lambda: {})
    code_patterns = Column(JSON, default=lambda: [])
    user_behavior_patterns = Column(JSON, default=lambda: [])
    
    # Prevention Suggestions
    prevention_steps = Column(JSON, default=lambda: [])
    recommended_actions = Column(JSON, default=lambda: [])
    
    # Outcome
    prediction_occurred = Column(Boolean)
    actual_error_id = Column(UUID(as_uuid=True), ForeignKey('error_occurrences.id'))
    prevented = Column(Boolean)
    
    # Metadata
    predicted_at = Column(DateTime(timezone=True), server_default=func.now())
    outcome_recorded_at = Column(DateTime(timezone=True))
    
    __table_args__ = (
        Index('idx_error_predictions_session', 'session_id'),
        Index('idx_error_predictions_confidence', 'prediction_confidence'),
        Index('idx_error_predictions_outcome', 'prediction_occurred', 'prevented'),
    )


# Pydantic Models for API

class ErrorPatternCreate(BaseModel):
    """Schema for creating error patterns."""
    error_type: str = Field(..., min_length=1, max_length=100)
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    error_message: str = Field(..., min_length=1)
    error_code: Optional[str] = Field(None, max_length=50)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    primary_solution: Optional[str] = None
    solution_steps: List[str] = Field(default_factory=list)
    
    @field_validator('error_message')
    def validate_error_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Error message cannot be empty")
        return v.strip()


class ErrorOccurrenceCreate(BaseModel):
    """Schema for recording error occurrences."""
    error_message: str = Field(..., min_length=1)
    stack_trace: Optional[str] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    system_state: Dict[str, Any] = Field(default_factory=dict)
    user_actions: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: Optional[str] = None


class ErrorSolutionCreate(BaseModel):
    """Schema for creating error solutions."""
    pattern_id: str
    solution_text: str = Field(..., min_length=1)
    solution_code: Optional[str] = None
    solution_steps: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)


class ErrorFeedbackCreate(BaseModel):
    """Schema for error feedback."""
    pattern_id: str
    helpful: bool
    feedback_text: Optional[str] = None
    suggested_improvement: Optional[str] = None
    actual_solution: Optional[str] = None
    resolution_time: Optional[float] = Field(None, ge=0)
    occurrence_id: Optional[str] = None
    solution_id: Optional[str] = None


class ErrorPatternResponse(BaseModel):
    """Schema for error pattern responses."""
    id: str
    pattern_hash: str
    error_type: str
    error_category: ErrorCategory
    error_message: str
    error_code: Optional[str]
    severity: ErrorSeverity
    primary_solution: Optional[str]
    alternative_solutions: List[str]
    solution_steps: List[str]
    occurrences: int
    success_rate: float
    confidence_score: float
    first_seen: datetime
    last_seen: datetime
    
    model_config = ConfigDict(from_attributes = True)


class ErrorPredictionResponse(BaseModel):
    """Schema for error prediction responses."""
    id: str
    predicted_pattern: Optional[ErrorPatternResponse]
    prediction_confidence: float
    risk_factors: List[str]
    prevention_steps: List[str]
    recommended_actions: List[str]
    
    model_config = ConfigDict(from_attributes = True)


class ErrorAnalytics(BaseModel):
    """Schema for error analytics."""
    total_patterns: int
    total_occurrences: int
    patterns_by_category: Dict[str, int]
    patterns_by_severity: Dict[str, int]
    top_errors: List[Dict[str, Any]]
    resolution_stats: Dict[str, float]
    learning_progress: Dict[str, Any]
    prediction_accuracy: float