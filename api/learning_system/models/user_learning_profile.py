"""User Learning Profile Model

Tracks persistent learning characteristics and preferences for users,
enabling personalized learning across sessions.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from ...models.base import Base


class LearningStyleType(PyEnum):
    """Types of learning styles"""
    VISUAL = "visual"              # Prefers visual information
    AUDITORY = "auditory"          # Prefers auditory information
    KINESTHETIC = "kinesthetic"    # Prefers hands-on learning
    READING = "reading"            # Prefers reading/writing
    MIXED = "mixed"                # Mixed learning style


class PreferenceLevel(PyEnum):
    """Levels of user preferences"""
    STRONG = "strong"      # Strong preference
    MODERATE = "moderate"  # Moderate preference
    WEAK = "weak"         # Weak preference
    NONE = "none"         # No preference


class UserLearningProfile(Base):
    """Model for tracking user learning profiles across sessions"""
    
    __tablename__ = 'user_learning_profiles'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # User identification
    user_id = Column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment='Unique identifier for the user'
    )
    
    # Learning characteristics
    learning_style = Column(
        String(20),
        nullable=True,
        default=LearningStyleType.MIXED.value,
        comment='Primary learning style of the user'
    )
    learning_pace = Column(
        String(20),
        nullable=True,
        default='medium',
        comment='Learning pace preference (slow, medium, fast)'
    )
    complexity_preference = Column(
        String(20),
        nullable=True,
        default='medium',
        comment='Complexity preference (simple, medium, complex)'
    )
    
    # Learning preferences
    preferred_explanation_style = Column(
        String(50),
        nullable=True,
        default='detailed',
        comment='Preferred explanation style (brief, detailed, comprehensive)'
    )
    preferred_code_style = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Preferred code style and patterns'
    )
    preferred_tools = Column(
        ARRAY(String(50)),
        nullable=True,
        default=list,
        comment='Preferred development tools'
    )
    
    # Learning behavior patterns
    learning_patterns = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='Learned patterns about user behavior'
    )
    interaction_patterns = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='Patterns in how user interacts with the system'
    )
    feedback_patterns = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment='Patterns in user feedback behavior'
    )
    
    # Learning metrics
    total_learning_sessions = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Total number of learning sessions'
    )
    total_patterns_learned = Column(
        Integer,
        nullable=False,
        default=0,
        comment='Total patterns learned by the user'
    )
    average_learning_effectiveness = Column(
        Float,
        nullable=True,
        comment='Average learning effectiveness across sessions'
    )
    knowledge_retention_rate = Column(
        Float,
        nullable=True,
        comment='Rate of knowledge retention (0.0 to 1.0)'
    )
    
    # Adaptation settings
    adaptation_aggressiveness = Column(
        Float,
        nullable=False,
        default=0.5,
        comment='How aggressively to adapt to user (0.0 to 1.0)'
    )
    pattern_sensitivity = Column(
        Float,
        nullable=False,
        default=0.7,
        comment='Sensitivity to pattern recognition (0.0 to 1.0)'
    )
    feedback_responsiveness = Column(
        Float,
        nullable=False,
        default=0.8,
        comment='How responsive to be to user feedback (0.0 to 1.0)'
    )
    
    # Contextual preferences
    domain_expertise = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Expertise levels in different domains'
    )
    communication_preferences = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Communication style preferences'
    )
    error_tolerance = Column(
        Float,
        nullable=False,
        default=0.5,
        comment='User tolerance for errors (0.0 to 1.0)'
    )
    
    # Learning history summary
    successful_patterns = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Summary of successful learning patterns'
    )
    failed_patterns = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Summary of failed learning attempts'
    )
    improvement_areas = Column(
        JSONB,
        nullable=True,
        default=dict,
        comment='Identified areas for improvement'
    )
    
    # Profile status
    profile_completeness = Column(
        Float,
        nullable=False,
        default=0.0,
        comment='Completeness of the profile (0.0 to 1.0)'
    )
    last_updated_by_session = Column(
        PGUUID(as_uuid=True),
        nullable=True,
        comment='Last session that updated this profile'
    )
    auto_update_enabled = Column(
        Boolean,
        nullable=False,
        default=True,
        comment='Whether profile auto-updates are enabled'
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
    last_interaction_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment='Last time user interacted with learning system'
    )
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_profile_user_id', 'user_id'),
        Index('idx_user_profile_updated', 'updated_at'),
        Index('idx_user_profile_completeness', 'profile_completeness'),
        Index('idx_user_profile_last_interaction', 'last_interaction_at'),
    )
    
    def update_interaction_timestamp(self):
        """Update last interaction timestamp"""
        self.last_interaction_at = datetime.now(timezone.utc)
    
    def add_learning_pattern(self, pattern_type: str, pattern_data: Dict[str, Any]):
        """Add a learning pattern to the profile"""
        if self.learning_patterns is None:
            self.learning_patterns = {}
        
        if pattern_type not in self.learning_patterns:
            self.learning_patterns[pattern_type] = []
        
        self.learning_patterns[pattern_type].append({
            'pattern_data': pattern_data,
            'added_at': datetime.now(timezone.utc).isoformat(),
            'confidence': pattern_data.get('confidence', 0.5)
        })
    
    def update_learning_effectiveness(self, session_effectiveness: float):
        """Update average learning effectiveness"""
        if self.average_learning_effectiveness is None:
            self.average_learning_effectiveness = session_effectiveness
        else:
            # Weighted average with more weight on recent sessions
            weight = 0.3  # Weight for new session
            self.average_learning_effectiveness = (
                self.average_learning_effectiveness * (1 - weight) +
                session_effectiveness * weight
            )
    
    def update_domain_expertise(self, domain: str, expertise_level: float):
        """Update expertise level for a domain"""
        if self.domain_expertise is None:
            self.domain_expertise = {}
        
        self.domain_expertise[domain] = {
            'level': expertise_level,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
    
    def add_successful_pattern(self, pattern_type: str, pattern_summary: Dict[str, Any]):
        """Add a successful pattern to the profile"""
        if self.successful_patterns is None:
            self.successful_patterns = {}
        
        if pattern_type not in self.successful_patterns:
            self.successful_patterns[pattern_type] = []
        
        self.successful_patterns[pattern_type].append({
            'pattern_summary': pattern_summary,
            'success_date': datetime.now(timezone.utc).isoformat()
        })
    
    def add_failed_pattern(self, pattern_type: str, failure_reason: str, pattern_data: Dict[str, Any]):
        """Add a failed pattern to learn from"""
        if self.failed_patterns is None:
            self.failed_patterns = {}
        
        if pattern_type not in self.failed_patterns:
            self.failed_patterns[pattern_type] = []
        
        self.failed_patterns[pattern_type].append({
            'failure_reason': failure_reason,
            'pattern_data': pattern_data,
            'failed_at': datetime.now(timezone.utc).isoformat()
        })
    
    def calculate_profile_completeness(self) -> float:
        """Calculate profile completeness score"""
        completeness_factors = {
            'learning_style': 0.1,
            'learning_pace': 0.05,
            'complexity_preference': 0.05,
            'preferred_explanation_style': 0.05,
            'preferred_code_style': 0.1,
            'learning_patterns': 0.15,
            'interaction_patterns': 0.15,
            'domain_expertise': 0.15,
            'communication_preferences': 0.1,
            'successful_patterns': 0.1
        }
        
        score = 0.0
        for factor, weight in completeness_factors.items():
            value = getattr(self, factor)
            if value and (isinstance(value, str) or (isinstance(value, (dict, list)) and value)):
                score += weight
        
        # Add session-based completeness
        if self.total_learning_sessions > 0:
            session_factor = min(1.0, self.total_learning_sessions / 10) * 0.1
            score += session_factor
        
        self.profile_completeness = min(1.0, score)
        return self.profile_completeness
    
    def get_learning_recommendations(self) -> List[str]:
        """Get personalized learning recommendations"""
        recommendations = []
        
        # Learning pace recommendations
        if self.learning_pace == 'slow':
            recommendations.append("Provide more detailed explanations and examples")
        elif self.learning_pace == 'fast':
            recommendations.append("Provide concise explanations and focus on key points")
        
        # Complexity recommendations
        if self.complexity_preference == 'simple':
            recommendations.append("Use simpler examples and avoid advanced concepts initially")
        elif self.complexity_preference == 'complex':
            recommendations.append("Include advanced concepts and comprehensive examples")
        
        # Learning style recommendations
        if self.learning_style == LearningStyleType.VISUAL.value:
            recommendations.append("Use visual aids, diagrams, and code examples")
        elif self.learning_style == LearningStyleType.KINESTHETIC.value:
            recommendations.append("Provide hands-on exercises and interactive examples")
        
        # Effectiveness-based recommendations
        if self.average_learning_effectiveness and self.average_learning_effectiveness < 0.6:
            recommendations.append("Adjust teaching approach based on user feedback")
            recommendations.append("Provide more reinforcement and practice opportunities")
        
        return recommendations
    
    def get_adaptation_settings(self) -> Dict[str, Any]:
        """Get current adaptation settings"""
        return {
            'adaptation_aggressiveness': self.adaptation_aggressiveness,
            'pattern_sensitivity': self.pattern_sensitivity,
            'feedback_responsiveness': self.feedback_responsiveness,
            'learning_style': self.learning_style,
            'learning_pace': self.learning_pace,
            'complexity_preference': self.complexity_preference,
            'error_tolerance': self.error_tolerance,
            'auto_update_enabled': self.auto_update_enabled
        }
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get a summary of the user profile"""
        return {
            'user_id': self.user_id,
            'learning_style': self.learning_style,
            'learning_pace': self.learning_pace,
            'complexity_preference': self.complexity_preference,
            'preferred_explanation_style': self.preferred_explanation_style,
            'total_learning_sessions': self.total_learning_sessions,
            'total_patterns_learned': self.total_patterns_learned,
            'average_learning_effectiveness': self.average_learning_effectiveness,
            'knowledge_retention_rate': self.knowledge_retention_rate,
            'profile_completeness': self.profile_completeness,
            'adaptation_aggressiveness': self.adaptation_aggressiveness,
            'pattern_sensitivity': self.pattern_sensitivity,
            'feedback_responsiveness': self.feedback_responsiveness,
            'error_tolerance': self.error_tolerance,
            'domain_expertise_count': len(self.domain_expertise) if self.domain_expertise else 0,
            'successful_patterns_count': len(self.successful_patterns) if self.successful_patterns else 0,
            'failed_patterns_count': len(self.failed_patterns) if self.failed_patterns else 0,
            'days_since_last_interaction': (
                (datetime.now(timezone.utc) - self.last_interaction_at).days
                if self.last_interaction_at else None
            ),
            'profile_age_days': (datetime.now(timezone.utc) - self.created_at).days
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'user_id': self.user_id,
            'learning_style': self.learning_style,
            'learning_pace': self.learning_pace,
            'complexity_preference': self.complexity_preference,
            'preferred_explanation_style': self.preferred_explanation_style,
            'preferred_code_style': self.preferred_code_style,
            'preferred_tools': self.preferred_tools,
            'learning_patterns': self.learning_patterns,
            'interaction_patterns': self.interaction_patterns,
            'feedback_patterns': self.feedback_patterns,
            'total_learning_sessions': self.total_learning_sessions,
            'total_patterns_learned': self.total_patterns_learned,
            'average_learning_effectiveness': self.average_learning_effectiveness,
            'knowledge_retention_rate': self.knowledge_retention_rate,
            'adaptation_aggressiveness': self.adaptation_aggressiveness,
            'pattern_sensitivity': self.pattern_sensitivity,
            'feedback_responsiveness': self.feedback_responsiveness,
            'domain_expertise': self.domain_expertise,
            'communication_preferences': self.communication_preferences,
            'error_tolerance': self.error_tolerance,
            'successful_patterns': self.successful_patterns,
            'failed_patterns': self.failed_patterns,
            'improvement_areas': self.improvement_areas,
            'profile_completeness': self.profile_completeness,
            'last_updated_by_session': str(self.last_updated_by_session) if self.last_updated_by_session else None,
            'auto_update_enabled': self.auto_update_enabled,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_interaction_at': self.last_interaction_at.isoformat() if self.last_interaction_at else None
        }
    
    def __repr__(self) -> str:
        return (
            f"<UserLearningProfile(user_id={self.user_id}, "
            f"style={self.learning_style}, completeness={self.profile_completeness:.2f})>"
        )