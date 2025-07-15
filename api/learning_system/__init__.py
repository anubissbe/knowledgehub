"""Advanced Learning System for AI Development Assistant

This system extends KnowledgeHub's memory capabilities with:
- Pattern learning from user interactions
- Feedback processing and adaptation
- Success tracking for decisions and recommendations
- Cross-session learning capabilities
- Behavioral adaptation based on learned patterns
"""

from .core.learning_engine import LearningEngine
from .services.pattern_learning import PatternLearningService
from .services.feedback_processor import FeedbackProcessor
from .services.success_tracker import SuccessTracker
# Removed SQLAlchemy model imports to avoid Pydantic issues
from .models.learning_pattern import PatternType
from .models.decision_outcome import OutcomeType
from .models.user_feedback import FeedbackType

__all__ = [
    'LearningEngine',
    'PatternLearningService',
    'FeedbackProcessor',
    'SuccessTracker',
    'PatternType',
    'OutcomeType',
    'FeedbackType'
]