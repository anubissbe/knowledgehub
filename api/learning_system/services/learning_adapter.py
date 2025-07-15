"""Learning Adapter Service

This service adapts system behavior based on learned patterns and feedback,
providing real-time adjustments to improve response quality.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, TYPE_CHECKING
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from collections import defaultdict
import asyncio
import json

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..models.learning_pattern import LearningPattern, PatternType
    from ..models.user_feedback import UserFeedback, FeedbackType
    from ..models.decision_outcome import DecisionOutcome
else:
    LearningPattern = Any
    PatternType = Any
    UserFeedback = Any
    FeedbackType = Any
    DecisionOutcome = Any
from .pattern_learning import PatternLearningService
from .correction_processor import CorrectionProcessor, CorrectionResult
from ...memory_system.models.memory import Memory, MemoryType
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


from enum import Enum

class AdaptationType(str, Enum):
    """Types of adaptations that can be applied"""
    RESPONSE_STYLE = "response_style"
    CONTENT_DEPTH = "content_depth"
    ERROR_HANDLING = "error_handling"
    CODE_GENERATION = "code_generation"
    EXPLANATION_LEVEL = "explanation_level"
    FORMATTING = "formatting"
    TOOL_SELECTION = "tool_selection"
    CONTEXT_USAGE = "context_usage"


class AdaptationRule(BaseModel):
    """A rule for adapting behavior"""
    rule_id: UUID = Field(default_factory=uuid4)
    adaptation_type: AdaptationType
    trigger_patterns: List[str] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    adjustments: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    success_rate: float = Field(0.0, ge=0.0, le=1.0)
    usage_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None


class AdaptationContext(BaseModel):
    """Context for applying adaptations"""
    user_input: str
    session_id: UUID
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    current_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    recent_feedback: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AppliedAdaptation(BaseModel):
    """Record of an applied adaptation"""
    adaptation_id: UUID = Field(default_factory=uuid4)
    rule_id: UUID
    adaptation_type: AdaptationType
    adjustments_applied: Dict[str, Any]
    context_snapshot: Dict[str, Any]
    confidence: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LearningAdapter:
    """Service for adapting behavior based on learned patterns"""
    
    def __init__(self, db: Session):
        """Initialize the learning adapter"""
        self.db = db
        self.pattern_learner = PatternLearningService(db)
        self.correction_processor = CorrectionProcessor(db)
        
        # Configuration
        self.min_confidence_threshold = 0.6
        self.max_adaptations_per_request = 5
        self.adaptation_cache_ttl = 3600  # 1 hour
        self.pattern_match_threshold = 0.7
        
        # In-memory caches
        self._adaptation_rules = self._load_default_rules()
        self._user_profiles = {}
        self._active_adaptations = {}
        
        # Start background tasks
        asyncio.create_task(self._periodic_rule_optimization())
    
    async def adapt_response(
        self,
        context: AdaptationContext,
        initial_response: str
    ) -> Tuple[str, List[AppliedAdaptation]]:
        """Adapt a response based on learned patterns and user preferences
        
        Args:
            context: Context for adaptation
            initial_response: The initial system response
            
        Returns:
            Tuple of (adapted_response, applied_adaptations)
        """
        try:
            # Get user profile
            user_profile = await self._get_or_create_user_profile(context.session_id)
            
            # Find applicable adaptation rules
            applicable_rules = await self._find_applicable_rules(context, user_profile)
            
            # Sort by confidence and success rate
            applicable_rules.sort(
                key=lambda r: (r.confidence * r.success_rate, r.usage_count),
                reverse=True
            )
            
            # Apply adaptations
            adapted_response = initial_response
            applied_adaptations = []
            
            for rule in applicable_rules[:self.max_adaptations_per_request]:
                # Apply the adaptation
                result = await self._apply_adaptation_rule(
                    adapted_response,
                    rule,
                    context
                )
                
                if result['success']:
                    adapted_response = result['adapted_text']
                    
                    # Record the adaptation
                    adaptation = AppliedAdaptation(
                        rule_id=rule.rule_id,
                        adaptation_type=rule.adaptation_type,
                        adjustments_applied=result['adjustments'],
                        context_snapshot={
                            'user_input': context.user_input,
                            'pattern_matches': result.get('pattern_matches', [])
                        },
                        confidence=rule.confidence
                    )
                    
                    applied_adaptations.append(adaptation)
                    
                    # Update rule usage
                    rule.usage_count += 1
                    rule.last_used = datetime.now(timezone.utc)
            
            # Apply learned corrections
            if context.recent_feedback:
                correction_result = await self.correction_processor.apply_learned_corrections(
                    adapted_response,
                    {'session_id': str(context.session_id)}
                )
                
                if correction_result.corrections_applied:
                    adapted_response = correction_result.corrected_text
                    
                    # Add correction adaptation record
                    adaptation = AppliedAdaptation(
                        rule_id=uuid4(),  # Synthetic rule ID
                        adaptation_type=AdaptationType.ERROR_HANDLING,
                        adjustments_applied={
                            'corrections': correction_result.corrections_applied
                        },
                        context_snapshot={'correction_confidence': correction_result.confidence},
                        confidence=correction_result.confidence
                    )
                    applied_adaptations.append(adaptation)
            
            # Store adaptations for tracking
            await self._store_adaptations(context.session_id, applied_adaptations)
            
            # Update user profile
            await self._update_user_profile(context.session_id, applied_adaptations)
            
            return adapted_response, applied_adaptations
            
        except Exception as e:
            logger.error(f"Error adapting response: {e}")
            return initial_response, []
    
    async def learn_from_adaptation_outcome(
        self,
        session_id: UUID,
        adaptation_id: UUID,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from the outcome of an applied adaptation
        
        Args:
            session_id: Session ID
            adaptation_id: ID of the applied adaptation
            outcome: Outcome data (success, user_satisfaction, etc.)
            
        Returns:
            Learning result
        """
        try:
            # Find the adaptation
            adaptations = self._active_adaptations.get(str(session_id), [])
            adaptation = next(
                (a for a in adaptations if a.adaptation_id == adaptation_id),
                None
            )
            
            if not adaptation:
                return {"error": "Adaptation not found"}
            
            # Find the rule
            rule = next(
                (r for r in self._adaptation_rules if r.rule_id == adaptation.rule_id),
                None
            )
            
            if not rule:
                return {"error": "Rule not found"}
            
            # Update rule success rate
            success = outcome.get('success', False)
            satisfaction = outcome.get('user_satisfaction', 0.5)
            
            # Weighted average update
            alpha = 0.1  # Learning rate
            new_success_rate = (1 - alpha) * rule.success_rate + alpha * (
                1.0 if success else 0.0
            )
            rule.success_rate = new_success_rate
            
            # Update confidence based on satisfaction
            if satisfaction > 0.7:
                rule.confidence = min(1.0, rule.confidence + 0.05)
            elif satisfaction < 0.3:
                rule.confidence = max(0.0, rule.confidence - 0.1)
            
            # Store learning outcome
            await self._store_adaptation_outcome(adaptation, outcome)
            
            # If rule is performing poorly, consider disabling it
            if rule.success_rate < 0.3 and rule.usage_count > 10:
                rule.confidence = 0.0  # Effectively disable
                logger.info(f"Disabled poorly performing rule: {rule.rule_id}")
            
            return {
                "rule_id": str(rule.rule_id),
                "new_success_rate": rule.success_rate,
                "new_confidence": rule.confidence,
                "rule_active": rule.confidence >= self.min_confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error learning from adaptation outcome: {e}")
            return {"error": str(e)}
    
    async def get_adaptation_suggestions(
        self,
        context: AdaptationContext
    ) -> List[Dict[str, Any]]:
        """Get suggestions for potential adaptations
        
        Args:
            context: Current context
            
        Returns:
            List of adaptation suggestions
        """
        try:
            suggestions = []
            
            # Analyze recent patterns
            pattern_analysis = await self._analyze_recent_patterns(context)
            
            # Check for response style improvements
            if pattern_analysis.get('frequent_corrections', {}).get('style'):
                suggestions.append({
                    "type": AdaptationType.RESPONSE_STYLE,
                    "description": "Adjust response style based on user corrections",
                    "confidence": 0.8,
                    "examples": pattern_analysis['frequent_corrections']['style'][:3]
                })
            
            # Check for content depth preferences
            feedback_analysis = await self._analyze_feedback_preferences(context)
            if feedback_analysis.get('prefers_detailed'):
                suggestions.append({
                    "type": AdaptationType.CONTENT_DEPTH,
                    "description": "Provide more detailed explanations",
                    "confidence": 0.7,
                    "reason": "User frequently asks for more details"
                })
            
            # Check for code generation improvements
            if pattern_analysis.get('code_patterns'):
                suggestions.append({
                    "type": AdaptationType.CODE_GENERATION,
                    "description": "Adjust code generation patterns",
                    "confidence": 0.85,
                    "patterns": pattern_analysis['code_patterns'][:5]
                })
            
            # Check for error handling patterns
            error_patterns = await self._analyze_error_patterns(context)
            if error_patterns:
                suggestions.append({
                    "type": AdaptationType.ERROR_HANDLING,
                    "description": "Improve error handling based on past issues",
                    "confidence": 0.9,
                    "common_errors": error_patterns[:3]
                })
            
            # Sort by confidence
            suggestions.sort(key=lambda s: s['confidence'], reverse=True)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting adaptation suggestions: {e}")
            return []
    
    async def create_custom_rule(
        self,
        rule_definition: Dict[str, Any]
    ) -> AdaptationRule:
        """Create a custom adaptation rule
        
        Args:
            rule_definition: Rule definition
            
        Returns:
            Created adaptation rule
        """
        try:
            rule = AdaptationRule(
                adaptation_type=rule_definition['type'],
                trigger_patterns=rule_definition.get('triggers', []),
                conditions=rule_definition.get('conditions', {}),
                adjustments=rule_definition.get('adjustments', {}),
                confidence=rule_definition.get('initial_confidence', 0.5)
            )
            
            # Validate rule
            if not self._validate_rule(rule):
                raise ValueError("Invalid rule definition")
            
            # Add to rules
            self._adaptation_rules.append(rule)
            
            # Cache rule
            await self._cache_rule(rule)
            
            return rule
            
        except Exception as e:
            logger.error(f"Error creating custom rule: {e}")
            raise
    
    async def get_adaptation_metrics(
        self,
        time_period: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Get metrics about adaptation effectiveness
        
        Args:
            time_period: Time period for metrics
            
        Returns:
            Adaptation metrics
        """
        try:
            metrics = {
                "total_adaptations": 0,
                "adaptation_types": defaultdict(int),
                "average_confidence": 0.0,
                "success_rate": 0.0,
                "most_effective_rules": [],
                "user_satisfaction": 0.0,
                "improvement_areas": []
            }
            
            # Get recent adaptations from cache
            recent_adaptations = await self._get_recent_adaptations(time_period)
            
            if not recent_adaptations:
                return metrics
            
            metrics["total_adaptations"] = len(recent_adaptations)
            
            # Analyze by type
            confidences = []
            for adaptation in recent_adaptations:
                metrics["adaptation_types"][adaptation['type']] += 1
                confidences.append(adaptation['confidence'])
            
            metrics["average_confidence"] = sum(confidences) / len(confidences)
            
            # Get rule effectiveness
            rule_stats = defaultdict(lambda: {'successes': 0, 'total': 0})
            
            for rule in self._adaptation_rules:
                if rule.usage_count > 0:
                    rule_stats[str(rule.rule_id)] = {
                        'rule_id': str(rule.rule_id),
                        'type': rule.adaptation_type,
                        'success_rate': rule.success_rate,
                        'usage_count': rule.usage_count,
                        'confidence': rule.confidence
                    }
            
            # Sort by effectiveness
            sorted_rules = sorted(
                rule_stats.values(),
                key=lambda r: r['success_rate'] * r['confidence'],
                reverse=True
            )
            
            metrics["most_effective_rules"] = sorted_rules[:5]
            
            # Calculate overall success rate
            if sorted_rules:
                total_usage = sum(r['usage_count'] for r in sorted_rules)
                weighted_success = sum(
                    r['success_rate'] * r['usage_count'] 
                    for r in sorted_rules
                )
                metrics["success_rate"] = weighted_success / total_usage if total_usage > 0 else 0.0
            
            # Identify improvement areas
            low_performing = [
                r for r in sorted_rules 
                if r['success_rate'] < 0.5 and r['usage_count'] > 5
            ]
            
            metrics["improvement_areas"] = [
                {
                    "type": r['type'],
                    "current_success_rate": r['success_rate'],
                    "recommendation": self._get_improvement_recommendation(r)
                }
                for r in low_performing[:3]
            ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting adaptation metrics: {e}")
            return {}
    
    # Private helper methods
    
    def _load_default_rules(self) -> List[AdaptationRule]:
        """Load default adaptation rules"""
        return [
            # Response style rules
            AdaptationRule(
                adaptation_type=AdaptationType.RESPONSE_STYLE,
                trigger_patterns=["brief", "concise", "short"],
                conditions={"input_contains": ["explain", "describe", "how"]},
                adjustments={"style": "concise", "max_length": 200},
                confidence=0.8
            ),
            AdaptationRule(
                adaptation_type=AdaptationType.RESPONSE_STYLE,
                trigger_patterns=["detailed", "comprehensive", "thorough"],
                conditions={"input_contains": ["explain", "describe", "how"]},
                adjustments={"style": "detailed", "include_examples": True},
                confidence=0.8
            ),
            
            # Code generation rules
            AdaptationRule(
                adaptation_type=AdaptationType.CODE_GENERATION,
                trigger_patterns=["async", "asynchronous", "await"],
                conditions={"language": "python", "context": "code_generation"},
                adjustments={"prefer_async": True, "include_error_handling": True},
                confidence=0.9
            ),
            AdaptationRule(
                adaptation_type=AdaptationType.CODE_GENERATION,
                trigger_patterns=["type", "typed", "typing"],
                conditions={"language": "python"},
                adjustments={"include_type_hints": True, "strict_typing": True},
                confidence=0.85
            ),
            
            # Error handling rules
            AdaptationRule(
                adaptation_type=AdaptationType.ERROR_HANDLING,
                trigger_patterns=["error", "exception", "handle"],
                conditions={"context": "code_generation"},
                adjustments={"comprehensive_error_handling": True, "include_logging": True},
                confidence=0.9
            ),
            
            # Explanation level rules
            AdaptationRule(
                adaptation_type=AdaptationType.EXPLANATION_LEVEL,
                trigger_patterns=["beginner", "new", "learning"],
                conditions={},
                adjustments={"explanation_level": "beginner", "include_analogies": True},
                confidence=0.75
            ),
            AdaptationRule(
                adaptation_type=AdaptationType.EXPLANATION_LEVEL,
                trigger_patterns=["expert", "advanced", "professional"],
                conditions={},
                adjustments={"explanation_level": "expert", "technical_depth": "high"},
                confidence=0.75
            )
        ]
    
    async def _get_or_create_user_profile(
        self,
        session_id: UUID
    ) -> Dict[str, Any]:
        """Get or create user profile"""
        session_key = str(session_id)
        
        if session_key in self._user_profiles:
            return self._user_profiles[session_key]
        
        # Check cache
        if redis_client.client:
            try:
                cached = await redis_client.get(f"user_profile:{session_key}")
                if cached:
                    self._user_profiles[session_key] = cached
                    return cached
            except Exception as e:
                logger.warning(f"Cache error: {e}")
        
        # Create new profile
        profile = {
            "session_id": session_key,
            "preferences": {
                "response_style": "balanced",
                "explanation_level": "intermediate",
                "code_style": "modern",
                "error_detail": "moderate"
            },
            "learned_patterns": [],
            "feedback_history": [],
            "adaptation_history": [],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._user_profiles[session_key] = profile
        
        # Cache profile
        if redis_client.client:
            try:
                await redis_client.set(
                    f"user_profile:{session_key}",
                    profile,
                    expiry=86400  # 24 hours
                )
            except Exception as e:
                logger.warning(f"Cache error: {e}")
        
        return profile
    
    async def _find_applicable_rules(
        self,
        context: AdaptationContext,
        user_profile: Dict[str, Any]
    ) -> List[AdaptationRule]:
        """Find rules that apply to the current context"""
        applicable = []
        
        for rule in self._adaptation_rules:
            # Check confidence threshold
            if rule.confidence < self.min_confidence_threshold:
                continue
            
            # Check trigger patterns
            if rule.trigger_patterns:
                input_lower = context.user_input.lower()
                if not any(trigger in input_lower for trigger in rule.trigger_patterns):
                    continue
            
            # Check conditions
            if rule.conditions:
                if not self._check_conditions(rule.conditions, context, user_profile):
                    continue
            
            # Check pattern matching
            if context.current_patterns:
                pattern_match = self._check_pattern_match(rule, context.current_patterns)
                if pattern_match < self.pattern_match_threshold:
                    continue
            
            applicable.append(rule)
        
        return applicable
    
    def _check_conditions(
        self,
        conditions: Dict[str, Any],
        context: AdaptationContext,
        user_profile: Dict[str, Any]
    ) -> bool:
        """Check if conditions are met"""
        for key, value in conditions.items():
            if key == "input_contains":
                input_lower = context.user_input.lower()
                if not any(term in input_lower for term in value):
                    return False
            
            elif key == "language":
                # Check if language is mentioned or detected
                if value.lower() not in context.user_input.lower():
                    # Could add more sophisticated language detection
                    return False
            
            elif key == "context":
                if context.metadata.get("context_type") != value:
                    return False
            
            elif key == "user_preference":
                if user_profile["preferences"].get(value[0]) != value[1]:
                    return False
        
        return True
    
    def _check_pattern_match(
        self,
        rule: AdaptationRule,
        patterns: List[LearningPattern]
    ) -> float:
        """Check how well patterns match the rule"""
        if not patterns:
            return 0.0
        
        matches = 0
        for pattern in patterns:
            # Check if pattern type aligns with adaptation type
            if self._pattern_aligns_with_adaptation(pattern, rule.adaptation_type):
                matches += pattern.confidence_score
        
        return matches / len(patterns)
    
    def _pattern_aligns_with_adaptation(
        self,
        pattern: LearningPattern,
        adaptation_type: AdaptationType
    ) -> bool:
        """Check if pattern aligns with adaptation type"""
        alignments = {
            AdaptationType.CODE_GENERATION: [
                PatternType.CODE_GENERATION,
                PatternType.LANGUAGE_PREFERENCE
            ],
            AdaptationType.ERROR_HANDLING: [
                PatternType.ERROR_PATTERN,
                PatternType.DEBUGGING_APPROACH
            ],
            AdaptationType.RESPONSE_STYLE: [
                PatternType.COMMUNICATION_STYLE,
                PatternType.INTERACTION_PATTERN
            ],
            AdaptationType.EXPLANATION_LEVEL: [
                PatternType.TASK_APPROACH,
                PatternType.LEARNING_STYLE
            ]
        }
        
        return pattern.pattern_type in alignments.get(adaptation_type, [])
    
    async def _apply_adaptation_rule(
        self,
        text: str,
        rule: AdaptationRule,
        context: AdaptationContext
    ) -> Dict[str, Any]:
        """Apply an adaptation rule to text"""
        try:
            adjusted_text = text
            adjustments_made = {}
            
            # Apply adjustments based on type
            if rule.adaptation_type == AdaptationType.RESPONSE_STYLE:
                adjusted_text = self._adjust_response_style(
                    text,
                    rule.adjustments
                )
                adjustments_made = {"style": rule.adjustments.get("style")}
            
            elif rule.adaptation_type == AdaptationType.CONTENT_DEPTH:
                adjusted_text = self._adjust_content_depth(
                    text,
                    rule.adjustments
                )
                adjustments_made = {"depth": rule.adjustments.get("depth", "moderate")}
            
            elif rule.adaptation_type == AdaptationType.CODE_GENERATION:
                adjusted_text = self._adjust_code_generation(
                    text,
                    rule.adjustments
                )
                adjustments_made = rule.adjustments
            
            elif rule.adaptation_type == AdaptationType.EXPLANATION_LEVEL:
                adjusted_text = self._adjust_explanation_level(
                    text,
                    rule.adjustments
                )
                adjustments_made = {"level": rule.adjustments.get("explanation_level")}
            
            return {
                "success": adjusted_text != text,
                "adapted_text": adjusted_text,
                "adjustments": adjustments_made,
                "pattern_matches": []  # Could include actual matches
            }
            
        except Exception as e:
            logger.error(f"Error applying adaptation rule: {e}")
            return {
                "success": False,
                "adapted_text": text,
                "adjustments": {},
                "error": str(e)
            }
    
    def _adjust_response_style(
        self,
        text: str,
        adjustments: Dict[str, Any]
    ) -> str:
        """Adjust response style"""
        style = adjustments.get("style", "balanced")
        
        if style == "concise":
            # Simplify and shorten
            max_length = adjustments.get("max_length", 200)
            if len(text) > max_length:
                # Find a good cut point
                sentences = text.split(". ")
                adjusted = ""
                for sentence in sentences:
                    if len(adjusted) + len(sentence) < max_length:
                        adjusted += sentence + ". "
                    else:
                        break
                return adjusted.strip()
        
        elif style == "detailed":
            # Add examples if not present
            if adjustments.get("include_examples") and "example" not in text.lower():
                text += "\n\nFor example, this could be applied in practice by..."
        
        return text
    
    def _adjust_content_depth(
        self,
        text: str,
        adjustments: Dict[str, Any]
    ) -> str:
        """Adjust content depth"""
        depth = adjustments.get("depth", "moderate")
        
        if depth == "shallow":
            # Remove technical details
            # This is a simplified implementation
            lines = text.split("\n")
            adjusted = []
            for line in lines:
                if not any(term in line.lower() for term in ["technically", "specifically", "internally"]):
                    adjusted.append(line)
            return "\n".join(adjusted)
        
        elif depth == "deep":
            # Add technical context
            if "```" in text and adjustments.get("add_explanations"):
                # Add explanations after code blocks
                parts = text.split("```")
                adjusted_parts = []
                for i, part in enumerate(parts):
                    adjusted_parts.append(part)
                    if i % 2 == 1:  # Code block
                        adjusted_parts.append("\n\nThis code works by...")
                return "```".join(adjusted_parts)
        
        return text
    
    def _adjust_code_generation(
        self,
        text: str,
        adjustments: Dict[str, Any]
    ) -> str:
        """Adjust code generation patterns"""
        if "```" not in text:
            return text
        
        # Extract code blocks
        import re
        code_pattern = r'```(\w*)\n(.*?)```'
        
        def adjust_code(match):
            language = match.group(1)
            code = match.group(2)
            
            if language == "python" and adjustments.get("prefer_async"):
                # Add async to functions
                code = re.sub(r'def (\w+)', r'async def \1', code)
                # Add await to calls
                code = re.sub(r'(\w+)\(', r'await \1(', code)
            
            if adjustments.get("include_type_hints") and language == "python":
                # Add basic type hints
                code = re.sub(r'def (\w+)\((.*?)\):', r'def \1(\2) -> None:', code)
            
            if adjustments.get("include_error_handling"):
                # Wrap in try-except
                if "try:" not in code:
                    code = f"try:\n    {code.replace(chr(10), chr(10) + '    ')}\nexcept Exception as e:\n    logger.error(f'Error: {{e}}')"
            
            return f"```{language}\n{code}```"
        
        return re.sub(code_pattern, adjust_code, text, flags=re.DOTALL)
    
    def _adjust_explanation_level(
        self,
        text: str,
        adjustments: Dict[str, Any]
    ) -> str:
        """Adjust explanation level"""
        level = adjustments.get("explanation_level", "intermediate")
        
        if level == "beginner":
            # Add clarifications
            technical_terms = {
                "API": "API (Application Programming Interface - a way for programs to talk to each other)",
                "async": "async (asynchronous - allowing multiple operations at once)",
                "callback": "callback (a function that runs after something else finishes)"
            }
            
            adjusted = text
            for term, explanation in technical_terms.items():
                if term in adjusted and explanation not in adjusted:
                    adjusted = adjusted.replace(term, explanation, 1)
            
            return adjusted
        
        elif level == "expert":
            # Remove basic explanations
            # This is simplified - in practice would be more sophisticated
            adjusted = re.sub(r'\(.*?basic.*?\)', '', text)
            adjusted = re.sub(r'\(.*?simple.*?\)', '', adjusted)
            return adjusted
        
        return text
    
    async def _store_adaptations(
        self,
        session_id: UUID,
        adaptations: List[AppliedAdaptation]
    ):
        """Store applied adaptations"""
        session_key = str(session_id)
        
        if session_key not in self._active_adaptations:
            self._active_adaptations[session_key] = []
        
        self._active_adaptations[session_key].extend(adaptations)
        
        # Limit storage
        self._active_adaptations[session_key] = self._active_adaptations[session_key][-50:]
        
        # Cache adaptations
        if redis_client.client:
            try:
                await redis_client.set(
                    f"adaptations:{session_key}",
                    [a.dict() for a in adaptations],
                    expiry=self.adaptation_cache_ttl
                )
            except Exception as e:
                logger.warning(f"Cache error: {e}")
    
    async def _update_user_profile(
        self,
        session_id: UUID,
        adaptations: List[AppliedAdaptation]
    ):
        """Update user profile based on adaptations"""
        profile = await self._get_or_create_user_profile(session_id)
        
        # Update adaptation history
        profile["adaptation_history"].extend([
            {
                "type": a.adaptation_type,
                "timestamp": a.timestamp.isoformat(),
                "confidence": a.confidence
            }
            for a in adaptations
        ])
        
        # Keep recent history
        profile["adaptation_history"] = profile["adaptation_history"][-100:]
        
        # Update preferences based on successful adaptations
        # This is simplified - would be more sophisticated in practice
        adaptation_counts = defaultdict(int)
        for adaptation in profile["adaptation_history"]:
            adaptation_counts[adaptation["type"]] += 1
        
        # Update preferences based on most common adaptations
        if adaptation_counts:
            most_common = max(adaptation_counts.items(), key=lambda x: x[1])
            if most_common[0] == AdaptationType.RESPONSE_STYLE:
                profile["preferences"]["response_style"] = "detailed"
            elif most_common[0] == AdaptationType.CODE_GENERATION:
                profile["preferences"]["code_style"] = "async_first"
    
    def _validate_rule(self, rule: AdaptationRule) -> bool:
        """Validate an adaptation rule"""
        # Check required fields
        if not rule.adaptation_type or not rule.adjustments:
            return False
        
        # Check adjustment validity
        valid_adjustments = {
            AdaptationType.RESPONSE_STYLE: ["style", "max_length", "tone"],
            AdaptationType.CONTENT_DEPTH: ["depth", "include_examples"],
            AdaptationType.CODE_GENERATION: ["prefer_async", "include_type_hints"],
            AdaptationType.EXPLANATION_LEVEL: ["explanation_level", "technical_depth"]
        }
        
        allowed_adjustments = valid_adjustments.get(rule.adaptation_type, [])
        for adjustment in rule.adjustments:
            if adjustment not in allowed_adjustments:
                logger.warning(f"Invalid adjustment '{adjustment}' for {rule.adaptation_type}")
                return False
        
        return True
    
    async def _cache_rule(self, rule: AdaptationRule):
        """Cache an adaptation rule"""
        if redis_client.client:
            try:
                await redis_client.set(
                    f"adaptation_rule:{rule.rule_id}",
                    rule.dict(),
                    expiry=86400  # 24 hours
                )
            except Exception as e:
                logger.warning(f"Cache error: {e}")
    
    async def _get_recent_adaptations(
        self,
        time_period: timedelta
    ) -> List[Dict[str, Any]]:
        """Get recent adaptations from all sessions"""
        recent = []
        cutoff = datetime.now(timezone.utc) - time_period
        
        for session_key, adaptations in self._active_adaptations.items():
            for adaptation in adaptations:
                if adaptation.timestamp >= cutoff:
                    recent.append({
                        "session": session_key,
                        "type": adaptation.adaptation_type,
                        "confidence": adaptation.confidence,
                        "timestamp": adaptation.timestamp
                    })
        
        return recent
    
    def _get_improvement_recommendation(
        self,
        rule_stats: Dict[str, Any]
    ) -> str:
        """Get improvement recommendation for a rule"""
        if rule_stats["success_rate"] < 0.3:
            return "Consider revising rule conditions or adjustments"
        elif rule_stats["success_rate"] < 0.5:
            return "Fine-tune rule parameters based on user feedback"
        else:
            return "Monitor performance and gather more data"
    
    async def _store_adaptation_outcome(
        self,
        adaptation: AppliedAdaptation,
        outcome: Dict[str, Any]
    ):
        """Store adaptation outcome for analysis"""
        # In production, this would store to database
        # For now, just log
        logger.info(f"Adaptation outcome: {adaptation.adaptation_id} - {outcome}")
    
    async def _analyze_recent_patterns(
        self,
        context: AdaptationContext
    ) -> Dict[str, Any]:
        """Analyze recent patterns for adaptation opportunities"""
        analysis = {
            "frequent_corrections": defaultdict(list),
            "code_patterns": [],
            "interaction_patterns": []
        }
        
        # Analyze patterns
        for pattern in context.current_patterns[:20]:  # Recent patterns
            if pattern.pattern_type == PatternType.CORRECTION:
                correction_type = pattern.pattern_data.get("correction_type", "general")
                analysis["frequent_corrections"][correction_type].append(
                    pattern.pattern_data
                )
            
            elif pattern.pattern_type == PatternType.CODE_GENERATION:
                analysis["code_patterns"].append({
                    "language": pattern.pattern_data.get("language"),
                    "patterns": pattern.pattern_data.get("patterns", [])
                })
        
        return analysis
    
    async def _analyze_feedback_preferences(
        self,
        context: AdaptationContext
    ) -> Dict[str, Any]:
        """Analyze feedback to determine preferences"""
        preferences = {
            "prefers_detailed": False,
            "prefers_examples": False,
            "technical_level": "intermediate"
        }
        
        # Analyze recent feedback
        positive_feedback = [
            f for f in context.recent_feedback
            if f.feedback_type == FeedbackType.RATING and f.get_rating() >= 4
        ]
        
        negative_feedback = [
            f for f in context.recent_feedback
            if f.feedback_type == FeedbackType.RATING and f.get_rating() <= 2
        ]
        
        # Simple heuristics
        if len(positive_feedback) > len(negative_feedback):
            # Current style is working
            pass
        else:
            # Need adjustments
            preferences["prefers_detailed"] = True
        
        return preferences
    
    async def _analyze_error_patterns(
        self,
        context: AdaptationContext
    ) -> List[Dict[str, Any]]:
        """Analyze error patterns"""
        error_patterns = []
        
        # Look for error-related patterns
        for pattern in context.current_patterns:
            if pattern.pattern_type == PatternType.ERROR_PATTERN:
                error_patterns.append({
                    "error_type": pattern.pattern_data.get("error_type"),
                    "frequency": pattern.pattern_data.get("frequency", 1),
                    "context": pattern.pattern_data.get("context")
                })
        
        # Sort by frequency
        error_patterns.sort(key=lambda p: p["frequency"], reverse=True)
        
        return error_patterns
    
    async def _periodic_rule_optimization(self):
        """Periodically optimize adaptation rules"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Analyze rule performance
                for rule in self._adaptation_rules:
                    if rule.usage_count > 100:
                        # Decay old rules
                        if rule.last_used:
                            days_old = (
                                datetime.now(timezone.utc) - rule.last_used
                            ).days
                            if days_old > 30:
                                rule.confidence *= 0.95  # Decay confidence
                        
                        # Boost high-performing rules
                        if rule.success_rate > 0.8 and rule.confidence < 0.9:
                            rule.confidence = min(0.95, rule.confidence + 0.05)
                
                # Remove very low confidence rules
                self._adaptation_rules = [
                    r for r in self._adaptation_rules
                    if r.confidence > 0.1 or r.usage_count < 10
                ]
                
                logger.info(f"Rule optimization complete. Active rules: {len(self._adaptation_rules)}")
                
            except Exception as e:
                logger.error(f"Error in rule optimization: {e}")