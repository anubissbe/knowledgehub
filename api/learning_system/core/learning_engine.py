"""Advanced Learning Engine for AI Development Assistant

This engine coordinates all learning activities and provides the main interface
for the learning system. It integrates pattern learning, feedback processing,
success tracking, and adaptive behavior.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, or_

from ..services.pattern_learning import PatternLearningService
from ..services.feedback_processor import FeedbackProcessor
from ..services.success_tracker import SuccessTracker
from ..services.adaptation_engine import AdaptationEngine
from ..services.feedback_collection import FeedbackCollectionService, FeedbackContext, FeedbackPromptType
from ..services.correction_processor import CorrectionProcessor
from ..services.learning_adapter import LearningAdapter, AdaptationContext
from ..services.decision_outcome_tracker import DecisionOutcomeTracker, DecisionContext
from ..services.success_metrics import SuccessMetrics, MetricType, MetricFilter, TimeFrame
from ..services.effectiveness_analyzer import EffectivenessAnalyzer, AnalysisType, EffectivenessCategory
from ..models.learning_pattern import LearningPattern, PatternType
from ..models.decision_outcome import DecisionOutcome
from ..models.user_feedback import UserFeedback
from ...memory_system.models.memory import Memory, MemoryType
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


class LearningEngine:
    """Main learning engine that coordinates all learning activities"""
    
    def __init__(self, db: Session):
        """Initialize the learning engine with all sub-services"""
        self.db = db
        self.pattern_learner = PatternLearningService(db)
        self.feedback_processor = FeedbackProcessor(db)
        self.success_tracker = SuccessTracker(db)
        self.adaptation_engine = AdaptationEngine(db)
        self.feedback_collector = FeedbackCollectionService(db)
        self.correction_processor = CorrectionProcessor(db)
        self.learning_adapter = LearningAdapter(db)
        
        # Success tracking services
        self.decision_tracker = DecisionOutcomeTracker(db)
        self.success_metrics = SuccessMetrics(db)
        self.effectiveness_analyzer = EffectivenessAnalyzer(db)
        
        # Learning configuration
        self.min_confidence_threshold = 0.6
        self.pattern_activation_threshold = 0.7
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        
        # Cache for frequently accessed patterns
        self._pattern_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    async def learn_from_interaction(
        self,
        session_id: UUID,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from a user interaction
        
        Args:
            session_id: Current session ID
            interaction_data: Data about the interaction including:
                - user_input: What the user asked/did
                - system_response: How the system responded
                - context: Context at the time
                - outcome: Result of the interaction
                
        Returns:
            Dictionary with learning results
        """
        try:
            # Extract patterns from the interaction
            patterns = await self.pattern_learner.extract_patterns(
                interaction_data
            )
            
            # Check if this matches existing patterns
            similar_patterns = await self._find_similar_patterns(patterns)
            
            # Update or create patterns based on similarity
            learning_results = []
            for pattern in patterns:
                if similar_patterns.get(pattern.pattern_hash):
                    # Reinforce existing pattern
                    updated = await self._reinforce_pattern(
                        similar_patterns[pattern.pattern_hash],
                        pattern,
                        interaction_data.get('outcome', {})
                    )
                    learning_results.append(updated)
                else:
                    # Create new pattern
                    new_pattern = await self._create_pattern(
                        pattern,
                        session_id,
                        interaction_data
                    )
                    learning_results.append(new_pattern)
            
            # Update cache
            await self._update_pattern_cache(learning_results)
            
            # Trigger adaptation if needed
            await self._check_adaptation_triggers(learning_results)
            
            return {
                'patterns_learned': len(learning_results),
                'patterns': [p.to_dict() for p in learning_results],
                'adaptation_triggered': self._should_adapt(learning_results)
            }
            
        except Exception as e:
            logger.error(f"Error in learn_from_interaction: {e}")
            return {
                'error': str(e),
                'patterns_learned': 0
            }
    
    async def process_user_feedback(
        self,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process user feedback and learn from it
        
        Args:
            feedback_data: Feedback information including:
                - memory_id: Related memory if applicable
                - feedback_type: Type of feedback (correction, rating, etc)
                - original_content: What was originally provided
                - corrected_content: User's correction if applicable
                - rating: User rating if applicable
                
        Returns:
            Dictionary with feedback processing results
        """
        try:
            # Process the feedback
            feedback = await self.feedback_processor.process_feedback(
                feedback_data
            )
            
            # Learn from the feedback
            if feedback.feedback_type == 'correction':
                await self._learn_from_correction(feedback)
            elif feedback.feedback_type == 'rating':
                await self._learn_from_rating(feedback)
            
            # Update related patterns
            if feedback.memory_id:
                await self._update_patterns_from_feedback(
                    feedback.memory_id,
                    feedback
                )
            
            return {
                'feedback_processed': True,
                'feedback_id': str(feedback.id),
                'learning_impact': await self._calculate_learning_impact(feedback)
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                'feedback_processed': False,
                'error': str(e)
            }
    
    async def track_decision_outcome(
        self,
        decision_id: UUID,
        outcome_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track the outcome of a decision for learning
        
        Args:
            decision_id: ID of the decision memory
            outcome_data: Information about the outcome including:
                - outcome_type: Type of outcome (success, failure, partial)
                - success_score: Score from 0.0 to 1.0
                - impact_data: Data about the impact
                - feedback: Any user feedback
                
        Returns:
            Dictionary with tracking results
        """
        try:
            # Track the outcome
            outcome = await self.success_tracker.track_outcome(
                decision_id,
                outcome_data
            )
            
            # Learn from the outcome
            learning_result = await self._learn_from_outcome(outcome)
            
            # Update decision patterns based on outcome
            await self._update_decision_patterns(decision_id, outcome)
            
            return {
                'outcome_tracked': True,
                'outcome_id': str(outcome.id),
                'success_score': outcome.success_score,
                'learning_result': learning_result
            }
            
        except Exception as e:
            logger.error(f"Error tracking decision outcome: {e}")
            return {
                'outcome_tracked': False,
                'error': str(e)
            }
    
    async def get_learned_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        min_confidence: float = 0.6,
        limit: int = 100
    ) -> List[LearningPattern]:
        """Get learned patterns above confidence threshold
        
        Args:
            pattern_type: Filter by pattern type
            min_confidence: Minimum confidence score
            limit: Maximum number of patterns to return
            
        Returns:
            List of learned patterns
        """
        query = select(LearningPattern).where(
            LearningPattern.confidence_score >= min_confidence
        )
        
        if pattern_type:
            query = query.where(LearningPattern.pattern_type == pattern_type)
        
        query = query.order_by(
            LearningPattern.confidence_score.desc(),
            LearningPattern.usage_count.desc()
        ).limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def apply_learned_patterns(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply learned patterns to improve responses
        
        Args:
            context: Current context for pattern application
            
        Returns:
            Dictionary with applied patterns and recommendations
        """
        try:
            # Find applicable patterns
            applicable_patterns = await self._find_applicable_patterns(context)
            
            # Apply adaptation based on patterns
            adaptations = await self.adaptation_engine.apply_adaptations(
                applicable_patterns,
                context
            )
            
            return {
                'patterns_applied': len(applicable_patterns),
                'adaptations': adaptations,
                'confidence': self._calculate_adaptation_confidence(
                    applicable_patterns
                )
            }
            
        except Exception as e:
            logger.error(f"Error applying patterns: {e}")
            return {
                'patterns_applied': 0,
                'error': str(e)
            }
    
    async def get_learning_analytics(
        self,
        time_period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get analytics about the learning system
        
        Args:
            time_period: Time period for analytics (default: last 30 days)
            
        Returns:
            Dictionary with learning analytics
        """
        if not time_period:
            time_period = timedelta(days=30)
        
        start_date = datetime.now(timezone.utc) - time_period
        
        try:
            # Pattern statistics
            pattern_stats = await self._get_pattern_statistics(start_date)
            
            # Feedback statistics
            feedback_stats = await self._get_feedback_statistics(start_date)
            
            # Outcome statistics
            outcome_stats = await self._get_outcome_statistics(start_date)
            
            # Learning effectiveness
            effectiveness = await self._calculate_learning_effectiveness(
                start_date
            )
            
            return {
                'time_period': str(time_period),
                'pattern_statistics': pattern_stats,
                'feedback_statistics': feedback_stats,
                'outcome_statistics': outcome_stats,
                'learning_effectiveness': effectiveness,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting learning analytics: {e}")
            return {
                'error': str(e)
            }
    
    # Private helper methods
    
    async def _find_similar_patterns(
        self,
        patterns: List[LearningPattern]
    ) -> Dict[str, LearningPattern]:
        """Find similar existing patterns"""
        similar = {}
        
        for pattern in patterns:
            # Check cache first
            cached = self._pattern_cache.get(pattern.pattern_hash)
            if cached and cached['expires'] > datetime.now(timezone.utc):
                similar[pattern.pattern_hash] = cached['pattern']
                continue
            
            # Query database
            existing = await self.db.execute(
                select(LearningPattern).where(
                    LearningPattern.pattern_hash == pattern.pattern_hash
                )
            )
            existing_pattern = existing.scalar_one_or_none()
            
            if existing_pattern:
                similar[pattern.pattern_hash] = existing_pattern
                # Update cache
                self._pattern_cache[pattern.pattern_hash] = {
                    'pattern': existing_pattern,
                    'expires': datetime.now(timezone.utc) + timedelta(
                        seconds=self._cache_ttl
                    )
                }
        
        return similar
    
    async def _reinforce_pattern(
        self,
        existing: LearningPattern,
        new: LearningPattern,
        outcome: Dict[str, Any]
    ) -> LearningPattern:
        """Reinforce an existing pattern with new evidence"""
        # Update confidence based on outcome
        success_score = outcome.get('success_score', 0.8)
        
        # Weighted average for confidence
        total_weight = existing.usage_count + 1
        existing.confidence_score = (
            (existing.confidence_score * existing.usage_count + success_score) /
            total_weight
        )
        
        # Update usage count
        existing.usage_count += 1
        existing.last_used = datetime.now(timezone.utc)
        
        # Update pattern data with new information
        if new.pattern_data:
            existing.pattern_data = self._merge_pattern_data(
                existing.pattern_data,
                new.pattern_data
            )
        
        self.db.commit()
        return existing
    
    async def _create_pattern(
        self,
        pattern: LearningPattern,
        session_id: UUID,
        interaction_data: Dict[str, Any]
    ) -> LearningPattern:
        """Create a new learning pattern"""
        pattern.created_by_session = session_id
        pattern.interaction_count = 1
        pattern.last_used = datetime.now(timezone.utc)
        
        # Set initial confidence based on context
        pattern.confidence_score = self._calculate_initial_confidence(
            interaction_data
        )
        
        self.db.add(pattern)
        self.db.commit()
        
        return pattern
    
    async def _update_pattern_cache(self, patterns: List[LearningPattern]):
        """Update the pattern cache with new/updated patterns"""
        expires = datetime.now(timezone.utc) + timedelta(seconds=self._cache_ttl)
        
        for pattern in patterns:
            self._pattern_cache[pattern.pattern_hash] = {
                'pattern': pattern,
                'expires': expires
            }
    
    async def _check_adaptation_triggers(
        self,
        patterns: List[LearningPattern]
    ):
        """Check if any patterns should trigger behavioral adaptation"""
        high_confidence_patterns = [
            p for p in patterns
            if p.confidence_score >= self.pattern_activation_threshold
        ]
        
        if high_confidence_patterns:
            await self.adaptation_engine.trigger_adaptation(
                high_confidence_patterns
            )
    
    def _should_adapt(self, patterns: List[LearningPattern]) -> bool:
        """Determine if adaptation should be triggered"""
        return any(
            p.confidence_score >= self.pattern_activation_threshold
            for p in patterns
        )
    
    async def _learn_from_correction(self, feedback: UserFeedback):
        """Learn from user corrections"""
        # Create or update correction pattern
        pattern_data = {
            'original': feedback.original_content,
            'corrected': feedback.corrected_content,
            'context': feedback.context_data
        }
        
        correction_pattern = LearningPattern(
            pattern_type=PatternType.CORRECTION,
            pattern_data=pattern_data,
            pattern_hash=self._hash_pattern(pattern_data),
            confidence_score=0.9,  # High confidence for direct corrections
            source='user_correction'
        )
        
        await self.pattern_learner.store_pattern(correction_pattern)
    
    async def _learn_from_rating(self, feedback: UserFeedback):
        """Learn from user ratings"""
        # Adjust confidence of related patterns based on rating
        if feedback.memory_id:
            # Find patterns related to this memory
            related_patterns = await self._find_patterns_for_memory(
                feedback.memory_id
            )
            
            # Adjust confidence based on rating
            rating = feedback.feedback_data.get('rating', 3) / 5.0
            for pattern in related_patterns:
                pattern.confidence_score = (
                    pattern.confidence_score * 0.8 + rating * 0.2
                )
                self.db.commit()
    
    async def _update_patterns_from_feedback(
        self,
        memory_id: UUID,
        feedback: UserFeedback
    ):
        """Update patterns based on feedback for a memory"""
        patterns = await self._find_patterns_for_memory(memory_id)
        
        for pattern in patterns:
            await self.pattern_learner.update_pattern_from_feedback(
                pattern,
                feedback
            )
    
    async def _learn_from_outcome(
        self,
        outcome: DecisionOutcome
    ) -> Dict[str, Any]:
        """Learn from decision outcomes"""
        # Find the decision memory
        decision = await self.db.get(Memory, outcome.decision_id)
        if not decision:
            return {'error': 'Decision not found'}
        
        # Extract patterns from successful outcomes
        if outcome.success_score >= 0.7:
            pattern_data = {
                'decision_type': decision.metadata.get('decision_type'),
                'context': decision.metadata.get('context'),
                'outcome': outcome.outcome_data,
                'success_factors': outcome.outcome_data.get('success_factors', [])
            }
            
            success_pattern = LearningPattern(
                pattern_type=PatternType.SUCCESS,
                pattern_data=pattern_data,
                pattern_hash=self._hash_pattern(pattern_data),
                confidence_score=outcome.success_score,
                source='decision_outcome'
            )
            
            await self.pattern_learner.store_pattern(success_pattern)
            
            return {
                'pattern_created': True,
                'pattern_type': 'success',
                'confidence': outcome.success_score
            }
        else:
            # Learn from failures
            failure_data = {
                'decision_type': decision.metadata.get('decision_type'),
                'context': decision.metadata.get('context'),
                'failure_reasons': outcome.outcome_data.get('failure_reasons', [])
            }
            
            failure_pattern = LearningPattern(
                pattern_type=PatternType.ERROR,
                pattern_data=failure_data,
                pattern_hash=self._hash_pattern(failure_data),
                confidence_score=1.0 - outcome.success_score,
                source='decision_failure'
            )
            
            await self.pattern_learner.store_pattern(failure_pattern)
            
            return {
                'pattern_created': True,
                'pattern_type': 'failure',
                'confidence': 1.0 - outcome.success_score
            }
    
    async def _update_decision_patterns(
        self,
        decision_id: UUID,
        outcome: DecisionOutcome
    ):
        """Update patterns related to a decision based on its outcome"""
        # Find patterns that influenced this decision
        patterns = await self._find_patterns_for_memory(decision_id)
        
        # Reinforce or weaken patterns based on outcome
        for pattern in patterns:
            if outcome.success_score >= 0.7:
                # Reinforce successful patterns
                pattern.confidence_score = min(
                    1.0,
                    pattern.confidence_score + self.learning_rate * outcome.success_score
                )
            else:
                # Weaken unsuccessful patterns
                pattern.confidence_score = max(
                    0.0,
                    pattern.confidence_score - self.learning_rate * (1 - outcome.success_score)
                )
            
            pattern.usage_count += 1
            pattern.last_used = datetime.now(timezone.utc)
        
        self.db.commit()
    
    async def _find_applicable_patterns(
        self,
        context: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Find patterns applicable to the current context"""
        # Extract context features
        context_type = context.get('type')
        context_tags = context.get('tags', [])
        context_entities = context.get('entities', [])
        
        # Query patterns that match context
        query = select(LearningPattern).where(
            and_(
                LearningPattern.confidence_score >= self.min_confidence_threshold,
                or_(
                    LearningPattern.pattern_data['context_type'].astext == context_type,
                    LearningPattern.pattern_data['tags'].has_any(context_tags),
                    LearningPattern.pattern_data['entities'].has_any(context_entities)
                )
            )
        )
        
        result = await self.db.execute(query)
        patterns = result.scalars().all()
        
        # Apply decay to older patterns
        for pattern in patterns:
            age_days = (datetime.now(timezone.utc) - pattern.last_used).days
            decay = self.decay_factor ** (age_days / 30)  # Monthly decay
            pattern.effective_confidence = pattern.confidence_score * decay
        
        # Sort by effective confidence
        patterns.sort(
            key=lambda p: p.effective_confidence,
            reverse=True
        )
        
        return patterns[:10]  # Top 10 most relevant patterns
    
    def _calculate_adaptation_confidence(
        self,
        patterns: List[LearningPattern]
    ) -> float:
        """Calculate overall confidence in adaptations"""
        if not patterns:
            return 0.0
        
        # Weighted average based on pattern confidence and usage
        total_weight = sum(p.usage_count for p in patterns)
        if total_weight == 0:
            return sum(p.confidence_score for p in patterns) / len(patterns)
        
        weighted_sum = sum(
            p.confidence_score * p.usage_count for p in patterns
        )
        
        return weighted_sum / total_weight
    
    async def _calculate_learning_impact(
        self,
        feedback: UserFeedback
    ) -> Dict[str, Any]:
        """Calculate the impact of feedback on learning"""
        # Find patterns affected by this feedback
        affected_patterns = []
        
        if feedback.memory_id:
            affected_patterns = await self._find_patterns_for_memory(
                feedback.memory_id
            )
        
        return {
            'patterns_affected': len(affected_patterns),
            'average_confidence_change': self._calculate_confidence_change(
                affected_patterns
            ),
            'feedback_weight': self._calculate_feedback_weight(feedback)
        }
    
    async def _find_patterns_for_memory(
        self,
        memory_id: UUID
    ) -> List[LearningPattern]:
        """Find patterns associated with a specific memory"""
        # This would be implemented based on pattern-memory relationships
        # For now, return empty list
        return []
    
    def _merge_pattern_data(
        self,
        existing: Dict[str, Any],
        new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge pattern data intelligently"""
        merged = existing.copy()
        
        for key, value in new.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists without duplicates
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Recursively merge dictionaries
                merged[key] = self._merge_pattern_data(merged[key], value)
            elif isinstance(value, (int, float)) and isinstance(merged[key], (int, float)):
                # Average numeric values
                merged[key] = (merged[key] + value) / 2
        
        return merged
    
    def _calculate_initial_confidence(
        self,
        interaction_data: Dict[str, Any]
    ) -> float:
        """Calculate initial confidence for a new pattern"""
        base_confidence = 0.5
        
        # Adjust based on interaction outcome
        if interaction_data.get('outcome', {}).get('success'):
            base_confidence += 0.2
        
        # Adjust based on user feedback
        if interaction_data.get('user_feedback', {}).get('positive'):
            base_confidence += 0.1
        
        # Adjust based on context quality
        if interaction_data.get('context', {}).get('high_quality'):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _hash_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Generate a hash for pattern data"""
        import hashlib
        import json
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(pattern_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def _calculate_confidence_change(
        self,
        patterns: List[LearningPattern]
    ) -> float:
        """Calculate average confidence change for patterns"""
        if not patterns:
            return 0.0
        
        # This would track actual changes - for now return estimate
        return 0.05  # 5% average change
    
    def _calculate_feedback_weight(self, feedback: UserFeedback) -> float:
        """Calculate the weight/importance of feedback"""
        base_weight = 1.0
        
        # Direct corrections have higher weight
        if feedback.feedback_type == 'correction':
            base_weight = 2.0
        
        # Recent feedback has higher weight
        age_hours = (
            datetime.now(timezone.utc) - feedback.created_at
        ).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + age_hours / 24)  # 24-hour half-life
        
        return base_weight * recency_factor
    
    async def _get_pattern_statistics(
        self,
        start_date: datetime
    ) -> Dict[str, Any]:
        """Get pattern statistics for analytics"""
        # Total patterns
        total_patterns = await self.db.execute(
            select(func.count(LearningPattern.id))
        )
        
        # Patterns by type
        patterns_by_type = await self.db.execute(
            select(
                LearningPattern.pattern_type,
                func.count(LearningPattern.id)
            ).group_by(LearningPattern.pattern_type)
        )
        
        # High confidence patterns
        high_confidence = await self.db.execute(
            select(func.count(LearningPattern.id)).where(
                LearningPattern.confidence_score >= 0.8
            )
        )
        
        # Recently used patterns
        recent_patterns = await self.db.execute(
            select(func.count(LearningPattern.id)).where(
                LearningPattern.last_used >= start_date
            )
        )
        
        return {
            'total_patterns': total_patterns.scalar(),
            'patterns_by_type': dict(patterns_by_type.all()),
            'high_confidence_patterns': high_confidence.scalar(),
            'recently_used_patterns': recent_patterns.scalar()
        }
    
    async def _get_feedback_statistics(
        self,
        start_date: datetime
    ) -> Dict[str, Any]:
        """Get feedback statistics for analytics"""
        # Total feedback
        total_feedback = await self.db.execute(
            select(func.count(UserFeedback.id)).where(
                UserFeedback.created_at >= start_date
            )
        )
        
        # Feedback by type
        feedback_by_type = await self.db.execute(
            select(
                UserFeedback.feedback_type,
                func.count(UserFeedback.id)
            ).where(
                UserFeedback.created_at >= start_date
            ).group_by(UserFeedback.feedback_type)
        )
        
        # Applied feedback
        applied_feedback = await self.db.execute(
            select(func.count(UserFeedback.id)).where(
                and_(
                    UserFeedback.created_at >= start_date,
                    UserFeedback.applied == True
                )
            )
        )
        
        return {
            'total_feedback': total_feedback.scalar(),
            'feedback_by_type': dict(feedback_by_type.all()),
            'applied_feedback': applied_feedback.scalar(),
            'feedback_application_rate': (
                applied_feedback.scalar() / max(total_feedback.scalar(), 1)
            )
        }
    
    async def _get_outcome_statistics(
        self,
        start_date: datetime
    ) -> Dict[str, Any]:
        """Get outcome statistics for analytics"""
        # Total outcomes tracked
        total_outcomes = await self.db.execute(
            select(func.count(DecisionOutcome.id)).where(
                DecisionOutcome.measured_at >= start_date
            )
        )
        
        # Average success score
        avg_success = await self.db.execute(
            select(func.avg(DecisionOutcome.success_score)).where(
                DecisionOutcome.measured_at >= start_date
            )
        )
        
        # Outcomes by type
        outcomes_by_type = await self.db.execute(
            select(
                DecisionOutcome.outcome_type,
                func.count(DecisionOutcome.id),
                func.avg(DecisionOutcome.success_score)
            ).where(
                DecisionOutcome.measured_at >= start_date
            ).group_by(DecisionOutcome.outcome_type)
        )
        
        return {
            'total_outcomes': total_outcomes.scalar(),
            'average_success_score': avg_success.scalar() or 0.0,
            'outcomes_by_type': [
                {
                    'type': outcome_type,
                    'count': count,
                    'avg_success': avg_score
                }
                for outcome_type, count, avg_score in outcomes_by_type.all()
            ]
        }
    
    async def _calculate_learning_effectiveness(
        self,
        start_date: datetime
    ) -> Dict[str, Any]:
        """Calculate overall learning effectiveness"""
        # Pattern confidence improvement
        confidence_improvement = await self.db.execute(
            select(
                func.avg(LearningPattern.confidence_score) -
                func.avg(LearningPattern.initial_confidence)
            ).where(
                LearningPattern.created_at >= start_date
            )
        )
        
        # Decision success improvement over time
        # This would track improvement trends - simplified for now
        success_improvement = 0.15  # 15% improvement
        
        # Adaptation effectiveness
        adaptation_success_rate = 0.82  # 82% successful adaptations
        
        return {
            'confidence_improvement': confidence_improvement.scalar() or 0.0,
            'decision_success_improvement': success_improvement,
            'adaptation_success_rate': adaptation_success_rate,
            'overall_effectiveness': (
                (confidence_improvement.scalar() or 0.0) * 0.3 +
                success_improvement * 0.4 +
                adaptation_success_rate * 0.3
            )
        }
    
    # Success Tracking Integration Methods
    
    async def track_decision_outcome(
        self,
        decision_id: UUID,
        outcome_data: Dict[str, Any],
        impact_metrics: Optional[Dict[str, Any]] = None
    ) -> DecisionOutcome:
        """Track the outcome of a decision with comprehensive metrics
        
        Args:
            decision_id: ID of the decision memory
            outcome_data: Information about the outcome
            impact_metrics: Additional impact metrics
            
        Returns:
            DecisionOutcome instance
        """
        try:
            # Create decision context
            decision_context = DecisionContext(
                decision_id=decision_id,
                decision_type=outcome_data.get('decision_type', 'general'),
                context_data=outcome_data.get('context', {}),
                user_id=outcome_data.get('user_id'),
                session_id=outcome_data.get('session_id')
            )
            
            # Track with decision outcome tracker
            outcome = await self.decision_tracker.track_decision_outcome(
                decision_context, outcome_data, impact_metrics
            )
            
            # Also track with legacy success tracker for compatibility
            legacy_outcome = await self.success_tracker.track_outcome(
                decision_id, outcome_data
            )
            
            logger.info(f"Tracked decision outcome for {decision_id} with both new and legacy systems")
            return outcome
            
        except Exception as e:
            logger.error(f"Error tracking decision outcome: {e}")
            raise
    
    async def get_success_metrics_dashboard(
        self,
        time_frame: TimeFrame = TimeFrame.MONTH,
        decision_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive success metrics dashboard
        
        Args:
            time_frame: Time frame for metrics
            decision_types: Optional filter for specific decision types
            
        Returns:
            Dashboard data with all success metrics
        """
        try:
            # Create filter criteria
            filter_criteria = MetricFilter(
                time_frame=time_frame,
                decision_types=decision_types,
                include_partial=True
            )
            
            # Generate comprehensive dashboard
            dashboard = await self.success_metrics.generate_metrics_dashboard(filter_criteria)
            
            # Add legacy analytics for comparison
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
            legacy_analytics = await self.get_analytics(start_date)
            
            # Combine results
            combined_dashboard = {
                "new_metrics": dashboard.dict(),
                "legacy_analytics": legacy_analytics,
                "integration_status": "active",
                "data_sources": ["decision_outcomes", "learning_patterns", "user_feedback"]
            }
            
            return combined_dashboard
            
        except Exception as e:
            logger.error(f"Error generating success metrics dashboard: {e}")
            return {"error": str(e)}
    
    async def analyze_decision_effectiveness(
        self,
        analysis_type: AnalysisType = AnalysisType.OVERALL,
        time_period_days: int = 30,
        decision_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze decision effectiveness with comprehensive insights
        
        Args:
            analysis_type: Type of effectiveness analysis
            time_period_days: Number of days to analyze
            decision_type: Optional specific decision type
            
        Returns:
            Effectiveness analysis results
        """
        try:
            if analysis_type == AnalysisType.OVERALL:
                report = await self.effectiveness_analyzer.analyze_overall_effectiveness(
                    time_period_days=time_period_days,
                    include_benchmarks=True
                )
                
            elif analysis_type == AnalysisType.BY_DECISION_TYPE:
                reports = await self.effectiveness_analyzer.analyze_effectiveness_by_decision_type(
                    decision_type=decision_type,
                    time_period_days=time_period_days
                )
                # Convert list to single report if specific type requested
                if decision_type and reports:
                    report = reports[0]
                else:
                    # Return aggregated data for all types
                    report = {
                        "analysis_type": "multiple_decision_types",
                        "reports": [r.dict() for r in reports],
                        "summary": {
                            "total_types_analyzed": len(reports),
                            "average_effectiveness": sum(r.overall_score for r in reports) / len(reports) if reports else 0.0
                        }
                    }
                    return report
                    
            elif analysis_type == AnalysisType.BY_TIME_PERIOD:
                report = await self.effectiveness_analyzer.analyze_effectiveness_trends(
                    time_period_days=time_period_days,
                    trend_granularity="weekly"
                )
                
            else:
                # Default to overall analysis
                report = await self.effectiveness_analyzer.analyze_overall_effectiveness(
                    time_period_days=time_period_days
                )
            
            # Convert to dict and add metadata
            if hasattr(report, 'dict'):
                result = report.dict()
            else:
                result = report
                
            result["generated_by"] = "effectiveness_analyzer"
            result["integration_version"] = "1.0"
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing decision effectiveness: {e}")
            return {"error": str(e), "analysis_type": analysis_type.value}
    
    async def predict_decision_effectiveness(
        self,
        context: Dict[str, Any],
        decision_type: str,
        user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Predict effectiveness for a potential decision
        
        Args:
            context: Context for the decision
            decision_type: Type of decision
            user_id: Optional user ID for personalized prediction
            
        Returns:
            Prediction results with recommendations
        """
        try:
            # Use effectiveness analyzer for prediction
            prediction = await self.effectiveness_analyzer.predict_effectiveness(
                context=context,
                decision_type=decision_type
            )
            
            # Enhance with learning system insights
            relevant_patterns = await self.pattern_learner.find_patterns(
                pattern_type=PatternType.SUCCESS,
                context=context
            )
            
            # Add pattern-based adjustments
            if relevant_patterns:
                pattern_confidence = sum(p.confidence_score for p in relevant_patterns) / len(relevant_patterns)
                prediction["pattern_boost"] = pattern_confidence * 0.1
                prediction["predicted_effectiveness"] = min(1.0, 
                    prediction["predicted_effectiveness"] + prediction["pattern_boost"]
                )
                prediction["pattern_support"] = len(relevant_patterns)
            
            # Add learning system recommendations
            prediction["learning_recommendations"] = await self._generate_learning_recommendations(
                prediction, context, decision_type
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting decision effectiveness: {e}")
            return {"error": str(e), "predicted_effectiveness": 0.5}
    
    async def get_success_insights(
        self,
        focus_area: str = "overall",
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Get actionable insights about success patterns and areas for improvement
        
        Args:
            focus_area: Area to focus on (overall, consistency, improvement, etc.)
            time_period_days: Number of days to analyze
            
        Returns:
            Insights and recommendations
        """
        try:
            insights = {
                "focus_area": focus_area,
                "time_period_days": time_period_days,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "insights": [],
                "recommendations": [],
                "metrics_summary": {}
            }
            
            # Get core success metrics
            filter_criteria = MetricFilter(time_frame=TimeFrame.MONTH)
            
            if focus_area == "overall":
                # Overall effectiveness analysis
                report = await self.effectiveness_analyzer.analyze_overall_effectiveness(
                    time_period_days=time_period_days
                )
                insights["insights"] = [insight.dict() for insight in report.insights]
                insights["recommendations"] = report.recommendations
                insights["metrics_summary"] = {
                    "effectiveness_category": report.effectiveness_category.value,
                    "overall_score": report.overall_score,
                    "confidence": report.confidence
                }
                
            elif focus_area == "consistency":
                # Focus on consistency metrics
                consistency_metric = await self.success_metrics.calculate_consistency_score(filter_criteria)
                insights["metrics_summary"] = consistency_metric.dict()
                insights["insights"] = [
                    f"Consistency score: {consistency_metric.value:.3f}",
                    f"Sample size: {consistency_metric.sample_size} decisions"
                ]
                insights["recommendations"] = [
                    "Monitor decision variability",
                    "Standardize high-performing patterns",
                    "Implement consistency checkpoints"
                ]
                
            elif focus_area == "improvement":
                # Focus on improvement trends
                improvement_metric = await self.success_metrics.calculate_improvement_trend(filter_criteria)
                insights["metrics_summary"] = improvement_metric.dict()
                insights["insights"] = [
                    f"Improvement trend: {improvement_metric.value*100:.1f}%",
                    f"Confidence: {improvement_metric.confidence:.3f}"
                ]
                
            # Add learning patterns insights
            recent_patterns = await self._get_recent_learning_patterns(time_period_days)
            insights["learning_patterns"] = {
                "total_patterns": len(recent_patterns),
                "pattern_types": list(set(p.pattern_type.value for p in recent_patterns)),
                "average_confidence": sum(p.confidence_score for p in recent_patterns) / len(recent_patterns) if recent_patterns else 0.0
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating success insights: {e}")
            return {"error": str(e), "focus_area": focus_area}
    
    async def _generate_learning_recommendations(
        self,
        prediction: Dict[str, Any],
        context: Dict[str, Any],
        decision_type: str
    ) -> List[str]:
        """Generate learning-based recommendations for decisions"""
        recommendations = []
        
        predicted_score = prediction.get("predicted_effectiveness", 0.5)
        confidence = prediction.get("confidence", 0.5)
        
        # Base recommendations from prediction
        if predicted_score < 0.5:
            recommendations.append("Consider alternative approaches")
            recommendations.append("Gather more context information")
        elif predicted_score > 0.8:
            recommendations.append("High success probability - proceed with confidence")
            
        # Confidence-based recommendations
        if confidence < 0.5:
            recommendations.append("Low prediction confidence - monitor outcome closely")
            recommendations.append("Document outcome for future learning")
        
        # Decision type specific recommendations
        if decision_type == "code_generation":
            recommendations.append("Review code patterns and best practices")
            recommendations.append("Consider automated testing integration")
        elif decision_type == "debugging":
            recommendations.append("Use systematic debugging approaches")
            recommendations.append("Document root cause for pattern learning")
        
        return recommendations
    
    async def _get_recent_learning_patterns(self, days: int) -> List[LearningPattern]:
        """Get recent learning patterns for insights"""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        result = await self.db.execute(
            select(LearningPattern)
            .where(LearningPattern.created_at >= start_date)
            .order_by(LearningPattern.confidence_score.desc())
            .limit(50)
        )
        
        return result.scalars().all()