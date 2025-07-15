"""Feedback Processing Service

This service handles user feedback processing, including corrections, ratings,
and confirmations. It learns from feedback to improve future interactions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError

from ..models.user_feedback import UserFeedback, FeedbackType
from ..models.learning_pattern import LearningPattern
from ...memory_system.models.memory import Memory

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """Service for processing and learning from user feedback"""
    
    def __init__(self, db: Session):
        """Initialize the feedback processor"""
        self.db = db
        
        # Feedback processing configuration
        self.min_confidence_adjustment = 0.05
        self.max_confidence_adjustment = 0.2
        self.feedback_weight_decay = 0.9  # Older feedback has less impact
        
        # Feedback type handlers
        self.feedback_handlers = {
            FeedbackType.CORRECTION: self._process_correction_feedback,
            FeedbackType.RATING: self._process_rating_feedback,
            FeedbackType.CONFIRMATION: self._process_confirmation_feedback,
            FeedbackType.REJECTION: self._process_rejection_feedback
        }
    
    async def process_feedback(
        self,
        feedback_data: Dict[str, Any]
    ) -> UserFeedback:
        """Process user feedback and store it
        
        Args:
            feedback_data: Feedback information including:
                - memory_id: Related memory if applicable
                - feedback_type: Type of feedback
                - original_content: What was originally provided
                - corrected_content: User's correction if applicable
                - rating: User rating if applicable
                - context_data: Additional context
                
        Returns:
            Processed UserFeedback instance
        """
        try:
            # Create feedback record
            feedback = UserFeedback(
                id=uuid4(),
                memory_id=feedback_data.get('memory_id'),
                feedback_type=feedback_data.get('feedback_type', FeedbackType.RATING.value),
                original_content=feedback_data.get('original_content'),
                corrected_content=feedback_data.get('corrected_content'),
                feedback_data=feedback_data.get('feedback_data', {}),
                context_data=feedback_data.get('context_data', {}),
                created_at=datetime.now(timezone.utc),
                applied=False
            )
            
            # Validate feedback
            validation_result = await self._validate_feedback(feedback)
            if not validation_result['valid']:
                raise ValueError(f"Invalid feedback: {validation_result['reason']}")
            
            # Process feedback based on type
            feedback_type = FeedbackType(feedback.feedback_type)
            if feedback_type in self.feedback_handlers:
                handler = self.feedback_handlers[feedback_type]
                processing_result = await handler(feedback)
                feedback.processing_result = processing_result
            
            # Store feedback
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            
            # Apply feedback learnings
            await self._apply_feedback_learnings(feedback)
            
            logger.info(f"Processed {feedback.feedback_type} feedback: {feedback.id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            self.db.rollback()
            raise
    
    async def get_feedback_summary(
        self,
        time_period_days: int = 30,
        feedback_type: Optional[FeedbackType] = None
    ) -> Dict[str, Any]:
        """Get summary of feedback received
        
        Args:
            time_period_days: Number of days to look back
            feedback_type: Optional specific feedback type to filter
            
        Returns:
            Dictionary with feedback summary statistics
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=time_period_days)
            
            # Base query
            query = select(UserFeedback).where(
                UserFeedback.created_at >= start_date
            )
            
            if feedback_type:
                query = query.where(UserFeedback.feedback_type == feedback_type.value)
            
            result = await self.db.execute(query)
            feedbacks = result.scalars().all()
            
            # Calculate statistics
            summary = {
                'total_feedback': len(feedbacks),
                'feedback_by_type': await self._get_feedback_by_type_stats(start_date),
                'applied_feedback': len([f for f in feedbacks if f.applied]),
                'application_rate': len([f for f in feedbacks if f.applied]) / max(len(feedbacks), 1),
                'average_processing_time': await self._calculate_avg_processing_time(feedbacks),
                'common_corrections': await self._get_common_corrections(start_date),
                'rating_distribution': await self._get_rating_distribution(start_date),
                'feedback_trends': await self._analyze_feedback_trends(feedbacks)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting feedback summary: {e}")
            return {'error': str(e)}
    
    async def apply_feedback_to_memory(
        self,
        feedback_id: UUID,
        memory_id: UUID
    ) -> Dict[str, Any]:
        """Apply specific feedback to a memory
        
        Args:
            feedback_id: ID of the feedback to apply
            memory_id: ID of the memory to update
            
        Returns:
            Dictionary with application results
        """
        try:
            # Get feedback and memory
            feedback = await self.db.get(UserFeedback, feedback_id)
            memory = await self.db.get(Memory, memory_id)
            
            if not feedback:
                return {'error': 'Feedback not found'}
            if not memory:
                return {'error': 'Memory not found'}
            
            # Apply feedback based on type
            if feedback.feedback_type == FeedbackType.CORRECTION.value:
                memory.content = feedback.corrected_content
                memory.metadata['corrected'] = True
                memory.metadata['correction_date'] = datetime.now(timezone.utc).isoformat()
            
            elif feedback.feedback_type == FeedbackType.RATING.value:
                rating = feedback.feedback_data.get('rating', 3)
                # Adjust importance based on rating
                memory.importance = (memory.importance * 0.7 + (rating / 5.0) * 0.3)
            
            # Mark feedback as applied
            feedback.applied = True
            feedback.applied_at = datetime.now(timezone.utc)
            
            self.db.commit()
            
            return {
                'success': True,
                'memory_updated': True,
                'feedback_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error applying feedback to memory: {e}")
            self.db.rollback()
            return {'error': str(e)}
    
    async def get_feedback_for_memory(
        self,
        memory_id: UUID
    ) -> List[UserFeedback]:
        """Get all feedback for a specific memory
        
        Args:
            memory_id: Memory ID to get feedback for
            
        Returns:
            List of feedback records
        """
        result = await self.db.execute(
            select(UserFeedback).where(
                UserFeedback.memory_id == memory_id
            ).order_by(UserFeedback.created_at.desc())
        )
        return result.scalars().all()
    
    # Private methods
    
    async def _validate_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Validate feedback data"""
        # Check required fields based on feedback type
        if feedback.feedback_type == FeedbackType.CORRECTION.value:
            if not feedback.original_content or not feedback.corrected_content:
                return {
                    'valid': False,
                    'reason': 'Correction feedback requires original and corrected content'
                }
            if feedback.original_content == feedback.corrected_content:
                return {
                    'valid': False,
                    'reason': 'Corrected content must be different from original'
                }
        
        elif feedback.feedback_type == FeedbackType.RATING.value:
            rating = feedback.feedback_data.get('rating')
            if rating is None or not (1 <= rating <= 5):
                return {
                    'valid': False,
                    'reason': 'Rating feedback requires a rating between 1 and 5'
                }
        
        return {'valid': True}
    
    async def _process_correction_feedback(
        self,
        feedback: UserFeedback
    ) -> Dict[str, Any]:
        """Process correction-type feedback"""
        result = {
            'type': 'correction',
            'processed': True
        }
        
        # Analyze the correction
        correction_analysis = await self._analyze_correction(
            feedback.original_content,
            feedback.corrected_content
        )
        
        result['analysis'] = correction_analysis
        
        # Extract patterns from correction
        if correction_analysis['pattern_identified']:
            result['pattern'] = correction_analysis['pattern']
            # Store pattern for future use
            await self._store_correction_pattern(correction_analysis['pattern'])
        
        # Calculate confidence impact
        result['confidence_impact'] = -self.min_confidence_adjustment * 2
        
        return result
    
    async def _process_rating_feedback(
        self,
        feedback: UserFeedback
    ) -> Dict[str, Any]:
        """Process rating-type feedback"""
        rating = feedback.feedback_data.get('rating', 3)
        
        result = {
            'type': 'rating',
            'processed': True,
            'rating': rating
        }
        
        # Calculate confidence impact based on rating
        # 5 stars = positive impact, 1 star = negative impact
        normalized_rating = (rating - 3) / 2  # Range: -1 to 1
        result['confidence_impact'] = normalized_rating * self.min_confidence_adjustment
        
        # Analyze rating context
        if rating <= 2:
            result['analysis'] = 'negative_feedback'
            result['action'] = 'review_similar_responses'
        elif rating >= 4:
            result['analysis'] = 'positive_feedback'
            result['action'] = 'reinforce_approach'
        else:
            result['analysis'] = 'neutral_feedback'
            result['action'] = 'monitor'
        
        return result
    
    async def _process_confirmation_feedback(
        self,
        feedback: UserFeedback
    ) -> Dict[str, Any]:
        """Process confirmation-type feedback"""
        result = {
            'type': 'confirmation',
            'processed': True
        }
        
        # Confirmations increase confidence
        result['confidence_impact'] = self.min_confidence_adjustment
        
        # Mark any associated patterns as validated
        result['patterns_validated'] = True
        
        return result
    
    async def _process_rejection_feedback(
        self,
        feedback: UserFeedback
    ) -> Dict[str, Any]:
        """Process rejection-type feedback"""
        result = {
            'type': 'rejection',
            'processed': True
        }
        
        # Rejections significantly decrease confidence
        result['confidence_impact'] = -self.max_confidence_adjustment
        
        # Analyze rejection reason if provided
        reason = feedback.feedback_data.get('reason', 'unspecified')
        result['rejection_reason'] = reason
        
        # Mark for pattern review
        result['requires_pattern_review'] = True
        
        return result
    
    async def _apply_feedback_learnings(self, feedback: UserFeedback):
        """Apply learnings from feedback to improve future responses"""
        if not feedback.processing_result:
            return
        
        confidence_impact = feedback.processing_result.get('confidence_impact', 0)
        
        # Update related patterns if memory is associated
        if feedback.memory_id:
            await self._update_memory_patterns(feedback.memory_id, confidence_impact)
        
        # Store feedback patterns for future reference
        if feedback.feedback_type == FeedbackType.CORRECTION.value:
            await self._learn_from_correction(feedback)
        
        # Update global metrics
        await self._update_feedback_metrics(feedback)
    
    async def _analyze_correction(
        self,
        original: str,
        corrected: str
    ) -> Dict[str, Any]:
        """Analyze a correction to identify patterns"""
        analysis = {
            'pattern_identified': False,
            'correction_type': 'unknown'
        }
        
        # Simple pattern detection
        if len(original) > 0 and len(corrected) > 0:
            # Check for case corrections
            if original.lower() == corrected.lower() and original != corrected:
                analysis['correction_type'] = 'case_correction'
                analysis['pattern_identified'] = True
                analysis['pattern'] = {
                    'type': 'case_sensitivity',
                    'example': {'original': original, 'corrected': corrected}
                }
            
            # Check for typo corrections (edit distance)
            elif self._calculate_edit_distance(original, corrected) <= 3:
                analysis['correction_type'] = 'typo_correction'
                analysis['pattern_identified'] = True
                analysis['pattern'] = {
                    'type': 'typo',
                    'example': {'original': original, 'corrected': corrected}
                }
            
            # Check for structural corrections
            elif self._is_structural_correction(original, corrected):
                analysis['correction_type'] = 'structural_correction'
                analysis['pattern_identified'] = True
                analysis['pattern'] = {
                    'type': 'structure',
                    'example': {'original': original, 'corrected': corrected}
                }
        
        return analysis
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _is_structural_correction(self, original: str, corrected: str) -> bool:
        """Check if correction is structural (e.g., formatting, organization)"""
        # Check for indentation changes
        if original.strip() == corrected.strip():
            return True
        
        # Check for line break changes
        if original.replace('\n', ' ') == corrected.replace('\n', ' '):
            return True
        
        # Check for punctuation changes
        import string
        if original.translate(str.maketrans('', '', string.punctuation)) == \
           corrected.translate(str.maketrans('', '', string.punctuation)):
            return True
        
        return False
    
    async def _store_correction_pattern(self, pattern: Dict[str, Any]):
        """Store a correction pattern for future use"""
        # This would store the pattern in a dedicated correction patterns table
        # For now, log it
        logger.info(f"Storing correction pattern: {pattern}")
    
    async def _update_memory_patterns(
        self,
        memory_id: UUID,
        confidence_impact: float
    ):
        """Update patterns associated with a memory based on feedback"""
        # This would update pattern confidence scores
        # For now, log the update
        logger.info(f"Updating patterns for memory {memory_id} with impact {confidence_impact}")
    
    async def _learn_from_correction(self, feedback: UserFeedback):
        """Learn from correction feedback"""
        if feedback.processing_result and feedback.processing_result.get('pattern'):
            pattern = feedback.processing_result['pattern']
            
            # Create a learning pattern from the correction
            learning_pattern = LearningPattern(
                pattern_type=PatternType.CORRECTION,
                pattern_data={
                    'correction_type': pattern['type'],
                    'examples': [pattern['example']],
                    'context': feedback.context_data
                },
                pattern_hash=self._hash_correction(pattern['example']),
                confidence_score=0.9,  # High confidence for direct corrections
                source='user_correction'
            )
            
            # Store for future reference
            self.db.add(learning_pattern)
            self.db.commit()
    
    def _hash_correction(self, example: Dict[str, str]) -> str:
        """Generate hash for a correction example"""
        import hashlib
        import json
        
        data = json.dumps(example, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def _update_feedback_metrics(self, feedback: UserFeedback):
        """Update global feedback metrics"""
        # This would update metrics in a metrics table
        # For now, log the metrics
        logger.info(f"Updating metrics for {feedback.feedback_type} feedback")
    
    async def _get_feedback_by_type_stats(
        self,
        start_date: datetime
    ) -> Dict[str, int]:
        """Get feedback counts by type"""
        result = await self.db.execute(
            select(
                UserFeedback.feedback_type,
                func.count(UserFeedback.id)
            ).where(
                UserFeedback.created_at >= start_date
            ).group_by(UserFeedback.feedback_type)
        )
        
        return dict(result.all())
    
    async def _calculate_avg_processing_time(
        self,
        feedbacks: List[UserFeedback]
    ) -> float:
        """Calculate average time to apply feedback"""
        processing_times = []
        
        for feedback in feedbacks:
            if feedback.applied and feedback.applied_at:
                processing_time = (
                    feedback.applied_at - feedback.created_at
                ).total_seconds() / 60  # Convert to minutes
                processing_times.append(processing_time)
        
        if processing_times:
            return sum(processing_times) / len(processing_times)
        
        return 0.0
    
    async def _get_common_corrections(
        self,
        start_date: datetime,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most common corrections"""
        result = await self.db.execute(
            select(UserFeedback).where(
                and_(
                    UserFeedback.created_at >= start_date,
                    UserFeedback.feedback_type == FeedbackType.CORRECTION.value
                )
            ).limit(limit)
        )
        
        corrections = []
        for feedback in result.scalars().all():
            corrections.append({
                'original': feedback.original_content[:50] + '...' if len(feedback.original_content) > 50 else feedback.original_content,
                'corrected': feedback.corrected_content[:50] + '...' if len(feedback.corrected_content) > 50 else feedback.corrected_content,
                'date': feedback.created_at.isoformat()
            })
        
        return corrections
    
    async def _get_rating_distribution(
        self,
        start_date: datetime
    ) -> Dict[int, int]:
        """Get distribution of ratings"""
        result = await self.db.execute(
            select(UserFeedback).where(
                and_(
                    UserFeedback.created_at >= start_date,
                    UserFeedback.feedback_type == FeedbackType.RATING.value
                )
            )
        )
        
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for feedback in result.scalars().all():
            rating = feedback.feedback_data.get('rating')
            if rating and 1 <= rating <= 5:
                distribution[rating] += 1
        
        return distribution
    
    async def _analyze_feedback_trends(
        self,
        feedbacks: List[UserFeedback]
    ) -> Dict[str, Any]:
        """Analyze trends in feedback"""
        if not feedbacks:
            return {'trend': 'no_data'}
        
        # Sort by date
        sorted_feedbacks = sorted(feedbacks, key=lambda f: f.created_at)
        
        # Calculate trend for ratings
        recent_ratings = []
        older_ratings = []
        
        midpoint = len(sorted_feedbacks) // 2
        
        for i, feedback in enumerate(sorted_feedbacks):
            if feedback.feedback_type == FeedbackType.RATING.value:
                rating = feedback.feedback_data.get('rating', 3)
                if i < midpoint:
                    older_ratings.append(rating)
                else:
                    recent_ratings.append(rating)
        
        trend = {
            'direction': 'stable',
            'confidence': 0.5
        }
        
        if recent_ratings and older_ratings:
            recent_avg = sum(recent_ratings) / len(recent_ratings)
            older_avg = sum(older_ratings) / len(older_ratings)
            
            if recent_avg > older_avg + 0.5:
                trend['direction'] = 'improving'
                trend['confidence'] = min((recent_avg - older_avg) / 2, 1.0)
            elif recent_avg < older_avg - 0.5:
                trend['direction'] = 'declining'
                trend['confidence'] = min((older_avg - recent_avg) / 2, 1.0)
        
        return trend