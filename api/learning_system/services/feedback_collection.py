"""Feedback Collection Service

This service provides enhanced feedback collection capabilities, making it easy
for users to provide corrections, ratings, and suggestions inline with responses.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
import asyncio
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func
from pydantic import BaseModel, Field

from ..models.user_feedback import UserFeedback, FeedbackType
from ..models.learning_pattern import LearningPattern, PatternType
from .feedback_processor import FeedbackProcessor
from ...memory_system.models.memory import Memory
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


class FeedbackPromptType(str, Enum):
    """Types of feedback prompts"""
    INLINE_CORRECTION = "inline_correction"
    RATING_REQUEST = "rating_request"
    IMPROVEMENT_SUGGESTION = "improvement_suggestion"
    CONFIRMATION_REQUEST = "confirmation_request"
    ERROR_REPORT = "error_report"


class FeedbackContext(BaseModel):
    """Context for feedback collection"""
    session_id: UUID
    memory_id: Optional[UUID] = None
    interaction_id: Optional[UUID] = None
    response_segment: Optional[str] = None
    original_input: str
    system_response: str
    response_metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackPrompt(BaseModel):
    """A prompt for collecting feedback"""
    id: UUID = Field(default_factory=uuid4)
    prompt_type: FeedbackPromptType
    message: str
    context: FeedbackContext
    options: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


class CollectedFeedback(BaseModel):
    """Collected feedback from user"""
    prompt_id: UUID
    feedback_type: FeedbackType
    content: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    correction: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackCollectionService:
    """Service for collecting user feedback with enhanced UX"""
    
    def __init__(self, db: Session):
        """Initialize the feedback collection service"""
        self.db = db
        self.feedback_processor = FeedbackProcessor(db)
        
        # Configuration
        self.auto_prompt_threshold = 0.6  # Confidence below this triggers prompts
        self.prompt_frequency = timedelta(minutes=10)  # Min time between prompts
        self.max_prompts_per_session = 5
        
        # Cache for active prompts
        self._active_prompts = {}
        self._last_prompt_time = {}
        self._prompt_count = {}
    
    async def analyze_response_for_feedback(
        self,
        context: FeedbackContext
    ) -> List[FeedbackPrompt]:
        """Analyze a response and determine if feedback should be collected
        
        Args:
            context: Context of the interaction
            
        Returns:
            List of feedback prompts to show to the user
        """
        prompts = []
        
        try:
            # Check if we should prompt (rate limiting)
            if not self._should_prompt(context.session_id):
                return prompts
            
            # Analyze response quality and confidence
            analysis = await self._analyze_response_quality(context)
            
            # Generate prompts based on analysis
            if analysis['confidence'] < self.auto_prompt_threshold:
                # Low confidence - ask for confirmation
                prompt = FeedbackPrompt(
                    prompt_type=FeedbackPromptType.CONFIRMATION_REQUEST,
                    message="Is this response accurate and helpful?",
                    context=context,
                    options=["Yes, it's correct", "No, needs correction", "Partially correct"]
                )
                prompts.append(prompt)
            
            # Check for potential errors
            if analysis.get('potential_errors'):
                prompt = FeedbackPrompt(
                    prompt_type=FeedbackPromptType.ERROR_REPORT,
                    message="Did you notice any errors in this response?",
                    context=context,
                    options=["No errors", "Minor issues", "Significant errors"]
                )
                prompts.append(prompt)
            
            # Periodic rating request
            if self._should_request_rating(context.session_id):
                prompt = FeedbackPrompt(
                    prompt_type=FeedbackPromptType.RATING_REQUEST,
                    message="How would you rate this response?",
                    context=context,
                    options=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
                )
                prompts.append(prompt)
            
            # Store active prompts
            for prompt in prompts:
                self._active_prompts[str(prompt.id)] = prompt
                
            # Update prompt tracking
            if prompts:
                self._last_prompt_time[str(context.session_id)] = datetime.now(timezone.utc)
                self._prompt_count[str(context.session_id)] = self._prompt_count.get(
                    str(context.session_id), 0
                ) + len(prompts)
            
            return prompts
            
        except Exception as e:
            logger.error(f"Error analyzing response for feedback: {e}")
            return []
    
    async def create_inline_correction_prompt(
        self,
        context: FeedbackContext,
        highlighted_text: str,
        suggestion: Optional[str] = None
    ) -> FeedbackPrompt:
        """Create an inline correction prompt for specific text
        
        Args:
            context: Feedback context
            highlighted_text: Text that may need correction
            suggestion: Optional suggested correction
            
        Returns:
            FeedbackPrompt for inline correction
        """
        message = f"Would you like to correct this part: '{highlighted_text}'?"
        if suggestion:
            message += f"\nSuggested: '{suggestion}'"
        
        prompt = FeedbackPrompt(
            prompt_type=FeedbackPromptType.INLINE_CORRECTION,
            message=message,
            context=context,
            options=["Accept suggestion", "Provide correction", "Keep original"]
        )
        
        self._active_prompts[str(prompt.id)] = prompt
        return prompt
    
    async def collect_feedback(
        self,
        feedback: CollectedFeedback
    ) -> Dict[str, Any]:
        """Process collected feedback from user
        
        Args:
            feedback: Collected feedback data
            
        Returns:
            Processing result
        """
        try:
            # Retrieve the original prompt
            prompt = self._active_prompts.get(str(feedback.prompt_id))
            if not prompt:
                return {"error": "Invalid or expired prompt"}
            
            # Prepare feedback data based on type
            feedback_data = self._prepare_feedback_data(prompt, feedback)
            
            # Process through feedback processor
            result = await self.feedback_processor.process_feedback(feedback_data)
            
            # Learn from the feedback immediately
            await self._learn_from_feedback(prompt.context, feedback, result)
            
            # Clean up
            del self._active_prompts[str(feedback.prompt_id)]
            
            # Cache feedback for pattern analysis
            await self._cache_feedback(prompt.context.session_id, feedback)
            
            return {
                "success": True,
                "feedback_id": result.get('feedback_id'),
                "impact": result.get('learning_impact', {}),
                "message": self._get_thank_you_message(feedback.feedback_type)
            }
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return {"error": str(e)}
    
    async def get_feedback_suggestions(
        self,
        session_id: UUID,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get suggested improvements based on collected feedback
        
        Args:
            session_id: Current session ID
            limit: Maximum suggestions to return
            
        Returns:
            List of improvement suggestions
        """
        try:
            # Get recent feedback for the session
            recent_feedback = await self._get_recent_feedback(session_id)
            
            # Analyze patterns
            patterns = await self._analyze_feedback_patterns(recent_feedback)
            
            # Generate suggestions
            suggestions = []
            
            for pattern in patterns[:limit]:
                suggestion = {
                    "type": pattern['type'],
                    "description": pattern['description'],
                    "examples": pattern['examples'],
                    "confidence": pattern['confidence'],
                    "action": self._get_improvement_action(pattern)
                }
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting feedback suggestions: {e}")
            return []
    
    async def get_feedback_statistics(
        self,
        session_id: Optional[UUID] = None,
        time_period: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Get statistics about collected feedback
        
        Args:
            session_id: Optional session filter
            time_period: Time period for statistics
            
        Returns:
            Feedback statistics
        """
        try:
            start_date = datetime.now(timezone.utc) - time_period
            
            # Build query
            query = select(UserFeedback).where(
                UserFeedback.created_at >= start_date
            )
            
            if session_id:
                query = query.where(UserFeedback.session_id == session_id)
            
            result = await self.db.execute(query)
            feedbacks = result.scalars().all()
            
            # Calculate statistics
            stats = {
                "total_feedback": len(feedbacks),
                "feedback_types": {},
                "average_rating": 0.0,
                "correction_rate": 0.0,
                "response_time": self._calculate_avg_response_time(feedbacks),
                "most_corrected_topics": await self._get_most_corrected_topics(feedbacks),
                "improvement_trend": await self._calculate_improvement_trend(feedbacks)
            }
            
            # Count by type
            ratings = []
            corrections = 0
            
            for feedback in feedbacks:
                feedback_type = feedback.feedback_type
                stats["feedback_types"][feedback_type] = stats["feedback_types"].get(
                    feedback_type, 0
                ) + 1
                
                if feedback_type == FeedbackType.RATING.value:
                    rating = feedback.get_rating()
                    if rating:
                        ratings.append(rating)
                elif feedback_type == FeedbackType.CORRECTION.value:
                    corrections += 1
            
            # Calculate averages
            if ratings:
                stats["average_rating"] = sum(ratings) / len(ratings)
            
            if feedbacks:
                stats["correction_rate"] = corrections / len(feedbacks)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting feedback statistics: {e}")
            return {}
    
    # Private helper methods
    
    def _should_prompt(self, session_id: UUID) -> bool:
        """Check if we should show prompts to this session"""
        session_key = str(session_id)
        
        # Check prompt count
        prompt_count = self._prompt_count.get(session_key, 0)
        if prompt_count >= self.max_prompts_per_session:
            return False
        
        # Check time since last prompt
        last_prompt = self._last_prompt_time.get(session_key)
        if last_prompt:
            time_since = datetime.now(timezone.utc) - last_prompt
            if time_since < self.prompt_frequency:
                return False
        
        return True
    
    async def _analyze_response_quality(
        self,
        context: FeedbackContext
    ) -> Dict[str, Any]:
        """Analyze the quality and confidence of a response"""
        analysis = {
            "confidence": 0.8,  # Default confidence
            "potential_errors": [],
            "improvement_areas": []
        }
        
        # Check response length
        response_length = len(context.system_response)
        if response_length < 50:
            analysis["confidence"] -= 0.2
            analysis["improvement_areas"].append("response_brevity")
        
        # Check for uncertainty markers
        uncertainty_markers = [
            "might", "possibly", "perhaps", "not sure",
            "I think", "probably", "it seems"
        ]
        
        response_lower = context.system_response.lower()
        uncertainty_count = sum(
            1 for marker in uncertainty_markers
            if marker in response_lower
        )
        
        if uncertainty_count > 2:
            analysis["confidence"] -= 0.1 * uncertainty_count
            analysis["potential_errors"].append("high_uncertainty")
        
        # Check for common error patterns
        if "error" in response_lower or "exception" in response_lower:
            analysis["potential_errors"].append("error_mentioned")
        
        # Ensure confidence is between 0 and 1
        analysis["confidence"] = max(0.0, min(1.0, analysis["confidence"]))
        
        return analysis
    
    def _should_request_rating(self, session_id: UUID) -> bool:
        """Determine if we should request a rating"""
        # Simple heuristic: request rating 20% of the time
        import random
        return random.random() < 0.2
    
    def _prepare_feedback_data(
        self,
        prompt: FeedbackPrompt,
        feedback: CollectedFeedback
    ) -> Dict[str, Any]:
        """Prepare feedback data for processing"""
        feedback_data = {
            "session_id": prompt.context.session_id,
            "memory_id": prompt.context.memory_id,
            "feedback_type": feedback.feedback_type.value,
            "context_data": {
                "prompt_type": prompt.prompt_type.value,
                "interaction_id": prompt.context.interaction_id,
                "response_metadata": prompt.context.response_metadata
            }
        }
        
        # Add type-specific data
        if feedback.feedback_type == FeedbackType.CORRECTION:
            feedback_data["original_content"] = prompt.context.system_response
            feedback_data["corrected_content"] = feedback.correction or feedback.content
        
        elif feedback.feedback_type == FeedbackType.RATING:
            feedback_data["feedback_data"] = {"rating": feedback.rating}
        
        elif feedback.feedback_type == FeedbackType.CONFIRMATION:
            feedback_data["feedback_data"] = {
                "confirmed": feedback.content == "Yes, it's correct"
            }
        
        return feedback_data
    
    async def _learn_from_feedback(
        self,
        context: FeedbackContext,
        feedback: CollectedFeedback,
        result: Dict[str, Any]
    ):
        """Immediately learn from the feedback"""
        try:
            # Create learning pattern from feedback
            if feedback.feedback_type == FeedbackType.CORRECTION:
                pattern_data = {
                    "correction_type": "user_provided",
                    "original": context.system_response,
                    "corrected": feedback.correction,
                    "context": {
                        "input": context.original_input,
                        "metadata": context.response_metadata
                    }
                }
                
                pattern = LearningPattern(
                    pattern_type=PatternType.CORRECTION,
                    pattern_data=pattern_data,
                    pattern_hash=self._hash_pattern(pattern_data),
                    confidence_score=0.95,  # High confidence for direct corrections
                    source='user_feedback_collection'
                )
                
                self.db.add(pattern)
                self.db.commit()
                
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    def _hash_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Generate hash for pattern data"""
        import hashlib
        import json
        
        sorted_data = json.dumps(pattern_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    async def _cache_feedback(self, session_id: UUID, feedback: CollectedFeedback):
        """Cache feedback for quick access"""
        if redis_client.client:
            try:
                key = f"feedback:{session_id}:{feedback.prompt_id}"
                await redis_client.set(
                    key,
                    feedback.dict(),
                    expiry=3600  # 1 hour
                )
            except Exception as e:
                logger.warning(f"Failed to cache feedback: {e}")
    
    def _get_thank_you_message(self, feedback_type: FeedbackType) -> str:
        """Get appropriate thank you message"""
        messages = {
            FeedbackType.CORRECTION: "Thank you! I'll learn from this correction.",
            FeedbackType.RATING: "Thanks for rating! Your feedback helps me improve.",
            FeedbackType.CONFIRMATION: "Thank you for confirming!",
            FeedbackType.REJECTION: "I appreciate your feedback. I'll work on improving."
        }
        return messages.get(feedback_type, "Thank you for your feedback!")
    
    async def _get_recent_feedback(
        self,
        session_id: UUID,
        limit: int = 20
    ) -> List[UserFeedback]:
        """Get recent feedback for a session"""
        result = await self.db.execute(
            select(UserFeedback).where(
                UserFeedback.session_id == session_id
            ).order_by(
                UserFeedback.created_at.desc()
            ).limit(limit)
        )
        return result.scalars().all()
    
    async def _analyze_feedback_patterns(
        self,
        feedbacks: List[UserFeedback]
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in feedback"""
        patterns = []
        
        # Group corrections by type
        corrections = [f for f in feedbacks if f.feedback_type == FeedbackType.CORRECTION.value]
        
        if corrections:
            # Find common correction patterns
            correction_types = {}
            for correction in corrections:
                # Simple pattern detection
                if correction.original_content and correction.corrected_content:
                    if len(correction.corrected_content) > len(correction.original_content):
                        pattern_type = "expansion_needed"
                    elif "async" in correction.corrected_content and "async" not in correction.original_content:
                        pattern_type = "async_missing"
                    else:
                        pattern_type = "general_correction"
                    
                    correction_types[pattern_type] = correction_types.get(pattern_type, 0) + 1
            
            # Create patterns from types
            for pattern_type, count in correction_types.items():
                patterns.append({
                    "type": pattern_type,
                    "description": self._get_pattern_description(pattern_type),
                    "examples": [],  # Would include actual examples
                    "confidence": min(0.9, count * 0.1),
                    "frequency": count
                })
        
        # Sort by frequency
        patterns.sort(key=lambda p: p['frequency'], reverse=True)
        
        return patterns
    
    def _get_pattern_description(self, pattern_type: str) -> str:
        """Get human-readable description for pattern type"""
        descriptions = {
            "expansion_needed": "Responses often need more detail",
            "async_missing": "Async/await patterns frequently missing",
            "general_correction": "General accuracy improvements needed"
        }
        return descriptions.get(pattern_type, "Pattern detected in corrections")
    
    def _get_improvement_action(self, pattern: Dict[str, Any]) -> str:
        """Get suggested action for improvement"""
        actions = {
            "expansion_needed": "Provide more comprehensive responses with examples",
            "async_missing": "Default to async/await patterns for I/O operations",
            "general_correction": "Review and improve accuracy in this area"
        }
        return actions.get(pattern['type'], "Review and improve based on feedback")
    
    def _calculate_avg_response_time(self, feedbacks: List[UserFeedback]) -> float:
        """Calculate average time to respond to prompts"""
        # This would require tracking prompt display time
        # For now, return a placeholder
        return 5.2  # seconds
    
    async def _get_most_corrected_topics(
        self,
        feedbacks: List[UserFeedback]
    ) -> List[str]:
        """Get topics that receive the most corrections"""
        topics = {}
        
        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.CORRECTION.value:
                # Extract topics from context
                context = feedback.context_data or {}
                topic = context.get('topic', 'general')
                topics[topic] = topics.get(topic, 0) + 1
        
        # Sort and return top topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:5]]
    
    async def _calculate_improvement_trend(
        self,
        feedbacks: List[UserFeedback]
    ) -> Dict[str, Any]:
        """Calculate improvement trend based on feedback"""
        if not feedbacks:
            return {"trend": "insufficient_data"}
        
        # Sort by date
        sorted_feedback = sorted(feedbacks, key=lambda f: f.created_at)
        
        # Split into periods
        midpoint = len(sorted_feedback) // 2
        first_half = sorted_feedback[:midpoint]
        second_half = sorted_feedback[midpoint:]
        
        # Calculate metrics for each period
        first_metrics = self._calculate_period_metrics(first_half)
        second_metrics = self._calculate_period_metrics(second_half)
        
        # Determine trend
        trend = {
            "direction": "stable",
            "rating_change": second_metrics['avg_rating'] - first_metrics['avg_rating'],
            "correction_change": second_metrics['correction_rate'] - first_metrics['correction_rate'],
            "confidence": 0.7
        }
        
        if trend['rating_change'] > 0.2:
            trend['direction'] = "improving"
        elif trend['rating_change'] < -0.2:
            trend['direction'] = "declining"
        
        return trend
    
    def _calculate_period_metrics(self, feedbacks: List[UserFeedback]) -> Dict[str, float]:
        """Calculate metrics for a period of feedback"""
        metrics = {
            "avg_rating": 0.0,
            "correction_rate": 0.0
        }
        
        if not feedbacks:
            return metrics
        
        ratings = []
        corrections = 0
        
        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.RATING.value:
                rating = feedback.get_rating()
                if rating:
                    ratings.append(rating)
            elif feedback.feedback_type == FeedbackType.CORRECTION.value:
                corrections += 1
        
        if ratings:
            metrics["avg_rating"] = sum(ratings) / len(ratings)
        
        metrics["correction_rate"] = corrections / len(feedbacks)
        
        return metrics