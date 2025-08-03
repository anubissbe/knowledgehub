"""
AI Feedback Loop System.

Provides continuous learning and improvement capabilities for AI integrations
through feedback collection, analysis, and model adaptation.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_, or_

from ..models.memory import Memory
from ..models.analytics import Metric
from ..database import get_db_session
from .memory_service import MemoryService
from .ai_service import AIService

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    POSITIVE = "positive"           # User liked the suggestion/result
    NEGATIVE = "negative"           # User disliked the suggestion/result
    MODIFICATION = "modification"   # User modified the suggestion
    REJECTION = "rejection"         # User completely rejected suggestion
    ACCEPTANCE = "acceptance"       # User accepted suggestion as-is
    PARTIAL = "partial"            # User used part of the suggestion
    TIMEOUT = "timeout"            # No response within timeout period
    ERROR = "error"                # Suggestion caused an error


class LearningSignal(Enum):
    """Types of learning signals extracted from feedback."""
    CONTEXT_RELEVANCE = "context_relevance"     # How relevant context was
    SUGGESTION_QUALITY = "suggestion_quality"   # Quality of AI suggestion
    TIMING_APPROPRIATENESS = "timing"           # Was timing appropriate
    USER_PREFERENCE = "user_preference"         # User's style preferences
    PATTERN_EFFECTIVENESS = "pattern_effectiveness"  # How well patterns worked
    MODEL_CONFIDENCE = "model_confidence"       # Model confidence calibration


@dataclass
class FeedbackEvent:
    """Individual feedback event."""
    id: str
    feedback_type: FeedbackType
    timestamp: datetime
    source: str                    # Which AI tool/service generated feedback
    user_id: Optional[str]
    project_id: Optional[str]
    suggestion_id: Optional[str]
    context: Dict[str, Any]        # Context when feedback was given
    details: Dict[str, Any]        # Specific feedback details
    metadata: Dict[str, Any]       # Additional metadata


@dataclass
class LearningInsight:
    """Insight extracted from feedback analysis."""
    signal_type: LearningSignal
    confidence: float
    insight: str
    supporting_data: Dict[str, Any]
    actionable_recommendations: List[str]
    affected_components: List[str]


class AIFeedbackLoop:
    """
    Main feedback loop system for continuous AI improvement.
    
    Collects feedback from various AI integrations, analyzes patterns,
    extracts learning insights, and applies improvements to enhance
    future AI performance.
    """
    
    def __init__(self):
        self.memory_service = MemoryService()
        self.ai_service = AIService()
        self.feedback_buffer: List[FeedbackEvent] = []
        self.learning_insights: List[LearningInsight] = []
        
        # Configuration
        self.buffer_size = 100
        self.analysis_interval = 300  # 5 minutes
        self.min_feedback_for_analysis = 5
        
        # Background task for periodic analysis
        self._analysis_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the feedback loop system."""
        self._running = True
        self._analysis_task = asyncio.create_task(self._periodic_analysis())
        logger.info("AI Feedback Loop system started")
    
    async def stop(self):
        """Stop the feedback loop system."""
        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("AI Feedback Loop system stopped")
    
    async def record_feedback(
        self,
        feedback_type: FeedbackType,
        source: str,
        context: Dict[str, Any],
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        suggestion_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a feedback event.
        
        Args:
            feedback_type: Type of feedback
            source: Source system that generated the feedback
            context: Context when feedback was given
            details: Specific feedback details
            user_id: User who provided feedback
            project_id: Project context
            suggestion_id: ID of suggestion being rated
            metadata: Additional metadata
            
        Returns:
            Feedback event ID
        """
        try:
            # Create feedback event
            event = FeedbackEvent(
                id=str(uuid4()),
                feedback_type=feedback_type,
                timestamp=datetime.utcnow(),
                source=source,
                user_id=user_id,
                project_id=project_id,
                suggestion_id=suggestion_id,
                context=context,
                details=details,
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.feedback_buffer.append(event)
            
            # Store in database for persistence
            await self._store_feedback_event(event)
            
            # Trigger immediate analysis if buffer is full
            if len(self.feedback_buffer) >= self.buffer_size:
                await self._process_feedback_buffer()
            
            logger.info(f"Recorded feedback event: {event.id} ({feedback_type.value})")
            return event.id
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            raise
    
    async def analyze_feedback_patterns(
        self,
        time_window_hours: int = 24,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feedback patterns for insights.
        
        Args:
            time_window_hours: Time window for analysis
            user_id: Filter by specific user
            project_id: Filter by specific project
            source: Filter by specific source system
            
        Returns:
            Analysis results with patterns and insights
        """
        try:
            # Get feedback events from the specified time window
            since = datetime.utcnow() - timedelta(hours=time_window_hours)
            events = await self._get_feedback_events(
                since=since,
                user_id=user_id,
                project_id=project_id,
                source=source
            )
            
            if len(events) < self.min_feedback_for_analysis:
                return {
                    "status": "insufficient_data",
                    "events_found": len(events),
                    "min_required": self.min_feedback_for_analysis
                }
            
            # Analyze patterns
            patterns = await self._analyze_patterns(events)
            
            # Extract learning insights
            insights = await self._extract_learning_insights(events, patterns)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(insights)
            
            return {
                "status": "analysis_complete",
                "time_window_hours": time_window_hours,
                "events_analyzed": len(events),
                "patterns": patterns,
                "insights": [asdict(insight) for insight in insights],
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            raise
    
    async def apply_learning_insights(
        self,
        insights: List[LearningInsight],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply learning insights to improve AI systems.
        
        Args:
            insights: Learning insights to apply
            dry_run: If True, only simulate changes without applying
            
        Returns:
            Results of applying insights
        """
        try:
            results = {
                "applied_insights": 0,
                "failed_insights": 0,
                "changes_made": [],
                "errors": []
            }
            
            for insight in insights:
                try:
                    # Apply insight based on its type
                    if insight.signal_type == LearningSignal.CONTEXT_RELEVANCE:
                        await self._apply_context_relevance_insight(insight, dry_run)
                    elif insight.signal_type == LearningSignal.SUGGESTION_QUALITY:
                        await self._apply_suggestion_quality_insight(insight, dry_run)
                    elif insight.signal_type == LearningSignal.USER_PREFERENCE:
                        await self._apply_user_preference_insight(insight, dry_run)
                    elif insight.signal_type == LearningSignal.PATTERN_EFFECTIVENESS:
                        await self._apply_pattern_effectiveness_insight(insight, dry_run)
                    elif insight.signal_type == LearningSignal.MODEL_CONFIDENCE:
                        await self._apply_model_confidence_insight(insight, dry_run)
                    
                    results["applied_insights"] += 1
                    results["changes_made"].extend(insight.actionable_recommendations)
                    
                except Exception as e:
                    logger.error(f"Error applying insight {insight.signal_type}: {e}")
                    results["failed_insights"] += 1
                    results["errors"].append(str(e))
            
            # Store applied insights as memories for future reference
            if not dry_run:
                await self._store_applied_insights(insights)
            
            logger.info(f"Applied {results['applied_insights']} learning insights")
            return results
            
        except Exception as e:
            logger.error(f"Error applying learning insights: {e}")
            raise
    
    async def get_user_feedback_summary(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get feedback summary for a specific user."""
        try:
            since = datetime.utcnow() - timedelta(days=days)
            events = await self._get_feedback_events(since=since, user_id=user_id)
            
            # Calculate summary statistics
            total_events = len(events)
            feedback_types = {}
            sources = {}
            
            for event in events:
                # Count by feedback type
                feedback_type = event.feedback_type.value
                feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
                
                # Count by source
                source = event.source
                sources[source] = sources.get(source, 0) + 1
            
            # Calculate satisfaction metrics
            positive_feedback = sum(
                count for feedback_type, count in feedback_types.items()
                if feedback_type in ["positive", "acceptance"]
            )
            satisfaction_rate = positive_feedback / total_events if total_events > 0 else 0
            
            return {
                "user_id": user_id,
                "period_days": days,
                "total_feedback_events": total_events,
                "satisfaction_rate": satisfaction_rate,
                "feedback_by_type": feedback_types,
                "feedback_by_source": sources,
                "summary_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user feedback summary: {e}")
            raise
    
    async def _periodic_analysis(self):
        """Periodic analysis task that runs in the background."""
        while self._running:
            try:
                await asyncio.sleep(self.analysis_interval)
                
                if len(self.feedback_buffer) >= self.min_feedback_for_analysis:
                    await self._process_feedback_buffer()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic analysis: {e}")
    
    async def _process_feedback_buffer(self):
        """Process the current feedback buffer."""
        try:
            if not self.feedback_buffer:
                return
            
            # Analyze current buffer
            patterns = await self._analyze_patterns(self.feedback_buffer)
            insights = await self._extract_learning_insights(self.feedback_buffer, patterns)
            
            # Auto-apply high-confidence insights
            high_confidence_insights = [
                insight for insight in insights
                if insight.confidence >= 0.8
            ]
            
            if high_confidence_insights:
                await self.apply_learning_insights(high_confidence_insights)
            
            # Store all insights for future reference
            self.learning_insights.extend(insights)
            
            # Clear buffer
            self.feedback_buffer.clear()
            
            logger.info(f"Processed feedback buffer: {len(insights)} insights extracted")
            
        except Exception as e:
            logger.error(f"Error processing feedback buffer: {e}")
    
    async def _analyze_patterns(self, events: List[FeedbackEvent]) -> Dict[str, Any]:
        """Analyze patterns in feedback events."""
        patterns = {
            "feedback_type_distribution": {},
            "source_performance": {},
            "temporal_patterns": {},
            "user_patterns": {},
            "context_patterns": {}
        }
        
        # Analyze feedback type distribution
        for event in events:
            feedback_type = event.feedback_type.value
            patterns["feedback_type_distribution"][feedback_type] = \
                patterns["feedback_type_distribution"].get(feedback_type, 0) + 1
        
        # Analyze source performance
        for event in events:
            source = event.source
            if source not in patterns["source_performance"]:
                patterns["source_performance"][source] = {
                    "total": 0, "positive": 0, "negative": 0
                }
            
            patterns["source_performance"][source]["total"] += 1
            
            if event.feedback_type in [FeedbackType.POSITIVE, FeedbackType.ACCEPTANCE]:
                patterns["source_performance"][source]["positive"] += 1
            elif event.feedback_type in [FeedbackType.NEGATIVE, FeedbackType.REJECTION]:
                patterns["source_performance"][source]["negative"] += 1
        
        # Calculate performance ratios
        for source, stats in patterns["source_performance"].items():
            if stats["total"] > 0:
                stats["positive_ratio"] = stats["positive"] / stats["total"]
                stats["negative_ratio"] = stats["negative"] / stats["total"]
        
        # Analyze temporal patterns (by hour of day)
        hourly_feedback = {}
        for event in events:
            hour = event.timestamp.hour
            hourly_feedback[hour] = hourly_feedback.get(hour, 0) + 1
        patterns["temporal_patterns"]["by_hour"] = hourly_feedback
        
        return patterns
    
    async def _extract_learning_insights(
        self,
        events: List[FeedbackEvent],
        patterns: Dict[str, Any]
    ) -> List[LearningInsight]:
        """Extract learning insights from events and patterns."""
        insights = []
        
        # Insight 1: Context relevance from modification patterns
        modification_events = [e for e in events if e.feedback_type == FeedbackType.MODIFICATION]
        if modification_events:
            # Analyze what context was missing or irrelevant
            insight = await self._analyze_context_relevance(modification_events)
            if insight:
                insights.append(insight)
        
        # Insight 2: Source performance insights
        source_performance = patterns.get("source_performance", {})
        for source, stats in source_performance.items():
            if stats["total"] >= 5:  # Minimum events for reliable insight
                if stats["positive_ratio"] < 0.3:  # Low performance
                    insights.append(LearningInsight(
                        signal_type=LearningSignal.SUGGESTION_QUALITY,
                        confidence=0.7,
                        insight=f"Source {source} has low user satisfaction ({stats['positive_ratio']:.1%})",
                        supporting_data={"source": source, "stats": stats},
                        actionable_recommendations=[
                            f"Review {source} suggestion algorithm",
                            f"Improve context gathering for {source}",
                            f"Add more user preference learning for {source}"
                        ],
                        affected_components=[source]
                    ))
        
        # Insight 3: User preference patterns
        user_patterns = await self._analyze_user_preferences(events)
        insights.extend(user_patterns)
        
        # Insight 4: Temporal effectiveness
        temporal_insights = await self._analyze_temporal_effectiveness(events, patterns)
        insights.extend(temporal_insights)
        
        return insights
    
    async def _analyze_context_relevance(
        self,
        modification_events: List[FeedbackEvent]
    ) -> Optional[LearningInsight]:
        """Analyze context relevance from modification patterns."""
        try:
            if len(modification_events) < 3:
                return None
            
            # Analyze what context was commonly modified
            context_issues = {}
            for event in modification_events:
                details = event.details
                if "modified_parts" in details:
                    for part in details["modified_parts"]:
                        context_issues[part] = context_issues.get(part, 0) + 1
            
            if context_issues:
                most_modified = max(context_issues.items(), key=lambda x: x[1])
                
                return LearningInsight(
                    signal_type=LearningSignal.CONTEXT_RELEVANCE,
                    confidence=0.6,
                    insight=f"Context component '{most_modified[0]}' frequently modified by users",
                    supporting_data={"modification_counts": context_issues},
                    actionable_recommendations=[
                        f"Improve quality of {most_modified[0]} context",
                        f"Reduce weight of {most_modified[0]} in suggestions",
                        f"Add user preferences for {most_modified[0]} context"
                    ],
                    affected_components=["context_injector", "suggestion_enhancer"]
                )
        except Exception as e:
            logger.error(f"Error analyzing context relevance: {e}")
        
        return None
    
    async def _analyze_user_preferences(
        self,
        events: List[FeedbackEvent]
    ) -> List[LearningInsight]:
        """Analyze user preference patterns."""
        insights = []
        
        # Group events by user
        user_events = {}
        for event in events:
            if event.user_id:
                if event.user_id not in user_events:
                    user_events[event.user_id] = []
                user_events[event.user_id].append(event)
        
        # Analyze each user's patterns
        for user_id, user_event_list in user_events.items():
            if len(user_event_list) >= 5:  # Minimum for pattern detection
                # Check for consistent preferences
                positive_events = [
                    e for e in user_event_list
                    if e.feedback_type in [FeedbackType.POSITIVE, FeedbackType.ACCEPTANCE]
                ]
                
                if len(positive_events) >= 3:
                    # Extract common patterns from positive feedback
                    patterns = self._extract_user_patterns(positive_events)
                    if patterns:
                        insights.append(LearningInsight(
                            signal_type=LearningSignal.USER_PREFERENCE,
                            confidence=0.5,
                            insight=f"User {user_id} prefers specific patterns",
                            supporting_data={"user_id": user_id, "patterns": patterns},
                            actionable_recommendations=[
                                f"Personalize suggestions for user {user_id}",
                                "Store user preference patterns",
                                "Weight suggestions based on user history"
                            ],
                            affected_components=["suggestion_enhancer", "context_injector"]
                        ))
        
        return insights
    
    async def _analyze_temporal_effectiveness(
        self,
        events: List[FeedbackEvent],
        patterns: Dict[str, Any]
    ) -> List[LearningInsight]:
        """Analyze temporal effectiveness patterns."""
        insights = []
        
        # Check if certain hours have consistently bad feedback
        hourly_patterns = patterns.get("temporal_patterns", {}).get("by_hour", {})
        
        if hourly_patterns:
            # Calculate feedback quality by hour
            hourly_quality = {}
            for event in events:
                hour = event.timestamp.hour
                if hour not in hourly_quality:
                    hourly_quality[hour] = {"total": 0, "positive": 0}
                
                hourly_quality[hour]["total"] += 1
                if event.feedback_type in [FeedbackType.POSITIVE, FeedbackType.ACCEPTANCE]:
                    hourly_quality[hour]["positive"] += 1
            
            # Find hours with poor performance
            poor_hours = []
            for hour, stats in hourly_quality.items():
                if stats["total"] >= 3:  # Minimum events
                    quality_ratio = stats["positive"] / stats["total"]
                    if quality_ratio < 0.4:  # Poor performance threshold
                        poor_hours.append((hour, quality_ratio))
            
            if poor_hours:
                insights.append(LearningInsight(
                    signal_type=LearningSignal.TIMING_APPROPRIATENESS,
                    confidence=0.4,
                    insight=f"Poor AI performance during hours: {[h[0] for h in poor_hours]}",
                    supporting_data={"poor_hours": poor_hours},
                    actionable_recommendations=[
                        "Investigate context quality during poor-performing hours",
                        "Adjust suggestion algorithms for time-sensitive patterns",
                        "Consider user fatigue factors during certain hours"
                    ],
                    affected_components=["suggestion_enhancer", "context_injector"]
                ))
        
        return insights
    
    def _extract_user_patterns(self, positive_events: List[FeedbackEvent]) -> Dict[str, Any]:
        """Extract common patterns from user's positive feedback."""
        patterns = {}
        
        # Extract common context elements
        context_elements = {}
        for event in positive_events:
            context = event.context
            for key, value in context.items():
                if key not in context_elements:
                    context_elements[key] = {}
                str_value = str(value)
                context_elements[key][str_value] = context_elements[key].get(str_value, 0) + 1
        
        # Find frequently occurring context elements
        for key, values in context_elements.items():
            if len(values) > 1:  # Multiple different values
                most_common = max(values.items(), key=lambda x: x[1])
                if most_common[1] >= len(positive_events) * 0.6:  # Appears in 60%+ of events
                    patterns[key] = most_common[0]
        
        return patterns
    
    async def _apply_context_relevance_insight(
        self,
        insight: LearningInsight,
        dry_run: bool
    ) -> None:
        """Apply context relevance learning insight."""
        if not dry_run:
            # Store insight as memory for context injector to use
            await self.memory_service.create_memory(
                content=f"Context relevance insight: {insight.insight}",
                memory_type="learning",
                metadata={
                    "insight_type": insight.signal_type.value,
                    "confidence": insight.confidence,
                    "recommendations": insight.actionable_recommendations,
                    "supporting_data": insight.supporting_data
                }
            )
    
    async def _apply_suggestion_quality_insight(
        self,
        insight: LearningInsight,
        dry_run: bool
    ) -> None:
        """Apply suggestion quality learning insight."""
        if not dry_run:
            # Store quality improvement guidance
            await self.memory_service.create_memory(
                content=f"Suggestion quality insight: {insight.insight}",
                memory_type="learning",
                metadata={
                    "insight_type": insight.signal_type.value,
                    "confidence": insight.confidence,
                    "affected_components": insight.affected_components,
                    "recommendations": insight.actionable_recommendations
                }
            )
    
    async def _apply_user_preference_insight(
        self,
        insight: LearningInsight,
        dry_run: bool
    ) -> None:
        """Apply user preference learning insight."""
        if not dry_run:
            supporting_data = insight.supporting_data
            user_id = supporting_data.get("user_id")
            patterns = supporting_data.get("patterns", {})
            
            if user_id and patterns:
                # Store user-specific preference patterns
                await self.memory_service.create_memory(
                    content=f"User preference patterns for {user_id}",
                    memory_type="user_preference",
                    user_id=user_id,
                    metadata={
                        "patterns": patterns,
                        "confidence": insight.confidence,
                        "recommendations": insight.actionable_recommendations
                    }
                )
    
    async def _apply_pattern_effectiveness_insight(
        self,
        insight: LearningInsight,
        dry_run: bool
    ) -> None:
        """Apply pattern effectiveness learning insight."""
        if not dry_run:
            # Store pattern effectiveness data
            await self.memory_service.create_memory(
                content=f"Pattern effectiveness insight: {insight.insight}",
                memory_type="learning",
                metadata={
                    "insight_type": insight.signal_type.value,
                    "confidence": insight.confidence,
                    "supporting_data": insight.supporting_data
                }
            )
    
    async def _apply_model_confidence_insight(
        self,
        insight: LearningInsight,
        dry_run: bool
    ) -> None:
        """Apply model confidence calibration insight."""
        if not dry_run:
            # Store confidence calibration data
            await self.memory_service.create_memory(
                content=f"Model confidence insight: {insight.insight}",
                memory_type="learning",
                metadata={
                    "insight_type": insight.signal_type.value,
                    "confidence": insight.confidence,
                    "calibration_data": insight.supporting_data
                }
            )
    
    async def _store_feedback_event(self, event: FeedbackEvent) -> None:
        """Store feedback event in database."""
        try:
            async with get_db_session() as session:
                # Store as a metric for analytics
                metric = Metric(
                    name="ai_feedback",
                    value=1.0,
                    metric_type="counter",
                    tags={
                        "feedback_type": event.feedback_type.value,
                        "source": event.source,
                        "user_id": event.user_id or "anonymous",
                        "project_id": event.project_id or "none"
                    },
                    metadata={
                        "feedback_event": asdict(event),
                        "context": event.context,
                        "details": event.details
                    }
                )
                
                session.add(metric)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error storing feedback event: {e}")
    
    async def _store_applied_insights(self, insights: List[LearningInsight]) -> None:
        """Store applied insights as memories."""
        try:
            for insight in insights:
                await self.memory_service.create_memory(
                    content=f"Applied learning insight: {insight.insight}",
                    memory_type="applied_learning",
                    metadata={
                        "signal_type": insight.signal_type.value,
                        "confidence": insight.confidence,
                        "recommendations": insight.actionable_recommendations,
                        "affected_components": insight.affected_components,
                        "applied_timestamp": datetime.utcnow().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Error storing applied insights: {e}")
    
    async def _get_feedback_events(
        self,
        since: Optional[datetime] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 1000
    ) -> List[FeedbackEvent]:
        """Get feedback events from database."""
        try:
            async with get_db_session() as session:
                query = select(Metric).where(Metric.name == "ai_feedback")
                
                if since:
                    query = query.where(Metric.created_at >= since)
                
                if user_id:
                    query = query.where(Metric.tags["user_id"].astext == user_id)
                
                if project_id:
                    query = query.where(Metric.tags["project_id"].astext == project_id)
                
                if source:
                    query = query.where(Metric.tags["source"].astext == source)
                
                query = query.order_by(desc(Metric.created_at)).limit(limit)
                
                result = await session.execute(query)
                metrics = result.scalars().all()
                
                # Convert metrics back to feedback events
                events = []
                for metric in metrics:
                    try:
                        event_data = metric.metadata.get("feedback_event", {})
                        if event_data:
                            # Reconstruct FeedbackEvent from stored data
                            event = FeedbackEvent(
                                id=event_data["id"],
                                feedback_type=FeedbackType(event_data["feedback_type"]),
                                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                                source=event_data["source"],
                                user_id=event_data.get("user_id"),
                                project_id=event_data.get("project_id"),
                                suggestion_id=event_data.get("suggestion_id"),
                                context=event_data.get("context", {}),
                                details=event_data.get("details", {}),
                                metadata=event_data.get("metadata", {})
                            )
                            events.append(event)
                    except Exception as e:
                        logger.error(f"Error reconstructing feedback event: {e}")
                        continue
                
                return events
                
        except Exception as e:
            logger.error(f"Error getting feedback events: {e}")
            return []
    
    async def _generate_recommendations(
        self,
        insights: List[LearningInsight]
    ) -> List[str]:
        """Generate actionable recommendations from insights."""
        recommendations = []
        
        # Collect all recommendations from insights
        for insight in insights:
            recommendations.extend(insight.actionable_recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations