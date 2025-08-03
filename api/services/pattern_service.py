"""
Pattern Recognition and Analysis Service.

This service provides comprehensive pattern recognition capabilities:
- Code pattern analysis
- Usage pattern detection
- Performance pattern identification
- Pattern-based recommendations
- Pattern learning and adaptation
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import json

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, and_, or_, desc

from ..models.base import get_db_context
from ..models.memory import MemoryItem
from ..models.session import Session
from ..models.error_pattern import ErrorOccurrence
from ..models.enhanced_decision import EnhancedDecision
from ..models.workflow import WorkflowPattern
from ..ml.pattern_recognition import (
    pattern_recognition_engine, PatternMatch, PatternType, 
    PatternCategory, CodeMetrics
)
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("pattern_service")


class PatternService:
    """
    Comprehensive pattern recognition and analysis service.
    
    Features:
    - Multi-dimensional pattern detection
    - Real-time pattern analysis
    - Pattern-based recommendations
    - Performance optimization suggestions
    - Learning from user feedback
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.analytics_service = TimeSeriesAnalyticsService()
        self._initialized = False
        
        # Pattern analysis configuration
        self.analysis_config = {
            "code_analysis_batch_size": 100,
            "usage_analysis_window_hours": 168,  # 1 week
            "performance_analysis_window_hours": 24,
            "pattern_cache_ttl": 3600,  # 1 hour
            "min_confidence_threshold": 0.6
        }
        
        # Pattern learning weights
        self.learning_weights = {
            "user_feedback": 0.4,
            "historical_accuracy": 0.3,
            "frequency": 0.2,
            "recency": 0.1
        }
        
        logger.info("Initialized PatternService")
    
    async def initialize(self):
        """Initialize the pattern service."""
        if self._initialized:
            return
        
        try:
            await self.analytics_service.initialize()
            await redis_client.initialize()
            
            self._initialized = True
            logger.info("PatternService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PatternService: {e}")
            raise
    
    async def analyze_code_patterns(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        file_patterns: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze code patterns for a user or project.
        
        Args:
            user_id: Filter by user ID
            project_id: Filter by project ID
            file_patterns: File patterns to analyze
            force_refresh: Force fresh analysis
            
        Returns:
            Analysis results with patterns and recommendations
        """
        try:
            # Check cache first
            cache_key = f"code_patterns:{user_id}:{project_id}"
            if not force_refresh:
                cached_result = await redis_client.get(cache_key)
                if cached_result:
                    logger.debug("Using cached code pattern analysis")
                    return json.loads(cached_result)
            
            # Get code samples
            code_samples = await self._get_code_samples(
                user_id, project_id, file_patterns
            )
            
            if not code_samples:
                return {"patterns": [], "recommendations": [], "metrics": {}}
            
            # Analyze patterns
            patterns = await pattern_recognition_engine.analyze_code_patterns(
                code_samples, user_id, project_id
            )
            
            # Generate recommendations
            recommendations = await pattern_recognition_engine.suggest_optimizations(
                patterns
            )
            
            # Calculate overall metrics
            metrics = self._calculate_code_metrics(patterns, code_samples)
            
            # Prepare result
            result = {
                "patterns": [self._pattern_to_dict(p) for p in patterns],
                "recommendations": recommendations,
                "metrics": metrics,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "samples_analyzed": len(code_samples)
            }
            
            # Cache result
            await redis_client.setex(
                cache_key,
                self.analysis_config["pattern_cache_ttl"],
                json.dumps(result)
            )
            
            # Record analytics
            await self.analytics_service.record_metric(
                metric_type="code_patterns_analyzed",
                value=len(patterns),
                tags={
                    "user_id": user_id or "all",
                    "project_id": project_id or "all"
                }
            )
            
            logger.info(f"Analyzed {len(code_samples)} code samples, found {len(patterns)} patterns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze code patterns: {e}")
            return {"patterns": [], "recommendations": [], "metrics": {}, "error": str(e)}
    
    async def analyze_usage_patterns(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze user behavior and usage patterns.
        
        Args:
            user_id: Filter by user ID
            project_id: Filter by project ID
            time_window_hours: Analysis time window
            
        Returns:
            Usage pattern analysis results
        """
        try:
            window_hours = time_window_hours or self.analysis_config["usage_analysis_window_hours"]
            
            # Get user activities
            activities = await self._get_user_activities(
                user_id, project_id, window_hours
            )
            
            if not activities:
                return {"patterns": [], "insights": [], "metrics": {}}
            
            # Analyze patterns
            patterns = await pattern_recognition_engine.analyze_usage_patterns(
                activities, window_hours
            )
            
            # Generate insights
            insights = self._generate_usage_insights(patterns, activities)
            
            # Calculate usage metrics
            metrics = self._calculate_usage_metrics(activities)
            
            result = {
                "patterns": [self._pattern_to_dict(p) for p in patterns],
                "insights": insights,
                "metrics": metrics,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "activities_analyzed": len(activities),
                "time_window_hours": window_hours
            }
            
            # Record analytics
            await self.analytics_service.record_metric(
                metric_type="usage_patterns_analyzed",
                value=len(patterns),
                tags={
                    "user_id": user_id or "all",
                    "project_id": project_id or "all"
                }
            )
            
            logger.info(f"Analyzed {len(activities)} activities, found {len(patterns)} usage patterns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze usage patterns: {e}")
            return {"patterns": [], "insights": [], "metrics": {}, "error": str(e)}
    
    async def analyze_performance_patterns(
        self,
        system_id: Optional[str] = None,
        metric_types: Optional[List[str]] = None,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance patterns and bottlenecks.
        
        Args:
            system_id: Filter by system ID
            metric_types: Types of metrics to analyze
            time_window_hours: Analysis time window
            
        Returns:
            Performance pattern analysis results
        """
        try:
            window_hours = time_window_hours or self.analysis_config["performance_analysis_window_hours"]
            
            # Get performance data
            performance_data = await self._get_performance_data(
                system_id, metric_types, window_hours
            )
            
            if not performance_data:
                return {"patterns": [], "bottlenecks": [], "recommendations": []}
            
            # Analyze patterns
            patterns = await pattern_recognition_engine.analyze_performance_patterns(
                performance_data
            )
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(performance_data, patterns)
            
            # Generate performance recommendations
            recommendations = self._generate_performance_recommendations(
                patterns, bottlenecks
            )
            
            result = {
                "patterns": [self._pattern_to_dict(p) for p in patterns],
                "bottlenecks": bottlenecks,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "data_points_analyzed": len(performance_data),
                "time_window_hours": window_hours
            }
            
            # Record analytics
            await self.analytics_service.record_metric(
                metric_type="performance_patterns_analyzed",
                value=len(patterns),
                tags={
                    "system_id": system_id or "all"
                }
            )
            
            logger.info(f"Analyzed {len(performance_data)} performance data points, found {len(patterns)} patterns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze performance patterns: {e}")
            return {"patterns": [], "bottlenecks": [], "recommendations": [], "error": str(e)}
    
    async def analyze_error_patterns(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        error_types: Optional[List[str]] = None,
        time_window_hours: int = 72
    ) -> Dict[str, Any]:
        """
        Analyze error patterns and sequences.
        
        Args:
            user_id: Filter by user ID
            project_id: Filter by project ID
            error_types: Types of errors to analyze
            time_window_hours: Analysis time window
            
        Returns:
            Error pattern analysis results
        """
        try:
            # Get error data
            errors = await self._get_error_data(
                user_id, project_id, error_types, time_window_hours
            )
            
            if not errors:
                return {"patterns": [], "hotspots": [], "prevention_strategies": []}
            
            # Analyze patterns
            patterns = await pattern_recognition_engine.analyze_error_patterns(errors)
            
            # Identify error hotspots
            hotspots = self._identify_error_hotspots(errors)
            
            # Generate prevention strategies
            prevention_strategies = self._generate_prevention_strategies(patterns, errors)
            
            result = {
                "patterns": [self._pattern_to_dict(p) for p in patterns],
                "hotspots": hotspots,
                "prevention_strategies": prevention_strategies,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "errors_analyzed": len(errors),
                "time_window_hours": time_window_hours
            }
            
            # Record analytics
            await self.analytics_service.record_metric(
                metric_type="error_patterns_analyzed",
                value=len(patterns),
                tags={
                    "user_id": user_id or "all",
                    "project_id": project_id or "all"
                }
            )
            
            logger.info(f"Analyzed {len(errors)} errors, found {len(patterns)} error patterns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze error patterns: {e}")
            return {"patterns": [], "hotspots": [], "prevention_strategies": [], "error": str(e)}
    
    async def get_pattern_recommendations(
        self,
        pattern_types: Optional[List[PatternType]] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get pattern-based recommendations.
        
        Args:
            pattern_types: Types of patterns to consider
            user_id: Filter by user ID
            project_id: Filter by project ID
            limit: Maximum number of recommendations
            
        Returns:
            List of pattern-based recommendations
        """
        try:
            # Get recent patterns from cache
            patterns = await self._get_cached_patterns(
                pattern_types, user_id, project_id
            )
            
            if not patterns:
                return []
            
            # Generate recommendations
            recommendations = await pattern_recognition_engine.suggest_optimizations(
                patterns
            )
            
            # Sort by impact score and limit
            recommendations.sort(
                key=lambda x: x.get("impact_score", 0), 
                reverse=True
            )
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get pattern recommendations: {e}")
            return []
    
    async def provide_pattern_feedback(
        self,
        pattern_id: str,
        feedback_type: str,  # "helpful", "not_helpful", "incorrect"
        user_id: str,
        feedback_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Provide feedback on pattern detection accuracy.
        
        Args:
            pattern_id: ID of the pattern
            feedback_type: Type of feedback
            user_id: User providing feedback
            feedback_details: Additional feedback details
            
        Returns:
            Success status
        """
        try:
            # Store feedback
            feedback_data = {
                "pattern_id": pattern_id,
                "feedback_type": feedback_type,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": feedback_details or {}
            }
            
            feedback_key = f"pattern_feedback:{pattern_id}:{user_id}"
            await redis_client.setex(
                feedback_key,
                86400 * 7,  # Keep for 7 days
                json.dumps(feedback_data)
            )
            
            # Update pattern confidence based on feedback
            await self._update_pattern_confidence(pattern_id, feedback_type)
            
            # Record analytics
            await self.analytics_service.record_metric(
                metric_type="pattern_feedback",
                value=1,
                tags={
                    "feedback_type": feedback_type,
                    "user_id": user_id
                }
            )
            
            logger.info(f"Recorded pattern feedback: {pattern_id} - {feedback_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to provide pattern feedback: {e}")
            return False
    
    async def get_pattern_analytics(
        self,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get pattern recognition analytics.
        
        Args:
            time_window_days: Time window for analytics
            
        Returns:
            Pattern analytics data
        """
        try:
            # Get pattern detection metrics
            detection_metrics = await self._get_pattern_detection_metrics(time_window_days)
            
            # Get feedback metrics
            feedback_metrics = await self._get_pattern_feedback_metrics(time_window_days)
            
            # Get accuracy metrics
            accuracy_metrics = await self._get_pattern_accuracy_metrics(time_window_days)
            
            # Get usage metrics
            usage_metrics = await self._get_pattern_usage_metrics(time_window_days)
            
            return {
                "detection_metrics": detection_metrics,
                "feedback_metrics": feedback_metrics,
                "accuracy_metrics": accuracy_metrics,
                "usage_metrics": usage_metrics,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "time_window_days": time_window_days
            }
            
        except Exception as e:
            logger.error(f"Failed to get pattern analytics: {e}")
            return {"error": str(e)}
    
    # Internal methods
    
    async def _get_code_samples(
        self,
        user_id: Optional[str],
        project_id: Optional[str],
        file_patterns: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Get code samples for analysis."""
        try:
            with get_db_context() as db:
                query = db.query(MemoryItem).filter(
                    MemoryItem.memory_type == "code"
                )
                
                if user_id:
                    query = query.filter(MemoryItem.user_id == user_id)
                
                if project_id:
                    query = query.filter(
                        MemoryItem.metadata["project_id"].astext == project_id
                    )
                
                # Get recent code samples
                samples = query.order_by(
                    MemoryItem.created_at.desc()
                ).limit(self.analysis_config["code_analysis_batch_size"]).all()
                
                code_samples = []
                for sample in samples:
                    code_samples.append({
                        "id": str(sample.id),
                        "content": sample.content,
                        "file": sample.metadata.get("file", "unknown"),
                        "language": sample.metadata.get("language", "unknown"),
                        "user_id": sample.user_id,
                        "project_id": sample.metadata.get("project_id"),
                        "created_at": sample.created_at,
                        "metadata": sample.metadata
                    })
                
                return code_samples
                
        except Exception as e:
            logger.error(f"Failed to get code samples: {e}")
            return []
    
    async def _get_user_activities(
        self,
        user_id: Optional[str],
        project_id: Optional[str],
        window_hours: int
    ) -> List[Dict[str, Any]]:
        """Get user activities for analysis."""
        try:
            with get_db_context() as db:
                window_start = datetime.utcnow() - timedelta(hours=window_hours)
                
                query = db.query(MemoryItem).filter(
                    MemoryItem.created_at >= window_start
                )
                
                if user_id:
                    query = query.filter(MemoryItem.user_id == user_id)
                
                if project_id:
                    query = query.filter(
                        MemoryItem.metadata["project_id"].astext == project_id
                    )
                
                activities = query.order_by(MemoryItem.created_at).all()
                
                activity_data = []
                for activity in activities:
                    activity_data.append({
                        "id": str(activity.id),
                        "user_id": activity.user_id,
                        "action": activity.memory_type,
                        "feature": activity.metadata.get("feature", "unknown"),
                        "project_id": activity.metadata.get("project_id"),
                        "session_id": activity.metadata.get("session_id"),
                        "timestamp": activity.created_at,
                        "content_length": len(activity.content),
                        "metadata": activity.metadata
                    })
                
                return activity_data
                
        except Exception as e:
            logger.error(f"Failed to get user activities: {e}")
            return []
    
    async def _get_performance_data(
        self,
        system_id: Optional[str],
        metric_types: Optional[List[str]],
        window_hours: int
    ) -> List[Dict[str, Any]]:
        """Get performance data for analysis."""
        try:
            # Get from analytics service
            performance_data = []
            
            metrics_to_analyze = metric_types or [
                "response_time", "cpu_usage", "memory_usage", "api_calls"
            ]
            
            for metric_type in metrics_to_analyze:
                history = await self.analytics_service.get_metric_history(
                    metric_type=metric_type,
                    hours=window_hours
                )
                
                for point in history:
                    performance_data.append({
                        "operation": metric_type,
                        "duration_ms": point.get("value", 0),
                        "cpu_usage": point.get("cpu_usage", 0),
                        "memory_usage": point.get("memory_usage", 0),
                        "timestamp": point.get("timestamp", datetime.utcnow()),
                        "system_id": system_id or "default",
                        "metadata": point
                    })
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get performance data: {e}")
            return []
    
    async def _get_error_data(
        self,
        user_id: Optional[str],
        project_id: Optional[str],
        error_types: Optional[List[str]],
        window_hours: int
    ) -> List[Dict[str, Any]]:
        """Get error data for analysis."""
        try:
            with get_db_context() as db:
                window_start = datetime.utcnow() - timedelta(hours=window_hours)
                
                query = db.query(ErrorOccurrence).filter(
                    ErrorOccurrence.timestamp >= window_start
                )
                
                if user_id:
                    query = query.filter(ErrorOccurrence.user_id == user_id)
                
                if project_id:
                    query = query.filter(
                        ErrorOccurrence.execution_context["project_id"].astext == project_id
                    )
                
                if error_types:
                    query = query.filter(
                        ErrorOccurrence.pattern.has(
                            error_type=error_types
                        )
                    )
                
                errors = query.order_by(ErrorOccurrence.timestamp).all()
                
                error_data = []
                for error in errors:
                    error_data.append({
                        "id": str(error.id),
                        "message": error.error_message,
                        "type": error.pattern.error_type if error.pattern else "unknown",
                        "timestamp": error.timestamp,
                        "user_id": error.user_id,
                        "project_id": error.execution_context.get("project_id"),
                        "resolved": error.resolved,
                        "stack_trace": error.stack_trace,
                        "execution_context": error.execution_context
                    })
                
                return error_data
                
        except Exception as e:
            logger.error(f"Failed to get error data: {e}")
            return []
    
    async def _get_cached_patterns(
        self,
        pattern_types: Optional[List[PatternType]],
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[PatternMatch]:
        """Get cached patterns for recommendations."""
        try:
            patterns = []
            
            # Try to get from various analysis caches
            cache_keys = [
                f"code_patterns:{user_id}:{project_id}",
                f"usage_patterns:{user_id}:{project_id}",
                f"performance_patterns:all:all",
                f"error_patterns:{user_id}:{project_id}"
            ]
            
            for cache_key in cache_keys:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    try:
                        data = json.loads(cached_data)
                        cached_patterns = data.get("patterns", [])
                        
                        for pattern_dict in cached_patterns:
                            # Convert dict back to PatternMatch
                            pattern = self._dict_to_pattern(pattern_dict)
                            if pattern and (not pattern_types or pattern.pattern_type in pattern_types):
                                patterns.append(pattern)
                                
                    except Exception as e:
                        logger.warning(f"Failed to parse cached patterns: {e}")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get cached patterns: {e}")
            return []
    
    def _pattern_to_dict(self, pattern: PatternMatch) -> Dict[str, Any]:
        """Convert PatternMatch to dictionary."""
        return {
            "pattern_type": pattern.pattern_type.value,
            "category": pattern.category.value,
            "confidence": pattern.confidence,
            "description": pattern.description,
            "examples": pattern.examples,
            "metadata": pattern.metadata,
            "improvement_suggestion": pattern.improvement_suggestion,
            "impact_level": pattern.impact_level
        }
    
    def _dict_to_pattern(self, pattern_dict: Dict[str, Any]) -> Optional[PatternMatch]:
        """Convert dictionary to PatternMatch."""
        try:
            return PatternMatch(
                pattern_type=PatternType(pattern_dict["pattern_type"]),
                category=PatternCategory(pattern_dict["category"]),
                confidence=pattern_dict["confidence"],
                description=pattern_dict["description"],
                examples=pattern_dict["examples"],
                metadata=pattern_dict["metadata"],
                improvement_suggestion=pattern_dict.get("improvement_suggestion"),
                impact_level=pattern_dict.get("impact_level", "medium")
            )
        except Exception as e:
            logger.warning(f"Failed to convert dict to pattern: {e}")
            return None
    
    def _calculate_code_metrics(
        self,
        patterns: List[PatternMatch],
        code_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall code metrics."""
        try:
            # Count patterns by category
            category_counts = {}
            for pattern in patterns:
                category = pattern.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Calculate quality score
            quality_issues = len([p for p in patterns if p.category == PatternCategory.QUALITY])
            quality_score = max(0, 100 - (quality_issues * 10))
            
            # Calculate complexity score
            complexity_patterns = [
                p for p in patterns 
                if "complexity" in p.description.lower()
            ]
            complexity_score = max(0, 100 - (len(complexity_patterns) * 15))
            
            return {
                "total_patterns": len(patterns),
                "category_breakdown": category_counts,
                "quality_score": quality_score,
                "complexity_score": complexity_score,
                "samples_analyzed": len(code_samples),
                "avg_confidence": sum(p.confidence for p in patterns) / len(patterns) if patterns else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate code metrics: {e}")
            return {}
    
    def _calculate_usage_metrics(
        self,
        activities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate usage metrics."""
        try:
            if not activities:
                return {}
            
            # Calculate activity frequency
            total_activities = len(activities)
            unique_users = len(set(a["user_id"] for a in activities))
            unique_features = len(set(a["feature"] for a in activities))
            
            # Calculate time span
            timestamps = [a["timestamp"] for a in activities]
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600  # hours
            
            # Calculate activity rate
            activity_rate = total_activities / max(time_span, 1)
            
            return {
                "total_activities": total_activities,
                "unique_users": unique_users,
                "unique_features": unique_features,
                "time_span_hours": time_span,
                "activity_rate_per_hour": activity_rate,
                "avg_content_length": sum(a["content_length"] for a in activities) / total_activities
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate usage metrics: {e}")
            return {}
    
    def _generate_usage_insights(
        self,
        patterns: List[PatternMatch],
        activities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate usage insights from patterns."""
        insights = []
        
        try:
            # Peak usage times
            behavioral_patterns = [
                p for p in patterns 
                if p.category == PatternCategory.BEHAVIORAL
            ]
            
            for pattern in behavioral_patterns:
                if "peak" in pattern.description.lower():
                    insights.append({
                        "type": "peak_usage",
                        "title": "Peak Usage Time Detected",
                        "description": pattern.description,
                        "recommendation": pattern.improvement_suggestion,
                        "confidence": pattern.confidence
                    })
                elif "collaboration" in pattern.description.lower():
                    insights.append({
                        "type": "collaboration",
                        "title": "Collaboration Pattern Found",
                        "description": pattern.description,
                        "recommendation": pattern.improvement_suggestion,
                        "confidence": pattern.confidence
                    })
            
            # Feature usage insights
            if activities:
                feature_usage = {}
                for activity in activities:
                    feature = activity["feature"]
                    feature_usage[feature] = feature_usage.get(feature, 0) + 1
                
                most_used = max(feature_usage, key=feature_usage.get)
                least_used = min(feature_usage, key=feature_usage.get)
                
                insights.append({
                    "type": "feature_usage",
                    "title": "Feature Usage Analysis",
                    "description": f"Most used: {most_used}, Least used: {least_used}",
                    "recommendation": "Consider optimizing most-used features and improving discoverability of least-used ones",
                    "confidence": 0.8
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate usage insights: {e}")
            return []
    
    def _identify_bottlenecks(
        self,
        performance_data: List[Dict[str, Any]],
        patterns: List[PatternMatch]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        try:
            # Group by operation
            operation_stats = {}
            for data_point in performance_data:
                operation = data_point["operation"]
                duration = data_point["duration_ms"]
                
                if operation not in operation_stats:
                    operation_stats[operation] = {
                        "durations": [],
                        "call_count": 0
                    }
                
                operation_stats[operation]["durations"].append(duration)
                operation_stats[operation]["call_count"] += 1
            
            # Find slow operations
            for operation, stats in operation_stats.items():
                if stats["call_count"] >= 5:  # Need enough samples
                    avg_duration = sum(stats["durations"]) / len(stats["durations"])
                    max_duration = max(stats["durations"])
                    
                    if avg_duration > 1000:  # More than 1 second
                        severity = "high" if avg_duration > 5000 else "medium"
                        
                        bottlenecks.append({
                            "type": "slow_operation",
                            "operation": operation,
                            "avg_duration_ms": avg_duration,
                            "max_duration_ms": max_duration,
                            "call_count": stats["call_count"],
                            "severity": severity,
                            "impact_score": avg_duration * stats["call_count"] / 1000
                        })
            
            # Sort by impact score
            bottlenecks.sort(key=lambda x: x["impact_score"], reverse=True)
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Failed to identify bottlenecks: {e}")
            return []
    
    def _generate_performance_recommendations(
        self,
        patterns: List[PatternMatch],
        bottlenecks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate performance recommendations."""
        recommendations = []
        
        try:
            # Recommendations from patterns
            for pattern in patterns:
                if pattern.pattern_type == PatternType.PERFORMANCE_BOTTLENECK:
                    recommendations.append({
                        "type": "pattern_based",
                        "title": f"Address {pattern.description}",
                        "description": pattern.improvement_suggestion,
                        "priority": pattern.impact_level,
                        "confidence": pattern.confidence
                    })
            
            # Recommendations from bottlenecks
            for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
                if bottleneck["severity"] == "high":
                    recommendations.append({
                        "type": "bottleneck",
                        "title": f"Optimize {bottleneck['operation']}",
                        "description": f"Operation taking {bottleneck['avg_duration_ms']:.0f}ms on average",
                        "priority": "high",
                        "confidence": 0.9
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate performance recommendations: {e}")
            return []
    
    def _identify_error_hotspots(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify error hotspots."""
        hotspots = []
        
        try:
            # Group by project
            project_errors = {}
            for error in errors:
                project_id = error.get("project_id", "unknown")
                if project_id not in project_errors:
                    project_errors[project_id] = []
                project_errors[project_id].append(error)
            
            # Find projects with high error rates
            for project_id, project_error_list in project_errors.items():
                if len(project_error_list) >= 5:  # At least 5 errors
                    hotspots.append({
                        "type": "project_hotspot",
                        "project_id": project_id,
                        "error_count": len(project_error_list),
                        "error_rate": len(project_error_list) / len(errors),
                        "severity": "high" if len(project_error_list) > 10 else "medium"
                    })
            
            return hotspots
            
        except Exception as e:
            logger.error(f"Failed to identify error hotspots: {e}")
            return []
    
    def _generate_prevention_strategies(
        self,
        patterns: List[PatternMatch],
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate error prevention strategies."""
        strategies = []
        
        try:
            # Strategies from patterns
            for pattern in patterns:
                if pattern.pattern_type == PatternType.ERROR_SEQUENCE:
                    strategies.append({
                        "type": "pattern_prevention",
                        "title": f"Prevent {pattern.description}",
                        "strategy": pattern.improvement_suggestion,
                        "priority": pattern.impact_level,
                        "confidence": pattern.confidence
                    })
            
            # General strategies based on error types
            error_types = set(error.get("type", "unknown") for error in errors)
            
            for error_type in error_types:
                if error_type != "unknown":
                    strategies.append({
                        "type": "error_type_prevention",
                        "title": f"Prevent {error_type} errors",
                        "strategy": f"Implement validation and error handling for {error_type}",
                        "priority": "medium",
                        "confidence": 0.7
                    })
            
            return strategies
            
        except Exception as e:
            logger.error(f"Failed to generate prevention strategies: {e}")
            return []
    
    async def _update_pattern_confidence(
        self,
        pattern_id: str,
        feedback_type: str
    ):
        """Update pattern confidence based on feedback."""
        try:
            # Get current confidence
            confidence_key = f"pattern_confidence:{pattern_id}"
            current_confidence = await redis_client.get(confidence_key)
            
            if current_confidence:
                confidence = float(current_confidence.decode())
            else:
                confidence = 0.5  # Default confidence
            
            # Adjust confidence based on feedback
            if feedback_type == "helpful":
                confidence = min(1.0, confidence + 0.1)
            elif feedback_type == "not_helpful":
                confidence = max(0.0, confidence - 0.05)
            elif feedback_type == "incorrect":
                confidence = max(0.0, confidence - 0.2)
            
            # Store updated confidence
            await redis_client.set(confidence_key, str(confidence))
            
        except Exception as e:
            logger.error(f"Failed to update pattern confidence: {e}")
    
    async def _get_pattern_detection_metrics(
        self,
        time_window_days: int
    ) -> Dict[str, Any]:
        """Get pattern detection metrics."""
        try:
            # This would typically query analytics database
            # For now, return simulated metrics
            
            return {
                "total_patterns_detected": 150,
                "patterns_by_type": {
                    "code_structure": 45,
                    "usage_behavior": 38,
                    "performance_bottleneck": 22,
                    "error_sequence": 28,
                    "optimization_opportunity": 17
                },
                "avg_confidence": 0.75,
                "detection_rate_per_day": 5.2
            }
            
        except Exception as e:
            logger.error(f"Failed to get detection metrics: {e}")
            return {}
    
    async def _get_pattern_feedback_metrics(
        self,
        time_window_days: int
    ) -> Dict[str, Any]:
        """Get pattern feedback metrics."""
        try:
            # This would typically aggregate feedback data
            # For now, return simulated metrics
            
            return {
                "total_feedback": 85,
                "feedback_breakdown": {
                    "helpful": 52,
                    "not_helpful": 23,
                    "incorrect": 10
                },
                "feedback_rate": 0.57,  # 57% of patterns get feedback
                "avg_time_to_feedback_hours": 2.5
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback metrics: {e}")
            return {}
    
    async def _get_pattern_accuracy_metrics(
        self,
        time_window_days: int
    ) -> Dict[str, Any]:
        """Get pattern accuracy metrics."""
        try:
            # This would calculate accuracy based on feedback
            # For now, return simulated metrics
            
            return {
                "overall_accuracy": 0.78,
                "accuracy_by_type": {
                    "code_structure": 0.82,
                    "usage_behavior": 0.75,
                    "performance_bottleneck": 0.85,
                    "error_sequence": 0.73,
                    "optimization_opportunity": 0.68
                },
                "precision": 0.76,
                "recall": 0.71,
                "f1_score": 0.73
            }
            
        except Exception as e:
            logger.error(f"Failed to get accuracy metrics: {e}")
            return {}
    
    async def _get_pattern_usage_metrics(
        self,
        time_window_days: int
    ) -> Dict[str, Any]:
        """Get pattern usage metrics."""
        try:
            # This would track how patterns are used
            # For now, return simulated metrics
            
            return {
                "patterns_acted_upon": 45,
                "patterns_ignored": 105,
                "action_rate": 0.30,
                "avg_time_to_action_hours": 4.2,
                "most_actionable_type": "optimization_opportunity"
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage metrics: {e}")
            return {}


# Global pattern service instance
pattern_service = PatternService()