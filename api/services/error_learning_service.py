"""
Advanced AI Error Learning Service Implementation.

This service provides intelligent error pattern recognition, solution discovery,
and predictive error prevention through machine learning.
"""

import logging
import asyncio
import hashlib
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text, and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError
import difflib

from ..models.error_pattern import (
    EnhancedErrorPattern, ErrorOccurrence, ErrorSolution, ErrorFeedback, ErrorPrediction,
    ErrorSeverity, ErrorCategory, SolutionStatus,
    ErrorPatternCreate, ErrorOccurrenceCreate, ErrorSolutionCreate, ErrorFeedbackCreate,
    ErrorPatternResponse, ErrorPredictionResponse, ErrorAnalytics
)
from ..models.base import get_db_context
from ..services.embedding_service import embedding_service, EmbeddingModel
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService, MetricType
from ..services.session_service import session_service
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("error_learning_service")


@dataclass
class ErrorMatchResult:
    """Result of error pattern matching."""
    pattern: EnhancedErrorPattern
    similarity_score: float
    match_type: str  # exact, fuzzy, semantic
    suggested_solutions: List[ErrorSolution]
    confidence: float


@dataclass
class ErrorPredictionResult:
    """Result of error prediction analysis."""
    predicted_patterns: List[EnhancedErrorPattern]
    risk_score: float
    prevention_steps: List[str]
    risk_factors: List[Dict[str, Any]]
    confidence: float


class ErrorLearningService:
    """
    Advanced AI Error Learning Service with pattern recognition and prediction.
    
    Features:
    - Error pattern recognition with fuzzy and semantic matching
    - Solution effectiveness tracking and ranking
    - Predictive error prevention
    - Learning from user feedback
    - Pattern evolution tracking
    - Error clustering and categorization
    - Cross-session error analysis
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.embedding_service = embedding_service
        self.analytics_service = TimeSeriesAnalyticsService()
        self._initialized = False
        
        # Configuration
        self.similarity_threshold = 0.3
        self.semantic_threshold = 0.7
        self.prediction_confidence_threshold = 0.6
        self.max_pattern_age_days = 180
        self.learning_rate = 0.1
        
        # Pattern matching weights
        self.match_weights = {
            "error_type": 0.3,
            "error_message": 0.4,
            "stack_trace": 0.2,
            "context": 0.1
        }
        
        logger.info("Initialized ErrorLearningService")
    
    async def initialize(self):
        """Initialize the error learning service."""
        if self._initialized:
            return
        
        try:
            # Initialize dependencies
            await self.embedding_service.initialize()
            await self.analytics_service.initialize()
            await redis_client.initialize()
            
            self._initialized = True
            logger.info("ErrorLearningService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ErrorLearningService: {e}")
            raise
    
    async def record_error(
        self,
        error_data: ErrorOccurrenceCreate,
        user_id: str,
        session_id: Optional[str] = None
    ) -> ErrorMatchResult:
        """
        Record a new error occurrence and find matching patterns.
        
        Args:
            error_data: Error occurrence data
            user_id: User ID
            session_id: Optional session ID
            
        Returns:
            Error match result with suggested solutions
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        try:
            with get_db_context() as db:
                # Find or create error pattern
                pattern_match = await self._find_or_create_pattern(db, error_data)
                
                # Record the occurrence
                occurrence = ErrorOccurrence(
                    pattern_id=pattern_match.pattern.id,
                    session_id=session_id,
                    user_id=user_id,
                    full_error_message=error_data.error_message,
                    full_stack_trace=error_data.stack_trace,
                    execution_context=error_data.execution_context,
                    system_state=error_data.system_state,
                    user_actions=error_data.user_actions
                )
                
                db.add(occurrence)
                db.flush()
                
                # Update pattern metrics
                pattern_match.pattern.occurrences += 1
                pattern_match.pattern.last_seen = datetime.utcnow()
                
                # Get ranked solutions
                solutions = await self._get_ranked_solutions(db, pattern_match.pattern.id)
                pattern_match.suggested_solutions = solutions
                
                db.commit()
                
                # Record analytics
                processing_time = (time.time() - start_time) * 1000
                await self._record_error_analytics(
                    pattern=pattern_match.pattern,
                    match_type=pattern_match.match_type,
                    processing_time=processing_time,
                    user_id=user_id
                )
                
                # Trigger predictive analysis in background
                asyncio.create_task(self._analyze_error_patterns(user_id, session_id))
                
                logger.info(f"Recorded error occurrence: {occurrence.id} for pattern {pattern_match.pattern.id}")
                return pattern_match
                
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
            raise
    
    async def provide_feedback(
        self,
        feedback_data: ErrorFeedbackCreate,
        user_id: str
    ) -> bool:
        """
        Provide feedback on error pattern or solution effectiveness.
        
        Args:
            feedback_data: Feedback data
            user_id: User ID
            
        Returns:
            Success status
        """
        try:
            with get_db_context() as db:
                # Create feedback record
                feedback = ErrorFeedback(
                    pattern_id=feedback_data.pattern_id,
                    occurrence_id=feedback_data.occurrence_id,
                    solution_id=feedback_data.solution_id,
                    user_id=user_id,
                    helpful=feedback_data.helpful,
                    feedback_text=feedback_data.feedback_text,
                    suggested_improvement=feedback_data.suggested_improvement,
                    actual_solution=feedback_data.actual_solution,
                    resolution_time=feedback_data.resolution_time
                )
                
                db.add(feedback)
                
                # Update pattern/solution effectiveness
                if feedback_data.solution_id:
                    solution = db.query(ErrorSolution).filter_by(
                        id=feedback_data.solution_id
                    ).first()
                    
                    if solution:
                        solution.update_effectiveness(
                            success=feedback_data.helpful,
                            resolution_time=feedback_data.resolution_time or 0.0
                        )
                
                # Update pattern metrics
                pattern = db.query(EnhancedErrorPattern).filter_by(
                    id=feedback_data.pattern_id
                ).first()
                
                if pattern:
                    pattern.update_metrics(
                        success=feedback_data.helpful,
                        resolution_time=feedback_data.resolution_time or 0.0
                    )
                
                # Mark occurrence as resolved if applicable
                if feedback_data.occurrence_id and feedback_data.helpful:
                    occurrence = db.query(ErrorOccurrence).filter_by(
                        id=feedback_data.occurrence_id
                    ).first()
                    
                    if occurrence:
                        occurrence.resolved = True
                        occurrence.resolution_time = feedback_data.resolution_time
                        occurrence.applied_solution_id = feedback_data.solution_id
                
                db.commit()
                
                logger.info(f"Recorded feedback for pattern {feedback_data.pattern_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False
    
    async def predict_errors(
        self,
        user_id: str,
        session_id: str,
        context: Dict[str, Any]
    ) -> ErrorPredictionResult:
        """
        Predict potential errors based on current context and history.
        
        Args:
            user_id: User ID
            session_id: Session ID
            context: Current execution context
            
        Returns:
            Error prediction result
        """
        try:
            with get_db_context() as db:
                # Get recent error patterns for user
                recent_patterns = db.query(EnhancedErrorPattern).join(
                    ErrorOccurrence
                ).filter(
                    ErrorOccurrence.user_id == user_id,
                    ErrorOccurrence.timestamp > datetime.utcnow() - timedelta(days=7)
                ).distinct().all()
                
                if not recent_patterns:
                    return ErrorPredictionResult(
                        predicted_patterns=[],
                        risk_score=0.0,
                        prevention_steps=[],
                        risk_factors=[],
                        confidence=0.0
                    )
                
                # Analyze patterns and context
                predictions = []
                risk_factors = []
                
                for pattern in recent_patterns:
                    # Calculate likelihood based on context similarity
                    likelihood = self._calculate_error_likelihood(pattern, context)
                    
                    if likelihood > self.prediction_confidence_threshold:
                        predictions.append({
                            "pattern": pattern,
                            "likelihood": likelihood
                        })
                        
                        risk_factors.append({
                            "pattern_type": pattern.error_type,
                            "category": pattern.error_category,
                            "recent_occurrences": pattern.occurrences,
                            "likelihood": likelihood
                        })
                
                # Sort by likelihood
                predictions.sort(key=lambda x: x["likelihood"], reverse=True)
                
                # Calculate overall risk score
                risk_score = max([p["likelihood"] for p in predictions]) if predictions else 0.0
                
                # Generate prevention steps
                prevention_steps = []
                for pred in predictions[:3]:  # Top 3 predictions
                    pattern = pred["pattern"]
                    if pattern.prerequisites:
                        prevention_steps.extend([
                            f"Avoid: {prereq}" 
                            for prereq in pattern.prerequisites.get("conditions", [])
                        ])
                    if pattern.primary_solution:
                        prevention_steps.append(f"Preventive: {pattern.primary_solution}")
                
                # Create prediction records
                for pred in predictions[:5]:  # Top 5 predictions
                    prediction = ErrorPrediction(
                        session_id=session_id,
                        user_id=user_id,
                        predicted_pattern_id=pred["pattern"].id,
                        prediction_confidence=pred["likelihood"],
                        risk_factors=[{
                            "context_match": 0.8,  # Simplified
                            "recent_frequency": pred["pattern"].occurrences / 7  # Per day
                        }],
                        context_snapshot=context,
                        prevention_steps=prevention_steps[:3]
                    )
                    db.add(prediction)
                
                db.commit()
                
                return ErrorPredictionResult(
                    predicted_patterns=[p["pattern"] for p in predictions[:5]],
                    risk_score=risk_score,
                    prevention_steps=list(set(prevention_steps))[:10],
                    risk_factors=risk_factors[:5],
                    confidence=risk_score * 0.9  # Adjusted confidence
                )
                
        except Exception as e:
            logger.error(f"Failed to predict errors: {e}")
            raise
    
    async def get_error_analytics(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        time_window_hours: int = 168  # 7 days
    ) -> ErrorAnalytics:
        """Get comprehensive error analytics."""
        
        try:
            with get_db_context() as db:
                # Base query
                base_query = db.query(EnhancedErrorPattern)
                occurrence_query = db.query(ErrorOccurrence)
                
                if user_id:
                    occurrence_query = occurrence_query.filter(ErrorOccurrence.user_id == user_id)
                
                # Time window filter
                time_threshold = datetime.utcnow() - timedelta(hours=time_window_hours)
                recent_occurrences = occurrence_query.filter(
                    ErrorOccurrence.timestamp >= time_threshold
                ).all()
                
                # Calculate metrics
                total_patterns = base_query.count()
                total_occurrences = len(recent_occurrences)
                
                # Patterns by category
                category_counts = db.query(
                    EnhancedErrorPattern.error_category,
                    func.count(EnhancedErrorPattern.id)
                ).group_by(EnhancedErrorPattern.error_category).all()
                
                patterns_by_category = {cat: count for cat, count in category_counts}
                
                # Patterns by severity
                severity_counts = db.query(
                    EnhancedErrorPattern.severity,
                    func.count(EnhancedErrorPattern.id)
                ).group_by(EnhancedErrorPattern.severity).all()
                
                patterns_by_severity = {sev: count for sev, count in severity_counts}
                
                # Top errors
                top_errors = db.query(
                    EnhancedErrorPattern,
                    func.count(ErrorOccurrence.id).label('occurrence_count')
                ).join(ErrorOccurrence).filter(
                    ErrorOccurrence.timestamp >= time_threshold
                ).group_by(EnhancedErrorPattern.id).order_by(
                    desc('occurrence_count')
                ).limit(10).all()
                
                top_errors_list = [
                    {
                        "pattern_id": str(pattern.id),
                        "error_type": pattern.error_type,
                        "error_message": pattern.error_message,
                        "occurrences": count,
                        "severity": pattern.severity,
                        "success_rate": pattern.success_rate
                    }
                    for pattern, count in top_errors
                ]
                
                # Resolution stats
                resolved_count = sum(1 for o in recent_occurrences if o.resolved)
                avg_resolution_time = np.mean([
                    o.resolution_time for o in recent_occurrences 
                    if o.resolved and o.resolution_time
                ]) if resolved_count > 0 else 0.0
                
                resolution_stats = {
                    "total_errors": total_occurrences,
                    "resolved_errors": resolved_count,
                    "resolution_rate": resolved_count / total_occurrences if total_occurrences > 0 else 0.0,
                    "avg_resolution_time": avg_resolution_time
                }
                
                # Learning progress
                patterns_with_solutions = db.query(EnhancedErrorPattern).join(
                    ErrorSolution
                ).filter(
                    ErrorSolution.status == SolutionStatus.VERIFIED.value
                ).distinct().count()
                
                learning_progress = {
                    "patterns_identified": total_patterns,
                    "patterns_with_solutions": patterns_with_solutions,
                    "solution_coverage": patterns_with_solutions / total_patterns if total_patterns > 0 else 0.0,
                    "avg_pattern_confidence": db.query(func.avg(EnhancedErrorPattern.confidence_score)).scalar() or 0.0
                }
                
                # Prediction accuracy
                predictions = db.query(ErrorPrediction).filter(
                    ErrorPrediction.prediction_occurred.isnot(None)
                ).all()
                
                accurate_predictions = sum(1 for p in predictions if p.prediction_occurred)
                prediction_accuracy = accurate_predictions / len(predictions) if predictions else 0.0
                
                return ErrorAnalytics(
                    total_patterns=total_patterns,
                    total_occurrences=total_occurrences,
                    patterns_by_category=patterns_by_category,
                    patterns_by_severity=patterns_by_severity,
                    top_errors=top_errors_list,
                    resolution_stats=resolution_stats,
                    learning_progress=learning_progress,
                    prediction_accuracy=prediction_accuracy
                )
                
        except Exception as e:
            logger.error(f"Failed to get error analytics: {e}")
            raise
    
    # Helper methods
    
    async def _find_or_create_pattern(
        self,
        db: DBSession,
        error_data: ErrorOccurrenceCreate
    ) -> ErrorMatchResult:
        """Find matching error pattern or create new one."""
        
        # Calculate pattern hash
        pattern_hash = self._calculate_pattern_hash(
            error_data.error_type or "Unknown",
            error_data.error_message,
            error_data.stack_trace
        )
        
        # Try exact match by hash
        exact_match = db.query(EnhancedErrorPattern).filter_by(
            pattern_hash=pattern_hash
        ).first()
        
        if exact_match:
            return ErrorMatchResult(
                pattern=exact_match,
                similarity_score=1.0,
                match_type="exact",
                suggested_solutions=[],
                confidence=exact_match.confidence_score
            )
        
        # Try fuzzy matching
        fuzzy_match = await self._fuzzy_match_pattern(db, error_data)
        
        if fuzzy_match and fuzzy_match.similarity_score > self.similarity_threshold:
            return fuzzy_match
        
        # Try semantic matching
        if self.embedding_service:
            semantic_match = await self._semantic_match_pattern(db, error_data)
            
            if semantic_match and semantic_match.similarity_score > self.semantic_threshold:
                return semantic_match
        
        # Create new pattern
        new_pattern = await self._create_new_pattern(db, error_data, pattern_hash)
        
        return ErrorMatchResult(
            pattern=new_pattern,
            similarity_score=1.0,
            match_type="new",
            suggested_solutions=[],
            confidence=0.5  # Initial confidence
        )
    
    def _calculate_pattern_hash(
        self,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str]
    ) -> str:
        """Calculate normalized pattern hash."""
        
        # Normalize error message
        normalized_message = re.sub(r'\d+', 'N', error_message.lower())
        normalized_message = re.sub(
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
            'UUID',
            normalized_message
        )
        
        # Normalize stack trace
        if stack_trace:
            normalized_trace = re.sub(r':\d+', '', stack_trace)
        else:
            normalized_trace = ""
        
        # Create pattern text
        pattern_text = f"{error_type}::{normalized_message}::{normalized_trace}"
        
        # Generate hash
        return hashlib.sha256(pattern_text.encode()).hexdigest()
    
    async def _fuzzy_match_pattern(
        self,
        db: DBSession,
        error_data: ErrorOccurrenceCreate
    ) -> Optional[ErrorMatchResult]:
        """Find pattern using fuzzy text matching."""
        
        # Use PostgreSQL trigram similarity
        result = db.execute(text("""
            SELECT * FROM find_similar_error_patterns(
                :search_message,
                :search_type,
                :threshold,
                :limit
            )
        """), {
            "search_message": error_data.error_message,
            "search_type": error_data.error_type,
            "threshold": self.similarity_threshold,
            "limit": 5
        })
        
        matches = result.fetchall()
        
        if matches:
            best_match = matches[0]
            pattern = db.query(EnhancedErrorPattern).filter_by(
                id=best_match[0]
            ).first()
            
            if pattern:
                return ErrorMatchResult(
                    pattern=pattern,
                    similarity_score=best_match[1],
                    match_type="fuzzy",
                    suggested_solutions=[],
                    confidence=pattern.confidence_score * best_match[1]
                )
        
        return None
    
    async def _semantic_match_pattern(
        self,
        db: DBSession,
        error_data: ErrorOccurrenceCreate
    ) -> Optional[ErrorMatchResult]:
        """Find pattern using semantic similarity."""
        
        try:
            # Generate embedding for error message
            embedding_result = await self.embedding_service.generate_embedding(
                text=error_data.error_message,
                context={
                    "error_type": error_data.error_type,
                    "type": "error_pattern"
                }
            )
            
            query_embedding = embedding_result.embeddings
            
            # Get patterns with embeddings
            patterns_with_embeddings = db.query(EnhancedErrorPattern).filter(
                EnhancedErrorPattern.embeddings.isnot(None)
            ).all()
            
            if not patterns_with_embeddings:
                return None
            
            # Calculate similarities
            best_match = None
            best_score = 0.0
            
            for pattern in patterns_with_embeddings:
                similarity = await self.embedding_service.calculate_similarity(
                    query_embedding,
                    pattern.embeddings,
                    method="cosine"
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = pattern
            
            if best_match and best_score > self.semantic_threshold:
                return ErrorMatchResult(
                    pattern=best_match,
                    similarity_score=best_score,
                    match_type="semantic",
                    suggested_solutions=[],
                    confidence=best_match.confidence_score * best_score
                )
        
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")
        
        return None
    
    async def _create_new_pattern(
        self,
        db: DBSession,
        error_data: ErrorOccurrenceCreate,
        pattern_hash: str
    ) -> EnhancedErrorPattern:
        """Create a new error pattern."""
        
        # Extract key indicators
        key_indicators = self._extract_key_indicators(error_data.error_message)
        
        # Determine category
        category = self._determine_error_category(
            error_data.error_type or "Unknown",
            error_data.error_message
        )
        
        # Generate embeddings
        embeddings = None
        if self.embedding_service:
            try:
                embedding_result = await self.embedding_service.generate_embedding(
                    text=error_data.error_message,
                    context={
                        "error_type": error_data.error_type,
                        "type": "error_pattern"
                    }
                )
                embeddings = embedding_result.embeddings
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")
        
        # Create pattern
        pattern = EnhancedErrorPattern(
            pattern_hash=pattern_hash,
            error_type=error_data.error_type or "Unknown",
            error_category=category.value,
            error_message=error_data.error_message,
            error_code=error_data.error_code,
            stack_trace=error_data.stack_trace,
            key_indicators=key_indicators,
            context=error_data.execution_context,
            embeddings=embeddings,
            created_by="system"
        )
        
        db.add(pattern)
        db.flush()
        
        logger.info(f"Created new error pattern: {pattern.id}")
        return pattern
    
    def _extract_key_indicators(self, error_message: str) -> List[str]:
        """Extract key indicators from error message."""
        
        # Common error keywords
        keywords = [
            "null", "undefined", "missing", "failed", "timeout",
            "connection", "permission", "denied", "invalid",
            "syntax", "reference", "type", "range", "overflow"
        ]
        
        indicators = []
        message_lower = error_message.lower()
        
        # Find present keywords
        for keyword in keywords:
            if keyword in message_lower:
                indicators.append(keyword)
        
        # Extract error codes
        error_codes = re.findall(r'\b[A-Z]+[0-9]+\b', error_message)
        indicators.extend(error_codes)
        
        return indicators[:10]  # Limit to 10 indicators
    
    def _determine_error_category(self, error_type: str, error_message: str) -> ErrorCategory:
        """Determine error category based on type and message."""
        
        type_lower = error_type.lower()
        message_lower = error_message.lower()
        
        # Category mapping
        category_indicators = {
            ErrorCategory.SYNTAX: ["syntax", "parse", "unexpected token", "indent"],
            ErrorCategory.RUNTIME: ["runtime", "null", "undefined", "reference"],
            ErrorCategory.NETWORK: ["connection", "timeout", "network", "socket"],
            ErrorCategory.DATABASE: ["database", "query", "sql", "transaction"],
            ErrorCategory.AUTHENTICATION: ["auth", "login", "token", "credential"],
            ErrorCategory.PERMISSION: ["permission", "denied", "forbidden", "access"],
            ErrorCategory.VALIDATION: ["validation", "invalid", "required", "format"],
            ErrorCategory.PERFORMANCE: ["timeout", "slow", "memory", "cpu"]
        }
        
        for category, indicators in category_indicators.items():
            for indicator in indicators:
                if indicator in type_lower or indicator in message_lower:
                    return category
        
        return ErrorCategory.UNKNOWN
    
    async def _get_ranked_solutions(
        self,
        db: DBSession,
        pattern_id: str,
        limit: int = 5
    ) -> List[ErrorSolution]:
        """Get ranked solutions for a pattern."""
        
        solutions = db.query(ErrorSolution).filter(
            ErrorSolution.pattern_id == pattern_id,
            ErrorSolution.status.in_([
                SolutionStatus.VERIFIED.value,
                SolutionStatus.SUGGESTED.value
            ])
        ).order_by(
            desc(ErrorSolution.effectiveness_score),
            desc(ErrorSolution.success_count)
        ).limit(limit).all()
        
        return solutions
    
    def _calculate_error_likelihood(
        self,
        pattern: EnhancedErrorPattern,
        context: Dict[str, Any]
    ) -> float:
        """Calculate likelihood of error occurring based on context."""
        
        likelihood = 0.0
        
        # Base likelihood from pattern frequency
        base_likelihood = min(pattern.occurrences / 100.0, 0.5)
        likelihood += base_likelihood
        
        # Context similarity
        if pattern.prerequisites:
            matching_conditions = 0
            total_conditions = len(pattern.prerequisites.get("conditions", []))
            
            for condition in pattern.prerequisites.get("conditions", []):
                if self._check_condition(condition, context):
                    matching_conditions += 1
            
            if total_conditions > 0:
                likelihood += (matching_conditions / total_conditions) * 0.3
        
        # Recent occurrence boost
        if pattern.last_seen:
            hours_since = (datetime.utcnow() - pattern.last_seen).total_seconds() / 3600
            recency_factor = 1.0 / (1.0 + hours_since / 24)  # Decay over days
            likelihood += recency_factor * 0.2
        
        return min(likelihood, 1.0)
    
    def _check_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Check if a condition is met in the context."""
        
        # Simple keyword matching for now
        condition_lower = condition.lower()
        context_str = json.dumps(context).lower()
        
        return condition_lower in context_str
    
    async def _analyze_error_patterns(self, user_id: str, session_id: Optional[str]):
        """Background task to analyze error patterns."""
        
        try:
            # Analyze patterns for predictive insights
            await asyncio.sleep(1)  # Don't block main thread
            
            if session_id:
                # Get session context
                session = await session_service.get_session(session_id)
                
                if session:
                    # Perform prediction
                    prediction = await self.predict_errors(
                        user_id=user_id,
                        session_id=session_id,
                        context={
                            "session_type": session.session_type,
                            "interaction_count": session.interaction_count,
                            "error_count": session.error_count
                        }
                    )
                    
                    if prediction.risk_score > 0.7:
                        logger.warning(f"High error risk detected for session {session_id}: {prediction.risk_score}")
        
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
    
    async def _record_error_analytics(
        self,
        pattern: EnhancedErrorPattern,
        match_type: str,
        processing_time: float,
        user_id: str
    ):
        """Record error analytics."""
        
        try:
            # Record in time-series analytics
            await self.analytics_service.record_metric(
                metric_type=MetricType.ERROR_OCCURRENCE,
                value=1.0,
                tags={
                    "error_type": pattern.error_type,
                    "category": pattern.error_category,
                    "severity": pattern.severity,
                    "match_type": match_type
                },
                metadata={
                    "pattern_id": str(pattern.id),
                    "user_id": user_id,
                    "processing_time_ms": processing_time
                }
            )
            
        except Exception as e:
            logger.warning(f"Analytics recording failed: {e}")
    
    async def cleanup(self):
        """Clean up service resources."""
        await self.embedding_service.cleanup()
        await self.analytics_service.cleanup()
        self._initialized = False
        logger.info("ErrorLearningService cleaned up")


# Global error learning service instance
error_learning_service = ErrorLearningService()