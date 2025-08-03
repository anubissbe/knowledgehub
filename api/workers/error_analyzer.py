"""
Background Error Analyzer Worker for continuous learning and pattern evolution.

This worker performs background analysis of error patterns, identifies trends,
and evolves the error learning system.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import func, and_, or_, text, desc
from sqlalchemy.orm import Session as DBSession

from ..models.base import get_db_context
from ..models.error_pattern import (
    EnhancedErrorPattern, ErrorOccurrence, ErrorSolution, 
    ErrorFeedback, ErrorPrediction,
    ErrorSeverity, ErrorCategory, SolutionStatus
)
from ..services.error_learning_service import error_learning_service
from ..services.embedding_service import embedding_service
from shared.logging import setup_logging

logger = setup_logging("error_analyzer_worker")


class ErrorAnalyzerWorker:
    """
    Background worker for analyzing error patterns and improving the learning system.
    
    Tasks:
    - Pattern consolidation: Merge similar patterns
    - Solution optimization: Identify and promote effective solutions
    - Prediction validation: Track prediction accuracy
    - Pattern evolution: Update patterns based on new occurrences
    - Stale pattern cleanup: Archive old unused patterns
    - Cross-user learning: Identify common patterns across users
    """
    
    def __init__(self):
        self.running = False
        self.analysis_interval = 3600  # 1 hour
        self.consolidation_threshold = 0.85
        self.min_pattern_occurrences = 5
        self.pattern_decay_days = 90
        self.confidence_boost_threshold = 0.8
        
    async def start(self):
        """Start the error analyzer worker."""
        if self.running:
            logger.warning("Error analyzer worker is already running")
            return
        
        self.running = True
        logger.info("Starting error analyzer worker")
        
        # Initialize services
        await error_learning_service.initialize()
        await embedding_service.initialize()
        
        # Start background tasks
        asyncio.create_task(self._analysis_loop())
        asyncio.create_task(self._prediction_validation_loop())
        asyncio.create_task(self._pattern_evolution_loop())
    
    async def stop(self):
        """Stop the error analyzer worker."""
        self.running = False
        logger.info("Stopping error analyzer worker")
    
    async def _analysis_loop(self):
        """Main analysis loop."""
        while self.running:
            try:
                logger.info("Starting error pattern analysis cycle")
                
                # Run analysis tasks
                await self._consolidate_similar_patterns()
                await self._optimize_solutions()
                await self._analyze_cross_user_patterns()
                await self._cleanup_stale_patterns()
                
                logger.info("Completed error pattern analysis cycle")
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(self.analysis_interval)
    
    async def _consolidate_similar_patterns(self):
        """Consolidate similar error patterns to reduce redundancy."""
        
        try:
            with get_db_context() as db:
                # Get patterns with embeddings
                patterns = db.query(EnhancedErrorPattern).filter(
                    EnhancedErrorPattern.embeddings.isnot(None),
                    EnhancedErrorPattern.occurrences >= self.min_pattern_occurrences
                ).all()
                
                if len(patterns) < 2:
                    return
                
                logger.info(f"Analyzing {len(patterns)} patterns for consolidation")
                
                # Group similar patterns
                consolidated_count = 0
                processed = set()
                
                for i, pattern1 in enumerate(patterns):
                    if pattern1.id in processed:
                        continue
                    
                    similar_patterns = []
                    
                    for j, pattern2 in enumerate(patterns[i+1:], i+1):
                        if pattern2.id in processed:
                            continue
                        
                        # Calculate similarity
                        similarity = await embedding_service.calculate_similarity(
                            pattern1.embeddings,
                            pattern2.embeddings,
                            method="cosine"
                        )
                        
                        if similarity > self.consolidation_threshold:
                            similar_patterns.append((pattern2, similarity))
                    
                    # Consolidate if similar patterns found
                    if similar_patterns:
                        await self._merge_patterns(db, pattern1, similar_patterns)
                        processed.add(pattern1.id)
                        processed.update(p[0].id for p in similar_patterns)
                        consolidated_count += len(similar_patterns)
                
                db.commit()
                logger.info(f"Consolidated {consolidated_count} similar patterns")
                
        except Exception as e:
            logger.error(f"Pattern consolidation failed: {e}")
    
    async def _merge_patterns(
        self,
        db: DBSession,
        primary_pattern: EnhancedErrorPattern,
        similar_patterns: List[Tuple[EnhancedErrorPattern, float]]
    ):
        """Merge similar patterns into primary pattern."""
        
        try:
            # Accumulate occurrences and metrics
            total_occurrences = primary_pattern.occurrences
            total_success_count = primary_pattern.occurrences * primary_pattern.success_rate
            
            for pattern, similarity in similar_patterns:
                # Move occurrences to primary pattern
                db.execute(text("""
                    UPDATE error_occurrences 
                    SET pattern_id = :primary_id 
                    WHERE pattern_id = :secondary_id
                """), {
                    "primary_id": primary_pattern.id,
                    "secondary_id": pattern.id
                })
                
                # Accumulate metrics
                total_occurrences += pattern.occurrences
                total_success_count += pattern.occurrences * pattern.success_rate
                
                # Merge solutions
                for solution in pattern.solutions:
                    solution.pattern_id = primary_pattern.id
                
                # Mark pattern as merged
                pattern.error_category = "merged"
                pattern.context["merged_into"] = str(primary_pattern.id)
                pattern.context["merge_similarity"] = similarity
            
            # Update primary pattern metrics
            primary_pattern.occurrences = total_occurrences
            primary_pattern.success_rate = total_success_count / total_occurrences if total_occurrences > 0 else 0
            primary_pattern.confidence_score = min(0.95, primary_pattern.confidence_score + 0.1)
            primary_pattern.last_updated = datetime.utcnow()
            
            logger.info(f"Merged {len(similar_patterns)} patterns into {primary_pattern.id}")
            
        except Exception as e:
            logger.error(f"Pattern merge failed: {e}")
    
    async def _optimize_solutions(self):
        """Optimize and rank solutions based on effectiveness."""
        
        try:
            with get_db_context() as db:
                # Get patterns with multiple solutions
                patterns_with_solutions = db.query(EnhancedErrorPattern).join(
                    ErrorSolution
                ).group_by(EnhancedErrorPattern.id).having(
                    func.count(ErrorSolution.id) > 1
                ).all()
                
                logger.info(f"Optimizing solutions for {len(patterns_with_solutions)} patterns")
                
                for pattern in patterns_with_solutions:
                    # Get solutions ordered by effectiveness
                    solutions = db.query(ErrorSolution).filter_by(
                        pattern_id=pattern.id
                    ).order_by(
                        desc(ErrorSolution.effectiveness_score),
                        desc(ErrorSolution.success_count)
                    ).all()
                    
                    if not solutions:
                        continue
                    
                    # Update primary solution if better one found
                    best_solution = solutions[0]
                    if best_solution.effectiveness_score > self.confidence_boost_threshold:
                        pattern.primary_solution = best_solution.solution_text
                        pattern.solution_steps = best_solution.solution_steps
                        
                        # Verify solution if highly effective
                        if best_solution.status != SolutionStatus.VERIFIED.value and \
                           best_solution.effectiveness_score > 0.9:
                            best_solution.status = SolutionStatus.VERIFIED.value
                            best_solution.verified_at = datetime.utcnow()
                            best_solution.verified_by = "auto_verification"
                    
                    # Deprecate ineffective solutions
                    for solution in solutions:
                        if solution.effectiveness_score < 0.3 and \
                           solution.success_count + solution.failure_count > 10:
                            solution.status = SolutionStatus.DEPRECATED.value
                
                db.commit()
                logger.info("Solution optimization completed")
                
        except Exception as e:
            logger.error(f"Solution optimization failed: {e}")
    
    async def _analyze_cross_user_patterns(self):
        """Analyze patterns across users to identify common issues."""
        
        try:
            with get_db_context() as db:
                # Find patterns occurring across multiple users
                cross_user_patterns = db.execute(text("""
                    SELECT 
                        p.id,
                        p.error_type,
                        p.error_message,
                        COUNT(DISTINCT o.user_id) as user_count,
                        COUNT(o.id) as total_occurrences
                    FROM enhanced_error_patterns p
                    JOIN error_occurrences o ON o.pattern_id = p.id
                    WHERE o.timestamp > NOW() - INTERVAL '30 days'
                    GROUP BY p.id, p.error_type, p.error_message
                    HAVING COUNT(DISTINCT o.user_id) > 3
                    ORDER BY user_count DESC, total_occurrences DESC
                    LIMIT 20
                """)).fetchall()
                
                logger.info(f"Found {len(cross_user_patterns)} cross-user patterns")
                
                for pattern_data in cross_user_patterns:
                    pattern = db.query(EnhancedErrorPattern).filter_by(
                        id=pattern_data[0]
                    ).first()
                    
                    if pattern:
                        # Boost confidence for widely occurring patterns
                        boost = min(0.2, pattern_data[3] * 0.02)  # 2% per user
                        pattern.confidence_score = min(0.95, pattern.confidence_score + boost)
                        
                        # Update context with cross-user info
                        pattern.context["cross_user_stats"] = {
                            "affected_users": pattern_data[3],
                            "total_occurrences": pattern_data[4],
                            "last_analyzed": datetime.utcnow().isoformat()
                        }
                        
                        # Increase severity if affecting many users
                        if pattern_data[3] > 10 and pattern.severity == ErrorSeverity.MEDIUM.value:
                            pattern.severity = ErrorSeverity.HIGH.value
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Cross-user analysis failed: {e}")
    
    async def _cleanup_stale_patterns(self):
        """Clean up old patterns with no recent activity."""
        
        try:
            with get_db_context() as db:
                stale_threshold = datetime.utcnow() - timedelta(days=self.pattern_decay_days)
                
                # Find stale patterns
                stale_patterns = db.query(EnhancedErrorPattern).filter(
                    EnhancedErrorPattern.last_seen < stale_threshold,
                    EnhancedErrorPattern.occurrences < 5,
                    EnhancedErrorPattern.confidence_score < 0.5
                ).all()
                
                logger.info(f"Found {len(stale_patterns)} stale patterns for cleanup")
                
                for pattern in stale_patterns:
                    # Archive instead of delete
                    pattern.context["archived"] = True
                    pattern.context["archived_at"] = datetime.utcnow().isoformat()
                    pattern.confidence_score = 0.1
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Stale pattern cleanup failed: {e}")
    
    async def _prediction_validation_loop(self):
        """Validate error predictions and update accuracy."""
        
        while self.running:
            try:
                await self._validate_predictions()
            except Exception as e:
                logger.error(f"Prediction validation failed: {e}")
            
            await asyncio.sleep(3600)  # Every hour
    
    async def _validate_predictions(self):
        """Validate predictions against actual occurrences."""
        
        try:
            with get_db_context() as db:
                # Get unvalidated predictions older than 1 hour
                validation_threshold = datetime.utcnow() - timedelta(hours=1)
                
                unvalidated_predictions = db.query(ErrorPrediction).filter(
                    ErrorPrediction.prediction_occurred.is_(None),
                    ErrorPrediction.predicted_at < validation_threshold
                ).limit(100).all()
                
                logger.info(f"Validating {len(unvalidated_predictions)} predictions")
                
                for prediction in unvalidated_predictions:
                    # Check if predicted error occurred
                    occurrence = db.query(ErrorOccurrence).filter(
                        ErrorOccurrence.pattern_id == prediction.predicted_pattern_id,
                        ErrorOccurrence.user_id == prediction.user_id,
                        ErrorOccurrence.timestamp > prediction.predicted_at,
                        ErrorOccurrence.timestamp < prediction.predicted_at + timedelta(hours=24)
                    ).first()
                    
                    if occurrence:
                        prediction.prediction_occurred = True
                        prediction.actual_error_id = occurrence.id
                        prediction.outcome_recorded_at = datetime.utcnow()
                        
                        # Check if it was prevented
                        if occurrence.resolved and occurrence.resolution_time < 60:
                            prediction.prevented = True
                    else:
                        # No occurrence within 24 hours
                        prediction.prediction_occurred = False
                        prediction.outcome_recorded_at = datetime.utcnow()
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Prediction validation failed: {e}")
    
    async def _pattern_evolution_loop(self):
        """Evolve patterns based on new data and feedback."""
        
        while self.running:
            try:
                await self._evolve_patterns()
            except Exception as e:
                logger.error(f"Pattern evolution failed: {e}")
            
            await asyncio.sleep(7200)  # Every 2 hours
    
    async def _evolve_patterns(self):
        """Evolve patterns based on feedback and new occurrences."""
        
        try:
            with get_db_context() as db:
                # Get patterns with recent feedback
                patterns_with_feedback = db.query(EnhancedErrorPattern).join(
                    ErrorFeedback
                ).filter(
                    ErrorFeedback.created_at > datetime.utcnow() - timedelta(days=7)
                ).distinct().all()
                
                logger.info(f"Evolving {len(patterns_with_feedback)} patterns based on feedback")
                
                for pattern in patterns_with_feedback:
                    # Get recent feedback
                    recent_feedback = db.query(ErrorFeedback).filter_by(
                        pattern_id=pattern.id
                    ).order_by(desc(ErrorFeedback.created_at)).limit(10).all()
                    
                    # Analyze feedback sentiment
                    positive_feedback = sum(1 for f in recent_feedback if f.helpful)
                    negative_feedback = len(recent_feedback) - positive_feedback
                    
                    # Update pattern confidence based on feedback
                    if positive_feedback > negative_feedback * 2:
                        pattern.confidence_score = min(0.95, pattern.confidence_score + 0.05)
                    elif negative_feedback > positive_feedback * 2:
                        pattern.confidence_score = max(0.1, pattern.confidence_score - 0.05)
                    
                    # Extract improvement suggestions
                    suggestions = [f.suggested_improvement for f in recent_feedback if f.suggested_improvement]
                    if suggestions:
                        pattern.context["user_suggestions"] = suggestions[:5]
                    
                    # Update false positive rate
                    if negative_feedback > 0:
                        pattern.false_positive_rate = negative_feedback / len(recent_feedback)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Pattern evolution failed: {e}")


# Global worker instance
error_analyzer_worker = ErrorAnalyzerWorker()


async def start_error_analyzer():
    """Start the error analyzer worker."""
    await error_analyzer_worker.start()


async def stop_error_analyzer():
    """Stop the error analyzer worker."""
    await error_analyzer_worker.stop()