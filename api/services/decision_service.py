"""
Advanced Decision Recording and Analysis Service.

This service provides comprehensive decision tracking, analysis, pattern mining,
and AI-powered recommendations for better decision making.
"""

import logging
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from uuid import UUID
import numpy as np
from dataclasses import dataclass
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text, and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from ..models.enhanced_decision import (
    EnhancedDecision, EnhancedAlternative, EnhancedDecisionOutcome,
    EnhancedDecisionFeedback, EnhancedDecisionRevision, DecisionPattern,
    DecisionType, DecisionStatus, OutcomeStatus, ImpactLevel,
    DecisionCreate, AlternativeCreate, OutcomeCreate, FeedbackCreate,
    DecisionResponse, DecisionTreeNode, DecisionAnalytics
)
from ..models.base import get_db_context
from ..services.embedding_service import embedding_service, EmbeddingModel
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService, MetricType
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("decision_service")


@dataclass
class DecisionRecommendation:
    """Decision recommendation based on patterns and history."""
    recommended_option: str
    confidence: float
    reasoning: str
    similar_decisions: List[EnhancedDecision]
    success_probability: float
    estimated_effort: float
    risks: List[str]
    best_practices: List[str]


@dataclass
class DecisionImpact:
    """Impact analysis for a decision."""
    direct_components: List[str]
    indirect_components: List[str]
    downstream_decisions: List[EnhancedDecision]
    estimated_scope: str  # minimal, moderate, extensive
    risk_assessment: Dict[str, Any]


class DecisionService:
    """
    Advanced Decision Recording and Analysis Service.
    
    Features:
    - Decision recording with alternatives and reasoning
    - Outcome tracking and validation
    - Pattern mining and recognition
    - AI-powered recommendations
    - Decision tree visualization
    - Impact analysis
    - Learning from historical decisions
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.embedding_service = embedding_service
        self.analytics_service = TimeSeriesAnalyticsService()
        self._initialized = False
        
        # Configuration
        self.similarity_threshold = 0.7
        self.pattern_confidence_threshold = 0.75
        self.min_pattern_occurrences = 3
        self.learning_rate = 0.1
        
        logger.info("Initialized DecisionService")
    
    async def initialize(self):
        """Initialize the decision service."""
        if self._initialized:
            return
        
        try:
            # Initialize dependencies
            await self.embedding_service.initialize()
            await self.analytics_service.initialize()
            await redis_client.initialize()
            
            self._initialized = True
            logger.info("DecisionService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DecisionService: {e}")
            raise
    
    async def record_decision(
        self,
        decision_data: DecisionCreate,
        user_id: str,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> EnhancedDecision:
        """
        Record a new decision with alternatives and reasoning.
        
        Args:
            decision_data: Decision data including alternatives
            user_id: User ID
            session_id: Optional session ID
            project_id: Optional project ID
            
        Returns:
            Created decision record
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            with get_db_context() as db:
                # Calculate pattern hash
                pattern_hash = self._calculate_pattern_hash(
                    decision_data.decision_type.value,
                    decision_data.context,
                    decision_data.constraints
                )
                
                # Generate embeddings for similarity search
                embeddings = None
                if self.embedding_service:
                    try:
                        embedding_result = await self.embedding_service.generate_embedding(
                            text=f"{decision_data.title} {decision_data.reasoning}",
                            context={
                                "type": "decision",
                                "decision_type": decision_data.decision_type.value
                            }
                        )
                        embeddings = embedding_result.embeddings
                    except Exception as e:
                        logger.warning(f"Failed to generate embeddings: {e}")
                
                # Predict success probability
                success_probability = await self._predict_success_probability(
                    decision_data.decision_type.value,
                    decision_data.confidence_score,
                    decision_data.impact_level.value,
                    decision_data.context
                )
                
                # Create decision record
                decision = EnhancedDecision(
                    decision_type=decision_data.decision_type.value,
                    category=decision_data.category,
                    title=decision_data.title,
                    description=decision_data.description,
                    chosen_option=decision_data.chosen_option,
                    reasoning=decision_data.reasoning,
                    confidence_score=decision_data.confidence_score,
                    context=decision_data.context,
                    constraints=decision_data.constraints,
                    assumptions=decision_data.assumptions,
                    risks=decision_data.risks,
                    dependencies=decision_data.dependencies,
                    impact_analysis=decision_data.impact_analysis,
                    impact_level=decision_data.impact_level.value,
                    affected_components=decision_data.affected_components,
                    estimated_effort=decision_data.estimated_effort,
                    tags=decision_data.tags,
                    parent_decision_id=UUID(decision_data.parent_decision_id) if decision_data.parent_decision_id else None,
                    status=DecisionStatus.DECIDED.value,
                    user_id=user_id,
                    session_id=UUID(session_id) if session_id else None,
                    project_id=project_id,
                    pattern_hash=pattern_hash,
                    embeddings=embeddings,
                    success_probability=success_probability
                )
                
                db.add(decision)
                db.flush()
                
                # Create alternatives
                for alt_data in decision_data.alternatives:
                    alternative = EnhancedAlternative(
                        decision_id=decision.id,
                        option=alt_data.option,
                        description=alt_data.description,
                        pros=alt_data.pros,
                        cons=alt_data.cons,
                        evaluation_score=alt_data.evaluation_score,
                        rejection_reason=alt_data.rejection_reason,
                        feasibility_score=alt_data.feasibility_score,
                        risk_score=alt_data.risk_score,
                        cost_estimate=alt_data.cost_estimate,
                        complexity_score=alt_data.complexity_score
                    )
                    db.add(alternative)
                
                db.commit()
                
                # Update patterns asynchronously
                asyncio.create_task(self._update_patterns(decision))
                
                # Record analytics
                await self._record_decision_analytics(decision)
                
                logger.info(f"Recorded decision: {decision.id}")
                return decision
                
        except Exception as e:
            logger.error(f"Failed to record decision: {e}")
            raise
    
    async def get_recommendations(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> DecisionRecommendation:
        """
        Get AI-powered recommendations based on similar past decisions.
        
        Args:
            decision_type: Type of decision
            context: Decision context
            constraints: Decision constraints
            user_id: Optional user ID for personalization
            project_id: Optional project ID
            
        Returns:
            Decision recommendation
        """
        try:
            with get_db_context() as db:
                # Find similar decisions
                similar_decisions = await self._find_similar_decisions(
                    db, decision_type.value, context, constraints
                )
                
                if not similar_decisions:
                    return DecisionRecommendation(
                        recommended_option="No historical data available",
                        confidence=0.0,
                        reasoning="No similar decisions found in history",
                        similar_decisions=[],
                        success_probability=0.5,
                        estimated_effort=0.0,
                        risks=[],
                        best_practices=[]
                    )
                
                # Analyze successful patterns
                successful_decisions = [
                    d for d in similar_decisions
                    if d.outcome and d.outcome.status == OutcomeStatus.SUCCESSFUL.value
                ]
                
                # Extract common successful choices
                option_success_rates = {}
                total_effort = 0.0
                effort_count = 0
                
                for decision in successful_decisions:
                    option = decision.chosen_option
                    if option not in option_success_rates:
                        option_success_rates[option] = []
                    
                    success_rating = decision.outcome.success_rating or 0.5
                    option_success_rates[option].append(success_rating)
                    
                    if decision.actual_effort:
                        total_effort += decision.actual_effort
                        effort_count += 1
                
                # Calculate average success rates
                best_option = None
                best_score = 0.0
                
                for option, ratings in option_success_rates.items():
                    avg_score = sum(ratings) / len(ratings)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_option = option
                
                # Get patterns and best practices
                patterns = await self._get_relevant_patterns(
                    db, decision_type.value, context
                )
                
                best_practices = []
                common_risks = []
                
                for pattern in patterns:
                    if pattern.best_practices:
                        best_practices.extend(pattern.best_practices)
                    if pattern.failure_indicators:
                        common_risks.extend([
                            f"Avoid: {indicator}"
                            for indicator in pattern.failure_indicators.get("risks", [])
                        ])
                
                # Calculate recommendation confidence
                confidence = min(0.95, best_score * len(successful_decisions) / max(len(similar_decisions), 1))
                
                # Estimate effort
                estimated_effort = total_effort / effort_count if effort_count > 0 else 40.0
                
                # Generate reasoning
                reasoning = self._generate_recommendation_reasoning(
                    best_option or "Consider multiple approaches",
                    similar_decisions,
                    patterns
                )
                
                return DecisionRecommendation(
                    recommended_option=best_option or "No clear recommendation",
                    confidence=confidence,
                    reasoning=reasoning,
                    similar_decisions=similar_decisions[:5],
                    success_probability=best_score,
                    estimated_effort=estimated_effort,
                    risks=list(set(common_risks))[:5],
                    best_practices=list(set(best_practices))[:5]
                )
                
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            raise
    
    async def record_outcome(
        self,
        outcome_data: OutcomeCreate,
        validated_by: Optional[str] = None
    ) -> EnhancedDecisionOutcome:
        """
        Record the outcome of a decision.
        
        Args:
            outcome_data: Outcome data
            validated_by: Optional validator ID
            
        Returns:
            Created outcome record
        """
        try:
            with get_db_context() as db:
                # Check if outcome already exists
                existing = db.query(EnhancedDecisionOutcome).filter_by(
                    decision_id=UUID(outcome_data.decision_id)
                ).first()
                
                if existing:
                    # Update existing outcome
                    for key, value in outcome_data.dict(exclude_unset=True).items():
                        if key != 'decision_id':
                            setattr(existing, key, value)
                    
                    if validated_by:
                        existing.validated_by = validated_by
                    
                    existing.updated_at = datetime.utcnow()
                    outcome = existing
                else:
                    # Create new outcome
                    outcome = EnhancedDecisionOutcome(
                        decision_id=UUID(outcome_data.decision_id),
                        status=outcome_data.status.value,
                        success_rating=outcome_data.success_rating,
                        description=outcome_data.description,
                        performance_metrics=outcome_data.performance_metrics,
                        quality_metrics=outcome_data.quality_metrics,
                        business_metrics=outcome_data.business_metrics,
                        lessons_learned=outcome_data.lessons_learned,
                        unexpected_consequences=outcome_data.unexpected_consequences,
                        positive_impacts=outcome_data.positive_impacts,
                        negative_impacts=outcome_data.negative_impacts,
                        validation_method=outcome_data.validation_method,
                        validation_data=outcome_data.validation_data,
                        validated_by=validated_by
                    )
                    db.add(outcome)
                
                # Update decision status
                decision = db.query(EnhancedDecision).filter_by(
                    id=UUID(outcome_data.decision_id)
                ).first()
                
                if decision:
                    if outcome_data.status == OutcomeStatus.SUCCESSFUL:
                        decision.status = DecisionStatus.VALIDATED.value
                        decision.validated_at = datetime.utcnow()
                    elif outcome_data.status == OutcomeStatus.FAILED:
                        decision.status = DecisionStatus.REVISED.value
                    
                    # Update actual effort if provided
                    if outcome_data.performance_metrics.get("actual_effort"):
                        decision.actual_effort = outcome_data.performance_metrics["actual_effort"]
                
                db.commit()
                
                # Update patterns based on outcome
                if decision:
                    asyncio.create_task(self._update_pattern_statistics(decision, outcome))
                
                logger.info(f"Recorded outcome for decision: {outcome_data.decision_id}")
                return outcome
                
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            raise
    
    async def get_decision_tree(
        self,
        root_decision_id: str,
        max_depth: int = 5
    ) -> DecisionTreeNode:
        """
        Get decision tree visualization data.
        
        Args:
            root_decision_id: Root decision ID
            max_depth: Maximum tree depth
            
        Returns:
            Decision tree structure
        """
        try:
            with get_db_context() as db:
                # Get decision tree from database
                tree_data = db.execute(text("""
                    SELECT * FROM get_decision_tree(:root_id, :max_depth)
                """), {
                    "root_id": root_decision_id,
                    "max_depth": max_depth
                }).fetchall()
                
                # Build tree structure
                nodes_by_id = {}
                root_node = None
                
                for row in tree_data:
                    node = DecisionTreeNode(
                        id=str(row.decision_id),
                        title=row.title,
                        decision_type=row.decision_type,
                        confidence_score=row.confidence_score,
                        status=row.status,
                        impact_level=self._get_decision_impact_level(db, row.decision_id),
                        children=[],
                        outcome_status=self._get_outcome_status(db, row.decision_id),
                        success_rating=self._get_success_rating(db, row.decision_id)
                    )
                    
                    nodes_by_id[str(row.decision_id)] = node
                    
                    if str(row.decision_id) == root_decision_id:
                        root_node = node
                    elif row.parent_id and str(row.parent_id) in nodes_by_id:
                        parent_node = nodes_by_id[str(row.parent_id)]
                        parent_node.children.append(node)
                
                return root_node or DecisionTreeNode(
                    id=root_decision_id,
                    title="Decision not found",
                    decision_type="unknown",
                    confidence_score=0.0,
                    status="unknown",
                    impact_level="unknown",
                    children=[]
                )
                
        except Exception as e:
            logger.error(f"Failed to get decision tree: {e}")
            raise
    
    async def analyze_decision_impact(
        self,
        decision_id: str
    ) -> DecisionImpact:
        """
        Analyze the impact of a decision.
        
        Args:
            decision_id: Decision ID
            
        Returns:
            Impact analysis
        """
        try:
            with get_db_context() as db:
                decision = db.query(EnhancedDecision).filter_by(
                    id=UUID(decision_id)
                ).first()
                
                if not decision:
                    raise ValueError(f"Decision {decision_id} not found")
                
                # Get impact metrics from database
                impact_data = db.execute(text("""
                    SELECT * FROM analyze_decision_impact(:decision_id)
                """), {"decision_id": decision_id}).fetchone()
                
                # Get downstream decisions
                downstream = db.query(EnhancedDecision).filter(
                    text(":decision_id = ANY(decision_path)"),
                    {"decision_id": decision_id}
                ).all()
                
                # Analyze indirect impact
                indirect_components = set()
                for d in downstream:
                    if d.affected_components:
                        indirect_components.update(d.affected_components)
                
                # Risk assessment
                risk_assessment = {
                    "complexity_risk": "high" if len(downstream) > 10 else "medium" if len(downstream) > 5 else "low",
                    "change_frequency": len(decision.revisions) if decision.revisions else 0,
                    "failure_probability": 1.0 - (decision.success_probability or 0.5),
                    "impact_severity": decision.impact_level
                }
                
                # Determine scope
                total_affected = impact_data.total_affected_components if impact_data else 0
                estimated_scope = (
                    "extensive" if total_affected > 20
                    else "moderate" if total_affected > 5
                    else "minimal"
                )
                
                return DecisionImpact(
                    direct_components=decision.affected_components or [],
                    indirect_components=list(indirect_components),
                    downstream_decisions=downstream,
                    estimated_scope=estimated_scope,
                    risk_assessment=risk_assessment
                )
                
        except Exception as e:
            logger.error(f"Failed to analyze decision impact: {e}")
            raise
    
    async def get_decision_analytics(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        time_window_days: int = 30
    ) -> DecisionAnalytics:
        """
        Get comprehensive decision analytics.
        
        Args:
            user_id: Optional user filter
            project_id: Optional project filter
            time_window_days: Time window for analysis
            
        Returns:
            Decision analytics
        """
        try:
            with get_db_context() as db:
                # Base query
                query = db.query(EnhancedDecision)
                
                if user_id:
                    query = query.filter(EnhancedDecision.user_id == user_id)
                if project_id:
                    query = query.filter(EnhancedDecision.project_id == project_id)
                
                # Time filter
                if time_window_days:
                    cutoff = datetime.utcnow() - timedelta(days=time_window_days)
                    query = query.filter(EnhancedDecision.created_at >= cutoff)
                
                decisions = query.all()
                
                # Calculate metrics
                total_decisions = len(decisions)
                
                # Decisions by type
                decisions_by_type = {}
                decisions_by_status = {}
                impact_distribution = {}
                
                confidence_scores = []
                successful_decisions = 0
                implementation_times = []
                
                for decision in decisions:
                    # Type distribution
                    decisions_by_type[decision.decision_type] = \
                        decisions_by_type.get(decision.decision_type, 0) + 1
                    
                    # Status distribution
                    decisions_by_status[decision.status] = \
                        decisions_by_status.get(decision.status, 0) + 1
                    
                    # Impact distribution
                    impact_distribution[decision.impact_level] = \
                        impact_distribution.get(decision.impact_level, 0) + 1
                    
                    # Confidence scores
                    confidence_scores.append(decision.confidence_score)
                    
                    # Success tracking
                    if decision.outcome and decision.outcome.status == OutcomeStatus.SUCCESSFUL.value:
                        successful_decisions += 1
                    
                    # Implementation time
                    if decision.implemented_at and decision.decided_at:
                        hours = (decision.implemented_at - decision.decided_at).total_seconds() / 3600
                        implementation_times.append(hours)
                
                # Calculate averages
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                success_rate = successful_decisions / total_decisions if total_decisions > 0 else 0.0
                avg_implementation_time = sum(implementation_times) / len(implementation_times) if implementation_times else 0.0
                
                # Get top patterns
                top_patterns = db.query(DecisionPattern).order_by(
                    desc(DecisionPattern.occurrence_count),
                    desc(DecisionPattern.success_rate)
                ).limit(5).all()
                
                pattern_list = [
                    {
                        "pattern_type": p.pattern_type,
                        "pattern_name": p.pattern_name,
                        "occurrences": p.occurrence_count,
                        "success_rate": p.success_rate,
                        "avg_confidence": p.avg_confidence
                    }
                    for p in top_patterns
                ]
                
                # Recent trends
                recent_week = datetime.utcnow() - timedelta(days=7)
                recent_decisions = [d for d in decisions if d.created_at >= recent_week]
                
                trends = {
                    "decisions_last_week": len(recent_decisions),
                    "avg_confidence_trend": sum(d.confidence_score for d in recent_decisions) / len(recent_decisions) if recent_decisions else 0.0,
                    "most_common_type": max(decisions_by_type.items(), key=lambda x: x[1])[0] if decisions_by_type else None
                }
                
                return DecisionAnalytics(
                    total_decisions=total_decisions,
                    decisions_by_type=decisions_by_type,
                    decisions_by_status=decisions_by_status,
                    avg_confidence_score=avg_confidence,
                    success_rate=success_rate,
                    avg_implementation_time_hours=avg_implementation_time,
                    top_patterns=pattern_list,
                    impact_distribution=impact_distribution,
                    recent_trends=trends
                )
                
        except Exception as e:
            logger.error(f"Failed to get decision analytics: {e}")
            raise
    
    # Helper methods
    
    def _calculate_pattern_hash(
        self,
        decision_type: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Calculate pattern hash for decision matching."""
        pattern_text = f"{decision_type}::{json.dumps(context, sort_keys=True)}::{json.dumps(constraints, sort_keys=True)}"
        return hashlib.sha256(pattern_text.encode()).hexdigest()
    
    async def _predict_success_probability(
        self,
        decision_type: str,
        confidence: float,
        impact_level: str,
        context: Dict[str, Any]
    ) -> float:
        """Predict success probability based on patterns."""
        base_probability = confidence * 0.6
        
        # Adjust based on impact level
        impact_factor = {
            "critical": -0.1,
            "high": -0.05,
            "medium": 0.0,
            "low": 0.05,
            "minimal": 0.1
        }.get(impact_level, 0.0)
        
        # Adjust based on context complexity
        context_complexity = len(context.get("constraints", {}))
        complexity_factor = -0.02 * min(context_complexity, 5)
        
        return max(0.1, min(0.95, base_probability + impact_factor + complexity_factor))
    
    async def _find_similar_decisions(
        self,
        db: DBSession,
        decision_type: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> List[EnhancedDecision]:
        """Find similar past decisions."""
        # Get pattern hash
        pattern_hash = self._calculate_pattern_hash(decision_type, context, constraints)
        
        # First try exact pattern match
        exact_matches = db.query(EnhancedDecision).filter_by(
            pattern_hash=pattern_hash
        ).limit(10).all()
        
        if exact_matches:
            return exact_matches
        
        # Try similar type and context
        similar = db.query(EnhancedDecision).filter(
            EnhancedDecision.decision_type == decision_type
        ).order_by(
            desc(EnhancedDecision.confidence_score)
        ).limit(20).all()
        
        # Filter by context similarity
        filtered = []
        for decision in similar:
            similarity = self._calculate_context_similarity(
                decision.context, context
            )
            if similarity > 0.5:
                filtered.append(decision)
        
        return filtered[:10]
    
    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two contexts."""
        if not context1 or not context2:
            return 0.0
        
        # Simple key overlap similarity
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        overlap = len(keys1.intersection(keys2))
        total = len(keys1.union(keys2))
        
        return overlap / total if total > 0 else 0.0
    
    async def _get_relevant_patterns(
        self,
        db: DBSession,
        decision_type: str,
        context: Dict[str, Any]
    ) -> List[DecisionPattern]:
        """Get relevant decision patterns."""
        patterns = db.query(DecisionPattern).filter(
            DecisionPattern.pattern_type == decision_type,
            DecisionPattern.occurrence_count >= self.min_pattern_occurrences,
            DecisionPattern.success_rate >= 0.6
        ).order_by(
            desc(DecisionPattern.occurrence_count)
        ).limit(5).all()
        
        return patterns
    
    def _generate_recommendation_reasoning(
        self,
        recommended_option: str,
        similar_decisions: List[EnhancedDecision],
        patterns: List[DecisionPattern]
    ) -> str:
        """Generate reasoning for recommendation."""
        reasoning_parts = []
        
        # Based on similar decisions
        if similar_decisions:
            success_count = sum(
                1 for d in similar_decisions
                if d.outcome and d.outcome.status == OutcomeStatus.SUCCESSFUL.value
            )
            reasoning_parts.append(
                f"Based on {len(similar_decisions)} similar past decisions with "
                f"{success_count}/{len(similar_decisions)} successful outcomes"
            )
        
        # Based on patterns
        if patterns:
            avg_success = sum(p.success_rate for p in patterns) / len(patterns)
            reasoning_parts.append(
                f"Pattern analysis shows {avg_success:.0%} average success rate "
                f"across {sum(p.occurrence_count for p in patterns)} occurrences"
            )
        
        # Recommendation
        if recommended_option != "No clear recommendation":
            reasoning_parts.append(
                f"'{recommended_option}' has shown the best results in similar contexts"
            )
        
        return ". ".join(reasoning_parts) or "Limited historical data available"
    
    async def _update_patterns(self, decision: EnhancedDecision):
        """Update decision patterns asynchronously."""
        try:
            await asyncio.sleep(1)  # Don't block main thread
            
            with get_db_context() as db:
                # Mine patterns from recent decisions
                patterns = db.execute(text("""
                    SELECT * FROM mine_decision_patterns(:min_occurrences, :min_success_rate)
                """), {
                    "min_occurrences": self.min_pattern_occurrences,
                    "min_success_rate": 0.6
                }).fetchall()
                
                # Update or create pattern records
                for pattern_data in patterns:
                    existing = db.query(DecisionPattern).filter_by(
                        pattern_type=pattern_data.pattern_type,
                        pattern_name=pattern_data.pattern_characteristics.get("pattern_hash")
                    ).first()
                    
                    if existing:
                        existing.occurrence_count = pattern_data.occurrence_count
                        existing.avg_confidence = pattern_data.avg_success_rate
                        existing.last_seen = datetime.utcnow()
                    else:
                        # Create new pattern
                        new_pattern = DecisionPattern(
                            pattern_type=pattern_data.pattern_type,
                            pattern_name=pattern_data.pattern_characteristics.get("pattern_hash"),
                            description=f"Pattern for {pattern_data.pattern_type} decisions",
                            context_patterns=pattern_data.common_context,
                            occurrence_count=pattern_data.occurrence_count,
                            success_rate=pattern_data.avg_success_rate
                        )
                        db.add(new_pattern)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Pattern update failed: {e}")
    
    async def _update_pattern_statistics(
        self,
        decision: EnhancedDecision,
        outcome: EnhancedDecisionOutcome
    ):
        """Update pattern statistics based on outcome."""
        try:
            with get_db_context() as db:
                # Find related pattern
                pattern = db.query(DecisionPattern).filter_by(
                    pattern_type=decision.decision_type,
                    pattern_name=decision.pattern_hash
                ).first()
                
                if pattern:
                    success = outcome.status == OutcomeStatus.SUCCESSFUL.value
                    implementation_hours = None
                    
                    if decision.actual_effort:
                        implementation_hours = decision.actual_effort
                    elif decision.implemented_at and decision.decided_at:
                        implementation_hours = (
                            decision.implemented_at - decision.decided_at
                        ).total_seconds() / 3600
                    
                    pattern.update_statistics(
                        new_decision_success=success,
                        confidence=decision.confidence_score,
                        implementation_hours=implementation_hours or 0.0
                    )
                    
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Pattern statistics update failed: {e}")
    
    def _get_decision_impact_level(self, db: DBSession, decision_id: UUID) -> str:
        """Get decision impact level."""
        decision = db.query(EnhancedDecision).filter_by(id=decision_id).first()
        return decision.impact_level if decision else "unknown"
    
    def _get_outcome_status(self, db: DBSession, decision_id: UUID) -> Optional[str]:
        """Get decision outcome status."""
        outcome = db.query(EnhancedDecisionOutcome).filter_by(decision_id=decision_id).first()
        return outcome.status if outcome else None
    
    def _get_success_rating(self, db: DBSession, decision_id: UUID) -> Optional[float]:
        """Get decision success rating."""
        outcome = db.query(EnhancedDecisionOutcome).filter_by(decision_id=decision_id).first()
        return outcome.success_rating if outcome else None
    
    async def _record_decision_analytics(self, decision: EnhancedDecision):
        """Record decision analytics."""
        try:
            await self.analytics_service.record_metric(
                metric_type=MetricType.CUSTOM,
                value=decision.confidence_score,
                tags={
                    "metric": "decision_confidence",
                    "decision_type": decision.decision_type,
                    "impact_level": decision.impact_level
                },
                metadata={
                    "decision_id": str(decision.id),
                    "user_id": decision.user_id,
                    "project_id": decision.project_id
                }
            )
        except Exception as e:
            logger.warning(f"Analytics recording failed: {e}")
    
    async def cleanup(self):
        """Clean up service resources."""
        await self.embedding_service.cleanup()
        await self.analytics_service.cleanup()
        self._initialized = False
        logger.info("DecisionService cleaned up")


# Global decision service instance
decision_service = DecisionService()