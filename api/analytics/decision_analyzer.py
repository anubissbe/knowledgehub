"""
Decision Analytics and Pattern Mining Module.

This module provides advanced analytics, pattern recognition, and visualization
for decision data to identify trends and improve decision-making.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
from uuid import UUID
import json

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text, func, and_, or_

from ..models.enhanced_decision import (
    EnhancedDecision, EnhancedDecisionOutcome, DecisionPattern,
    DecisionType, DecisionStatus, OutcomeStatus, ImpactLevel
)
from ..models.base import get_db_context
from shared.logging import setup_logging

logger = setup_logging("decision_analyzer")


@dataclass
class DecisionTrend:
    """Trend analysis for decisions."""
    period: str  # daily, weekly, monthly
    trend_type: str  # confidence, success_rate, volume
    values: List[float]
    timestamps: List[datetime]
    direction: str  # increasing, decreasing, stable
    change_percentage: float


@dataclass
class DecisionCluster:
    """Cluster of similar decisions."""
    cluster_id: str
    decision_type: str
    common_characteristics: Dict[str, Any]
    member_decisions: List[UUID]
    avg_success_rate: float
    avg_confidence: float
    best_practices: List[str]


@dataclass
class DecisionInsight:
    """Actionable insight from decision analysis."""
    insight_type: str  # recommendation, warning, opportunity
    title: str
    description: str
    confidence: float
    supporting_data: Dict[str, Any]
    recommended_actions: List[str]


class DecisionAnalyzer:
    """
    Advanced decision analytics and pattern mining.
    
    Features:
    - Trend analysis
    - Pattern recognition
    - Clustering similar decisions
    - Success factor identification
    - Predictive analytics
    - Insight generation
    """
    
    def __init__(self):
        self.min_cluster_size = 3
        self.trend_window_days = 90
        self.confidence_threshold = 0.7
        logger.info("Initialized DecisionAnalyzer")
    
    def analyze_trends(
        self,
        decision_type: Optional[DecisionType] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        period: str = "daily",
        window_days: int = 30
    ) -> List[DecisionTrend]:
        """
        Analyze decision trends over time.
        
        Args:
            decision_type: Optional filter by type
            user_id: Optional filter by user
            project_id: Optional filter by project
            period: Analysis period (daily, weekly, monthly)
            window_days: Time window for analysis
            
        Returns:
            List of trend analyses
        """
        try:
            with get_db_context() as db:
                # Base query
                query = db.query(EnhancedDecision)
                
                if decision_type:
                    query = query.filter(EnhancedDecision.decision_type == decision_type.value)
                if user_id:
                    query = query.filter(EnhancedDecision.user_id == user_id)
                if project_id:
                    query = query.filter(EnhancedDecision.project_id == project_id)
                
                # Time filter
                cutoff = datetime.utcnow() - timedelta(days=window_days)
                query = query.filter(EnhancedDecision.created_at >= cutoff)
                
                decisions = query.all()
                
                if not decisions:
                    return []
                
                trends = []
                
                # Volume trend
                volume_trend = self._analyze_volume_trend(decisions, period)
                if volume_trend:
                    trends.append(volume_trend)
                
                # Confidence trend
                confidence_trend = self._analyze_confidence_trend(decisions, period)
                if confidence_trend:
                    trends.append(confidence_trend)
                
                # Success rate trend
                success_trend = self._analyze_success_trend(decisions, period)
                if success_trend:
                    trends.append(success_trend)
                
                # Impact distribution trend
                impact_trend = self._analyze_impact_trend(decisions, period)
                if impact_trend:
                    trends.append(impact_trend)
                
                return trends
                
        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")
            return []
    
    def identify_success_factors(
        self,
        decision_type: Optional[DecisionType] = None,
        min_decisions: int = 10
    ) -> Dict[str, Any]:
        """
        Identify factors that contribute to decision success.
        
        Args:
            decision_type: Optional filter by type
            min_decisions: Minimum decisions for analysis
            
        Returns:
            Success factors analysis
        """
        try:
            with get_db_context() as db:
                # Get decisions with outcomes
                query = db.query(EnhancedDecision).join(
                    EnhancedDecisionOutcome
                ).filter(
                    EnhancedDecisionOutcome.status.in_([
                        OutcomeStatus.SUCCESSFUL.value,
                        OutcomeStatus.FAILED.value
                    ])
                )
                
                if decision_type:
                    query = query.filter(EnhancedDecision.decision_type == decision_type.value)
                
                decisions = query.all()
                
                if len(decisions) < min_decisions:
                    return {"error": "Insufficient data for analysis"}
                
                # Analyze success factors
                success_factors = {
                    "confidence_correlation": self._analyze_confidence_correlation(decisions),
                    "impact_level_success": self._analyze_impact_success(decisions),
                    "context_patterns": self._analyze_context_patterns(decisions),
                    "timing_factors": self._analyze_timing_factors(decisions),
                    "alternative_count_impact": self._analyze_alternative_impact(decisions)
                }
                
                # Key insights
                insights = self._extract_success_insights(success_factors)
                
                return {
                    "total_decisions_analyzed": len(decisions),
                    "success_rate": sum(
                        1 for d in decisions
                        if d.outcome.status == OutcomeStatus.SUCCESSFUL.value
                    ) / len(decisions),
                    "factors": success_factors,
                    "insights": insights
                }
                
        except Exception as e:
            logger.error(f"Failed to identify success factors: {e}")
            return {"error": str(e)}
    
    def cluster_decisions(
        self,
        decision_type: Optional[DecisionType] = None,
        max_clusters: int = 10
    ) -> List[DecisionCluster]:
        """
        Cluster similar decisions to identify patterns.
        
        Args:
            decision_type: Optional filter by type
            max_clusters: Maximum number of clusters
            
        Returns:
            List of decision clusters
        """
        try:
            with get_db_context() as db:
                query = db.query(EnhancedDecision)
                
                if decision_type:
                    query = query.filter(EnhancedDecision.decision_type == decision_type.value)
                
                decisions = query.all()
                
                if len(decisions) < self.min_cluster_size:
                    return []
                
                # Group by pattern hash initially
                pattern_groups = defaultdict(list)
                for decision in decisions:
                    if decision.pattern_hash:
                        pattern_groups[decision.pattern_hash].append(decision)
                
                # Create clusters from groups
                clusters = []
                
                for pattern_hash, group_decisions in pattern_groups.items():
                    if len(group_decisions) >= self.min_cluster_size:
                        cluster = self._create_cluster(pattern_hash, group_decisions)
                        if cluster:
                            clusters.append(cluster)
                
                # Sort by member count and limit
                clusters.sort(key=lambda c: len(c.member_decisions), reverse=True)
                
                return clusters[:max_clusters]
                
        except Exception as e:
            logger.error(f"Failed to cluster decisions: {e}")
            return []
    
    def generate_insights(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        lookback_days: int = 30
    ) -> List[DecisionInsight]:
        """
        Generate actionable insights from decision history.
        
        Args:
            user_id: Optional user filter
            project_id: Optional project filter
            lookback_days: Days to look back
            
        Returns:
            List of actionable insights
        """
        try:
            with get_db_context() as db:
                insights = []
                
                # Get recent decisions
                query = db.query(EnhancedDecision)
                
                if user_id:
                    query = query.filter(EnhancedDecision.user_id == user_id)
                if project_id:
                    query = query.filter(EnhancedDecision.project_id == project_id)
                
                cutoff = datetime.utcnow() - timedelta(days=lookback_days)
                recent_decisions = query.filter(
                    EnhancedDecision.created_at >= cutoff
                ).all()
                
                if not recent_decisions:
                    return []
                
                # Low confidence pattern
                low_confidence_insight = self._check_low_confidence_pattern(recent_decisions)
                if low_confidence_insight:
                    insights.append(low_confidence_insight)
                
                # High revision rate
                revision_insight = self._check_high_revision_rate(recent_decisions)
                if revision_insight:
                    insights.append(revision_insight)
                
                # Success rate deviation
                success_insight = self._check_success_rate_deviation(recent_decisions)
                if success_insight:
                    insights.append(success_insight)
                
                # Decision velocity
                velocity_insight = self._check_decision_velocity(recent_decisions)
                if velocity_insight:
                    insights.append(velocity_insight)
                
                # Impact distribution
                impact_insight = self._check_impact_distribution(recent_decisions)
                if impact_insight:
                    insights.append(impact_insight)
                
                return insights
                
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []
    
    def predict_decision_outcome(
        self,
        decision_type: DecisionType,
        confidence_score: float,
        impact_level: ImpactLevel,
        context: Dict[str, Any],
        alternatives_count: int
    ) -> Dict[str, Any]:
        """
        Predict likely outcome of a decision based on historical patterns.
        
        Args:
            decision_type: Type of decision
            confidence_score: Decision confidence
            impact_level: Impact level
            context: Decision context
            alternatives_count: Number of alternatives considered
            
        Returns:
            Prediction with confidence and factors
        """
        try:
            with get_db_context() as db:
                # Find similar past decisions
                similar_decisions = db.query(EnhancedDecision).filter(
                    EnhancedDecision.decision_type == decision_type.value,
                    EnhancedDecision.impact_level == impact_level.value
                ).limit(100).all()
                
                if not similar_decisions:
                    return {
                        "prediction": "unknown",
                        "confidence": 0.0,
                        "reason": "No historical data available"
                    }
                
                # Calculate success factors
                success_count = 0
                confidence_weighted_success = 0.0
                
                for decision in similar_decisions:
                    if decision.outcome:
                        if decision.outcome.status == OutcomeStatus.SUCCESSFUL.value:
                            success_count += 1
                            # Weight by confidence similarity
                            conf_similarity = 1.0 - abs(decision.confidence_score - confidence_score)
                            confidence_weighted_success += conf_similarity
                
                # Base success rate
                base_success_rate = success_count / len(similar_decisions) if similar_decisions else 0.5
                
                # Adjust for confidence
                confidence_factor = confidence_score * 0.3
                
                # Adjust for alternatives
                alternatives_factor = 0.1 if alternatives_count >= 3 else -0.05
                
                # Adjust for impact
                impact_factors = {
                    ImpactLevel.CRITICAL: -0.15,
                    ImpactLevel.HIGH: -0.1,
                    ImpactLevel.MEDIUM: 0.0,
                    ImpactLevel.LOW: 0.05,
                    ImpactLevel.MINIMAL: 0.1
                }
                impact_factor = impact_factors.get(impact_level, 0.0)
                
                # Calculate final prediction
                success_probability = base_success_rate + confidence_factor + alternatives_factor + impact_factor
                success_probability = max(0.0, min(1.0, success_probability))
                
                # Determine prediction
                if success_probability > 0.7:
                    prediction = "likely_successful"
                elif success_probability > 0.5:
                    prediction = "moderately_successful"
                elif success_probability > 0.3:
                    prediction = "uncertain"
                else:
                    prediction = "likely_challenging"
                
                return {
                    "prediction": prediction,
                    "success_probability": success_probability,
                    "confidence": min(0.9, len(similar_decisions) / 20),  # Confidence in prediction
                    "factors": {
                        "base_success_rate": base_success_rate,
                        "confidence_boost": confidence_factor,
                        "alternatives_boost": alternatives_factor,
                        "impact_adjustment": impact_factor
                    },
                    "similar_decisions_analyzed": len(similar_decisions)
                }
                
        except Exception as e:
            logger.error(f"Failed to predict decision outcome: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    # Helper methods
    
    def _analyze_volume_trend(
        self,
        decisions: List[EnhancedDecision],
        period: str
    ) -> Optional[DecisionTrend]:
        """Analyze decision volume trend."""
        # Group decisions by period
        period_counts = defaultdict(int)
        
        for decision in decisions:
            period_key = self._get_period_key(decision.created_at, period)
            period_counts[period_key] += 1
        
        if len(period_counts) < 2:
            return None
        
        # Sort by date
        sorted_periods = sorted(period_counts.items())
        timestamps = [self._parse_period_key(k, period) for k, _ in sorted_periods]
        values = [float(v) for _, v in sorted_periods]
        
        # Calculate trend
        direction, change = self._calculate_trend_direction(values)
        
        return DecisionTrend(
            period=period,
            trend_type="volume",
            values=values,
            timestamps=timestamps,
            direction=direction,
            change_percentage=change
        )
    
    def _analyze_confidence_trend(
        self,
        decisions: List[EnhancedDecision],
        period: str
    ) -> Optional[DecisionTrend]:
        """Analyze confidence score trend."""
        period_confidence = defaultdict(list)
        
        for decision in decisions:
            period_key = self._get_period_key(decision.created_at, period)
            period_confidence[period_key].append(decision.confidence_score)
        
        if len(period_confidence) < 2:
            return None
        
        # Calculate averages
        sorted_periods = sorted(period_confidence.items())
        timestamps = [self._parse_period_key(k, period) for k, _ in sorted_periods]
        values = [sum(scores) / len(scores) for _, scores in sorted_periods]
        
        direction, change = self._calculate_trend_direction(values)
        
        return DecisionTrend(
            period=period,
            trend_type="confidence",
            values=values,
            timestamps=timestamps,
            direction=direction,
            change_percentage=change
        )
    
    def _analyze_success_trend(
        self,
        decisions: List[EnhancedDecision],
        period: str
    ) -> Optional[DecisionTrend]:
        """Analyze success rate trend."""
        period_outcomes = defaultdict(lambda: {"success": 0, "total": 0})
        
        for decision in decisions:
            if decision.outcome:
                period_key = self._get_period_key(decision.created_at, period)
                period_outcomes[period_key]["total"] += 1
                if decision.outcome.status == OutcomeStatus.SUCCESSFUL.value:
                    period_outcomes[period_key]["success"] += 1
        
        if len(period_outcomes) < 2:
            return None
        
        # Calculate success rates
        sorted_periods = sorted(period_outcomes.items())
        timestamps = [self._parse_period_key(k, period) for k, _ in sorted_periods]
        values = [
            data["success"] / data["total"] if data["total"] > 0 else 0.0
            for _, data in sorted_periods
        ]
        
        direction, change = self._calculate_trend_direction(values)
        
        return DecisionTrend(
            period=period,
            trend_type="success_rate",
            values=values,
            timestamps=timestamps,
            direction=direction,
            change_percentage=change
        )
    
    def _analyze_impact_trend(
        self,
        decisions: List[EnhancedDecision],
        period: str
    ) -> Optional[DecisionTrend]:
        """Analyze impact level trend."""
        impact_scores = {
            ImpactLevel.CRITICAL.value: 5,
            ImpactLevel.HIGH.value: 4,
            ImpactLevel.MEDIUM.value: 3,
            ImpactLevel.LOW.value: 2,
            ImpactLevel.MINIMAL.value: 1
        }
        
        period_impacts = defaultdict(list)
        
        for decision in decisions:
            period_key = self._get_period_key(decision.created_at, period)
            impact_score = impact_scores.get(decision.impact_level, 3)
            period_impacts[period_key].append(impact_score)
        
        if len(period_impacts) < 2:
            return None
        
        # Calculate averages
        sorted_periods = sorted(period_impacts.items())
        timestamps = [self._parse_period_key(k, period) for k, _ in sorted_periods]
        values = [sum(scores) / len(scores) for _, scores in sorted_periods]
        
        direction, change = self._calculate_trend_direction(values)
        
        return DecisionTrend(
            period=period,
            trend_type="impact_level",
            values=values,
            timestamps=timestamps,
            direction=direction,
            change_percentage=change
        )
    
    def _get_period_key(self, dt: datetime, period: str) -> str:
        """Get period key for grouping."""
        if period == "daily":
            return dt.strftime("%Y-%m-%d")
        elif period == "weekly":
            return dt.strftime("%Y-W%U")
        elif period == "monthly":
            return dt.strftime("%Y-%m")
        else:
            return dt.strftime("%Y-%m-%d")
    
    def _parse_period_key(self, key: str, period: str) -> datetime:
        """Parse period key back to datetime."""
        if period == "daily":
            return datetime.strptime(key, "%Y-%m-%d")
        elif period == "weekly":
            return datetime.strptime(key + "-1", "%Y-W%U-%w")
        elif period == "monthly":
            return datetime.strptime(key + "-01", "%Y-%m-%d")
        else:
            return datetime.strptime(key, "%Y-%m-%d")
    
    def _calculate_trend_direction(
        self,
        values: List[float]
    ) -> Tuple[str, float]:
        """Calculate trend direction and change percentage."""
        if len(values) < 2:
            return "stable", 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Calculate percentage change
        if values[0] != 0:
            change_pct = ((values[-1] - values[0]) / values[0]) * 100
        else:
            change_pct = 0.0
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return direction, change_pct
    
    def _analyze_confidence_correlation(
        self,
        decisions: List[EnhancedDecision]
    ) -> Dict[str, float]:
        """Analyze correlation between confidence and success."""
        confidence_success = []
        
        for decision in decisions:
            if decision.outcome:
                success = 1.0 if decision.outcome.status == OutcomeStatus.SUCCESSFUL.value else 0.0
                confidence_success.append((decision.confidence_score, success))
        
        if not confidence_success:
            return {"correlation": 0.0, "significance": "none"}
        
        # Calculate correlation
        confidences, successes = zip(*confidence_success)
        
        if len(set(successes)) == 1:  # All same outcome
            correlation = 0.0
        else:
            correlation = np.corrcoef(confidences, successes)[0, 1]
        
        # Determine significance
        if abs(correlation) > 0.7:
            significance = "strong"
        elif abs(correlation) > 0.4:
            significance = "moderate"
        elif abs(correlation) > 0.2:
            significance = "weak"
        else:
            significance = "none"
        
        return {
            "correlation": float(correlation),
            "significance": significance,
            "sample_size": len(confidence_success)
        }
    
    def _analyze_impact_success(
        self,
        decisions: List[EnhancedDecision]
    ) -> Dict[str, float]:
        """Analyze success rates by impact level."""
        impact_outcomes = defaultdict(lambda: {"success": 0, "total": 0})
        
        for decision in decisions:
            if decision.outcome:
                impact_outcomes[decision.impact_level]["total"] += 1
                if decision.outcome.status == OutcomeStatus.SUCCESSFUL.value:
                    impact_outcomes[decision.impact_level]["success"] += 1
        
        success_rates = {}
        for impact, data in impact_outcomes.items():
            if data["total"] > 0:
                success_rates[impact] = data["success"] / data["total"]
        
        return success_rates
    
    def _analyze_context_patterns(
        self,
        decisions: List[EnhancedDecision]
    ) -> Dict[str, Any]:
        """Analyze patterns in decision context."""
        successful_contexts = []
        failed_contexts = []
        
        for decision in decisions:
            if decision.outcome and decision.context:
                if decision.outcome.status == OutcomeStatus.SUCCESSFUL.value:
                    successful_contexts.append(decision.context)
                else:
                    failed_contexts.append(decision.context)
        
        # Find common keys
        if successful_contexts:
            common_success_keys = set.intersection(
                *[set(ctx.keys()) for ctx in successful_contexts]
            )
        else:
            common_success_keys = set()
        
        if failed_contexts:
            common_failure_keys = set.intersection(
                *[set(ctx.keys()) for ctx in failed_contexts]
            )
        else:
            common_failure_keys = set()
        
        return {
            "common_success_factors": list(common_success_keys),
            "common_failure_factors": list(common_failure_keys),
            "unique_success_factors": list(common_success_keys - common_failure_keys),
            "unique_failure_factors": list(common_failure_keys - common_success_keys)
        }
    
    def _analyze_timing_factors(
        self,
        decisions: List[EnhancedDecision]
    ) -> Dict[str, Any]:
        """Analyze timing-related success factors."""
        implementation_times = []
        
        for decision in decisions:
            if decision.outcome and decision.implemented_at and decision.decided_at:
                hours = (decision.implemented_at - decision.decided_at).total_seconds() / 3600
                success = decision.outcome.status == OutcomeStatus.SUCCESSFUL.value
                implementation_times.append((hours, success))
        
        if not implementation_times:
            return {"average_implementation_time": None}
        
        # Analyze by success
        successful_times = [t for t, s in implementation_times if s]
        failed_times = [t for t, s in implementation_times if not s]
        
        return {
            "avg_successful_implementation_hours": np.mean(successful_times) if successful_times else None,
            "avg_failed_implementation_hours": np.mean(failed_times) if failed_times else None,
            "optimal_implementation_window": self._find_optimal_window(implementation_times)
        }
    
    def _find_optimal_window(
        self,
        implementation_times: List[Tuple[float, bool]]
    ) -> Dict[str, Any]:
        """Find optimal implementation time window."""
        if not implementation_times:
            return {"min_hours": None, "max_hours": None}
        
        # Sort by time
        sorted_times = sorted(implementation_times, key=lambda x: x[0])
        
        # Find window with highest success rate
        best_window = None
        best_rate = 0.0
        window_size = max(1, len(sorted_times) // 4)  # 25% window
        
        for i in range(len(sorted_times) - window_size + 1):
            window = sorted_times[i:i + window_size]
            success_rate = sum(1 for _, s in window if s) / len(window)
            
            if success_rate > best_rate:
                best_rate = success_rate
                best_window = (window[0][0], window[-1][0])
        
        if best_window:
            return {
                "min_hours": best_window[0],
                "max_hours": best_window[1],
                "success_rate": best_rate
            }
        else:
            return {"min_hours": None, "max_hours": None}
    
    def _analyze_alternative_impact(
        self,
        decisions: List[EnhancedDecision]
    ) -> Dict[str, float]:
        """Analyze impact of considering alternatives."""
        with_alternatives = {"success": 0, "total": 0}
        without_alternatives = {"success": 0, "total": 0}
        
        for decision in decisions:
            if decision.outcome:
                has_alternatives = len(decision.alternatives) > 0
                
                if has_alternatives:
                    with_alternatives["total"] += 1
                    if decision.outcome.status == OutcomeStatus.SUCCESSFUL.value:
                        with_alternatives["success"] += 1
                else:
                    without_alternatives["total"] += 1
                    if decision.outcome.status == OutcomeStatus.SUCCESSFUL.value:
                        without_alternatives["success"] += 1
        
        results = {}
        
        if with_alternatives["total"] > 0:
            results["success_rate_with_alternatives"] = (
                with_alternatives["success"] / with_alternatives["total"]
            )
        
        if without_alternatives["total"] > 0:
            results["success_rate_without_alternatives"] = (
                without_alternatives["success"] / without_alternatives["total"]
            )
        
        return results
    
    def _extract_success_insights(
        self,
        success_factors: Dict[str, Any]
    ) -> List[str]:
        """Extract key insights from success factors."""
        insights = []
        
        # Confidence correlation
        conf_corr = success_factors.get("confidence_correlation", {})
        if conf_corr.get("significance") == "strong":
            if conf_corr.get("correlation", 0) > 0:
                insights.append(
                    "Higher confidence strongly correlates with success"
                )
            else:
                insights.append(
                    "Lower confidence decisions tend to be more successful - "
                    "possible overconfidence bias"
                )
        
        # Impact level
        impact_success = success_factors.get("impact_level_success", {})
        if impact_success:
            best_impact = max(impact_success.items(), key=lambda x: x[1])
            worst_impact = min(impact_success.items(), key=lambda x: x[1])
            
            if best_impact[1] > 0.8:
                insights.append(
                    f"{best_impact[0]} impact decisions have {best_impact[1]:.0%} success rate"
                )
            if worst_impact[1] < 0.5:
                insights.append(
                    f"{worst_impact[0]} impact decisions need more careful consideration "
                    f"(only {worst_impact[1]:.0%} success)"
                )
        
        # Alternatives
        alt_impact = success_factors.get("alternative_count_impact", {})
        with_alt = alt_impact.get("success_rate_with_alternatives", 0)
        without_alt = alt_impact.get("success_rate_without_alternatives", 0)
        
        if with_alt > without_alt * 1.2:
            insights.append(
                f"Considering alternatives improves success by "
                f"{(with_alt - without_alt) * 100:.0f}%"
            )
        
        # Timing
        timing = success_factors.get("timing_factors", {})
        optimal = timing.get("optimal_implementation_window", {})
        if optimal.get("min_hours") is not None:
            insights.append(
                f"Optimal implementation window: {optimal['min_hours']:.0f}-"
                f"{optimal['max_hours']:.0f} hours after decision"
            )
        
        return insights
    
    def _create_cluster(
        self,
        pattern_hash: str,
        decisions: List[EnhancedDecision]
    ) -> Optional[DecisionCluster]:
        """Create a decision cluster."""
        if not decisions:
            return None
        
        # Extract common characteristics
        common_type = decisions[0].decision_type
        
        # Collect all contexts
        all_contexts = [d.context for d in decisions if d.context]
        common_keys = set()
        if all_contexts:
            common_keys = set.intersection(*[set(ctx.keys()) for ctx in all_contexts])
        
        # Calculate metrics
        success_count = sum(
            1 for d in decisions
            if d.outcome and d.outcome.status == OutcomeStatus.SUCCESSFUL.value
        )
        success_rate = success_count / len(decisions) if decisions else 0.0
        
        avg_confidence = sum(d.confidence_score for d in decisions) / len(decisions)
        
        # Extract best practices
        best_practices = []
        successful_decisions = [
            d for d in decisions
            if d.outcome and d.outcome.status == OutcomeStatus.SUCCESSFUL.value
        ]
        
        if successful_decisions:
            # Common reasoning patterns
            reasoning_words = Counter()
            for d in successful_decisions:
                words = d.reasoning.lower().split()
                reasoning_words.update(words)
            
            # Top reasoning keywords
            top_words = [word for word, _ in reasoning_words.most_common(5)]
            if top_words:
                best_practices.append(
                    f"Successful decisions often mention: {', '.join(top_words)}"
                )
        
        return DecisionCluster(
            cluster_id=pattern_hash[:8],
            decision_type=common_type,
            common_characteristics={
                "common_context_keys": list(common_keys),
                "pattern_hash": pattern_hash
            },
            member_decisions=[d.id for d in decisions],
            avg_success_rate=success_rate,
            avg_confidence=avg_confidence,
            best_practices=best_practices
        )
    
    def _check_low_confidence_pattern(
        self,
        decisions: List[EnhancedDecision]
    ) -> Optional[DecisionInsight]:
        """Check for pattern of low confidence decisions."""
        low_conf_decisions = [
            d for d in decisions
            if d.confidence_score < 0.6
        ]
        
        if len(low_conf_decisions) > len(decisions) * 0.3:
            return DecisionInsight(
                insight_type="warning",
                title="High frequency of low-confidence decisions",
                description=(
                    f"{len(low_conf_decisions)} out of {len(decisions)} recent decisions "
                    f"have confidence below 60%"
                ),
                confidence=0.8,
                supporting_data={
                    "low_confidence_count": len(low_conf_decisions),
                    "total_decisions": len(decisions),
                    "average_confidence": sum(d.confidence_score for d in decisions) / len(decisions)
                },
                recommended_actions=[
                    "Consider more thorough analysis before decisions",
                    "Gather additional data to increase confidence",
                    "Review decision-making process for gaps"
                ]
            )
        
        return None
    
    def _check_high_revision_rate(
        self,
        decisions: List[EnhancedDecision]
    ) -> Optional[DecisionInsight]:
        """Check for high revision rate."""
        revised_count = sum(1 for d in decisions if d.revisions)
        revision_rate = revised_count / len(decisions) if decisions else 0.0
        
        if revision_rate > 0.3:
            return DecisionInsight(
                insight_type="warning",
                title="High decision revision rate",
                description=(
                    f"{revision_rate:.0%} of recent decisions have been revised, "
                    f"indicating potential issues with initial decision quality"
                ),
                confidence=0.85,
                supporting_data={
                    "revised_decisions": revised_count,
                    "total_decisions": len(decisions),
                    "revision_rate": revision_rate
                },
                recommended_actions=[
                    "Analyze common revision reasons",
                    "Improve initial decision analysis",
                    "Consider more alternatives upfront"
                ]
            )
        
        return None
    
    def _check_success_rate_deviation(
        self,
        decisions: List[EnhancedDecision]
    ) -> Optional[DecisionInsight]:
        """Check for success rate deviations."""
        decisions_with_outcomes = [d for d in decisions if d.outcome]
        
        if len(decisions_with_outcomes) < 5:
            return None
        
        success_count = sum(
            1 for d in decisions_with_outcomes
            if d.outcome.status == OutcomeStatus.SUCCESSFUL.value
        )
        success_rate = success_count / len(decisions_with_outcomes)
        
        if success_rate < 0.5:
            return DecisionInsight(
                insight_type="warning",
                title="Below-average decision success rate",
                description=(
                    f"Only {success_rate:.0%} of decisions with outcomes were successful, "
                    f"below the expected rate"
                ),
                confidence=0.9,
                supporting_data={
                    "successful_decisions": success_count,
                    "total_with_outcomes": len(decisions_with_outcomes),
                    "success_rate": success_rate
                },
                recommended_actions=[
                    "Review failed decisions for common patterns",
                    "Implement decision review process",
                    "Consider external consultation for critical decisions"
                ]
            )
        elif success_rate > 0.9:
            return DecisionInsight(
                insight_type="opportunity",
                title="Exceptional decision success rate",
                description=(
                    f"{success_rate:.0%} of decisions were successful - "
                    f"consider documenting and sharing best practices"
                ),
                confidence=0.9,
                supporting_data={
                    "successful_decisions": success_count,
                    "total_with_outcomes": len(decisions_with_outcomes),
                    "success_rate": success_rate
                },
                recommended_actions=[
                    "Document successful decision patterns",
                    "Share learnings with team",
                    "Consider taking on more ambitious decisions"
                ]
            )
        
        return None
    
    def _check_decision_velocity(
        self,
        decisions: List[EnhancedDecision]
    ) -> Optional[DecisionInsight]:
        """Check decision-making velocity."""
        if len(decisions) < 2:
            return None
        
        # Calculate decisions per week
        time_span = (decisions[-1].created_at - decisions[0].created_at).days
        if time_span == 0:
            return None
        
        decisions_per_week = len(decisions) / (time_span / 7)
        
        if decisions_per_week > 20:
            return DecisionInsight(
                insight_type="warning",
                title="High decision-making velocity",
                description=(
                    f"Making {decisions_per_week:.0f} decisions per week - "
                    f"ensure adequate time for analysis"
                ),
                confidence=0.7,
                supporting_data={
                    "decisions_per_week": decisions_per_week,
                    "total_decisions": len(decisions),
                    "time_span_days": time_span
                },
                recommended_actions=[
                    "Review if all decisions need immediate action",
                    "Consider batching similar decisions",
                    "Ensure decision quality isn't compromised"
                ]
            )
        elif decisions_per_week < 1:
            return DecisionInsight(
                insight_type="opportunity",
                title="Low decision-making velocity",
                description=(
                    f"Only {decisions_per_week:.1f} decisions per week - "
                    f"consider if analysis paralysis is occurring"
                ),
                confidence=0.7,
                supporting_data={
                    "decisions_per_week": decisions_per_week,
                    "total_decisions": len(decisions),
                    "time_span_days": time_span
                },
                recommended_actions=[
                    "Set decision deadlines",
                    "Use time-boxing for analysis",
                    "Consider 'good enough' decisions for low-impact items"
                ]
            )
        
        return None
    
    def _check_impact_distribution(
        self,
        decisions: List[EnhancedDecision]
    ) -> Optional[DecisionInsight]:
        """Check decision impact distribution."""
        impact_counts = Counter(d.impact_level for d in decisions)
        
        critical_high = (
            impact_counts.get(ImpactLevel.CRITICAL.value, 0) +
            impact_counts.get(ImpactLevel.HIGH.value, 0)
        )
        
        critical_high_ratio = critical_high / len(decisions) if decisions else 0.0
        
        if critical_high_ratio > 0.5:
            return DecisionInsight(
                insight_type="recommendation",
                title="High proportion of critical decisions",
                description=(
                    f"{critical_high_ratio:.0%} of decisions are high or critical impact - "
                    f"consider risk mitigation strategies"
                ),
                confidence=0.8,
                supporting_data={
                    "critical_high_count": critical_high,
                    "total_decisions": len(decisions),
                    "impact_distribution": dict(impact_counts)
                },
                recommended_actions=[
                    "Implement formal review process for high-impact decisions",
                    "Consider breaking down critical decisions",
                    "Ensure adequate stakeholder involvement"
                ]
            )
        
        return None


# Global analyzer instance
decision_analyzer = DecisionAnalyzer()