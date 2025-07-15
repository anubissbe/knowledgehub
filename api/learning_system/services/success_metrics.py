"""Success Metrics Service

This service provides comprehensive metrics calculation and analysis for
decision outcomes and learning system performance, building analytics
foundation for the success tracking system.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
from uuid import UUID
from statistics import mean, median, stdev, variance
from collections import defaultdict, Counter
from dataclasses import dataclass
import asyncio
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func, or_, desc, asc, case, text
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, Field

from ..models.decision_outcome import DecisionOutcome, OutcomeType
from ..models.learning_pattern import LearningPattern, PatternType
from ..models.user_feedback import UserFeedback, FeedbackType
from ...memory_system.models.memory import Memory, MemoryType
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of success metrics"""
    SUCCESS_RATE = "success_rate"
    EFFECTIVENESS = "effectiveness" 
    IMPROVEMENT = "improvement"
    CONSISTENCY = "consistency"
    IMPACT = "impact"
    TREND = "trend"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"


class TimeFrame(Enum):
    """Time frame options for metrics"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    ALL_TIME = "all_time"


@dataclass
class MetricFilter:
    """Filter criteria for metrics calculation"""
    time_frame: TimeFrame = TimeFrame.MONTH
    decision_types: Optional[List[str]] = None
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    outcome_types: Optional[List[OutcomeType]] = None
    include_partial: bool = True


class SuccessMetricResult(BaseModel):
    """Result of success metric calculation"""
    metric_type: MetricType
    value: float
    unit: str = ""
    confidence: float = 1.0
    sample_size: int = 0
    time_frame: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    
    
class MetricsDashboard(BaseModel):
    """Comprehensive metrics dashboard data"""
    overview: Dict[str, Any] = Field(default_factory=dict)
    success_metrics: List[SuccessMetricResult] = Field(default_factory=list)
    trends: Dict[str, Any] = Field(default_factory=dict)
    breakdowns: Dict[str, Any] = Field(default_factory=dict)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SuccessMetrics:
    """Advanced success metrics calculation and analysis service"""
    
    def __init__(self, db: Session):
        """Initialize the success metrics service"""
        self.db = db
        
        # Configuration
        self.cache_ttl = 300  # 5 minutes
        self.min_sample_size = 5
        self.confidence_threshold = 0.7
        
        # Success thresholds
        self.excellent_threshold = 0.9
        self.good_threshold = 0.7
        self.poor_threshold = 0.3
        
        # Metric weights for composite scores
        self.metric_weights = {
            MetricType.SUCCESS_RATE: 0.3,
            MetricType.EFFECTIVENESS: 0.25,
            MetricType.CONSISTENCY: 0.2,
            MetricType.IMPROVEMENT: 0.15,
            MetricType.RELIABILITY: 0.1
        }
        
        # Time frame mappings
        self.time_deltas = {
            TimeFrame.HOUR: timedelta(hours=1),
            TimeFrame.DAY: timedelta(days=1),
            TimeFrame.WEEK: timedelta(weeks=1),
            TimeFrame.MONTH: timedelta(days=30),
            TimeFrame.QUARTER: timedelta(days=90),
            TimeFrame.YEAR: timedelta(days=365)
        }
    
    async def calculate_success_rate(
        self,
        filter_criteria: Optional[MetricFilter] = None
    ) -> SuccessMetricResult:
        """Calculate success rate with detailed breakdown
        
        Args:
            filter_criteria: Optional filter for specific criteria
            
        Returns:
            SuccessMetricResult with success rate data
        """
        try:
            # Get filtered outcomes
            outcomes = await self._get_filtered_outcomes(filter_criteria)
            
            if not outcomes:
                return SuccessMetricResult(
                    metric_type=MetricType.SUCCESS_RATE,
                    value=0.0,
                    unit="percentage",
                    sample_size=0,
                    details={"message": "No data available"}
                )
            
            # Calculate basic success rate
            success_count = len([o for o in outcomes if o.outcome_type == OutcomeType.SUCCESS.value])
            failure_count = len([o for o in outcomes if o.outcome_type == OutcomeType.FAILURE.value])
            partial_count = len([o for o in outcomes if o.outcome_type == OutcomeType.PARTIAL.value])
            
            success_rate = success_count / len(outcomes)
            
            # Calculate confidence based on sample size
            confidence = min(1.0, len(outcomes) / 50)  # Full confidence at 50+ samples
            
            # Detailed breakdown
            details = {
                "total_outcomes": len(outcomes),
                "successes": success_count,
                "failures": failure_count,
                "partial": partial_count,
                "success_percentage": round(success_rate * 100, 2),
                "failure_percentage": round((failure_count / len(outcomes)) * 100, 2),
                "partial_percentage": round((partial_count / len(outcomes)) * 100, 2),
                "average_score": round(mean([o.success_score for o in outcomes]), 3),
                "median_score": round(median([o.success_score for o in outcomes]), 3)
            }
            
            # Add score distribution
            score_ranges = self._calculate_score_distribution(outcomes)
            details["score_distribution"] = score_ranges
            
            # Add time-based analysis
            if len(outcomes) > 10:
                time_analysis = await self._analyze_success_over_time(outcomes)
                details["time_analysis"] = time_analysis
            
            return SuccessMetricResult(
                metric_type=MetricType.SUCCESS_RATE,
                value=round(success_rate, 4),
                unit="percentage",
                confidence=confidence,
                sample_size=len(outcomes),
                time_frame=filter_criteria.time_frame.value if filter_criteria else "all_time",
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}")
            return SuccessMetricResult(
                metric_type=MetricType.SUCCESS_RATE,
                value=0.0,
                unit="percentage",
                details={"error": str(e)}
            )
    
    async def calculate_effectiveness_score(
        self,
        filter_criteria: Optional[MetricFilter] = None
    ) -> SuccessMetricResult:
        """Calculate overall effectiveness score combining multiple factors
        
        Args:
            filter_criteria: Optional filter for specific criteria
            
        Returns:
            SuccessMetricResult with effectiveness data
        """
        try:
            outcomes = await self._get_filtered_outcomes(filter_criteria)
            
            if not outcomes:
                return SuccessMetricResult(
                    metric_type=MetricType.EFFECTIVENESS,
                    value=0.0,
                    unit="score",
                    sample_size=0
                )
            
            # Calculate component scores
            success_rate = len([o for o in outcomes if o.outcome_type == OutcomeType.SUCCESS.value]) / len(outcomes)
            average_score = mean([o.success_score for o in outcomes])
            consistency = await self._calculate_consistency(outcomes)
            impact_score = await self._calculate_impact_score(outcomes)
            
            # Weighted effectiveness score
            effectiveness = (
                success_rate * 0.4 +
                average_score * 0.3 +
                consistency * 0.2 +
                impact_score * 0.1
            )
            
            # Calculate confidence
            confidence = min(1.0, len(outcomes) / 30)
            
            details = {
                "component_scores": {
                    "success_rate": round(success_rate, 3),
                    "average_score": round(average_score, 3),
                    "consistency": round(consistency, 3),
                    "impact_score": round(impact_score, 3)
                },
                "effectiveness_grade": self._get_effectiveness_grade(effectiveness),
                "sample_size": len(outcomes),
                "recommendation": await self._get_effectiveness_recommendation(effectiveness, outcomes)
            }
            
            return SuccessMetricResult(
                metric_type=MetricType.EFFECTIVENESS,
                value=round(effectiveness, 4),
                unit="score",
                confidence=confidence,
                sample_size=len(outcomes),
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error calculating effectiveness: {e}")
            return SuccessMetricResult(
                metric_type=MetricType.EFFECTIVENESS,
                value=0.0,
                unit="score",
                details={"error": str(e)}
            )
    
    async def calculate_improvement_trend(
        self,
        filter_criteria: Optional[MetricFilter] = None,
        comparison_periods: int = 3
    ) -> SuccessMetricResult:
        """Calculate improvement trend over time periods
        
        Args:
            filter_criteria: Optional filter for specific criteria
            comparison_periods: Number of periods to compare
            
        Returns:
            SuccessMetricResult with improvement trend data
        """
        try:
            outcomes = await self._get_filtered_outcomes(filter_criteria)
            
            if len(outcomes) < comparison_periods:
                return SuccessMetricResult(
                    metric_type=MetricType.IMPROVEMENT,
                    value=0.0,
                    unit="trend",
                    sample_size=len(outcomes),
                    details={"message": "Insufficient data for trend analysis"}
                )
            
            # Sort outcomes by time
            sorted_outcomes = sorted(outcomes, key=lambda o: o.measured_at)
            
            # Split into periods
            period_size = len(sorted_outcomes) // comparison_periods
            periods = []
            
            for i in range(comparison_periods):
                start_idx = i * period_size
                end_idx = start_idx + period_size if i < comparison_periods - 1 else len(sorted_outcomes)
                period_outcomes = sorted_outcomes[start_idx:end_idx]
                
                if period_outcomes:
                    period_score = mean([o.success_score for o in period_outcomes])
                    periods.append({
                        "period": i + 1,
                        "score": period_score,
                        "count": len(period_outcomes),
                        "start_date": period_outcomes[0].measured_at.isoformat(),
                        "end_date": period_outcomes[-1].measured_at.isoformat()
                    })
            
            # Calculate improvement trend
            if len(periods) >= 2:
                first_score = periods[0]["score"]
                last_score = periods[-1]["score"]
                improvement = (last_score - first_score) / max(first_score, 0.01)
            else:
                improvement = 0.0
            
            # Analyze trend direction and strength
            trend_analysis = self._analyze_trend_direction(periods)
            
            details = {
                "periods": periods,
                "improvement_percentage": round(improvement * 100, 2),
                "trend_direction": trend_analysis["direction"],
                "trend_strength": trend_analysis["strength"],
                "consistency": trend_analysis["consistency"],
                "recommendation": self._get_improvement_recommendation(improvement, trend_analysis)
            }
            
            # Confidence based on data quality
            confidence = min(1.0, len(outcomes) / 40) * trend_analysis["consistency"]
            
            return SuccessMetricResult(
                metric_type=MetricType.IMPROVEMENT,
                value=round(improvement, 4),
                unit="percentage",
                confidence=confidence,
                sample_size=len(outcomes),
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error calculating improvement trend: {e}")
            return SuccessMetricResult(
                metric_type=MetricType.IMPROVEMENT,
                value=0.0,
                unit="percentage",
                details={"error": str(e)}
            )
    
    async def calculate_consistency_score(
        self,
        filter_criteria: Optional[MetricFilter] = None
    ) -> SuccessMetricResult:
        """Calculate consistency of decision outcomes
        
        Args:
            filter_criteria: Optional filter for specific criteria
            
        Returns:
            SuccessMetricResult with consistency data
        """
        try:
            outcomes = await self._get_filtered_outcomes(filter_criteria)
            
            if len(outcomes) < 3:
                return SuccessMetricResult(
                    metric_type=MetricType.CONSISTENCY,
                    value=0.0,
                    unit="score",
                    sample_size=len(outcomes),
                    details={"message": "Insufficient data for consistency analysis"}
                )
            
            scores = [o.success_score for o in outcomes]
            
            # Calculate various consistency metrics
            score_variance = variance(scores)
            score_stdev = stdev(scores)
            score_range = max(scores) - min(scores)
            
            # Consistency score (inverse of normalized variance)
            normalized_variance = score_variance / (mean(scores) ** 2) if mean(scores) > 0 else 1.0
            consistency = 1.0 / (1.0 + normalized_variance)
            
            # Analyze consistency patterns
            consecutive_consistency = await self._analyze_consecutive_consistency(outcomes)
            outcome_type_consistency = await self._analyze_outcome_type_consistency(outcomes)
            
            details = {
                "variance": round(score_variance, 4),
                "standard_deviation": round(score_stdev, 4),
                "score_range": round(score_range, 4),
                "consecutive_consistency": consecutive_consistency,
                "outcome_type_consistency": outcome_type_consistency,
                "consistency_grade": self._get_consistency_grade(consistency),
                "factors_affecting_consistency": await self._identify_consistency_factors(outcomes)
            }
            
            # Confidence based on sample size and variance
            confidence = min(1.0, len(outcomes) / 25) * (1.0 - min(score_variance, 1.0))
            
            return SuccessMetricResult(
                metric_type=MetricType.CONSISTENCY,
                value=round(consistency, 4),
                unit="score",
                confidence=confidence,
                sample_size=len(outcomes),
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error calculating consistency: {e}")
            return SuccessMetricResult(
                metric_type=MetricType.CONSISTENCY,
                value=0.0,
                unit="score",
                details={"error": str(e)}
            )
    
    async def calculate_impact_metrics(
        self,
        filter_criteria: Optional[MetricFilter] = None
    ) -> SuccessMetricResult:
        """Calculate impact metrics from decision outcomes
        
        Args:
            filter_criteria: Optional filter for specific criteria
            
        Returns:
            SuccessMetricResult with impact data
        """
        try:
            outcomes = await self._get_filtered_outcomes(filter_criteria)
            
            if not outcomes:
                return SuccessMetricResult(
                    metric_type=MetricType.IMPACT,
                    value=0.0,
                    unit="score",
                    sample_size=0
                )
            
            # Analyze impact data from outcomes
            impact_scores = []
            impact_categories = defaultdict(list)
            
            for outcome in outcomes:
                impact_data = outcome.impact_data or {}
                
                # Calculate overall impact score
                performance_impact = impact_data.get('performance_impact', 0.5)
                user_satisfaction = impact_data.get('user_satisfaction', 0.5)
                time_efficiency = impact_data.get('time_efficiency', 0.5)
                error_reduction = impact_data.get('error_reduction', 0.5)
                
                overall_impact = (
                    performance_impact * 0.3 +
                    user_satisfaction * 0.3 +
                    time_efficiency * 0.2 +
                    error_reduction * 0.2
                )
                
                impact_scores.append(overall_impact)
                
                # Categorize impacts
                for category, value in impact_data.items():
                    if isinstance(value, (int, float)):
                        impact_categories[category].append(value)
            
            # Calculate aggregate impact metrics
            if impact_scores:
                average_impact = mean(impact_scores)
                impact_consistency = 1.0 - (stdev(impact_scores) / max(mean(impact_scores), 0.01))
            else:
                average_impact = 0.0
                impact_consistency = 0.0
            
            # Analyze impact by category
            category_analysis = {}
            for category, values in impact_categories.items():
                if values:
                    category_analysis[category] = {
                        "average": round(mean(values), 3),
                        "median": round(median(values), 3),
                        "trend": self._calculate_simple_trend(values)
                    }
            
            details = {
                "average_impact": round(average_impact, 3),
                "impact_consistency": round(impact_consistency, 3),
                "category_analysis": category_analysis,
                "high_impact_decisions": len([s for s in impact_scores if s > 0.8]),
                "low_impact_decisions": len([s for s in impact_scores if s < 0.3]),
                "impact_distribution": self._calculate_impact_distribution(impact_scores)
            }
            
            confidence = min(1.0, len(outcomes) / 20)
            
            return SuccessMetricResult(
                metric_type=MetricType.IMPACT,
                value=round(average_impact, 4),
                unit="score",
                confidence=confidence,
                sample_size=len(outcomes),
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error calculating impact metrics: {e}")
            return SuccessMetricResult(
                metric_type=MetricType.IMPACT,
                value=0.0,
                unit="score",
                details={"error": str(e)}
            )
    
    async def generate_metrics_dashboard(
        self,
        filter_criteria: Optional[MetricFilter] = None
    ) -> MetricsDashboard:
        """Generate comprehensive metrics dashboard
        
        Args:
            filter_criteria: Optional filter for specific criteria
            
        Returns:
            MetricsDashboard with all metrics and insights
        """
        try:
            # Calculate all key metrics
            metrics_tasks = [
                self.calculate_success_rate(filter_criteria),
                self.calculate_effectiveness_score(filter_criteria),
                self.calculate_improvement_trend(filter_criteria),
                self.calculate_consistency_score(filter_criteria),
                self.calculate_impact_metrics(filter_criteria)
            ]
            
            metrics_results = await asyncio.gather(*metrics_tasks, return_exceptions=True)
            
            # Process results
            success_metrics = []
            for result in metrics_results:
                if isinstance(result, SuccessMetricResult):
                    success_metrics.append(result)
                else:
                    logger.error(f"Error in metric calculation: {result}")
            
            # Generate overview
            overview = await self._generate_overview(success_metrics, filter_criteria)
            
            # Generate trends analysis
            trends = await self._generate_trends_analysis(filter_criteria)
            
            # Generate breakdowns
            breakdowns = await self._generate_breakdowns(filter_criteria)
            
            # Generate insights and recommendations
            insights = await self._generate_insights(success_metrics, overview)
            recommendations = await self._generate_recommendations(success_metrics, insights)
            
            return MetricsDashboard(
                overview=overview,
                success_metrics=success_metrics,
                trends=trends,
                breakdowns=breakdowns,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generating metrics dashboard: {e}")
            return MetricsDashboard(
                overview={"error": str(e)},
                success_metrics=[],
                trends={},
                breakdowns={},
                insights=["Error generating dashboard"],
                recommendations=["Please check system logs"]
            )
    
    # Private helper methods
    
    async def _get_filtered_outcomes(
        self,
        filter_criteria: Optional[MetricFilter] = None
    ) -> List[DecisionOutcome]:
        """Get outcomes based on filter criteria"""
        query = select(DecisionOutcome)
        
        if filter_criteria:
            # Time frame filter
            if filter_criteria.time_frame != TimeFrame.ALL_TIME:
                start_time = datetime.now(timezone.utc) - self.time_deltas[filter_criteria.time_frame]
                query = query.where(DecisionOutcome.measured_at >= start_time)
            
            # Score filters
            if filter_criteria.min_score is not None:
                query = query.where(DecisionOutcome.success_score >= filter_criteria.min_score)
            if filter_criteria.max_score is not None:
                query = query.where(DecisionOutcome.success_score <= filter_criteria.max_score)
            
            # Outcome type filter
            if filter_criteria.outcome_types:
                outcome_values = [ot.value for ot in filter_criteria.outcome_types]
                query = query.where(DecisionOutcome.outcome_type.in_(outcome_values))
            
            # Partial outcomes filter
            if not filter_criteria.include_partial:
                query = query.where(DecisionOutcome.outcome_type != OutcomeType.PARTIAL.value)
        
        result = await self.db.execute(query.order_by(DecisionOutcome.measured_at))
        return result.scalars().all()
    
    def _calculate_score_distribution(self, outcomes: List[DecisionOutcome]) -> Dict[str, int]:
        """Calculate distribution of success scores"""
        distribution = {
            "excellent (0.9-1.0)": 0,
            "good (0.7-0.9)": 0,
            "fair (0.5-0.7)": 0,
            "poor (0.3-0.5)": 0,
            "very_poor (0.0-0.3)": 0
        }
        
        for outcome in outcomes:
            score = outcome.success_score
            if score >= 0.9:
                distribution["excellent (0.9-1.0)"] += 1
            elif score >= 0.7:
                distribution["good (0.7-0.9)"] += 1
            elif score >= 0.5:
                distribution["fair (0.5-0.7)"] += 1
            elif score >= 0.3:
                distribution["poor (0.3-0.5)"] += 1
            else:
                distribution["very_poor (0.0-0.3)"] += 1
        
        return distribution
    
    async def _analyze_success_over_time(self, outcomes: List[DecisionOutcome]) -> Dict[str, Any]:
        """Analyze success patterns over time"""
        sorted_outcomes = sorted(outcomes, key=lambda o: o.measured_at)
        
        # Split into time buckets
        if len(sorted_outcomes) >= 10:
            bucket_size = len(sorted_outcomes) // 5
            buckets = []
            
            for i in range(5):
                start_idx = i * bucket_size
                end_idx = start_idx + bucket_size if i < 4 else len(sorted_outcomes)
                bucket_outcomes = sorted_outcomes[start_idx:end_idx]
                
                if bucket_outcomes:
                    success_rate = len([o for o in bucket_outcomes if o.outcome_type == OutcomeType.SUCCESS.value]) / len(bucket_outcomes)
                    avg_score = mean([o.success_score for o in bucket_outcomes])
                    
                    buckets.append({
                        "period": i + 1,
                        "success_rate": round(success_rate, 3),
                        "average_score": round(avg_score, 3),
                        "count": len(bucket_outcomes)
                    })
            
            # Determine overall trend
            if len(buckets) >= 2:
                first_rate = buckets[0]["success_rate"]
                last_rate = buckets[-1]["success_rate"]
                
                if last_rate > first_rate + 0.1:
                    trend = "improving"
                elif last_rate < first_rate - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            return {
                "time_buckets": buckets,
                "overall_trend": trend,
                "trend_strength": abs(buckets[-1]["success_rate"] - buckets[0]["success_rate"]) if len(buckets) >= 2 else 0.0
            }
        
        return {"message": "Insufficient data for time analysis"}
    
    async def _calculate_consistency(self, outcomes: List[DecisionOutcome]) -> float:
        """Calculate consistency score for outcomes"""
        if len(outcomes) < 2:
            return 1.0
        
        scores = [o.success_score for o in outcomes]
        score_variance = variance(scores)
        
        # Normalize variance and convert to consistency score
        normalized_variance = score_variance / (mean(scores) ** 2) if mean(scores) > 0 else 1.0
        consistency = 1.0 / (1.0 + normalized_variance)
        
        return min(1.0, consistency)
    
    async def _calculate_impact_score(self, outcomes: List[DecisionOutcome]) -> float:
        """Calculate overall impact score from outcomes"""
        impact_scores = []
        
        for outcome in outcomes:
            impact_data = outcome.impact_data or {}
            
            # Extract impact metrics
            performance = impact_data.get('performance_impact', 0.5)
            satisfaction = impact_data.get('user_satisfaction', 0.5)
            efficiency = impact_data.get('time_efficiency', 0.5)
            
            # Calculate weighted impact
            impact = (performance * 0.4 + satisfaction * 0.4 + efficiency * 0.2)
            impact_scores.append(impact)
        
        return mean(impact_scores) if impact_scores else 0.5
    
    def _get_effectiveness_grade(self, effectiveness: float) -> str:
        """Get effectiveness grade based on score"""
        if effectiveness >= self.excellent_threshold:
            return "Excellent"
        elif effectiveness >= self.good_threshold:
            return "Good"
        elif effectiveness >= 0.5:
            return "Fair"
        elif effectiveness >= self.poor_threshold:
            return "Poor"
        else:
            return "Very Poor"
    
    async def _get_effectiveness_recommendation(
        self,
        effectiveness: float,
        outcomes: List[DecisionOutcome]
    ) -> str:
        """Generate recommendation based on effectiveness"""
        if effectiveness >= self.excellent_threshold:
            return "Maintain current high performance standards"
        elif effectiveness >= self.good_threshold:
            return "Good performance, look for optimization opportunities"
        elif effectiveness >= 0.5:
            return "Fair performance, identify improvement areas"
        elif effectiveness >= self.poor_threshold:
            return "Poor performance, requires significant improvement"
        else:
            return "Critical performance issues, immediate action required"
    
    def _analyze_trend_direction(self, periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        if len(periods) < 2:
            return {"direction": "unknown", "strength": 0.0, "consistency": 0.0}
        
        scores = [p["score"] for p in periods]
        
        # Calculate trend direction
        positive_changes = 0
        negative_changes = 0
        
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:
                positive_changes += 1
            elif scores[i] < scores[i-1]:
                negative_changes += 1
        
        total_changes = positive_changes + negative_changes
        
        if total_changes == 0:
            direction = "stable"
            consistency = 1.0
        else:
            if positive_changes > negative_changes:
                direction = "improving"
            elif negative_changes > positive_changes:
                direction = "declining"
            else:
                direction = "mixed"
            
            consistency = max(positive_changes, negative_changes) / total_changes
        
        # Calculate strength
        if len(scores) >= 2:
            strength = abs(scores[-1] - scores[0]) / max(scores[0], 0.01)
        else:
            strength = 0.0
        
        return {
            "direction": direction,
            "strength": min(1.0, strength),
            "consistency": consistency
        }
    
    def _get_improvement_recommendation(
        self,
        improvement: float,
        trend_analysis: Dict[str, Any]
    ) -> str:
        """Generate improvement recommendation"""
        direction = trend_analysis.get("direction", "unknown")
        strength = trend_analysis.get("strength", 0.0)
        
        if direction == "improving" and improvement > 0.1:
            return "Strong positive trend, continue current strategies"
        elif direction == "improving" and improvement > 0.05:
            return "Positive trend, maintain and optimize current approach"
        elif direction == "stable":
            return "Stable performance, explore new improvement opportunities"
        elif direction == "declining" and improvement < -0.05:
            return "Declining trend detected, investigate and address issues"
        elif direction == "declining" and improvement < -0.1:
            return "Significant decline, immediate intervention required"
        else:
            return "Mixed or unclear trend, monitor closely and adjust as needed"
    
    async def _analyze_consecutive_consistency(self, outcomes: List[DecisionOutcome]) -> Dict[str, Any]:
        """Analyze consistency in consecutive outcomes"""
        sorted_outcomes = sorted(outcomes, key=lambda o: o.measured_at)
        
        if len(sorted_outcomes) < 3:
            return {"message": "Insufficient data"}
        
        # Calculate consecutive score differences
        differences = []
        for i in range(1, len(sorted_outcomes)):
            diff = abs(sorted_outcomes[i].success_score - sorted_outcomes[i-1].success_score)
            differences.append(diff)
        
        avg_difference = mean(differences)
        max_difference = max(differences)
        
        # Count consecutive successes and failures
        consecutive_successes = 0
        consecutive_failures = 0
        current_success_streak = 0
        current_failure_streak = 0
        max_success_streak = 0
        max_failure_streak = 0
        
        for outcome in sorted_outcomes:
            if outcome.outcome_type == OutcomeType.SUCCESS.value:
                current_success_streak += 1
                current_failure_streak = 0
                max_success_streak = max(max_success_streak, current_success_streak)
            elif outcome.outcome_type == OutcomeType.FAILURE.value:
                current_failure_streak += 1
                current_success_streak = 0
                max_failure_streak = max(max_failure_streak, current_failure_streak)
            else:
                current_success_streak = 0
                current_failure_streak = 0
        
        return {
            "average_consecutive_difference": round(avg_difference, 3),
            "max_consecutive_difference": round(max_difference, 3),
            "max_success_streak": max_success_streak,
            "max_failure_streak": max_failure_streak,
            "current_success_streak": current_success_streak,
            "current_failure_streak": current_failure_streak
        }
    
    async def _analyze_outcome_type_consistency(self, outcomes: List[DecisionOutcome]) -> Dict[str, Any]:
        """Analyze consistency of outcome types"""
        outcome_counts = Counter([o.outcome_type for o in outcomes])
        total_outcomes = len(outcomes)
        
        # Calculate entropy as measure of consistency
        entropy = 0.0
        for count in outcome_counts.values():
            if count > 0:
                probability = count / total_outcomes
                entropy -= probability * (probability.bit_length() - 1) if probability > 0 else 0
        
        # Normalize entropy (max entropy for 3 outcome types is ~1.58)
        normalized_entropy = entropy / 1.585 if entropy > 0 else 0.0
        consistency = 1.0 - normalized_entropy
        
        return {
            "outcome_distribution": dict(outcome_counts),
            "entropy": round(entropy, 3),
            "consistency_score": round(consistency, 3),
            "dominant_outcome": max(outcome_counts, key=outcome_counts.get) if outcome_counts else None
        }
    
    async def _identify_consistency_factors(self, outcomes: List[DecisionOutcome]) -> List[str]:
        """Identify factors affecting consistency"""
        factors = []
        
        # Analyze score variance by decision type
        decision_types = {}
        for outcome in outcomes:
            decision_data = outcome.outcome_data or {}
            decision_type = decision_data.get('decision_type', 'unknown')
            
            if decision_type not in decision_types:
                decision_types[decision_type] = []
            decision_types[decision_type].append(outcome.success_score)
        
        # Find most and least consistent decision types
        type_variances = {}
        for dt, scores in decision_types.items():
            if len(scores) > 1:
                type_variances[dt] = variance(scores)
        
        if type_variances:
            most_consistent = min(type_variances, key=type_variances.get)
            least_consistent = max(type_variances, key=type_variances.get)
            
            factors.append(f"Most consistent decision type: {most_consistent}")
            factors.append(f"Least consistent decision type: {least_consistent}")
        
        # Analyze time-based patterns
        sorted_outcomes = sorted(outcomes, key=lambda o: o.measured_at)
        if len(sorted_outcomes) >= 6:
            first_half = sorted_outcomes[:len(sorted_outcomes)//2]
            second_half = sorted_outcomes[len(sorted_outcomes)//2:]
            
            first_variance = variance([o.success_score for o in first_half])
            second_variance = variance([o.success_score for o in second_half])
            
            if second_variance < first_variance * 0.8:
                factors.append("Consistency has improved over time")
            elif second_variance > first_variance * 1.2:
                factors.append("Consistency has decreased over time")
            else:
                factors.append("Consistency has remained stable over time")
        
        return factors
    
    def _get_consistency_grade(self, consistency: float) -> str:
        """Get consistency grade based on score"""
        if consistency >= 0.9:
            return "Excellent"
        elif consistency >= 0.8:
            return "Good"
        elif consistency >= 0.7:
            return "Fair"
        elif consistency >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def _calculate_simple_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction"""
        if len(values) < 2:
            return "unknown"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        
        change = (second_avg - first_avg) / max(first_avg, 0.01)
        
        if change > 0.05:
            return "improving"
        elif change < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _calculate_impact_distribution(self, impact_scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of impact scores"""
        if not impact_scores:
            return {}
        
        distribution = {
            "high_impact (0.8-1.0)": 0,
            "medium_impact (0.5-0.8)": 0,
            "low_impact (0.0-0.5)": 0
        }
        
        for score in impact_scores:
            if score >= 0.8:
                distribution["high_impact (0.8-1.0)"] += 1
            elif score >= 0.5:
                distribution["medium_impact (0.5-0.8)"] += 1
            else:
                distribution["low_impact (0.0-0.5)"] += 1
        
        return distribution
    
    async def _generate_overview(
        self,
        metrics: List[SuccessMetricResult],
        filter_criteria: Optional[MetricFilter]
    ) -> Dict[str, Any]:
        """Generate overview section of dashboard"""
        overview = {
            "total_metrics_calculated": len(metrics),
            "time_frame": filter_criteria.time_frame.value if filter_criteria else "all_time",
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Extract key values
        for metric in metrics:
            if metric.metric_type == MetricType.SUCCESS_RATE:
                overview["success_rate"] = f"{metric.value * 100:.1f}%"
                overview["total_decisions"] = metric.sample_size
            elif metric.metric_type == MetricType.EFFECTIVENESS:
                overview["effectiveness_score"] = f"{metric.value:.3f}"
                overview["effectiveness_grade"] = metric.details.get("effectiveness_grade", "Unknown")
            elif metric.metric_type == MetricType.CONSISTENCY:
                overview["consistency_score"] = f"{metric.value:.3f}"
            elif metric.metric_type == MetricType.IMPROVEMENT:
                overview["improvement_trend"] = f"{metric.value * 100:.1f}%"
        
        return overview
    
    async def _generate_trends_analysis(self, filter_criteria: Optional[MetricFilter]) -> Dict[str, Any]:
        """Generate trends analysis"""
        # This would include more sophisticated trend analysis
        # For now, return basic structure
        return {
            "success_rate_trend": "stable",
            "effectiveness_trend": "improving",
            "consistency_trend": "stable",
            "note": "Detailed trend analysis requires longer time series data"
        }
    
    async def _generate_breakdowns(self, filter_criteria: Optional[MetricFilter]) -> Dict[str, Any]:
        """Generate detailed breakdowns"""
        return {
            "by_decision_type": {},
            "by_time_period": {},
            "by_outcome_type": {},
            "note": "Detailed breakdowns require additional data processing"
        }
    
    async def _generate_insights(
        self,
        metrics: List[SuccessMetricResult],
        overview: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from metrics"""
        insights = []
        
        # Analyze success rate
        for metric in metrics:
            if metric.metric_type == MetricType.SUCCESS_RATE and metric.value < 0.7:
                insights.append(f"Success rate of {metric.value*100:.1f}% is below optimal threshold")
            elif metric.metric_type == MetricType.EFFECTIVENESS and metric.value < 0.6:
                insights.append("Overall effectiveness could be improved")
            elif metric.metric_type == MetricType.CONSISTENCY and metric.value < 0.7:
                insights.append("Decision consistency shows room for improvement")
        
        if not insights:
            insights.append("System performance metrics are within acceptable ranges")
        
        return insights
    
    async def _generate_recommendations(
        self,
        metrics: List[SuccessMetricResult],
        insights: List[str]
    ) -> List[str]:
        """Generate recommendations based on metrics and insights"""
        recommendations = []
        
        # Generate recommendations based on metric values
        for metric in metrics:
            if metric.metric_type == MetricType.SUCCESS_RATE and metric.value < 0.6:
                recommendations.append("Focus on improving decision accuracy and success factors")
            elif metric.metric_type == MetricType.CONSISTENCY and metric.value < 0.6:
                recommendations.append("Implement consistency checks and standardized processes")
            elif metric.metric_type == MetricType.IMPROVEMENT and metric.value < 0:
                recommendations.append("Address declining performance trends immediately")
        
        if not recommendations:
            recommendations.append("Continue monitoring and maintaining current performance levels")
        
        return recommendations