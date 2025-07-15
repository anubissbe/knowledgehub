"""Effectiveness Analyzer Service

This service provides deep analysis of decision effectiveness, combining
outcome data with contextual factors to identify patterns and provide
actionable insights for improving decision-making processes.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from statistics import mean, median, stdev, variance, correlation
from collections import defaultdict, Counter
from dataclasses import dataclass
import asyncio
import json
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


class EffectivenessCategory(Enum):
    """Categories of effectiveness analysis"""
    HIGHLY_EFFECTIVE = "highly_effective"
    EFFECTIVE = "effective"
    MODERATELY_EFFECTIVE = "moderately_effective"
    INEFFECTIVE = "ineffective"
    CRITICALLY_INEFFECTIVE = "critically_ineffective"


class AnalysisType(Enum):
    """Types of effectiveness analysis"""
    OVERALL = "overall"
    BY_DECISION_TYPE = "by_decision_type"
    BY_CONTEXT = "by_context"
    BY_TIME_PERIOD = "by_time_period"
    BY_USER = "by_user"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"


class EffectivenessFactors(BaseModel):
    """Factors influencing decision effectiveness"""
    decision_complexity: Optional[float] = None
    context_clarity: Optional[float] = None
    available_information: Optional[float] = None
    time_pressure: Optional[float] = None
    user_experience: Optional[float] = None
    system_confidence: Optional[float] = None
    resource_availability: Optional[float] = None
    
    
class EffectivenessInsight(BaseModel):
    """Individual effectiveness insight"""
    insight_type: str
    title: str
    description: str
    confidence: float
    impact_level: str  # high, medium, low
    actionable: bool = True
    recommendations: List[str] = Field(default_factory=list)
    supporting_data: Dict[str, Any] = Field(default_factory=dict)
    

class EffectivenessReport(BaseModel):
    """Comprehensive effectiveness analysis report"""
    analysis_id: UUID = Field(default_factory=uuid4)
    analysis_type: AnalysisType
    effectiveness_category: EffectivenessCategory
    overall_score: float
    confidence: float
    sample_size: int
    
    # Core metrics
    success_rate: float
    average_score: float
    consistency_score: float
    improvement_trend: float
    
    # Factor analysis
    key_success_factors: List[str] = Field(default_factory=list)
    key_failure_factors: List[str] = Field(default_factory=list)
    contextual_factors: Dict[str, float] = Field(default_factory=dict)
    
    # Insights and recommendations
    insights: List[EffectivenessInsight] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Comparative analysis
    benchmarks: Dict[str, float] = Field(default_factory=dict)
    peer_comparison: Dict[str, Any] = Field(default_factory=dict)
    
    # Time analysis
    temporal_patterns: Dict[str, Any] = Field(default_factory=dict)
    seasonal_effects: Dict[str, Any] = Field(default_factory=dict)
    
    # Risk assessment
    risk_factors: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EffectivenessAnalyzer:
    """Advanced effectiveness analysis service for decision outcomes"""
    
    def __init__(self, db: Session):
        """Initialize the effectiveness analyzer"""
        self.db = db
        
        # Configuration
        self.cache_ttl = 600  # 10 minutes
        self.min_sample_size = 10
        self.confidence_threshold = 0.8
        
        # Effectiveness thresholds
        self.thresholds = {
            EffectivenessCategory.HIGHLY_EFFECTIVE: 0.85,
            EffectivenessCategory.EFFECTIVE: 0.70,
            EffectivenessCategory.MODERATELY_EFFECTIVE: 0.55,
            EffectivenessCategory.INEFFECTIVE: 0.40,
            EffectivenessCategory.CRITICALLY_INEFFECTIVE: 0.0
        }
        
        # Factor importance weights
        self.factor_weights = {
            'success_rate': 0.25,
            'average_score': 0.20,
            'consistency': 0.15,
            'improvement_trend': 0.15,
            'user_satisfaction': 0.10,
            'impact_magnitude': 0.10,
            'time_efficiency': 0.05
        }
        
        # Context categories for analysis
        self.context_categories = {
            'code_generation': 'Code Generation',
            'debugging': 'Debugging & Troubleshooting',
            'architecture': 'Architecture & Design',
            'optimization': 'Performance Optimization',
            'testing': 'Testing & Validation',
            'documentation': 'Documentation',
            'deployment': 'Deployment & DevOps',
            'research': 'Research & Analysis'
        }
    
    async def analyze_overall_effectiveness(
        self,
        time_period_days: int = 30,
        include_benchmarks: bool = True
    ) -> EffectivenessReport:
        """Analyze overall system effectiveness
        
        Args:
            time_period_days: Number of days to analyze
            include_benchmarks: Whether to include benchmark comparisons
            
        Returns:
            EffectivenessReport with comprehensive analysis
        """
        try:
            # Get outcomes for analysis
            start_date = datetime.now(timezone.utc) - timedelta(days=time_period_days)
            outcomes = await self._get_outcomes_since(start_date)
            
            if len(outcomes) < self.min_sample_size:
                return await self._create_insufficient_data_report(AnalysisType.OVERALL, len(outcomes))
            
            # Calculate core metrics
            core_metrics = await self._calculate_core_metrics(outcomes)
            
            # Calculate overall effectiveness score
            overall_score = await self._calculate_weighted_effectiveness(core_metrics)
            
            # Categorize effectiveness
            effectiveness_category = self._categorize_effectiveness(overall_score)
            
            # Analyze factors
            success_factors, failure_factors = await self._analyze_success_failure_factors(outcomes)
            contextual_factors = await self._analyze_contextual_factors(outcomes)
            
            # Generate insights
            insights = await self._generate_effectiveness_insights(outcomes, core_metrics, overall_score)
            
            # Generate recommendations
            recommendations = await self._generate_effectiveness_recommendations(
                effectiveness_category, insights, core_metrics
            )
            
            # Temporal analysis
            temporal_patterns = await self._analyze_temporal_patterns(outcomes)
            
            # Risk assessment
            risk_factors, mitigation_strategies = await self._assess_risks(outcomes, core_metrics)
            
            # Benchmarks (if requested)
            benchmarks = {}
            if include_benchmarks:
                benchmarks = await self._calculate_benchmarks(core_metrics, time_period_days)
            
            # Calculate confidence
            confidence = self._calculate_analysis_confidence(len(outcomes), core_metrics)
            
            return EffectivenessReport(
                analysis_type=AnalysisType.OVERALL,
                effectiveness_category=effectiveness_category,
                overall_score=overall_score,
                confidence=confidence,
                sample_size=len(outcomes),
                success_rate=core_metrics['success_rate'],
                average_score=core_metrics['average_score'],
                consistency_score=core_metrics['consistency'],
                improvement_trend=core_metrics['improvement_trend'],
                key_success_factors=success_factors,
                key_failure_factors=failure_factors,
                contextual_factors=contextual_factors,
                insights=insights,
                recommendations=recommendations,
                benchmarks=benchmarks,
                temporal_patterns=temporal_patterns,
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies
            )
            
        except Exception as e:
            logger.error(f"Error in overall effectiveness analysis: {e}")
            return await self._create_error_report(AnalysisType.OVERALL, str(e))
    
    async def analyze_effectiveness_by_decision_type(
        self,
        decision_type: Optional[str] = None,
        time_period_days: int = 30
    ) -> List[EffectivenessReport]:
        """Analyze effectiveness broken down by decision type
        
        Args:
            decision_type: Specific decision type to analyze (None for all)
            time_period_days: Number of days to analyze
            
        Returns:
            List of EffectivenessReport for each decision type
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=time_period_days)
            outcomes = await self._get_outcomes_since(start_date)
            
            # Group outcomes by decision type
            grouped_outcomes = await self._group_outcomes_by_decision_type(outcomes)
            
            reports = []
            
            # Analyze each decision type
            for dt, dt_outcomes in grouped_outcomes.items():
                if decision_type and dt != decision_type:
                    continue
                    
                if len(dt_outcomes) < 5:  # Lower threshold for specific types
                    continue
                
                # Calculate metrics for this decision type
                core_metrics = await self._calculate_core_metrics(dt_outcomes)
                overall_score = await self._calculate_weighted_effectiveness(core_metrics)
                effectiveness_category = self._categorize_effectiveness(overall_score)
                
                # Analyze factors specific to this decision type
                success_factors, failure_factors = await self._analyze_success_failure_factors(dt_outcomes)
                contextual_factors = await self._analyze_contextual_factors(dt_outcomes)
                
                # Generate type-specific insights
                insights = await self._generate_decision_type_insights(dt, dt_outcomes, core_metrics)
                
                # Generate recommendations
                recommendations = await self._generate_decision_type_recommendations(
                    dt, effectiveness_category, insights
                )
                
                # Comparative analysis with other types
                peer_comparison = await self._compare_with_other_types(dt, core_metrics, grouped_outcomes)
                
                confidence = self._calculate_analysis_confidence(len(dt_outcomes), core_metrics)
                
                report = EffectivenessReport(
                    analysis_type=AnalysisType.BY_DECISION_TYPE,
                    effectiveness_category=effectiveness_category,
                    overall_score=overall_score,
                    confidence=confidence,
                    sample_size=len(dt_outcomes),
                    success_rate=core_metrics['success_rate'],
                    average_score=core_metrics['average_score'],
                    consistency_score=core_metrics['consistency'],
                    improvement_trend=core_metrics['improvement_trend'],
                    key_success_factors=success_factors,
                    key_failure_factors=failure_factors,
                    contextual_factors=contextual_factors,
                    insights=insights,
                    recommendations=recommendations,
                    peer_comparison=peer_comparison
                )
                
                # Add decision type to contextual factors
                report.contextual_factors['decision_type'] = dt
                
                reports.append(report)
            
            return reports
            
        except Exception as e:
            logger.error(f"Error in decision type effectiveness analysis: {e}")
            return [await self._create_error_report(AnalysisType.BY_DECISION_TYPE, str(e))]
    
    async def analyze_effectiveness_trends(
        self,
        time_period_days: int = 90,
        trend_granularity: str = "weekly"
    ) -> EffectivenessReport:
        """Analyze effectiveness trends over time
        
        Args:
            time_period_days: Number of days to analyze
            trend_granularity: Granularity of trend analysis (daily, weekly, monthly)
            
        Returns:
            EffectivenessReport with trend analysis
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=time_period_days)
            outcomes = await self._get_outcomes_since(start_date)
            
            if len(outcomes) < self.min_sample_size:
                return await self._create_insufficient_data_report(AnalysisType.BY_TIME_PERIOD, len(outcomes))
            
            # Group outcomes by time periods
            time_groups = await self._group_outcomes_by_time(outcomes, trend_granularity)
            
            # Calculate trends
            trend_analysis = await self._analyze_effectiveness_trends(time_groups)
            
            # Identify patterns
            temporal_patterns = await self._identify_temporal_patterns(time_groups)
            
            # Seasonal analysis
            seasonal_effects = await self._analyze_seasonal_effects(time_groups, trend_granularity)
            
            # Overall metrics for the period
            core_metrics = await self._calculate_core_metrics(outcomes)
            overall_score = await self._calculate_weighted_effectiveness(core_metrics)
            effectiveness_category = self._categorize_effectiveness(overall_score)
            
            # Generate trend insights
            insights = await self._generate_trend_insights(trend_analysis, temporal_patterns)
            
            # Generate trend recommendations
            recommendations = await self._generate_trend_recommendations(trend_analysis, seasonal_effects)
            
            confidence = self._calculate_analysis_confidence(len(outcomes), core_metrics)
            
            return EffectivenessReport(
                analysis_type=AnalysisType.BY_TIME_PERIOD,
                effectiveness_category=effectiveness_category,
                overall_score=overall_score,
                confidence=confidence,
                sample_size=len(outcomes),
                success_rate=core_metrics['success_rate'],
                average_score=core_metrics['average_score'],
                consistency_score=core_metrics['consistency'],
                improvement_trend=trend_analysis.get('overall_trend', 0.0),
                insights=insights,
                recommendations=recommendations,
                temporal_patterns=temporal_patterns,
                seasonal_effects=seasonal_effects
            )
            
        except Exception as e:
            logger.error(f"Error in trend effectiveness analysis: {e}")
            return await self._create_error_report(AnalysisType.BY_TIME_PERIOD, str(e))
    
    async def predict_effectiveness(
        self,
        context: Dict[str, Any],
        decision_type: str,
        factors: Optional[EffectivenessFactors] = None
    ) -> Dict[str, Any]:
        """Predict effectiveness for a given context and decision type
        
        Args:
            context: Context information for the decision
            decision_type: Type of decision being made
            factors: Optional effectiveness factors
            
        Returns:
            Prediction results with confidence and recommendations
        """
        try:
            # Get historical data for similar contexts
            similar_outcomes = await self._find_similar_contexts(context, decision_type)
            
            if len(similar_outcomes) < 5:
                return {
                    "predicted_effectiveness": 0.5,
                    "confidence": 0.0,
                    "message": "Insufficient historical data for prediction",
                    "recommendations": ["Proceed with caution", "Monitor outcome carefully"]
                }
            
            # Calculate base prediction from similar outcomes
            base_scores = [o.success_score for o in similar_outcomes]
            base_prediction = mean(base_scores)
            base_consistency = 1.0 - (stdev(base_scores) / max(mean(base_scores), 0.01))
            
            # Adjust prediction based on factors
            adjusted_prediction = base_prediction
            factor_adjustments = []
            
            if factors:
                adjustments = await self._calculate_factor_adjustments(factors, decision_type)
                for factor_name, adjustment in adjustments.items():
                    adjusted_prediction += adjustment
                    if abs(adjustment) > 0.05:
                        factor_adjustments.append(f"{factor_name}: {adjustment:+.2f}")
            
            # Ensure prediction is within bounds
            adjusted_prediction = max(0.0, min(1.0, adjusted_prediction))
            
            # Calculate confidence
            sample_confidence = min(1.0, len(similar_outcomes) / 20)
            consistency_confidence = base_consistency
            overall_confidence = (sample_confidence * 0.6 + consistency_confidence * 0.4)
            
            # Generate predictions for different outcome types
            success_probability = self._calculate_success_probability(adjusted_prediction)
            
            # Generate recommendations
            recommendations = await self._generate_prediction_recommendations(
                adjusted_prediction, overall_confidence, factor_adjustments
            )
            
            # Risk assessment
            risk_level = self._assess_prediction_risk(adjusted_prediction, overall_confidence)
            
            return {
                "predicted_effectiveness": round(adjusted_prediction, 3),
                "confidence": round(overall_confidence, 3),
                "success_probability": round(success_probability, 3),
                "risk_level": risk_level,
                "base_prediction": round(base_prediction, 3),
                "factor_adjustments": factor_adjustments,
                "historical_sample_size": len(similar_outcomes),
                "recommendations": recommendations,
                "prediction_details": {
                    "similar_outcomes_found": len(similar_outcomes),
                    "base_consistency": round(base_consistency, 3),
                    "sample_confidence": round(sample_confidence, 3),
                    "consistency_confidence": round(consistency_confidence, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in effectiveness prediction: {e}")
            return {
                "predicted_effectiveness": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "recommendations": ["Error in prediction", "Review system logs"]
            }
    
    async def compare_effectiveness(
        self,
        comparison_criteria: Dict[str, Any],
        time_period_days: int = 30
    ) -> EffectivenessReport:
        """Compare effectiveness across different criteria
        
        Args:
            comparison_criteria: Criteria for comparison (e.g., decision types, users, time periods)
            time_period_days: Number of days to analyze
            
        Returns:
            EffectivenessReport with comparative analysis
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=time_period_days)
            outcomes = await self._get_outcomes_since(start_date)
            
            # Perform comparisons based on criteria
            comparison_results = {}
            
            if 'decision_types' in comparison_criteria:
                comparison_results['by_decision_type'] = await self._compare_by_decision_type(
                    outcomes, comparison_criteria['decision_types']
                )
            
            if 'time_periods' in comparison_criteria:
                comparison_results['by_time_period'] = await self._compare_by_time_period(
                    outcomes, comparison_criteria['time_periods']
                )
            
            if 'contexts' in comparison_criteria:
                comparison_results['by_context'] = await self._compare_by_context(
                    outcomes, comparison_criteria['contexts']
                )
            
            # Calculate overall metrics
            core_metrics = await self._calculate_core_metrics(outcomes)
            overall_score = await self._calculate_weighted_effectiveness(core_metrics)
            effectiveness_category = self._categorize_effectiveness(overall_score)
            
            # Generate comparative insights
            insights = await self._generate_comparative_insights(comparison_results)
            
            # Generate comparative recommendations
            recommendations = await self._generate_comparative_recommendations(comparison_results)
            
            confidence = self._calculate_analysis_confidence(len(outcomes), core_metrics)
            
            return EffectivenessReport(
                analysis_type=AnalysisType.COMPARATIVE,
                effectiveness_category=effectiveness_category,
                overall_score=overall_score,
                confidence=confidence,
                sample_size=len(outcomes),
                success_rate=core_metrics['success_rate'],
                average_score=core_metrics['average_score'],
                consistency_score=core_metrics['consistency'],
                improvement_trend=core_metrics['improvement_trend'],
                insights=insights,
                recommendations=recommendations,
                peer_comparison=comparison_results
            )
            
        except Exception as e:
            logger.error(f"Error in comparative effectiveness analysis: {e}")
            return await self._create_error_report(AnalysisType.COMPARATIVE, str(e))
    
    # Private helper methods
    
    async def _get_outcomes_since(self, start_date: datetime) -> List[DecisionOutcome]:
        """Get all decision outcomes since the start date"""
        result = await self.db.execute(
            select(DecisionOutcome)
            .where(DecisionOutcome.measured_at >= start_date)
            .order_by(DecisionOutcome.measured_at)
        )
        return result.scalars().all()
    
    async def _calculate_core_metrics(self, outcomes: List[DecisionOutcome]) -> Dict[str, float]:
        """Calculate core effectiveness metrics"""
        if not outcomes:
            return {
                'success_rate': 0.0,
                'average_score': 0.0,
                'consistency': 0.0,
                'improvement_trend': 0.0
            }
        
        # Success rate
        success_count = len([o for o in outcomes if o.outcome_type == OutcomeType.SUCCESS.value])
        success_rate = success_count / len(outcomes)
        
        # Average score
        scores = [o.success_score for o in outcomes]
        average_score = mean(scores)
        
        # Consistency (inverse of coefficient of variation)
        if len(scores) > 1 and mean(scores) > 0:
            cv = stdev(scores) / mean(scores)
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 1.0
        
        # Improvement trend
        improvement_trend = await self._calculate_improvement_trend(outcomes)
        
        return {
            'success_rate': success_rate,
            'average_score': average_score,
            'consistency': consistency,
            'improvement_trend': improvement_trend
        }
    
    async def _calculate_weighted_effectiveness(self, core_metrics: Dict[str, float]) -> float:
        """Calculate weighted overall effectiveness score"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.factor_weights.items():
            if metric in core_metrics:
                weighted_score += core_metrics[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _categorize_effectiveness(self, score: float) -> EffectivenessCategory:
        """Categorize effectiveness based on score"""
        for category, threshold in self.thresholds.items():
            if score >= threshold:
                return category
        return EffectivenessCategory.CRITICALLY_INEFFECTIVE
    
    async def _analyze_success_failure_factors(
        self,
        outcomes: List[DecisionOutcome]
    ) -> Tuple[List[str], List[str]]:
        """Analyze factors contributing to success and failure"""
        success_factors = []
        failure_factors = []
        
        # Analyze successful outcomes
        successful_outcomes = [o for o in outcomes if o.outcome_type == OutcomeType.SUCCESS.value]
        success_factor_counts = Counter()
        
        for outcome in successful_outcomes:
            factors = outcome.outcome_data.get('success_factors', [])
            for factor in factors:
                success_factor_counts[factor] += 1
        
        # Get top success factors
        success_factors = [factor for factor, _ in success_factor_counts.most_common(5)]
        
        # Analyze failed outcomes
        failed_outcomes = [o for o in outcomes if o.outcome_type == OutcomeType.FAILURE.value]
        failure_factor_counts = Counter()
        
        for outcome in failed_outcomes:
            factors = outcome.outcome_data.get('failure_reasons', [])
            for factor in factors:
                failure_factor_counts[factor] += 1
        
        # Get top failure factors
        failure_factors = [factor for factor, _ in failure_factor_counts.most_common(5)]
        
        return success_factors, failure_factors
    
    async def _analyze_contextual_factors(self, outcomes: List[DecisionOutcome]) -> Dict[str, float]:
        """Analyze contextual factors and their impact"""
        contextual_factors = {}
        
        # Group by context and calculate average effectiveness
        context_groups = defaultdict(list)
        
        for outcome in outcomes:
            context_data = outcome.outcome_data.get('context', {})
            for key, value in context_data.items():
                if isinstance(value, str):
                    context_key = f"{key}:{value}"
                    context_groups[context_key].append(outcome.success_score)
        
        # Calculate average effectiveness for each context
        for context_key, scores in context_groups.items():
            if len(scores) >= 3:  # Minimum threshold
                contextual_factors[context_key] = mean(scores)
        
        return contextual_factors
    
    async def _generate_effectiveness_insights(
        self,
        outcomes: List[DecisionOutcome],
        core_metrics: Dict[str, float],
        overall_score: float
    ) -> List[EffectivenessInsight]:
        """Generate effectiveness insights"""
        insights = []
        
        # Success rate insight
        if core_metrics['success_rate'] < 0.6:
            insights.append(EffectivenessInsight(
                insight_type="performance",
                title="Low Success Rate Detected",
                description=f"Success rate of {core_metrics['success_rate']*100:.1f}% is below optimal threshold",
                confidence=0.9,
                impact_level="high",
                recommendations=[
                    "Review decision-making processes",
                    "Identify common failure patterns",
                    "Implement additional validation steps"
                ]
            ))
        elif core_metrics['success_rate'] > 0.85:
            insights.append(EffectivenessInsight(
                insight_type="performance",
                title="Excellent Success Rate",
                description=f"Success rate of {core_metrics['success_rate']*100:.1f}% indicates high effectiveness",
                confidence=0.9,
                impact_level="medium",
                recommendations=[
                    "Maintain current standards",
                    "Document best practices",
                    "Share successful patterns"
                ]
            ))
        
        # Consistency insight
        if core_metrics['consistency'] < 0.7:
            insights.append(EffectivenessInsight(
                insight_type="consistency",
                title="Inconsistent Performance",
                description="Decision outcomes show high variability",
                confidence=0.8,
                impact_level="medium",
                recommendations=[
                    "Standardize decision processes",
                    "Implement consistency checks",
                    "Provide additional training"
                ]
            ))
        
        # Improvement trend insight
        if core_metrics['improvement_trend'] < -0.1:
            insights.append(EffectivenessInsight(
                insight_type="trend",
                title="Declining Performance Trend",
                description="Effectiveness has been declining over time",
                confidence=0.8,
                impact_level="high",
                recommendations=[
                    "Investigate root causes",
                    "Implement improvement measures",
                    "Monitor progress closely"
                ]
            ))
        elif core_metrics['improvement_trend'] > 0.1:
            insights.append(EffectivenessInsight(
                insight_type="trend",
                title="Positive Improvement Trend",
                description="Effectiveness has been improving over time",
                confidence=0.8,
                impact_level="medium",
                recommendations=[
                    "Continue current improvements",
                    "Identify successful changes",
                    "Scale successful practices"
                ]
            ))
        
        return insights
    
    async def _generate_effectiveness_recommendations(
        self,
        category: EffectivenessCategory,
        insights: List[EffectivenessInsight],
        core_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate effectiveness recommendations"""
        recommendations = []
        
        # Category-based recommendations
        if category == EffectivenessCategory.CRITICALLY_INEFFECTIVE:
            recommendations.extend([
                "Immediate intervention required",
                "Review all decision processes",
                "Implement comprehensive monitoring",
                "Consider system redesign"
            ])
        elif category == EffectivenessCategory.INEFFECTIVE:
            recommendations.extend([
                "Identify and address root causes",
                "Improve validation processes",
                "Enhance training and support"
            ])
        elif category == EffectivenessCategory.MODERATELY_EFFECTIVE:
            recommendations.extend([
                "Focus on consistency improvements",
                "Identify optimization opportunities",
                "Standardize successful practices"
            ])
        elif category == EffectivenessCategory.EFFECTIVE:
            recommendations.extend([
                "Maintain current performance",
                "Optimize edge cases",
                "Share best practices"
            ])
        else:  # HIGHLY_EFFECTIVE
            recommendations.extend([
                "Document and preserve practices",
                "Mentor other teams",
                "Explore innovation opportunities"
            ])
        
        # Add insight-specific recommendations
        for insight in insights:
            recommendations.extend(insight.recommendations)
        
        # Remove duplicates and limit
        return list(dict.fromkeys(recommendations))[:10]
    
    async def _calculate_improvement_trend(self, outcomes: List[DecisionOutcome]) -> float:
        """Calculate improvement trend over time"""
        if len(outcomes) < 4:
            return 0.0
        
        # Sort by time
        sorted_outcomes = sorted(outcomes, key=lambda o: o.measured_at)
        
        # Split into halves
        midpoint = len(sorted_outcomes) // 2
        first_half = sorted_outcomes[:midpoint]
        second_half = sorted_outcomes[midpoint:]
        
        # Calculate average scores
        first_avg = mean([o.success_score for o in first_half])
        second_avg = mean([o.success_score for o in second_half])
        
        # Calculate improvement as percentage
        improvement = (second_avg - first_avg) / max(first_avg, 0.01)
        
        return improvement
    
    def _calculate_analysis_confidence(self, sample_size: int, core_metrics: Dict[str, float]) -> float:
        """Calculate confidence in the analysis"""
        # Base confidence on sample size
        size_confidence = min(1.0, sample_size / 50)
        
        # Adjust for consistency (more consistent results = higher confidence)
        consistency_confidence = core_metrics.get('consistency', 0.5)
        
        # Combined confidence
        overall_confidence = (size_confidence * 0.7 + consistency_confidence * 0.3)
        
        return overall_confidence
    
    async def _create_insufficient_data_report(
        self,
        analysis_type: AnalysisType,
        sample_size: int
    ) -> EffectivenessReport:
        """Create report for insufficient data cases"""
        return EffectivenessReport(
            analysis_type=analysis_type,
            effectiveness_category=EffectivenessCategory.MODERATELY_EFFECTIVE,
            overall_score=0.5,
            confidence=0.0,
            sample_size=sample_size,
            success_rate=0.0,
            average_score=0.0,
            consistency_score=0.0,
            improvement_trend=0.0,
            insights=[
                EffectivenessInsight(
                    insight_type="data",
                    title="Insufficient Data",
                    description=f"Only {sample_size} outcomes available, need at least {self.min_sample_size}",
                    confidence=1.0,
                    impact_level="high",
                    actionable=False,
                    recommendations=["Collect more decision outcome data"]
                )
            ],
            recommendations=["Continue collecting data", "Monitor system usage"]
        )
    
    async def _create_error_report(self, analysis_type: AnalysisType, error_message: str) -> EffectivenessReport:
        """Create report for error cases"""
        return EffectivenessReport(
            analysis_type=analysis_type,
            effectiveness_category=EffectivenessCategory.MODERATELY_EFFECTIVE,
            overall_score=0.0,
            confidence=0.0,
            sample_size=0,
            success_rate=0.0,
            average_score=0.0,
            consistency_score=0.0,
            improvement_trend=0.0,
            insights=[
                EffectivenessInsight(
                    insight_type="error",
                    title="Analysis Error",
                    description=f"Error during analysis: {error_message}",
                    confidence=1.0,
                    impact_level="high",
                    actionable=False,
                    recommendations=["Check system logs", "Contact support"]
                )
            ],
            recommendations=["Review system configuration", "Check data integrity"]
        )
    
    async def _group_outcomes_by_decision_type(
        self,
        outcomes: List[DecisionOutcome]
    ) -> Dict[str, List[DecisionOutcome]]:
        """Group outcomes by decision type"""
        grouped = defaultdict(list)
        
        for outcome in outcomes:
            decision_type = outcome.outcome_data.get('decision_type', 'unknown')
            grouped[decision_type].append(outcome)
        
        return dict(grouped)
    
    async def _generate_decision_type_insights(
        self,
        decision_type: str,
        outcomes: List[DecisionOutcome],
        core_metrics: Dict[str, float]
    ) -> List[EffectivenessInsight]:
        """Generate insights specific to a decision type"""
        insights = []
        
        # Decision type specific analysis
        type_name = self.context_categories.get(decision_type, decision_type)
        
        if core_metrics['success_rate'] < 0.5:
            insights.append(EffectivenessInsight(
                insight_type="decision_type",
                title=f"Low Effectiveness in {type_name}",
                description=f"{type_name} decisions show below-average success rate",
                confidence=0.8,
                impact_level="high",
                recommendations=[
                    f"Review {type_name} decision processes",
                    f"Identify common {type_name} failure patterns",
                    f"Enhance {type_name} validation methods"
                ]
            ))
        
        return insights
    
    async def _generate_decision_type_recommendations(
        self,
        decision_type: str,
        category: EffectivenessCategory,
        insights: List[EffectivenessInsight]
    ) -> List[str]:
        """Generate recommendations specific to decision type"""
        recommendations = []
        type_name = self.context_categories.get(decision_type, decision_type)
        
        # Type-specific recommendations
        if decision_type == 'code_generation':
            recommendations.extend([
                "Implement code quality checks",
                "Use established patterns and templates",
                "Add automated testing validation"
            ])
        elif decision_type == 'debugging':
            recommendations.extend([
                "Enhance diagnostic capabilities",
                "Improve error detection patterns",
                "Add comprehensive logging"
            ])
        elif decision_type == 'architecture':
            recommendations.extend([
                "Follow established design patterns",
                "Conduct design reviews",
                "Document architectural decisions"
            ])
        
        # Add insight recommendations
        for insight in insights:
            recommendations.extend(insight.recommendations)
        
        return list(dict.fromkeys(recommendations))[:8]
    
    async def _compare_with_other_types(
        self,
        decision_type: str,
        core_metrics: Dict[str, float],
        all_grouped: Dict[str, List[DecisionOutcome]]
    ) -> Dict[str, Any]:
        """Compare decision type with others"""
        comparison = {
            "decision_type": decision_type,
            "rank": 0,
            "total_types": len(all_grouped),
            "better_than": [],
            "worse_than": []
        }
        
        # Calculate ranking
        type_scores = {}
        for dt, outcomes in all_grouped.items():
            if len(outcomes) >= 3:
                scores = [o.success_score for o in outcomes]
                type_scores[dt] = mean(scores)
        
        if decision_type in type_scores:
            current_score = type_scores[decision_type]
            sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (dt, score) in enumerate(sorted_types, 1):
                if dt == decision_type:
                    comparison["rank"] = rank
                    break
            
            # Find better and worse performing types
            for dt, score in type_scores.items():
                if dt != decision_type:
                    if score > current_score:
                        comparison["better_than"].append(dt)
                    elif score < current_score:
                        comparison["worse_than"].append(dt)
        
        return comparison
    
    async def _analyze_temporal_patterns(self, outcomes: List[DecisionOutcome]) -> Dict[str, Any]:
        """Analyze temporal patterns in effectiveness"""
        # Basic temporal analysis
        return {
            "note": "Detailed temporal pattern analysis requires time series implementation",
            "sample_period": f"{len(outcomes)} outcomes analyzed"
        }
    
    async def _assess_risks(
        self,
        outcomes: List[DecisionOutcome],
        core_metrics: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """Assess risks and mitigation strategies"""
        risks = []
        mitigations = []
        
        if core_metrics['success_rate'] < 0.6:
            risks.append("High failure rate risk")
            mitigations.append("Implement additional validation steps")
        
        if core_metrics['consistency'] < 0.6:
            risks.append("Inconsistent performance risk")
            mitigations.append("Standardize processes and training")
        
        if core_metrics['improvement_trend'] < -0.1:
            risks.append("Declining performance risk")
            mitigations.append("Immediate performance review and intervention")
        
        return risks, mitigations
    
    async def _calculate_benchmarks(
        self,
        core_metrics: Dict[str, float],
        time_period_days: int
    ) -> Dict[str, float]:
        """Calculate performance benchmarks"""
        # Industry/internal benchmarks
        return {
            "industry_success_rate": 0.75,
            "target_success_rate": 0.80,
            "minimum_acceptable": 0.60,
            "excellence_threshold": 0.90
        }
    
    async def _find_similar_contexts(
        self,
        context: Dict[str, Any],
        decision_type: str
    ) -> List[DecisionOutcome]:
        """Find outcomes with similar contexts for prediction"""
        # This would implement similarity matching
        # For now, return outcomes of the same decision type
        result = await self.db.execute(
            select(DecisionOutcome)
            .where(DecisionOutcome.outcome_data['decision_type'].astext == decision_type)
            .order_by(DecisionOutcome.measured_at.desc())
            .limit(50)
        )
        return result.scalars().all()
    
    async def _calculate_factor_adjustments(
        self,
        factors: EffectivenessFactors,
        decision_type: str
    ) -> Dict[str, float]:
        """Calculate adjustments based on effectiveness factors"""
        adjustments = {}
        
        # Simple factor adjustments (would be more sophisticated in practice)
        if factors.decision_complexity is not None:
            # Higher complexity = lower effectiveness
            adjustments['complexity'] = -0.1 * factors.decision_complexity
        
        if factors.context_clarity is not None:
            # Higher clarity = higher effectiveness
            adjustments['clarity'] = 0.1 * factors.context_clarity
        
        if factors.time_pressure is not None:
            # Higher pressure = lower effectiveness
            adjustments['time_pressure'] = -0.05 * factors.time_pressure
        
        return adjustments
    
    def _calculate_success_probability(self, effectiveness_score: float) -> float:
        """Calculate probability of success based on effectiveness score"""
        # Simple mapping - could be more sophisticated
        return min(1.0, effectiveness_score * 1.2)
    
    async def _generate_prediction_recommendations(
        self,
        predicted_score: float,
        confidence: float,
        factor_adjustments: List[str]
    ) -> List[str]:
        """Generate recommendations for predictions"""
        recommendations = []
        
        if predicted_score < 0.5:
            recommendations.append("High risk decision - consider alternatives")
        elif predicted_score < 0.7:
            recommendations.append("Moderate risk - proceed with caution")
        else:
            recommendations.append("Good likelihood of success")
        
        if confidence < 0.5:
            recommendations.append("Low confidence - gather more data")
        
        if factor_adjustments:
            recommendations.append("Consider factor impacts: " + ", ".join(factor_adjustments))
        
        return recommendations
    
    def _assess_prediction_risk(self, predicted_score: float, confidence: float) -> str:
        """Assess risk level for predictions"""
        if predicted_score < 0.4 or confidence < 0.3:
            return "high"
        elif predicted_score < 0.6 or confidence < 0.6:
            return "medium"
        else:
            return "low"
    
    # Placeholder methods for comparative analysis
    async def _group_outcomes_by_time(self, outcomes: List[DecisionOutcome], granularity: str) -> Dict[str, List[DecisionOutcome]]:
        """Group outcomes by time periods"""
        return {"placeholder": outcomes}
    
    async def _analyze_effectiveness_trends(self, time_groups: Dict[str, List[DecisionOutcome]]) -> Dict[str, Any]:
        """Analyze effectiveness trends over time groups"""
        return {"overall_trend": 0.0}
    
    async def _identify_temporal_patterns(self, time_groups: Dict[str, List[DecisionOutcome]]) -> Dict[str, Any]:
        """Identify temporal patterns"""
        return {"patterns_found": []}
    
    async def _analyze_seasonal_effects(self, time_groups: Dict[str, List[DecisionOutcome]], granularity: str) -> Dict[str, Any]:
        """Analyze seasonal effects"""
        return {"seasonal_effects": []}
    
    async def _generate_trend_insights(self, trend_analysis: Dict[str, Any], temporal_patterns: Dict[str, Any]) -> List[EffectivenessInsight]:
        """Generate insights from trend analysis"""
        return []
    
    async def _generate_trend_recommendations(self, trend_analysis: Dict[str, Any], seasonal_effects: Dict[str, Any]) -> List[str]:
        """Generate recommendations from trend analysis"""
        return ["Monitor trends closely"]
    
    async def _compare_by_decision_type(self, outcomes: List[DecisionOutcome], decision_types: List[str]) -> Dict[str, Any]:
        """Compare effectiveness by decision type"""
        return {"comparison": "placeholder"}
    
    async def _compare_by_time_period(self, outcomes: List[DecisionOutcome], time_periods: List[str]) -> Dict[str, Any]:
        """Compare effectiveness by time period"""
        return {"comparison": "placeholder"}
    
    async def _compare_by_context(self, outcomes: List[DecisionOutcome], contexts: List[str]) -> Dict[str, Any]:
        """Compare effectiveness by context"""
        return {"comparison": "placeholder"}
    
    async def _generate_comparative_insights(self, comparison_results: Dict[str, Any]) -> List[EffectivenessInsight]:
        """Generate insights from comparative analysis"""
        return []
    
    async def _generate_comparative_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations from comparative analysis"""
        return ["Review comparative results"]