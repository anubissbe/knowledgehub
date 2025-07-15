"""Decision Outcome Tracker Service

This service provides comprehensive tracking and analysis of decision outcomes,
building upon the existing SuccessTracker to provide more detailed analytics
and decision effectiveness measurement.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from statistics import mean, median, stdev
from collections import defaultdict, Counter
import asyncio

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func, or_, desc, asc
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, Field

from ..models.decision_outcome import DecisionOutcome, OutcomeType
from ..models.learning_pattern import LearningPattern, PatternType
from ..models.user_feedback import UserFeedback, FeedbackType
from ...memory_system.models.memory import Memory, MemoryType
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


class DecisionContext(BaseModel):
    """Context information for a decision"""
    decision_id: UUID
    decision_type: str
    context_data: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OutcomeMetrics(BaseModel):
    """Comprehensive metrics for decision outcomes"""
    total_decisions: int = 0
    successful_decisions: int = 0
    failed_decisions: int = 0
    partial_decisions: int = 0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    average_success_score: float = 0.0
    median_success_score: float = 0.0
    score_distribution: Dict[str, int] = Field(default_factory=dict)
    time_to_outcome: Dict[str, float] = Field(default_factory=dict)
    impact_analysis: Dict[str, Any] = Field(default_factory=dict)


class DecisionTrend(BaseModel):
    """Trend analysis for decision outcomes"""
    time_period: str
    trend_direction: str  # improving, declining, stable
    trend_strength: float  # 0.0 to 1.0
    key_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class DecisionOutcomeTracker:
    """Advanced decision outcome tracking and analysis service"""
    
    def __init__(self, db: Session):
        """Initialize the decision outcome tracker"""
        self.db = db
        
        # Configuration
        self.success_threshold = 0.7
        self.failure_threshold = 0.3
        self.min_data_points = 5
        self.trend_analysis_window = timedelta(days=30)
        self.cache_ttl = 300  # 5 minutes
        
        # Decision type mappings
        self.decision_types = {
            'code_generation': 'Code Generation Decision',
            'architecture': 'Architecture Decision',
            'debugging': 'Debugging Decision',
            'optimization': 'Optimization Decision',
            'tool_selection': 'Tool Selection Decision',
            'approach_selection': 'Approach Selection Decision',
            'error_handling': 'Error Handling Decision',
            'refactoring': 'Refactoring Decision'
        }
        
        # Outcome impact factors
        self.impact_factors = {
            'user_satisfaction': 0.3,
            'task_completion': 0.25,
            'performance_impact': 0.2,
            'maintainability': 0.1,
            'error_reduction': 0.1,
            'time_efficiency': 0.05
        }
    
    async def track_decision_outcome(
        self,
        decision_context: DecisionContext,
        outcome_data: Dict[str, Any],
        impact_metrics: Optional[Dict[str, Any]] = None
    ) -> DecisionOutcome:
        """Track the outcome of a specific decision with detailed context
        
        Args:
            decision_context: Context information about the decision
            outcome_data: Detailed outcome information
            impact_metrics: Optional impact metrics
            
        Returns:
            DecisionOutcome instance
        """
        try:
            # Validate decision exists
            decision = await self.db.get(Memory, decision_context.decision_id)
            if not decision:
                raise ValueError(f"Decision {decision_context.decision_id} not found")
            
            # Calculate comprehensive success score
            success_score = await self._calculate_comprehensive_success_score(
                outcome_data,
                impact_metrics or {}
            )
            
            # Determine outcome type
            outcome_type = self._determine_outcome_type(success_score)
            
            # Analyze decision factors
            factor_analysis = await self._analyze_decision_factors(
                decision_context,
                outcome_data,
                impact_metrics or {}
            )
            
            # Create enhanced outcome record
            outcome = DecisionOutcome(
                id=uuid4(),
                decision_id=decision_context.decision_id,
                outcome_type=outcome_type,
                success_score=success_score,
                outcome_data={
                    **outcome_data,
                    'decision_context': decision_context.dict(),
                    'factor_analysis': factor_analysis,
                    'tracked_at': datetime.now(timezone.utc).isoformat()
                },
                impact_data=impact_metrics or {},
                measured_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc)
            )
            
            # Store outcome
            self.db.add(outcome)
            await self.db.commit()
            
            # Update cached metrics
            await self._update_cached_metrics(decision_context.decision_type)
            
            # Analyze for learning opportunities
            await self._analyze_for_learning_opportunities(outcome)
            
            logger.info(f"Tracked decision outcome: {outcome.id} (score: {success_score:.2f})")
            
            return outcome
            
        except Exception as e:
            logger.error(f"Error tracking decision outcome: {e}")
            await self.db.rollback()
            raise
    
    async def get_decision_effectiveness(
        self,
        decision_type: Optional[str] = None,
        time_period: Optional[timedelta] = None,
        user_id: Optional[UUID] = None
    ) -> OutcomeMetrics:
        """Get comprehensive effectiveness metrics for decisions
        
        Args:
            decision_type: Filter by decision type
            time_period: Time period for analysis
            user_id: Filter by user ID
            
        Returns:
            OutcomeMetrics with comprehensive analysis
        """
        try:
            # Build query filters
            filters = []
            
            if time_period:
                cutoff_date = datetime.now(timezone.utc) - time_period
                filters.append(DecisionOutcome.created_at >= cutoff_date)
            
            if decision_type:
                filters.append(
                    DecisionOutcome.outcome_data.op('->>')('decision_context').op('->>')('decision_type') == decision_type
                )
            
            if user_id:
                filters.append(
                    DecisionOutcome.outcome_data.op('->>')('decision_context').op('->>')('user_id') == str(user_id)
                )
            
            # Get outcomes
            query = select(DecisionOutcome)
            if filters:
                query = query.where(and_(*filters))
            
            result = await self.db.execute(query)
            outcomes = result.scalars().all()
            
            if not outcomes:
                return OutcomeMetrics()
            
            # Calculate metrics
            metrics = OutcomeMetrics(
                total_decisions=len(outcomes),
                successful_decisions=sum(1 for o in outcomes if o.is_successful()),
                failed_decisions=sum(1 for o in outcomes if o.is_failure()),
                partial_decisions=sum(1 for o in outcomes if not o.is_successful() and not o.is_failure())
            )
            
            # Calculate rates
            if metrics.total_decisions > 0:
                metrics.success_rate = metrics.successful_decisions / metrics.total_decisions
                metrics.failure_rate = metrics.failed_decisions / metrics.total_decisions
            
            # Score statistics
            scores = [o.success_score for o in outcomes]
            metrics.average_success_score = mean(scores)
            metrics.median_success_score = median(scores)
            
            # Score distribution
            score_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
            distribution = defaultdict(int)
            for score in scores:
                if score <= 0.2:
                    distribution['0.0-0.2'] += 1
                elif score <= 0.4:
                    distribution['0.2-0.4'] += 1
                elif score <= 0.6:
                    distribution['0.4-0.6'] += 1
                elif score <= 0.8:
                    distribution['0.6-0.8'] += 1
                else:
                    distribution['0.8-1.0'] += 1
            
            metrics.score_distribution = dict(distribution)
            
            # Time to outcome analysis
            metrics.time_to_outcome = await self._analyze_time_to_outcome(outcomes)
            
            # Impact analysis
            metrics.impact_analysis = await self._analyze_impact_metrics(outcomes)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting decision effectiveness: {e}")
            return OutcomeMetrics()
    
    async def get_decision_trends(
        self,
        decision_type: Optional[str] = None,
        time_period: Optional[timedelta] = None
    ) -> List[DecisionTrend]:
        """Analyze trends in decision outcomes
        
        Args:
            decision_type: Filter by decision type
            time_period: Time period for analysis
            
        Returns:
            List of DecisionTrend objects
        """
        try:
            period = time_period or self.trend_analysis_window
            
            # Get outcomes grouped by time periods
            trends = []
            
            # Weekly trends
            weekly_trend = await self._analyze_trend_for_period(
                decision_type,
                period,
                'weekly'
            )
            trends.append(weekly_trend)
            
            # Monthly trends
            monthly_trend = await self._analyze_trend_for_period(
                decision_type,
                period,
                'monthly'
            )
            trends.append(monthly_trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing decision trends: {e}")
            return []
    
    async def get_top_performing_decisions(
        self,
        limit: int = 10,
        decision_type: Optional[str] = None,
        time_period: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Get top performing decisions based on success scores
        
        Args:
            limit: Number of decisions to return
            decision_type: Filter by decision type
            time_period: Time period for analysis
            
        Returns:
            List of decision performance data
        """
        try:
            # Build query
            query = select(DecisionOutcome).order_by(desc(DecisionOutcome.success_score))
            
            # Add filters
            filters = []
            if decision_type:
                filters.append(
                    DecisionOutcome.outcome_data.op('->>')('decision_context').op('->>')('decision_type') == decision_type
                )
            
            if time_period:
                cutoff_date = datetime.now(timezone.utc) - time_period
                filters.append(DecisionOutcome.created_at >= cutoff_date)
            
            if filters:
                query = query.where(and_(*filters))
            
            query = query.limit(limit)
            
            result = await self.db.execute(query)
            outcomes = result.scalars().all()
            
            # Format results
            top_decisions = []
            for outcome in outcomes:
                decision_data = {
                    'outcome_id': str(outcome.id),
                    'decision_id': str(outcome.decision_id),
                    'success_score': outcome.success_score,
                    'outcome_type': outcome.outcome_type.value,
                    'decision_type': outcome.outcome_data.get('decision_context', {}).get('decision_type', 'unknown'),
                    'measured_at': outcome.measured_at.isoformat(),
                    'key_factors': outcome.outcome_data.get('factor_analysis', {}).get('positive_factors', []),
                    'impact_metrics': outcome.get_impact_metrics()
                }
                top_decisions.append(decision_data)
            
            return top_decisions
            
        except Exception as e:
            logger.error(f"Error getting top performing decisions: {e}")
            return []
    
    async def get_decision_failure_analysis(
        self,
        limit: int = 10,
        decision_type: Optional[str] = None,
        time_period: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Analyze failed decisions to identify patterns
        
        Args:
            limit: Number of failures to analyze
            decision_type: Filter by decision type
            time_period: Time period for analysis
            
        Returns:
            List of failure analysis data
        """
        try:
            # Build query for failures
            query = select(DecisionOutcome).where(
                DecisionOutcome.success_score <= self.failure_threshold
            ).order_by(asc(DecisionOutcome.success_score))
            
            # Add filters
            filters = []
            if decision_type:
                filters.append(
                    DecisionOutcome.outcome_data.op('->>')('decision_context').op('->>')('decision_type') == decision_type
                )
            
            if time_period:
                cutoff_date = datetime.now(timezone.utc) - time_period
                filters.append(DecisionOutcome.created_at >= cutoff_date)
            
            if filters:
                query = query.where(and_(*filters))
            
            query = query.limit(limit)
            
            result = await self.db.execute(query)
            failures = result.scalars().all()
            
            # Analyze failure patterns
            failure_analysis = []
            for failure in failures:
                analysis_data = {
                    'outcome_id': str(failure.id),
                    'decision_id': str(failure.decision_id),
                    'success_score': failure.success_score,
                    'decision_type': failure.outcome_data.get('decision_context', {}).get('decision_type', 'unknown'),
                    'failure_factors': failure.outcome_data.get('factor_analysis', {}).get('negative_factors', []),
                    'root_causes': await self._identify_root_causes(failure),
                    'improvement_suggestions': await self._generate_improvement_suggestions(failure),
                    'measured_at': failure.measured_at.isoformat()
                }
                failure_analysis.append(analysis_data)
            
            return failure_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing decision failures: {e}")
            return []
    
    async def get_decision_recommendations(
        self,
        decision_type: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get recommendations for improving decision outcomes
        
        Args:
            decision_type: Type of decision
            context: Decision context
            
        Returns:
            List of recommendations
        """
        try:
            # Analyze historical data for this decision type
            historical_metrics = await self.get_decision_effectiveness(
                decision_type=decision_type,
                time_period=timedelta(days=90)
            )
            
            # Get recent failures for pattern analysis
            recent_failures = await self.get_decision_failure_analysis(
                decision_type=decision_type,
                time_period=timedelta(days=30)
            )
            
            # Generate recommendations
            recommendations = []
            
            # Success rate recommendations
            if historical_metrics.success_rate < 0.7:
                recommendations.append({
                    'type': 'success_rate_improvement',
                    'priority': 'high',
                    'description': f'Success rate for {decision_type} decisions is {historical_metrics.success_rate:.2f}. Consider reviewing decision criteria.',
                    'suggested_actions': [
                        'Review decision-making process',
                        'Analyze successful decision patterns',
                        'Implement additional validation steps'
                    ]
                })
            
            # Failure pattern recommendations
            if recent_failures:
                common_factors = self._identify_common_failure_factors(recent_failures)
                if common_factors:
                    recommendations.append({
                        'type': 'failure_pattern_mitigation',
                        'priority': 'medium',
                        'description': f'Common failure factors identified: {", ".join(common_factors)}',
                        'suggested_actions': [
                            f'Address {factor}' for factor in common_factors[:3]
                        ]
                    })
            
            # Context-specific recommendations
            context_recommendations = await self._generate_context_specific_recommendations(
                decision_type,
                context,
                historical_metrics
            )
            recommendations.extend(context_recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating decision recommendations: {e}")
            return []
    
    # Private helper methods
    
    async def _calculate_comprehensive_success_score(
        self,
        outcome_data: Dict[str, Any],
        impact_metrics: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive success score using multiple factors"""
        try:
            # Base score from outcome data
            base_score = outcome_data.get('success_score', 0.5)
            
            # Adjust based on impact metrics
            score_adjustments = 0.0
            
            # User satisfaction impact
            if 'user_satisfaction' in impact_metrics:
                satisfaction = impact_metrics['user_satisfaction']
                score_adjustments += (satisfaction - 0.5) * self.impact_factors['user_satisfaction']
            
            # Task completion impact
            if 'task_completion' in impact_metrics:
                completion = impact_metrics['task_completion']
                score_adjustments += (completion - 0.5) * self.impact_factors['task_completion']
            
            # Performance impact
            if 'performance_impact' in impact_metrics:
                performance = impact_metrics['performance_impact']
                score_adjustments += (performance - 0.5) * self.impact_factors['performance_impact']
            
            # Error reduction impact
            if 'error_reduction' in impact_metrics:
                error_reduction = impact_metrics['error_reduction']
                score_adjustments += error_reduction * self.impact_factors['error_reduction']
            
            # Time efficiency impact
            if 'time_efficiency' in impact_metrics:
                efficiency = impact_metrics['time_efficiency']
                score_adjustments += (efficiency - 0.5) * self.impact_factors['time_efficiency']
            
            # Calculate final score
            final_score = base_score + score_adjustments
            
            # Ensure score is within bounds
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive success score: {e}")
            return 0.5
    
    def _determine_outcome_type(self, success_score: float) -> OutcomeType:
        """Determine outcome type based on success score"""
        if success_score >= self.success_threshold:
            return OutcomeType.SUCCESS
        elif success_score <= self.failure_threshold:
            return OutcomeType.FAILURE
        else:
            return OutcomeType.PARTIAL
    
    async def _analyze_decision_factors(
        self,
        decision_context: DecisionContext,
        outcome_data: Dict[str, Any],
        impact_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze factors that contributed to the decision outcome"""
        try:
            analysis = {
                'positive_factors': [],
                'negative_factors': [],
                'neutral_factors': [],
                'context_influence': {},
                'recommendation_factors': []
            }
            
            # Analyze success factors
            if outcome_data.get('success_score', 0) >= self.success_threshold:
                analysis['positive_factors'].extend([
                    'High success score achieved',
                    'Positive user feedback received' if outcome_data.get('user_feedback_positive') else None,
                    'Task completed successfully' if impact_metrics.get('task_completion', 0) >= 0.8 else None,
                    'Good performance metrics' if impact_metrics.get('performance_impact', 0) >= 0.7 else None
                ])
            
            # Analyze failure factors
            if outcome_data.get('success_score', 0) <= self.failure_threshold:
                analysis['negative_factors'].extend([
                    'Low success score',
                    'User dissatisfaction reported' if outcome_data.get('user_feedback_negative') else None,
                    'Task not completed' if impact_metrics.get('task_completion', 0) < 0.5 else None,
                    'Poor performance impact' if impact_metrics.get('performance_impact', 0) < 0.3 else None
                ])
            
            # Context influence analysis
            context_data = decision_context.context_data
            if context_data:
                analysis['context_influence'] = {
                    'complexity_level': context_data.get('complexity', 'medium'),
                    'time_pressure': context_data.get('time_pressure', 'normal'),
                    'resource_availability': context_data.get('resources', 'adequate'),
                    'user_experience_level': context_data.get('user_level', 'intermediate')
                }
            
            # Remove None values
            analysis['positive_factors'] = [f for f in analysis['positive_factors'] if f]
            analysis['negative_factors'] = [f for f in analysis['negative_factors'] if f]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing decision factors: {e}")
            return {'positive_factors': [], 'negative_factors': [], 'neutral_factors': []}
    
    async def _analyze_for_learning_opportunities(
        self,
        outcome: DecisionOutcome
    ):
        """Analyze outcome for learning opportunities"""
        try:
            # If it's a successful outcome, extract success patterns
            if outcome.is_successful():
                await self._extract_success_patterns(outcome)
            
            # If it's a failure, identify improvement opportunities
            elif outcome.is_failure():
                await self._identify_improvement_opportunities(outcome)
            
            # For partial outcomes, analyze what could be improved
            else:
                await self._analyze_partial_outcome(outcome)
                
        except Exception as e:
            logger.error(f"Error analyzing learning opportunities: {e}")
    
    async def _extract_success_patterns(self, outcome: DecisionOutcome):
        """Extract patterns from successful outcomes"""
        try:
            # Create a success pattern
            pattern_data = {
                'decision_type': outcome.outcome_data.get('decision_context', {}).get('decision_type'),
                'success_factors': outcome.outcome_data.get('factor_analysis', {}).get('positive_factors', []),
                'success_score': outcome.success_score,
                'context': outcome.outcome_data.get('decision_context', {}),
                'impact_metrics': outcome.get_impact_metrics()
            }
            
            # Store as learning pattern
            pattern = LearningPattern(
                id=uuid4(),
                pattern_type=PatternType.SUCCESS_PATTERN,
                pattern_data=pattern_data,
                pattern_hash=str(hash(str(pattern_data))),
                confidence_score=outcome.success_score,
                source='decision_outcome_tracker',
                created_at=datetime.now(timezone.utc)
            )
            
            self.db.add(pattern)
            logger.info(f"Created success pattern from outcome {outcome.id}")
            
        except Exception as e:
            logger.error(f"Error extracting success patterns: {e}")
    
    async def _identify_improvement_opportunities(self, outcome: DecisionOutcome):
        """Identify improvement opportunities from failed outcomes"""
        try:
            # Create improvement opportunity pattern
            pattern_data = {
                'decision_type': outcome.outcome_data.get('decision_context', {}).get('decision_type'),
                'failure_factors': outcome.outcome_data.get('factor_analysis', {}).get('negative_factors', []),
                'success_score': outcome.success_score,
                'improvement_suggestions': await self._generate_improvement_suggestions(outcome),
                'context': outcome.outcome_data.get('decision_context', {})
            }
            
            # Store as failure pattern for learning
            pattern = LearningPattern(
                id=uuid4(),
                pattern_type=PatternType.FAILURE_PATTERN,
                pattern_data=pattern_data,
                pattern_hash=str(hash(str(pattern_data))),
                confidence_score=1.0 - outcome.success_score,  # Higher confidence for clear failures
                source='decision_outcome_tracker',
                created_at=datetime.now(timezone.utc)
            )
            
            self.db.add(pattern)
            logger.info(f"Created failure pattern from outcome {outcome.id}")
            
        except Exception as e:
            logger.error(f"Error identifying improvement opportunities: {e}")
    
    async def _analyze_partial_outcome(self, outcome: DecisionOutcome):
        """Analyze partial outcomes for improvement potential"""
        try:
            # Analyze what prevented full success
            pattern_data = {
                'decision_type': outcome.outcome_data.get('decision_context', {}).get('decision_type'),
                'partial_factors': {
                    'positive': outcome.outcome_data.get('factor_analysis', {}).get('positive_factors', []),
                    'negative': outcome.outcome_data.get('factor_analysis', {}).get('negative_factors', [])
                },
                'success_score': outcome.success_score,
                'improvement_potential': self.success_threshold - outcome.success_score,
                'context': outcome.outcome_data.get('decision_context', {})
            }
            
            # Store as partial pattern
            pattern = LearningPattern(
                id=uuid4(),
                pattern_type=PatternType.DECISION_PATTERN,
                pattern_data=pattern_data,
                pattern_hash=str(hash(str(pattern_data))),
                confidence_score=outcome.success_score,
                source='decision_outcome_tracker',
                created_at=datetime.now(timezone.utc)
            )
            
            self.db.add(pattern)
            logger.info(f"Created partial outcome pattern from outcome {outcome.id}")
            
        except Exception as e:
            logger.error(f"Error analyzing partial outcome: {e}")
    
    async def _analyze_time_to_outcome(
        self,
        outcomes: List[DecisionOutcome]
    ) -> Dict[str, float]:
        """Analyze time-to-outcome metrics"""
        try:
            time_metrics = {
                'average_time_to_measure': 0.0,
                'median_time_to_measure': 0.0,
                'fastest_outcome': 0.0,
                'slowest_outcome': 0.0
            }
            
            # Calculate time differences
            time_diffs = []
            for outcome in outcomes:
                decision = await self.db.get(Memory, outcome.decision_id)
                if decision:
                    time_diff = (outcome.measured_at - decision.created_at).total_seconds() / 3600  # hours
                    time_diffs.append(time_diff)
            
            if time_diffs:
                time_metrics['average_time_to_measure'] = mean(time_diffs)
                time_metrics['median_time_to_measure'] = median(time_diffs)
                time_metrics['fastest_outcome'] = min(time_diffs)
                time_metrics['slowest_outcome'] = max(time_diffs)
            
            return time_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing time to outcome: {e}")
            return {}
    
    async def _analyze_impact_metrics(
        self,
        outcomes: List[DecisionOutcome]
    ) -> Dict[str, Any]:
        """Analyze impact metrics across outcomes"""
        try:
            impact_analysis = {
                'user_satisfaction': {
                    'average': 0.0,
                    'distribution': defaultdict(int)
                },
                'performance_impact': {
                    'average': 0.0,
                    'improvements': 0,
                    'degradations': 0
                },
                'error_impact': {
                    'total_errors_prevented': 0,
                    'average_error_reduction': 0.0
                },
                'efficiency_gains': {
                    'average_time_saved': 0.0,
                    'total_time_saved': 0.0
                }
            }
            
            # Analyze each outcome's impact
            satisfaction_scores = []
            performance_impacts = []
            error_reductions = []
            time_savings = []
            
            for outcome in outcomes:
                impact_data = outcome.impact_data or {}
                
                # User satisfaction
                if 'user_satisfaction' in impact_data:
                    satisfaction = impact_data['user_satisfaction']
                    satisfaction_scores.append(satisfaction)
                    
                    # Distribution
                    if satisfaction >= 0.8:
                        impact_analysis['user_satisfaction']['distribution']['high'] += 1
                    elif satisfaction >= 0.6:
                        impact_analysis['user_satisfaction']['distribution']['medium'] += 1
                    else:
                        impact_analysis['user_satisfaction']['distribution']['low'] += 1
                
                # Performance impact
                if 'performance_impact' in impact_data:
                    performance = impact_data['performance_impact']
                    performance_impacts.append(performance)
                    
                    if performance > 0.5:
                        impact_analysis['performance_impact']['improvements'] += 1
                    elif performance < 0.5:
                        impact_analysis['performance_impact']['degradations'] += 1
                
                # Error reduction
                if 'error_reduction' in impact_data:
                    error_reduction = impact_data['error_reduction']
                    error_reductions.append(error_reduction)
                
                # Time efficiency
                if 'time_saved' in impact_data:
                    time_saved = impact_data['time_saved']
                    time_savings.append(time_saved)
            
            # Calculate averages
            if satisfaction_scores:
                impact_analysis['user_satisfaction']['average'] = mean(satisfaction_scores)
            
            if performance_impacts:
                impact_analysis['performance_impact']['average'] = mean(performance_impacts)
            
            if error_reductions:
                impact_analysis['error_impact']['average_error_reduction'] = mean(error_reductions)
                impact_analysis['error_impact']['total_errors_prevented'] = sum(error_reductions)
            
            if time_savings:
                impact_analysis['efficiency_gains']['average_time_saved'] = mean(time_savings)
                impact_analysis['efficiency_gains']['total_time_saved'] = sum(time_savings)
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing impact metrics: {e}")
            return {}
    
    async def _analyze_trend_for_period(
        self,
        decision_type: Optional[str],
        time_period: timedelta,
        period_type: str
    ) -> DecisionTrend:
        """Analyze trend for a specific time period"""
        try:
            # Get outcomes for the period
            cutoff_date = datetime.now(timezone.utc) - time_period
            
            query = select(DecisionOutcome).where(
                DecisionOutcome.created_at >= cutoff_date
            )
            
            if decision_type:
                query = query.where(
                    DecisionOutcome.outcome_data.op('->>')('decision_context').op('->>')('decision_type') == decision_type
                )
            
            result = await self.db.execute(query)
            outcomes = result.scalars().all()
            
            if not outcomes:
                return DecisionTrend(
                    time_period=period_type,
                    trend_direction='stable',
                    trend_strength=0.0
                )
            
            # Split outcomes into periods for trend analysis
            outcomes_by_period = self._group_outcomes_by_period(outcomes, period_type)
            
            # Calculate trend
            if len(outcomes_by_period) < 2:
                return DecisionTrend(
                    time_period=period_type,
                    trend_direction='stable',
                    trend_strength=0.0
                )
            
            # Calculate success rates for each period
            success_rates = []
            for period_outcomes in outcomes_by_period:
                if period_outcomes:
                    success_rate = sum(1 for o in period_outcomes if o.is_successful()) / len(period_outcomes)
                    success_rates.append(success_rate)
            
            # Determine trend direction
            if len(success_rates) >= 2:
                recent_rate = success_rates[-1]
                previous_rate = success_rates[-2]
                rate_change = recent_rate - previous_rate
                
                if rate_change > 0.1:
                    trend_direction = 'improving'
                    trend_strength = min(1.0, rate_change * 2)
                elif rate_change < -0.1:
                    trend_direction = 'declining'
                    trend_strength = min(1.0, abs(rate_change) * 2)
                else:
                    trend_direction = 'stable'
                    trend_strength = 0.5
            else:
                trend_direction = 'stable'
                trend_strength = 0.5
            
            # Generate key factors and recommendations
            key_factors = await self._identify_trend_factors(outcomes)
            recommendations = await self._generate_trend_recommendations(
                trend_direction,
                trend_strength,
                key_factors
            )
            
            return DecisionTrend(
                time_period=period_type,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                key_factors=key_factors,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend for period: {e}")
            return DecisionTrend(
                time_period=period_type,
                trend_direction='stable',
                trend_strength=0.0
            )
    
    def _group_outcomes_by_period(
        self,
        outcomes: List[DecisionOutcome],
        period_type: str
    ) -> List[List[DecisionOutcome]]:
        """Group outcomes by time period"""
        if period_type == 'weekly':
            # Group by week
            groups = defaultdict(list)
            for outcome in outcomes:
                week_key = outcome.created_at.strftime('%Y-W%U')
                groups[week_key].append(outcome)
            return list(groups.values())
        
        elif period_type == 'monthly':
            # Group by month
            groups = defaultdict(list)
            for outcome in outcomes:
                month_key = outcome.created_at.strftime('%Y-%m')
                groups[month_key].append(outcome)
            return list(groups.values())
        
        else:
            return [outcomes]
    
    async def _identify_trend_factors(
        self,
        outcomes: List[DecisionOutcome]
    ) -> List[str]:
        """Identify key factors influencing trends"""
        try:
            factors = []
            
            # Analyze decision types
            decision_types = Counter()
            for outcome in outcomes:
                decision_type = outcome.outcome_data.get('decision_context', {}).get('decision_type', 'unknown')
                decision_types[decision_type] += 1
            
            # Most common decision type
            if decision_types:
                most_common = decision_types.most_common(1)[0]
                factors.append(f"Most common decision type: {most_common[0]} ({most_common[1]} decisions)")
            
            # Success patterns
            successful_outcomes = [o for o in outcomes if o.is_successful()]
            if successful_outcomes:
                factors.append(f"Success rate: {len(successful_outcomes)}/{len(outcomes)} ({len(successful_outcomes)/len(outcomes):.1%})")
            
            # Performance indicators
            avg_score = mean([o.success_score for o in outcomes])
            factors.append(f"Average success score: {avg_score:.2f}")
            
            return factors
            
        except Exception as e:
            logger.error(f"Error identifying trend factors: {e}")
            return []
    
    async def _generate_trend_recommendations(
        self,
        trend_direction: str,
        trend_strength: float,
        key_factors: List[str]
    ) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        if trend_direction == 'improving':
            recommendations.append("Continue current practices that are driving improvement")
            recommendations.append("Document successful patterns for future reference")
            if trend_strength > 0.7:
                recommendations.append("Consider scaling successful approaches to other areas")
        
        elif trend_direction == 'declining':
            recommendations.append("Investigate root causes of declining performance")
            recommendations.append("Review recent changes that may have impacted outcomes")
            if trend_strength > 0.7:
                recommendations.append("Implement immediate corrective measures")
        
        else:  # stable
            recommendations.append("Monitor for early indicators of change")
            recommendations.append("Consider optimization opportunities")
            recommendations.append("Evaluate if current performance meets goals")
        
        return recommendations
    
    async def _identify_root_causes(
        self,
        failure: DecisionOutcome
    ) -> List[str]:
        """Identify root causes of decision failure"""
        try:
            root_causes = []
            
            # Analyze failure factors
            factor_analysis = failure.outcome_data.get('factor_analysis', {})
            negative_factors = factor_analysis.get('negative_factors', [])
            
            # Map factors to root causes
            for factor in negative_factors:
                if 'user' in factor.lower():
                    root_causes.append('User experience issues')
                elif 'performance' in factor.lower():
                    root_causes.append('Performance problems')
                elif 'error' in factor.lower():
                    root_causes.append('Technical errors')
                elif 'time' in factor.lower():
                    root_causes.append('Time constraints')
                elif 'resource' in factor.lower():
                    root_causes.append('Resource limitations')
            
            # Analyze impact data
            impact_data = failure.impact_data or {}
            if impact_data.get('user_satisfaction', 1.0) < 0.3:
                root_causes.append('Poor user satisfaction')
            
            if impact_data.get('performance_impact', 1.0) < 0.3:
                root_causes.append('Performance degradation')
            
            # Remove duplicates
            return list(set(root_causes)) if root_causes else ['Unknown cause']
            
        except Exception as e:
            logger.error(f"Error identifying root causes: {e}")
            return ['Analysis error']
    
    async def _generate_improvement_suggestions(
        self,
        failure: DecisionOutcome
    ) -> List[str]:
        """Generate improvement suggestions for failed decisions"""
        try:
            suggestions = []
            
            # Analyze failure score
            if failure.success_score < 0.2:
                suggestions.append('Complete review of decision approach required')
            elif failure.success_score < 0.4:
                suggestions.append('Significant modifications needed')
            else:
                suggestions.append('Minor adjustments may resolve issues')
            
            # Context-specific suggestions
            decision_type = failure.outcome_data.get('decision_context', {}).get('decision_type', 'unknown')
            
            if decision_type == 'code_generation':
                suggestions.extend([
                    'Review code generation patterns',
                    'Validate against coding standards',
                    'Consider user feedback on code quality'
                ])
            elif decision_type == 'architecture':
                suggestions.extend([
                    'Reassess architectural requirements',
                    'Consider scalability implications',
                    'Review performance requirements'
                ])
            elif decision_type == 'debugging':
                suggestions.extend([
                    'Improve debugging methodology',
                    'Enhance error analysis techniques',
                    'Consider additional diagnostic tools'
                ])
            
            # Impact-based suggestions
            impact_data = failure.impact_data or {}
            if impact_data.get('user_satisfaction', 1.0) < 0.5:
                suggestions.append('Focus on user experience improvements')
            
            if impact_data.get('performance_impact', 1.0) < 0.5:
                suggestions.append('Optimize for better performance')
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return ['Review decision process']
    
    def _identify_common_failure_factors(
        self,
        failures: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify common factors across failures"""
        factor_counts = Counter()
        
        for failure in failures:
            failure_factors = failure.get('failure_factors', [])
            for factor in failure_factors:
                factor_counts[factor] += 1
        
        # Return factors that appear in at least 2 failures
        common_factors = [
            factor for factor, count in factor_counts.items()
            if count >= 2
        ]
        
        return common_factors
    
    async def _generate_context_specific_recommendations(
        self,
        decision_type: str,
        context: Dict[str, Any],
        historical_metrics: OutcomeMetrics
    ) -> List[Dict[str, Any]]:
        """Generate context-specific recommendations"""
        recommendations = []
        
        # Context complexity recommendations
        if context.get('complexity', 'medium') == 'high':
            recommendations.append({
                'type': 'complexity_management',
                'priority': 'high',
                'description': 'High complexity context detected',
                'suggested_actions': [
                    'Break down complex decisions into smaller parts',
                    'Implement staged decision making',
                    'Add additional validation steps'
                ]
            })
        
        # Time pressure recommendations
        if context.get('time_pressure', 'normal') == 'high':
            recommendations.append({
                'type': 'time_management',
                'priority': 'medium',
                'description': 'High time pressure may impact decision quality',
                'suggested_actions': [
                    'Prioritize critical decision factors',
                    'Use proven decision patterns',
                    'Consider parallel decision processes'
                ]
            })
        
        # Resource availability recommendations
        if context.get('resources', 'adequate') == 'limited':
            recommendations.append({
                'type': 'resource_optimization',
                'priority': 'medium',
                'description': 'Limited resources may constrain decision options',
                'suggested_actions': [
                    'Optimize resource allocation',
                    'Consider alternative approaches',
                    'Implement resource monitoring'
                ]
            })
        
        return recommendations
    
    async def _update_cached_metrics(self, decision_type: str):
        """Update cached metrics for performance"""
        try:
            if redis_client.client:
                # Cache key for decision type metrics
                cache_key = f"decision_metrics:{decision_type}"
                
                # Get fresh metrics
                metrics = await self.get_decision_effectiveness(
                    decision_type=decision_type,
                    time_period=timedelta(days=30)
                )
                
                # Cache the metrics
                await redis_client.set(
                    cache_key,
                    metrics.dict(),
                    expiry=self.cache_ttl
                )
                
        except Exception as e:
            logger.warning(f"Failed to update cached metrics: {e}")