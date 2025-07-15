"""Success Tracking Service

This service tracks the outcomes of decisions and recommendations,
measuring their effectiveness and learning from successes and failures.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from statistics import mean, median, stdev

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func, or_
from sqlalchemy.exc import IntegrityError

from ..models.decision_outcome import DecisionOutcome, OutcomeType
from ..models.learning_pattern import LearningPattern, PatternType
from ...memory_system.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


class SuccessTracker:
    """Service for tracking and analyzing decision outcomes"""
    
    def __init__(self, db: Session):
        """Initialize the success tracker"""
        self.db = db
        
        # Success tracking configuration
        self.success_threshold = 0.7  # Score above this is considered success
        self.failure_threshold = 0.3  # Score below this is considered failure
        self.min_data_points = 5  # Minimum outcomes before drawing conclusions
        
        # Outcome weights for different factors
        self.outcome_weights = {
            'task_completed': 0.3,
            'user_satisfied': 0.3,
            'performance_met': 0.2,
            'no_errors': 0.1,
            'timely_completion': 0.1
        }
    
    async def track_outcome(
        self,
        decision_id: UUID,
        outcome_data: Dict[str, Any]
    ) -> DecisionOutcome:
        """Track the outcome of a decision
        
        Args:
            decision_id: ID of the decision memory
            outcome_data: Information about the outcome including:
                - outcome_type: Type of outcome (success, failure, partial)
                - success_score: Score from 0.0 to 1.0
                - impact_data: Data about the impact
                - metrics: Performance metrics
                - feedback: Any user feedback
                
        Returns:
            DecisionOutcome instance
        """
        try:
            # Validate decision exists
            decision = await self.db.get(Memory, decision_id)
            if not decision or decision.memory_type != MemoryType.DECISION.value:
                raise ValueError(f"Decision {decision_id} not found or not a decision type")
            
            # Calculate success score if not provided
            if 'success_score' not in outcome_data:
                outcome_data['success_score'] = await self._calculate_success_score(
                    outcome_data
                )
            
            # Determine outcome type based on score
            success_score = outcome_data['success_score']
            if success_score >= self.success_threshold:
                outcome_type = OutcomeType.SUCCESS
            elif success_score <= self.failure_threshold:
                outcome_type = OutcomeType.FAILURE
            else:
                outcome_type = OutcomeType.PARTIAL
            
            # Create outcome record
            outcome = DecisionOutcome(
                id=uuid4(),
                decision_id=decision_id,
                outcome_type=outcome_type.value,
                success_score=success_score,
                outcome_data=outcome_data,
                impact_data=outcome_data.get('impact_data', {}),
                measured_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc)
            )
            
            # Analyze outcome factors
            outcome.outcome_data['analysis'] = await self._analyze_outcome_factors(
                outcome_data
            )
            
            # Store outcome
            self.db.add(outcome)
            self.db.commit()
            self.db.refresh(outcome)
            
            # Update decision memory with outcome reference
            await self._update_decision_memory(decision, outcome)
            
            # Learn from outcome
            await self._learn_from_outcome(decision, outcome)
            
            logger.info(
                f"Tracked {outcome_type.value} outcome for decision {decision_id} "
                f"(score: {success_score:.2f})"
            )
            
            return outcome
            
        except Exception as e:
            logger.error(f"Error tracking outcome: {e}")
            self.db.rollback()
            raise
    
    async def get_success_rate(
        self,
        decision_type: Optional[str] = None,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Get success rate statistics
        
        Args:
            decision_type: Optional specific decision type to filter
            time_period_days: Number of days to look back
            
        Returns:
            Dictionary with success rate statistics
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=time_period_days)
            
            # Base query
            query = select(DecisionOutcome).where(
                DecisionOutcome.measured_at >= start_date
            )
            
            # Filter by decision type if specified
            if decision_type:
                # This would join with Memory table to filter by decision type
                # For now, we'll filter by outcome data
                query = query.where(
                    DecisionOutcome.outcome_data['decision_type'].astext == decision_type
                )
            
            result = await self.db.execute(query)
            outcomes = result.scalars().all()
            
            if not outcomes:
                return {
                    'success_rate': 0.0,
                    'total_decisions': 0,
                    'message': 'No outcomes tracked in this period'
                }
            
            # Calculate statistics
            success_count = len([o for o in outcomes if o.outcome_type == OutcomeType.SUCCESS.value])
            failure_count = len([o for o in outcomes if o.outcome_type == OutcomeType.FAILURE.value])
            partial_count = len([o for o in outcomes if o.outcome_type == OutcomeType.PARTIAL.value])
            
            scores = [o.success_score for o in outcomes]
            
            stats = {
                'success_rate': success_count / len(outcomes),
                'failure_rate': failure_count / len(outcomes),
                'partial_rate': partial_count / len(outcomes),
                'total_decisions': len(outcomes),
                'average_score': mean(scores),
                'median_score': median(scores),
                'score_std_dev': stdev(scores) if len(scores) > 1 else 0.0,
                'breakdown': {
                    'successes': success_count,
                    'failures': failure_count,
                    'partial': partial_count
                },
                'time_period_days': time_period_days
            }
            
            # Add trend analysis
            stats['trend'] = await self._analyze_success_trend(outcomes)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}")
            return {'error': str(e)}
    
    async def get_decision_effectiveness(
        self,
        decision_id: UUID
    ) -> Dict[str, Any]:
        """Get effectiveness metrics for a specific decision
        
        Args:
            decision_id: ID of the decision to analyze
            
        Returns:
            Dictionary with effectiveness metrics
        """
        try:
            # Get all outcomes for this decision
            result = await self.db.execute(
                select(DecisionOutcome).where(
                    DecisionOutcome.decision_id == decision_id
                ).order_by(DecisionOutcome.measured_at)
            )
            outcomes = result.scalars().all()
            
            if not outcomes:
                return {
                    'effectiveness': 'unknown',
                    'message': 'No outcomes tracked for this decision'
                }
            
            # Calculate effectiveness metrics
            latest_outcome = outcomes[-1]
            scores = [o.success_score for o in outcomes]
            
            effectiveness = {
                'latest_score': latest_outcome.success_score,
                'latest_outcome': latest_outcome.outcome_type,
                'average_score': mean(scores),
                'score_trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'stable',
                'total_measurements': len(outcomes),
                'success_consistency': self._calculate_consistency(scores),
                'factors': await self._extract_success_factors(outcomes)
            }
            
            # Determine overall effectiveness
            if effectiveness['average_score'] >= self.success_threshold:
                effectiveness['overall'] = 'effective'
            elif effectiveness['average_score'] <= self.failure_threshold:
                effectiveness['overall'] = 'ineffective'
            else:
                effectiveness['overall'] = 'moderately_effective'
            
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error getting decision effectiveness: {e}")
            return {'error': str(e)}
    
    async def get_success_patterns(
        self,
        min_confidence: float = 0.7,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get patterns associated with successful outcomes
        
        Args:
            min_confidence: Minimum confidence score for patterns
            limit: Maximum number of patterns to return
            
        Returns:
            List of success patterns
        """
        try:
            # Get successful outcomes
            result = await self.db.execute(
                select(DecisionOutcome).where(
                    DecisionOutcome.outcome_type == OutcomeType.SUCCESS.value
                ).order_by(DecisionOutcome.success_score.desc()).limit(limit * 2)
            )
            successful_outcomes = result.scalars().all()
            
            # Extract patterns from successful outcomes
            patterns = []
            pattern_scores = {}
            
            for outcome in successful_outcomes:
                # Extract factors that contributed to success
                factors = outcome.outcome_data.get('success_factors', [])
                context = outcome.outcome_data.get('context', {})
                
                for factor in factors:
                    pattern_key = f"{factor}:{context.get('type', 'general')}"
                    if pattern_key not in pattern_scores:
                        pattern_scores[pattern_key] = []
                    pattern_scores[pattern_key].append(outcome.success_score)
            
            # Calculate pattern confidence
            for pattern_key, scores in pattern_scores.items():
                if len(scores) >= 3:  # Require at least 3 occurrences
                    factor, context_type = pattern_key.split(':', 1)
                    avg_score = mean(scores)
                    
                    if avg_score >= min_confidence:
                        patterns.append({
                            'pattern': factor,
                            'context_type': context_type,
                            'confidence': avg_score,
                            'occurrences': len(scores),
                            'consistency': self._calculate_consistency(scores)
                        })
            
            # Sort by confidence and limit
            patterns.sort(key=lambda p: p['confidence'], reverse=True)
            return patterns[:limit]
            
        except Exception as e:
            logger.error(f"Error getting success patterns: {e}")
            return []
    
    async def get_failure_patterns(
        self,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get patterns associated with failed outcomes
        
        Args:
            limit: Maximum number of patterns to return
            
        Returns:
            List of failure patterns
        """
        try:
            # Get failed outcomes
            result = await self.db.execute(
                select(DecisionOutcome).where(
                    DecisionOutcome.outcome_type == OutcomeType.FAILURE.value
                ).order_by(DecisionOutcome.measured_at.desc()).limit(limit * 2)
            )
            failed_outcomes = result.scalars().all()
            
            # Extract patterns from failures
            patterns = []
            failure_reasons = {}
            
            for outcome in failed_outcomes:
                # Extract failure reasons
                reasons = outcome.outcome_data.get('failure_reasons', [])
                context = outcome.outcome_data.get('context', {})
                
                for reason in reasons:
                    pattern_key = f"{reason}:{context.get('type', 'general')}"
                    if pattern_key not in failure_reasons:
                        failure_reasons[pattern_key] = 0
                    failure_reasons[pattern_key] += 1
            
            # Create pattern list
            for pattern_key, count in failure_reasons.items():
                if count >= 2:  # Require at least 2 occurrences
                    reason, context_type = pattern_key.split(':', 1)
                    patterns.append({
                        'pattern': reason,
                        'context_type': context_type,
                        'frequency': count,
                        'severity': 'high' if count > 5 else 'medium'
                    })
            
            # Sort by frequency
            patterns.sort(key=lambda p: p['frequency'], reverse=True)
            return patterns[:limit]
            
        except Exception as e:
            logger.error(f"Error getting failure patterns: {e}")
            return []
    
    # Private methods
    
    async def _calculate_success_score(
        self,
        outcome_data: Dict[str, Any]
    ) -> float:
        """Calculate success score based on outcome data"""
        score = 0.0
        total_weight = 0.0
        
        # Check each factor
        for factor, weight in self.outcome_weights.items():
            if factor in outcome_data:
                factor_value = outcome_data[factor]
                # Convert boolean to float
                if isinstance(factor_value, bool):
                    factor_value = 1.0 if factor_value else 0.0
                elif isinstance(factor_value, (int, float)):
                    factor_value = max(0.0, min(1.0, factor_value))
                else:
                    continue
                
                score += factor_value * weight
                total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            score = score / total_weight
        
        # Apply any modifiers
        if 'modifiers' in outcome_data:
            for modifier, value in outcome_data['modifiers'].items():
                if modifier == 'time_penalty' and value < 0:
                    score *= (1 + value)  # value is negative
                elif modifier == 'quality_bonus' and value > 0:
                    score = min(1.0, score * (1 + value))
        
        return max(0.0, min(1.0, score))
    
    async def _analyze_outcome_factors(
        self,
        outcome_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze factors that contributed to the outcome"""
        analysis = {
            'primary_factors': [],
            'missing_factors': [],
            'recommendations': []
        }
        
        # Identify primary contributing factors
        for factor, weight in self.outcome_weights.items():
            if factor in outcome_data:
                factor_value = outcome_data[factor]
                if isinstance(factor_value, bool) and factor_value:
                    analysis['primary_factors'].append(factor)
                elif isinstance(factor_value, (int, float)) and factor_value > 0.7:
                    analysis['primary_factors'].append(factor)
            else:
                analysis['missing_factors'].append(factor)
        
        # Generate recommendations based on analysis
        if 'task_completed' not in outcome_data or not outcome_data.get('task_completed'):
            analysis['recommendations'].append('Focus on task completion')
        
        if 'user_satisfied' not in outcome_data or not outcome_data.get('user_satisfied'):
            analysis['recommendations'].append('Improve user satisfaction')
        
        if 'performance_met' in outcome_data and outcome_data['performance_met'] < 0.5:
            analysis['recommendations'].append('Optimize performance')
        
        return analysis
    
    async def _update_decision_memory(
        self,
        decision: Memory,
        outcome: DecisionOutcome
    ):
        """Update decision memory with outcome information"""
        if not decision.memory_metadata:
            decision.memory_metadata = {}
        
        # Store outcome reference
        if 'outcomes' not in decision.memory_metadata:
            decision.memory_metadata['outcomes'] = []
        
        decision.memory_metadata['outcomes'].append({
            'outcome_id': str(outcome.id),
            'type': outcome.outcome_type,
            'score': outcome.success_score,
            'measured_at': outcome.measured_at.isoformat()
        })
        
        # Update latest outcome
        decision.memory_metadata['latest_outcome'] = {
            'type': outcome.outcome_type,
            'score': outcome.success_score,
            'measured_at': outcome.measured_at.isoformat()
        }
        
        # Update effectiveness rating
        if outcome.success_score >= self.success_threshold:
            decision.importance = min(1.0, decision.importance * 1.1)
        elif outcome.success_score <= self.failure_threshold:
            decision.importance = max(0.1, decision.importance * 0.9)
        
        self.db.commit()
    
    async def _learn_from_outcome(
        self,
        decision: Memory,
        outcome: DecisionOutcome
    ):
        """Learn patterns from the outcome"""
        # Extract decision context
        decision_context = decision.memory_metadata.get('context', {})
        decision_type = decision.memory_metadata.get('decision_type', 'general')
        
        # Create pattern based on outcome
        if outcome.outcome_type == OutcomeType.SUCCESS.value:
            # Learn success pattern
            pattern_data = {
                'decision_type': decision_type,
                'context': decision_context,
                'success_factors': outcome.outcome_data.get('analysis', {}).get('primary_factors', []),
                'score': outcome.success_score
            }
            
            pattern = LearningPattern(
                pattern_type=PatternType.SUCCESS,
                pattern_data=pattern_data,
                pattern_hash=self._hash_pattern(pattern_data),
                confidence_score=outcome.success_score,
                source='outcome_tracking'
            )
            
        elif outcome.outcome_type == OutcomeType.FAILURE.value:
            # Learn failure pattern
            pattern_data = {
                'decision_type': decision_type,
                'context': decision_context,
                'failure_reasons': outcome.outcome_data.get('failure_reasons', []),
                'missing_factors': outcome.outcome_data.get('analysis', {}).get('missing_factors', [])
            }
            
            pattern = LearningPattern(
                pattern_type=PatternType.ERROR,
                pattern_data=pattern_data,
                pattern_hash=self._hash_pattern(pattern_data),
                confidence_score=1.0 - outcome.success_score,
                source='failure_analysis'
            )
        else:
            # Partial success - learn what worked and what didn't
            return
        
        # Store pattern
        try:
            self.db.add(pattern)
            self.db.commit()
        except IntegrityError:
            # Pattern already exists, update it
            self.db.rollback()
    
    def _hash_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Generate hash for pattern data"""
        import hashlib
        import json
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(pattern_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    async def _analyze_success_trend(
        self,
        outcomes: List[DecisionOutcome]
    ) -> Dict[str, Any]:
        """Analyze trend in success rates"""
        if len(outcomes) < 2:
            return {'trend': 'insufficient_data'}
        
        # Sort by date
        sorted_outcomes = sorted(outcomes, key=lambda o: o.measured_at)
        
        # Split into halves
        midpoint = len(sorted_outcomes) // 2
        first_half = sorted_outcomes[:midpoint]
        second_half = sorted_outcomes[midpoint:]
        
        # Calculate success rates
        first_success_rate = len([o for o in first_half if o.outcome_type == OutcomeType.SUCCESS.value]) / len(first_half)
        second_success_rate = len([o for o in second_half if o.outcome_type == OutcomeType.SUCCESS.value]) / len(second_half)
        
        # Determine trend
        if second_success_rate > first_success_rate + 0.1:
            trend = 'improving'
        elif second_success_rate < first_success_rate - 0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'first_period_rate': first_success_rate,
            'second_period_rate': second_success_rate,
            'change': second_success_rate - first_success_rate
        }
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """Calculate consistency score (inverse of variance)"""
        if len(scores) < 2:
            return 1.0
        
        variance = stdev(scores) ** 2
        # Convert variance to consistency score (0-1)
        # Lower variance = higher consistency
        consistency = 1.0 / (1.0 + variance)
        
        return consistency
    
    async def _extract_success_factors(
        self,
        outcomes: List[DecisionOutcome]
    ) -> List[str]:
        """Extract common success factors from outcomes"""
        factor_counts = {}
        
        for outcome in outcomes:
            if outcome.outcome_type == OutcomeType.SUCCESS.value:
                factors = outcome.outcome_data.get('analysis', {}).get('primary_factors', [])
                for factor in factors:
                    factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Sort by frequency
        sorted_factors = sorted(
            factor_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top factors
        return [factor for factor, _ in sorted_factors[:5]]