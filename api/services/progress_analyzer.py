"""Progress Analyzer Service

Analyzes project progress patterns, predicts completion dates, and provides
insights for project management and timeline optimization.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
from statistics import mean, median, stdev
import math

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from .project_timeline_service import ProjectTimelineService
from ..models.project_timeline import ProjectTimeline, ProgressSnapshot, ProjectMilestone

logger = logging.getLogger(__name__)


class ProgressTrend:
    """Represents a progress trend analysis"""
    
    def __init__(self, trend_type: str, confidence: float, description: str, data: Dict[str, Any]):
        self.trend_type = trend_type  # 'improving', 'declining', 'stable', 'volatile'
        self.confidence = confidence  # 0.0 to 1.0
        self.description = description
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trend_type': self.trend_type,
            'confidence': self.confidence,
            'description': self.description,
            'data': self.data
        }


class ProgressAnalyzer:
    """Service for analyzing project progress and predicting outcomes"""
    
    def __init__(self, db: Session):
        self.db = db
        self.timeline_service = ProjectTimelineService(db)
    
    def analyze_project_progress(self, timeline_id: UUID) -> Dict[str, Any]:
        """Comprehensive project progress analysis"""
        
        timeline = self.timeline_service.get_project_timeline_by_uuid(timeline_id)
        if not timeline:
            return {}
        
        # Get progress history
        snapshots = self.timeline_service.get_progress_history(timeline_id, limit=100)
        milestones = self.timeline_service.list_milestones(timeline_id)
        
        # Perform various analyses
        velocity_analysis = self._analyze_velocity_trends(snapshots)
        completion_prediction = self._predict_completion_date(timeline, snapshots)
        milestone_analysis = self._analyze_milestone_progress(milestones)
        risk_assessment = self._assess_project_risks(timeline, snapshots, milestones)
        trend_analysis = self._analyze_progress_trends(snapshots)
        
        return {
            'timeline_id': str(timeline_id),
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'current_progress': timeline.progress_percentage,
            'velocity_analysis': velocity_analysis,
            'completion_prediction': completion_prediction,
            'milestone_analysis': milestone_analysis,
            'risk_assessment': risk_assessment,
            'trend_analysis': trend_analysis.to_dict() if trend_analysis else None,
            'recommendations': self._generate_recommendations(timeline, snapshots, milestones)
        }
    
    def _analyze_velocity_trends(self, snapshots: List[ProgressSnapshot]) -> Dict[str, Any]:
        """Analyze velocity trends over time"""
        
        if len(snapshots) < 3:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 3 snapshots for velocity analysis'
            }
        
        # Calculate velocity metrics
        velocities = [s.velocity for s in snapshots if s.velocity is not None]
        if not velocities:
            return {
                'status': 'no_velocity_data',
                'message': 'No velocity data available'
            }
        
        current_velocity = mean(velocities[:5]) if len(velocities) >= 5 else mean(velocities)
        historical_velocity = mean(velocities)
        
        # Calculate velocity trend
        if len(velocities) >= 10:
            recent_velocities = velocities[:5]
            older_velocities = velocities[5:10]
            
            recent_avg = mean(recent_velocities)
            older_avg = mean(older_velocities)
            
            if recent_avg > older_avg * 1.1:
                trend = 'improving'
            elif recent_avg < older_avg * 0.9:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        # Calculate velocity statistics
        velocity_stats = {
            'current_velocity': current_velocity,
            'historical_average': historical_velocity,
            'max_velocity': max(velocities),
            'min_velocity': min(velocities),
            'velocity_trend': trend,
            'velocity_consistency': self._calculate_velocity_consistency(velocities)
        }
        
        if len(velocities) > 1:
            velocity_stats['velocity_std_dev'] = stdev(velocities)
        
        return velocity_stats
    
    def _predict_completion_date(
        self,
        timeline: ProjectTimeline,
        snapshots: List[ProgressSnapshot]
    ) -> Dict[str, Any]:
        """Predict project completion date based on current progress"""
        
        if timeline.progress_percentage >= 100:
            return {
                'status': 'completed',
                'actual_completion': timeline.actual_end_date.isoformat() if timeline.actual_end_date else None
            }
        
        if len(snapshots) < 5:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 5 snapshots for prediction'
            }
        
        # Calculate remaining work
        remaining_progress = 100 - timeline.progress_percentage
        remaining_tasks = timeline.total_tasks - timeline.completed_tasks
        
        # Get recent velocity
        recent_snapshots = snapshots[:10]
        recent_velocities = [s.velocity for s in recent_snapshots if s.velocity > 0]
        
        if not recent_velocities:
            return {
                'status': 'no_velocity_data',
                'message': 'No recent velocity data for prediction'
            }
        
        # Calculate predictions using different methods
        predictions = []
        
        # Method 1: Average velocity
        avg_velocity = mean(recent_velocities)
        if avg_velocity > 0:
            days_remaining = remaining_tasks / avg_velocity
            completion_date = datetime.now(timezone.utc) + timedelta(days=days_remaining)
            predictions.append({
                'method': 'average_velocity',
                'completion_date': completion_date.isoformat(),
                'days_remaining': days_remaining,
                'confidence': 0.7
            })
        
        # Method 2: Linear regression (simplified)
        if len(recent_snapshots) >= 5:
            linear_prediction = self._linear_regression_prediction(recent_snapshots)
            if linear_prediction:
                predictions.append(linear_prediction)
        
        # Method 3: Exponential smoothing
        exponential_prediction = self._exponential_smoothing_prediction(recent_snapshots)
        if exponential_prediction:
            predictions.append(exponential_prediction)
        
        # Calculate consensus prediction
        consensus = self._calculate_consensus_prediction(predictions)
        
        # Compare with planned end date
        schedule_variance = None
        if timeline.planned_end_date and consensus:
            planned_date = timeline.planned_end_date
            predicted_date = datetime.fromisoformat(consensus['completion_date'].replace('Z', '+00:00'))
            schedule_variance = (predicted_date - planned_date).days
        
        return {
            'status': 'predicted',
            'consensus_prediction': consensus,
            'individual_predictions': predictions,
            'schedule_variance_days': schedule_variance,
            'confidence_score': consensus['confidence'] if consensus else 0.0
        }
    
    def _analyze_milestone_progress(self, milestones: List[ProjectMilestone]) -> Dict[str, Any]:
        """Analyze milestone progress and patterns"""
        
        if not milestones:
            return {
                'status': 'no_milestones',
                'message': 'No milestones found'
            }
        
        # Calculate milestone statistics
        total_milestones = len(milestones)
        completed_milestones = [m for m in milestones if m.is_completed]
        overdue_milestones = [m for m in milestones if m.is_overdue]
        
        completion_rate = len(completed_milestones) / total_milestones
        
        # Analyze milestone timing
        timing_analysis = self._analyze_milestone_timing(milestones)
        
        # Calculate milestone velocity
        milestone_velocity = self._calculate_milestone_velocity(completed_milestones)
        
        return {
            'total_milestones': total_milestones,
            'completed_milestones': len(completed_milestones),
            'overdue_milestones': len(overdue_milestones),
            'completion_rate': completion_rate,
            'timing_analysis': timing_analysis,
            'milestone_velocity': milestone_velocity,
            'next_milestone': self._get_next_milestone(milestones)
        }
    
    def _assess_project_risks(
        self,
        timeline: ProjectTimeline,
        snapshots: List[ProgressSnapshot],
        milestones: List[ProjectMilestone]
    ) -> Dict[str, Any]:
        """Assess project risks based on current status"""
        
        risks = []
        risk_score = 0.0
        
        # Schedule risk
        if timeline.planned_end_date:
            days_to_deadline = (timeline.planned_end_date - datetime.now(timezone.utc)).days
            if days_to_deadline < 0:
                risks.append({
                    'type': 'schedule_overrun',
                    'severity': 'high',
                    'description': f'Project is {abs(days_to_deadline)} days overdue',
                    'impact': 0.8
                })
                risk_score += 0.8
            elif days_to_deadline < 7 and timeline.progress_percentage < 90:
                risks.append({
                    'type': 'deadline_risk',
                    'severity': 'medium',
                    'description': f'Only {days_to_deadline} days remaining with {timeline.progress_percentage:.1f}% complete',
                    'impact': 0.6
                })
                risk_score += 0.6
        
        # Velocity risk
        if len(snapshots) >= 5:
            recent_velocities = [s.velocity for s in snapshots[:5] if s.velocity is not None]
            if recent_velocities:
                avg_velocity = mean(recent_velocities)
                if avg_velocity < 0.5:
                    risks.append({
                        'type': 'low_velocity',
                        'severity': 'medium',
                        'description': f'Current velocity is low ({avg_velocity:.2f} tasks/day)',
                        'impact': 0.4
                    })
                    risk_score += 0.4
        
        # Milestone risk
        overdue_milestones = [m for m in milestones if m.is_overdue]
        if overdue_milestones:
            risks.append({
                'type': 'milestone_delays',
                'severity': 'medium',
                'description': f'{len(overdue_milestones)} milestones are overdue',
                'impact': 0.5
            })
            risk_score += 0.5
        
        # Progress stagnation risk
        if len(snapshots) >= 7:
            recent_progress = [s.progress_percentage for s in snapshots[:7]]
            if max(recent_progress) - min(recent_progress) < 5:
                risks.append({
                    'type': 'progress_stagnation',
                    'severity': 'medium',
                    'description': 'Progress has been stagnant for the past week',
                    'impact': 0.3
                })
                risk_score += 0.3
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        return {
            'overall_risk_score': risk_score,
            'risk_level': self._categorize_risk_level(risk_score),
            'identified_risks': risks,
            'mitigation_suggestions': self._generate_mitigation_suggestions(risks)
        }
    
    def _analyze_progress_trends(self, snapshots: List[ProgressSnapshot]) -> Optional[ProgressTrend]:
        """Analyze overall progress trends"""
        
        if len(snapshots) < 5:
            return None
        
        # Get progress values
        progress_values = [s.progress_percentage for s in reversed(snapshots)]
        
        # Calculate trend
        if len(progress_values) >= 10:
            recent_progress = progress_values[-5:]
            older_progress = progress_values[-10:-5]
            
            recent_avg = mean(recent_progress)
            older_avg = mean(older_progress)
            
            if recent_avg > older_avg + 5:
                trend_type = 'improving'
                confidence = min(0.9, (recent_avg - older_avg) / 20)
            elif recent_avg < older_avg - 5:
                trend_type = 'declining'
                confidence = min(0.9, (older_avg - recent_avg) / 20)
            else:
                trend_type = 'stable'
                confidence = 0.7
        else:
            # Simple trend calculation
            first_half = progress_values[:len(progress_values)//2]
            second_half = progress_values[len(progress_values)//2:]
            
            first_avg = mean(first_half)
            second_avg = mean(second_half)
            
            if second_avg > first_avg + 3:
                trend_type = 'improving'
                confidence = 0.6
            elif second_avg < first_avg - 3:
                trend_type = 'declining'
                confidence = 0.6
            else:
                trend_type = 'stable'
                confidence = 0.7
        
        # Calculate volatility
        if len(progress_values) > 1:
            volatility = stdev(progress_values)
            if volatility > 10:
                trend_type = 'volatile'
                confidence = 0.5
        
        description = f"Progress trend is {trend_type}"
        
        return ProgressTrend(
            trend_type=trend_type,
            confidence=confidence,
            description=description,
            data={
                'recent_progress': progress_values[-5:] if len(progress_values) >= 5 else progress_values,
                'volatility': volatility if len(progress_values) > 1 else 0,
                'trend_strength': confidence
            }
        )
    
    def _generate_recommendations(
        self,
        timeline: ProjectTimeline,
        snapshots: List[ProgressSnapshot],
        milestones: List[ProjectMilestone]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Progress-based recommendations
        if timeline.progress_percentage < 30 and timeline.started_at:
            days_since_start = (datetime.now(timezone.utc) - timeline.started_at).days
            if days_since_start > 30:
                recommendations.append({
                    'type': 'progress_concern',
                    'priority': 'high',
                    'message': 'Project progress is slower than expected',
                    'action': 'Review project scope and resource allocation'
                })
        
        # Velocity-based recommendations
        if len(snapshots) >= 5:
            recent_velocities = [s.velocity for s in snapshots[:5] if s.velocity is not None]
            if recent_velocities and mean(recent_velocities) < 0.5:
                recommendations.append({
                    'type': 'velocity_improvement',
                    'priority': 'medium',
                    'message': 'Current velocity is below optimal levels',
                    'action': 'Consider increasing team size or removing blockers'
                })
        
        # Milestone-based recommendations
        overdue_milestones = [m for m in milestones if m.is_overdue]
        if overdue_milestones:
            recommendations.append({
                'type': 'milestone_attention',
                'priority': 'high',
                'message': f'{len(overdue_milestones)} milestones are overdue',
                'action': 'Review and update milestone deadlines or increase focus'
            })
        
        # Schedule-based recommendations
        if timeline.planned_end_date:
            days_remaining = (timeline.planned_end_date - datetime.now(timezone.utc)).days
            if days_remaining < 14 and timeline.progress_percentage < 85:
                recommendations.append({
                    'type': 'schedule_risk',
                    'priority': 'high',
                    'message': 'Project may miss deadline based on current progress',
                    'action': 'Consider scope reduction or deadline extension'
                })
        
        return recommendations
    
    # Helper methods
    
    def _calculate_velocity_consistency(self, velocities: List[float]) -> float:
        """Calculate how consistent the velocity is (0-1, higher is more consistent)"""
        
        if len(velocities) < 2:
            return 0.0
        
        avg_velocity = mean(velocities)
        if avg_velocity == 0:
            return 0.0
        
        std_dev = stdev(velocities)
        coefficient_of_variation = std_dev / avg_velocity
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency = max(0.0, 1.0 - coefficient_of_variation)
        return consistency
    
    def _linear_regression_prediction(self, snapshots: List[ProgressSnapshot]) -> Optional[Dict[str, Any]]:
        """Simple linear regression prediction"""
        
        if len(snapshots) < 5:
            return None
        
        # Convert to time series data
        x_values = []
        y_values = []
        
        base_time = snapshots[-1].snapshot_date
        for snapshot in reversed(snapshots):
            days_from_base = (snapshot.snapshot_date - base_time).days
            x_values.append(days_from_base)
            y_values.append(snapshot.progress_percentage)
        
        # Simple linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict completion (when y = 100)
        if slope > 0:
            days_to_completion = (100 - intercept) / slope
            completion_date = datetime.now(timezone.utc) + timedelta(days=days_to_completion)
            
            return {
                'method': 'linear_regression',
                'completion_date': completion_date.isoformat(),
                'days_remaining': days_to_completion,
                'confidence': 0.6
            }
        
        return None
    
    def _exponential_smoothing_prediction(self, snapshots: List[ProgressSnapshot]) -> Optional[Dict[str, Any]]:
        """Exponential smoothing prediction"""
        
        if len(snapshots) < 3:
            return None
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        smoothed_values = []
        
        progress_values = [s.progress_percentage for s in reversed(snapshots)]
        smoothed_values.append(progress_values[0])
        
        for i in range(1, len(progress_values)):
            smoothed_value = alpha * progress_values[i] + (1 - alpha) * smoothed_values[i-1]
            smoothed_values.append(smoothed_value)
        
        # Predict next few values
        current_value = smoothed_values[-1]
        if len(smoothed_values) >= 2:
            trend = smoothed_values[-1] - smoothed_values[-2]
            if trend > 0:
                days_to_completion = (100 - current_value) / trend
                completion_date = datetime.now(timezone.utc) + timedelta(days=days_to_completion)
                
                return {
                    'method': 'exponential_smoothing',
                    'completion_date': completion_date.isoformat(),
                    'days_remaining': days_to_completion,
                    'confidence': 0.5
                }
        
        return None
    
    def _calculate_consensus_prediction(self, predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calculate consensus prediction from multiple methods"""
        
        if not predictions:
            return None
        
        # Weight predictions by confidence
        total_weight = sum(p['confidence'] for p in predictions)
        if total_weight == 0:
            return None
        
        weighted_days = sum(p['days_remaining'] * p['confidence'] for p in predictions) / total_weight
        avg_confidence = mean(p['confidence'] for p in predictions)
        
        consensus_date = datetime.now(timezone.utc) + timedelta(days=weighted_days)
        
        return {
            'completion_date': consensus_date.isoformat(),
            'days_remaining': weighted_days,
            'confidence': avg_confidence,
            'method': 'consensus'
        }
    
    def _analyze_milestone_timing(self, milestones: List[ProjectMilestone]) -> Dict[str, Any]:
        """Analyze milestone timing patterns"""
        
        completed_milestones = [m for m in milestones if m.is_completed and m.planned_date and m.actual_date]
        
        if not completed_milestones:
            return {'status': 'no_completed_milestones'}
        
        # Calculate timing variances
        variances = []
        for milestone in completed_milestones:
            variance = (milestone.actual_date - milestone.planned_date).days
            variances.append(variance)
        
        avg_variance = mean(variances)
        
        return {
            'average_variance_days': avg_variance,
            'early_completions': len([v for v in variances if v < 0]),
            'late_completions': len([v for v in variances if v > 0]),
            'on_time_completions': len([v for v in variances if v == 0]),
            'worst_delay': max(variances) if variances else 0
        }
    
    def _calculate_milestone_velocity(self, completed_milestones: List[ProjectMilestone]) -> float:
        """Calculate milestone completion velocity"""
        
        if len(completed_milestones) < 2:
            return 0.0
        
        # Sort by completion date
        sorted_milestones = sorted(completed_milestones, key=lambda m: m.actual_date)
        
        # Calculate average time between milestone completions
        intervals = []
        for i in range(1, len(sorted_milestones)):
            interval = (sorted_milestones[i].actual_date - sorted_milestones[i-1].actual_date).days
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        avg_interval = mean(intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    def _get_next_milestone(self, milestones: List[ProjectMilestone]) -> Optional[Dict[str, Any]]:
        """Get the next upcoming milestone"""
        
        upcoming_milestones = [
            m for m in milestones 
            if not m.is_completed and m.planned_date and m.planned_date > datetime.now(timezone.utc)
        ]
        
        if not upcoming_milestones:
            return None
        
        next_milestone = min(upcoming_milestones, key=lambda m: m.planned_date)
        
        return {
            'id': str(next_milestone.id),
            'name': next_milestone.name,
            'planned_date': next_milestone.planned_date.isoformat(),
            'days_until_due': next_milestone.days_until_due,
            'progress_percentage': next_milestone.progress_percentage
        }
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level based on score"""
        
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _generate_mitigation_suggestions(self, risks: List[Dict[str, Any]]) -> List[str]:
        """Generate mitigation suggestions based on identified risks"""
        
        suggestions = []
        
        for risk in risks:
            risk_type = risk['type']
            
            if risk_type == 'schedule_overrun':
                suggestions.append("Consider extending deadline or reducing scope")
            elif risk_type == 'deadline_risk':
                suggestions.append("Increase team resources or focus on critical path")
            elif risk_type == 'low_velocity':
                suggestions.append("Identify and remove blockers, consider process improvements")
            elif risk_type == 'milestone_delays':
                suggestions.append("Review milestone dependencies and resource allocation")
            elif risk_type == 'progress_stagnation':
                suggestions.append("Conduct team retrospective and identify improvement areas")
        
        return suggestions