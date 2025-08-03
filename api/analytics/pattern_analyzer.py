"""
Advanced Pattern Analytics and Visualization.

This module provides comprehensive pattern analysis capabilities:
- Pattern trend analysis
- Pattern correlation detection
- Impact assessment
- Predictive pattern modeling
- Pattern lifecycle management
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, and_, or_

from ..models.base import get_db_context
from ..models.memory import MemoryItem
from ..models.session import Session
from ..models.error_pattern import ErrorOccurrence
from ..models.workflow import WorkflowPattern
from ..ml.pattern_recognition import PatternMatch, PatternType, PatternCategory
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService
from shared.logging import setup_logging

logger = setup_logging("pattern_analyzer")


class TrendDirection(str, Enum):
    """Pattern trend directions."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class PatternPhase(str, Enum):
    """Pattern lifecycle phases."""
    EMERGING = "emerging"
    GROWING = "growing"
    MATURE = "mature"
    DECLINING = "declining"
    OBSOLETE = "obsolete"


@dataclass
class PatternTrend:
    """Pattern trend analysis result."""
    pattern_id: str
    pattern_type: str
    trend_direction: TrendDirection
    confidence: float
    slope: float
    volatility: float
    data_points: List[Dict[str, Any]]
    time_window: str
    prediction_next_period: Optional[float] = None


@dataclass
class PatternCorrelation:
    """Pattern correlation analysis result."""
    pattern_a_id: str
    pattern_b_id: str
    correlation_coefficient: float
    correlation_type: str  # positive, negative, complex
    significance: float
    shared_contexts: List[str]
    potential_causality: Optional[str] = None


@dataclass
class PatternImpact:
    """Pattern impact assessment."""
    pattern_id: str
    impact_score: float
    affected_systems: List[str]
    affected_users: int
    business_impact: Dict[str, float]
    technical_impact: Dict[str, float]
    time_to_impact: Optional[float] = None  # hours


@dataclass
class PatternLifecycle:
    """Pattern lifecycle analysis."""
    pattern_id: str
    current_phase: PatternPhase
    phase_duration: float  # days
    expected_next_phase: Optional[PatternPhase]
    time_to_next_phase: Optional[float]  # days
    lifecycle_confidence: float


class PatternAnalyzer:
    """
    Advanced pattern analyzer for trend analysis and impact assessment.
    
    Features:
    - Multi-dimensional trend analysis
    - Pattern correlation detection
    - Impact assessment and prediction
    - Lifecycle management
    - Anomaly detection in patterns
    """
    
    def __init__(self):
        self.analytics_service = TimeSeriesAnalyticsService()
        
        # Analysis configuration
        self.config = {
            "trend_window_days": 30,
            "correlation_threshold": 0.5,
            "significance_level": 0.05,
            "min_data_points": 10,
            "volatility_threshold": 0.3
        }
        
        # Impact scoring weights
        self.impact_weights = {
            "frequency": 0.3,
            "severity": 0.4,
            "scope": 0.2,
            "trend": 0.1
        }
        
        logger.info("Initialized PatternAnalyzer")
    
    async def analyze_pattern_trends(
        self,
        pattern_ids: Optional[List[str]] = None,
        pattern_types: Optional[List[PatternType]] = None,
        time_window_days: int = 30
    ) -> List[PatternTrend]:
        """
        Analyze trends in pattern occurrence and characteristics.
        
        Args:
            pattern_ids: Specific pattern IDs to analyze
            pattern_types: Types of patterns to analyze
            time_window_days: Time window for analysis
            
        Returns:
            List of pattern trend analyses
        """
        try:
            trends = []
            
            # Get pattern data
            pattern_data = await self._get_pattern_time_series(
                pattern_ids, pattern_types, time_window_days
            )
            
            for pattern_id, time_series in pattern_data.items():
                if len(time_series) < self.config["min_data_points"]:
                    continue
                
                # Analyze trend
                trend = self._analyze_single_pattern_trend(
                    pattern_id, time_series, time_window_days
                )
                
                if trend:
                    trends.append(trend)
            
            logger.info(f"Analyzed trends for {len(trends)} patterns")
            return trends
            
        except Exception as e:
            logger.error(f"Failed to analyze pattern trends: {e}")
            return []
    
    async def detect_pattern_correlations(
        self,
        pattern_ids: Optional[List[str]] = None,
        min_correlation: float = 0.5,
        time_window_days: int = 30
    ) -> List[PatternCorrelation]:
        """
        Detect correlations between different patterns.
        
        Args:
            pattern_ids: Specific pattern IDs to analyze
            min_correlation: Minimum correlation threshold
            time_window_days: Time window for analysis
            
        Returns:
            List of pattern correlations
        """
        try:
            correlations = []
            
            # Get pattern data
            pattern_data = await self._get_pattern_time_series(
                pattern_ids, None, time_window_days
            )
            
            # Calculate pairwise correlations
            pattern_list = list(pattern_data.keys())
            
            for i in range(len(pattern_list)):
                for j in range(i + 1, len(pattern_list)):
                    pattern_a = pattern_list[i]
                    pattern_b = pattern_list[j]
                    
                    correlation = await self._calculate_pattern_correlation(
                        pattern_a, pattern_b, 
                        pattern_data[pattern_a], pattern_data[pattern_b]
                    )
                    
                    if correlation and abs(correlation.correlation_coefficient) >= min_correlation:
                        correlations.append(correlation)
            
            # Sort by correlation strength
            correlations.sort(
                key=lambda x: abs(x.correlation_coefficient), 
                reverse=True
            )
            
            logger.info(f"Detected {len(correlations)} pattern correlations")
            return correlations
            
        except Exception as e:
            logger.error(f"Failed to detect pattern correlations: {e}")
            return []
    
    async def assess_pattern_impacts(
        self,
        pattern_ids: Optional[List[str]] = None,
        include_predictions: bool = True
    ) -> List[PatternImpact]:
        """
        Assess the impact of patterns on system and users.
        
        Args:
            pattern_ids: Specific pattern IDs to analyze
            include_predictions: Include impact predictions
            
        Returns:
            List of pattern impact assessments
        """
        try:
            impacts = []
            
            # Get patterns to analyze
            patterns_to_analyze = await self._get_patterns_for_impact_analysis(pattern_ids)
            
            for pattern_data in patterns_to_analyze:
                impact = await self._assess_single_pattern_impact(
                    pattern_data, include_predictions
                )
                
                if impact:
                    impacts.append(impact)
            
            # Sort by impact score
            impacts.sort(key=lambda x: x.impact_score, reverse=True)
            
            logger.info(f"Assessed impact for {len(impacts)} patterns")
            return impacts
            
        except Exception as e:
            logger.error(f"Failed to assess pattern impacts: {e}")
            return []
    
    async def analyze_pattern_lifecycles(
        self,
        pattern_ids: Optional[List[str]] = None
    ) -> List[PatternLifecycle]:
        """
        Analyze pattern lifecycles and predict phase transitions.
        
        Args:
            pattern_ids: Specific pattern IDs to analyze
            
        Returns:
            List of pattern lifecycle analyses
        """
        try:
            lifecycles = []
            
            # Get pattern history
            pattern_histories = await self._get_pattern_histories(pattern_ids)
            
            for pattern_id, history in pattern_histories.items():
                lifecycle = self._analyze_pattern_lifecycle(pattern_id, history)
                
                if lifecycle:
                    lifecycles.append(lifecycle)
            
            logger.info(f"Analyzed lifecycle for {len(lifecycles)} patterns")
            return lifecycles
            
        except Exception as e:
            logger.error(f"Failed to analyze pattern lifecycles: {e}")
            return []
    
    async def predict_pattern_evolution(
        self,
        pattern_id: str,
        prediction_horizon_days: int = 30
    ) -> Dict[str, Any]:
        """
        Predict how a pattern will evolve over time.
        
        Args:
            pattern_id: Pattern ID to predict
            prediction_horizon_days: Prediction time horizon
            
        Returns:
            Pattern evolution prediction
        """
        try:
            # Get pattern history
            history = await self._get_single_pattern_history(pattern_id)
            
            if not history:
                return {"error": "No history available for pattern"}
            
            # Analyze current trend
            trend = self._analyze_single_pattern_trend(
                pattern_id, history, len(history)
            )
            
            if not trend:
                return {"error": "Unable to analyze pattern trend"}
            
            # Make predictions
            predictions = self._predict_pattern_future(
                history, trend, prediction_horizon_days
            )
            
            # Assess prediction confidence
            confidence = self._calculate_prediction_confidence(history, trend)
            
            return {
                "pattern_id": pattern_id,
                "current_trend": {
                    "direction": trend.trend_direction.value,
                    "slope": trend.slope,
                    "volatility": trend.volatility
                },
                "predictions": predictions,
                "confidence": confidence,
                "prediction_horizon_days": prediction_horizon_days,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict pattern evolution: {e}")
            return {"error": str(e)}
    
    async def detect_pattern_anomalies(
        self,
        time_window_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in pattern behavior.
        
        Args:
            time_window_days: Time window for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []
            
            # Get recent pattern data
            pattern_data = await self._get_pattern_time_series(
                None, None, time_window_days
            )
            
            for pattern_id, time_series in pattern_data.items():
                # Detect anomalies in this pattern
                pattern_anomalies = self._detect_single_pattern_anomalies(
                    pattern_id, time_series
                )
                
                anomalies.extend(pattern_anomalies)
            
            # Sort by anomaly score
            anomalies.sort(key=lambda x: x.get("anomaly_score", 0), reverse=True)
            
            logger.info(f"Detected {len(anomalies)} pattern anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Failed to detect pattern anomalies: {e}")
            return []
    
    async def generate_pattern_insights(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate comprehensive pattern insights.
        
        Args:
            user_id: Filter by user ID
            project_id: Filter by project ID
            time_window_days: Time window for analysis
            
        Returns:
            Comprehensive pattern insights
        """
        try:
            # Get various analyses
            trends = await self.analyze_pattern_trends(
                time_window_days=time_window_days
            )
            
            correlations = await self.detect_pattern_correlations(
                time_window_days=time_window_days
            )
            
            impacts = await self.assess_pattern_impacts()
            
            anomalies = await self.detect_pattern_anomalies(
                time_window_days=min(time_window_days, 7)
            )
            
            # Generate insights
            insights = {
                "summary": self._generate_insights_summary(
                    trends, correlations, impacts, anomalies
                ),
                "trends": {
                    "total_patterns_analyzed": len(trends),
                    "increasing_patterns": len([t for t in trends if t.trend_direction == TrendDirection.INCREASING]),
                    "decreasing_patterns": len([t for t in trends if t.trend_direction == TrendDirection.DECREASING]),
                    "stable_patterns": len([t for t in trends if t.trend_direction == TrendDirection.STABLE]),
                    "volatile_patterns": len([t for t in trends if t.trend_direction == TrendDirection.VOLATILE])
                },
                "correlations": {
                    "total_correlations": len(correlations),
                    "strong_positive": len([c for c in correlations if c.correlation_coefficient > 0.7]),
                    "strong_negative": len([c for c in correlations if c.correlation_coefficient < -0.7]),
                    "most_correlated_patterns": correlations[:5] if correlations else []
                },
                "impacts": {
                    "high_impact_patterns": len([i for i in impacts if i.impact_score > 0.8]),
                    "medium_impact_patterns": len([i for i in impacts if 0.4 < i.impact_score <= 0.8]),
                    "low_impact_patterns": len([i for i in impacts if i.impact_score <= 0.4]),
                    "total_affected_users": sum(i.affected_users for i in impacts)
                },
                "anomalies": {
                    "total_anomalies": len(anomalies),
                    "critical_anomalies": len([a for a in anomalies if a.get("anomaly_score", 0) > 0.8]),
                    "recent_anomalies": anomalies[:10] if anomalies else []
                },
                "recommendations": self._generate_pattern_recommendations(
                    trends, correlations, impacts, anomalies
                ),
                "analysis_metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "time_window_days": time_window_days,
                    "filters": {
                        "user_id": user_id,
                        "project_id": project_id
                    }
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate pattern insights: {e}")
            return {"error": str(e)}
    
    # Internal analysis methods
    
    async def _get_pattern_time_series(
        self,
        pattern_ids: Optional[List[str]],
        pattern_types: Optional[List[PatternType]],
        time_window_days: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get time series data for patterns."""
        try:
            pattern_data = {}
            
            # This would typically query pattern occurrence data
            # For now, simulate time series data
            
            # Generate some sample pattern IDs if none provided
            if not pattern_ids:
                pattern_ids = [f"pattern_{i}" for i in range(1, 11)]
            
            for pattern_id in pattern_ids:
                # Generate sample time series
                time_series = []
                start_date = datetime.utcnow() - timedelta(days=time_window_days)
                
                for day in range(time_window_days):
                    date = start_date + timedelta(days=day)
                    
                    # Simulate pattern occurrence with some trend
                    base_value = 10
                    trend_factor = day * 0.1  # Slight upward trend
                    noise = np.random.normal(0, 2)
                    value = max(0, base_value + trend_factor + noise)
                    
                    time_series.append({
                        "timestamp": date,
                        "value": value,
                        "occurrences": int(value),
                        "confidence": 0.7 + np.random.random() * 0.3
                    })
                
                pattern_data[pattern_id] = time_series
            
            return pattern_data
            
        except Exception as e:
            logger.error(f"Failed to get pattern time series: {e}")
            return {}
    
    def _analyze_single_pattern_trend(
        self,
        pattern_id: str,
        time_series: List[Dict[str, Any]],
        time_window_days: int
    ) -> Optional[PatternTrend]:
        """Analyze trend for a single pattern."""
        try:
            if len(time_series) < 3:
                return None
            
            # Extract values and timestamps
            values = [point["value"] for point in time_series]
            timestamps = [point["timestamp"] for point in time_series]
            
            # Calculate trend using linear regression
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Calculate volatility (standard deviation of residuals)
            predicted_values = slope * x + intercept
            residuals = values - predicted_values
            volatility = np.std(residuals) / np.mean(values) if np.mean(values) > 0 else 0
            
            # Determine trend direction
            if abs(slope) < 0.1:
                if volatility > self.config["volatility_threshold"]:
                    direction = TrendDirection.VOLATILE
                else:
                    direction = TrendDirection.STABLE
            elif slope > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING
            
            # Calculate confidence based on R-squared
            correlation = np.corrcoef(x, values)[0, 1]
            confidence = correlation ** 2 if not np.isnan(correlation) else 0
            
            # Predict next period
            next_value = slope * len(values) + intercept
            
            # Get pattern type (simplified)
            pattern_type = "unknown"  # Would extract from pattern data
            
            return PatternTrend(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                trend_direction=direction,
                confidence=confidence,
                slope=slope,
                volatility=volatility,
                data_points=time_series,
                time_window=f"{time_window_days}d",
                prediction_next_period=next_value
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze trend for pattern {pattern_id}: {e}")
            return None
    
    async def _calculate_pattern_correlation(
        self,
        pattern_a: str,
        pattern_b: str,
        series_a: List[Dict[str, Any]],
        series_b: List[Dict[str, Any]]
    ) -> Optional[PatternCorrelation]:
        """Calculate correlation between two patterns."""
        try:
            # Align time series
            min_length = min(len(series_a), len(series_b))
            values_a = [point["value"] for point in series_a[:min_length]]
            values_b = [point["value"] for point in series_b[:min_length]]
            
            if min_length < 3:
                return None
            
            # Calculate correlation coefficient
            correlation_matrix = np.corrcoef(values_a, values_b)
            correlation_coefficient = correlation_matrix[0, 1]
            
            if np.isnan(correlation_coefficient):
                return None
            
            # Determine correlation type
            if correlation_coefficient > 0.5:
                correlation_type = "positive"
            elif correlation_coefficient < -0.5:
                correlation_type = "negative"
            else:
                correlation_type = "weak"
            
            # Calculate significance (simplified)
            significance = abs(correlation_coefficient)
            
            # Find shared contexts (simplified)
            shared_contexts = ["temporal"]  # Would analyze actual contexts
            
            return PatternCorrelation(
                pattern_a_id=pattern_a,
                pattern_b_id=pattern_b,
                correlation_coefficient=correlation_coefficient,
                correlation_type=correlation_type,
                significance=significance,
                shared_contexts=shared_contexts,
                potential_causality=None  # Would require more analysis
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation: {e}")
            return None
    
    async def _get_patterns_for_impact_analysis(
        self,
        pattern_ids: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Get patterns for impact analysis."""
        try:
            patterns = []
            
            # This would typically query from database
            # For now, return sample pattern data
            
            sample_patterns = [
                {
                    "id": "pattern_1",
                    "type": "error_sequence",
                    "frequency": 25,
                    "severity": "high",
                    "affected_users": 150,
                    "affected_systems": ["api", "database"],
                    "recent_occurrences": 5
                },
                {
                    "id": "pattern_2",
                    "type": "performance_bottleneck",
                    "frequency": 10,
                    "severity": "medium",
                    "affected_users": 300,
                    "affected_systems": ["frontend", "api"],
                    "recent_occurrences": 3
                },
                {
                    "id": "pattern_3",
                    "type": "usage_behavior",
                    "frequency": 50,
                    "severity": "low",
                    "affected_users": 1000,
                    "affected_systems": ["ui"],
                    "recent_occurrences": 12
                }
            ]
            
            if pattern_ids:
                patterns = [p for p in sample_patterns if p["id"] in pattern_ids]
            else:
                patterns = sample_patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get patterns for impact analysis: {e}")
            return []
    
    async def _assess_single_pattern_impact(
        self,
        pattern_data: Dict[str, Any],
        include_predictions: bool
    ) -> Optional[PatternImpact]:
        """Assess impact of a single pattern."""
        try:
            pattern_id = pattern_data["id"]
            
            # Calculate impact score using weighted factors
            frequency_score = min(1.0, pattern_data["frequency"] / 100)
            
            severity_scores = {"low": 0.3, "medium": 0.6, "high": 1.0}
            severity_score = severity_scores.get(pattern_data["severity"], 0.5)
            
            scope_score = min(1.0, pattern_data["affected_users"] / 1000)
            
            # Calculate trend score (simplified)
            trend_score = min(1.0, pattern_data["recent_occurrences"] / 10)
            
            # Weighted impact score
            impact_score = (
                self.impact_weights["frequency"] * frequency_score +
                self.impact_weights["severity"] * severity_score +
                self.impact_weights["scope"] * scope_score +
                self.impact_weights["trend"] * trend_score
            )
            
            # Business and technical impact
            business_impact = {
                "user_satisfaction": -impact_score * 0.8,
                "productivity": -impact_score * 0.6,
                "revenue": -impact_score * 0.4
            }
            
            technical_impact = {
                "performance": -impact_score * 0.7,
                "reliability": -impact_score * 0.9,
                "maintainability": -impact_score * 0.5
            }
            
            # Time to impact (for predictions)
            time_to_impact = None
            if include_predictions and pattern_data["recent_occurrences"] > 0:
                # Simple prediction based on frequency
                hours_between_occurrences = 168 / pattern_data["recent_occurrences"]  # Weekly basis
                time_to_impact = hours_between_occurrences
            
            return PatternImpact(
                pattern_id=pattern_id,
                impact_score=impact_score,
                affected_systems=pattern_data["affected_systems"],
                affected_users=pattern_data["affected_users"],
                business_impact=business_impact,
                technical_impact=technical_impact,
                time_to_impact=time_to_impact
            )
            
        except Exception as e:
            logger.error(f"Failed to assess pattern impact: {e}")
            return None
    
    async def _get_pattern_histories(
        self,
        pattern_ids: Optional[List[str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get pattern histories for lifecycle analysis."""
        try:
            histories = {}
            
            # This would typically query pattern lifecycle data
            # For now, simulate lifecycle data
            
            sample_pattern_ids = pattern_ids or ["pattern_1", "pattern_2", "pattern_3"]
            
            for pattern_id in sample_pattern_ids:
                # Generate sample lifecycle history
                history = []
                start_date = datetime.utcnow() - timedelta(days=90)
                
                phases = [
                    ("emerging", 20),
                    ("growing", 30),
                    ("mature", 25),
                    ("declining", 15)
                ]
                
                current_date = start_date
                for phase, duration in phases:
                    for day in range(duration):
                        history.append({
                            "date": current_date + timedelta(days=day),
                            "phase": phase,
                            "occurrences": max(1, int(np.random.poisson(5))),
                            "confidence": 0.6 + np.random.random() * 0.4
                        })
                    current_date += timedelta(days=duration)
                
                histories[pattern_id] = history
            
            return histories
            
        except Exception as e:
            logger.error(f"Failed to get pattern histories: {e}")
            return {}
    
    def _analyze_pattern_lifecycle(
        self,
        pattern_id: str,
        history: List[Dict[str, Any]]
    ) -> Optional[PatternLifecycle]:
        """Analyze lifecycle of a single pattern."""
        try:
            if not history:
                return None
            
            # Get current phase
            current_phase_name = history[-1]["phase"]
            current_phase = PatternPhase(current_phase_name)
            
            # Calculate phase duration
            phase_start = None
            for entry in reversed(history):
                if entry["phase"] != current_phase_name:
                    break
                phase_start = entry["date"]
            
            if phase_start:
                phase_duration = (datetime.utcnow() - phase_start).days
            else:
                phase_duration = len(history)
            
            # Predict next phase
            phase_sequence = [
                PatternPhase.EMERGING,
                PatternPhase.GROWING,
                PatternPhase.MATURE,
                PatternPhase.DECLINING,
                PatternPhase.OBSOLETE
            ]
            
            current_index = phase_sequence.index(current_phase)
            expected_next_phase = None
            time_to_next_phase = None
            
            if current_index < len(phase_sequence) - 1:
                expected_next_phase = phase_sequence[current_index + 1]
                
                # Estimate time to next phase based on historical transitions
                avg_phase_duration = len(history) / 4  # Assume 4 phases so far
                time_to_next_phase = max(0, avg_phase_duration - phase_duration)
            
            # Calculate confidence based on pattern stability
            recent_entries = history[-10:] if len(history) >= 10 else history
            phase_consistency = sum(
                1 for entry in recent_entries 
                if entry["phase"] == current_phase_name
            ) / len(recent_entries)
            
            lifecycle_confidence = phase_consistency
            
            return PatternLifecycle(
                pattern_id=pattern_id,
                current_phase=current_phase,
                phase_duration=phase_duration,
                expected_next_phase=expected_next_phase,
                time_to_next_phase=time_to_next_phase,
                lifecycle_confidence=lifecycle_confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze pattern lifecycle: {e}")
            return None
    
    async def _get_single_pattern_history(
        self,
        pattern_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get history for a single pattern."""
        try:
            histories = await self._get_pattern_histories([pattern_id])
            return histories.get(pattern_id)
            
        except Exception as e:
            logger.error(f"Failed to get pattern history: {e}")
            return None
    
    def _predict_pattern_future(
        self,
        history: List[Dict[str, Any]],
        trend: PatternTrend,
        prediction_days: int
    ) -> List[Dict[str, Any]]:
        """Predict future pattern behavior."""
        try:
            predictions = []
            
            # Use trend information for prediction
            last_value = history[-1]["value"]
            
            for day in range(1, prediction_days + 1):
                # Simple linear prediction with noise
                predicted_value = last_value + (trend.slope * day)
                
                # Add uncertainty
                uncertainty = trend.volatility * np.sqrt(day)  # Uncertainty grows with time
                
                prediction_date = history[-1]["timestamp"] + timedelta(days=day)
                
                predictions.append({
                    "date": prediction_date.isoformat(),
                    "predicted_value": max(0, predicted_value),
                    "confidence_interval": [
                        max(0, predicted_value - uncertainty),
                        predicted_value + uncertainty
                    ],
                    "uncertainty": uncertainty
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict pattern future: {e}")
            return []
    
    def _calculate_prediction_confidence(
        self,
        history: List[Dict[str, Any]],
        trend: PatternTrend
    ) -> float:
        """Calculate confidence in predictions."""
        try:
            # Base confidence on trend confidence and data quality
            base_confidence = trend.confidence
            
            # Adjust for data quantity
            data_quality_factor = min(1.0, len(history) / 30)  # More data = higher confidence
            
            # Adjust for trend stability
            stability_factor = 1.0 - min(1.0, trend.volatility)
            
            # Combined confidence
            confidence = base_confidence * data_quality_factor * stability_factor
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate prediction confidence: {e}")
            return 0.5
    
    def _detect_single_pattern_anomalies(
        self,
        pattern_id: str,
        time_series: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in a single pattern."""
        try:
            anomalies = []
            
            if len(time_series) < 7:
                return anomalies
            
            values = [point["value"] for point in time_series]
            
            # Calculate statistics
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            if std_value == 0:
                return anomalies
            
            # Detect outliers using z-score
            z_threshold = 2.5
            
            for i, point in enumerate(time_series):
                z_score = abs((point["value"] - mean_value) / std_value)
                
                if z_score > z_threshold:
                    anomaly_score = min(1.0, z_score / 5.0)  # Normalize to 0-1
                    
                    anomalies.append({
                        "pattern_id": pattern_id,
                        "timestamp": point["timestamp"].isoformat(),
                        "value": point["value"],
                        "expected_range": [
                            mean_value - 2 * std_value,
                            mean_value + 2 * std_value
                        ],
                        "anomaly_score": anomaly_score,
                        "z_score": z_score,
                        "type": "statistical_outlier"
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies for pattern {pattern_id}: {e}")
            return []
    
    def _generate_insights_summary(
        self,
        trends: List[PatternTrend],
        correlations: List[PatternCorrelation],
        impacts: List[PatternImpact],
        anomalies: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate summary insights."""
        try:
            summary = {}
            
            # Trend insights
            if trends:
                increasing_count = len([t for t in trends if t.trend_direction == TrendDirection.INCREASING])
                if increasing_count > len(trends) * 0.3:
                    summary["trend_alert"] = f"{increasing_count} patterns showing increasing trend - monitor for potential issues"
            
            # Correlation insights
            if correlations:
                strong_correlations = [c for c in correlations if abs(c.correlation_coefficient) > 0.8]
                if strong_correlations:
                    summary["correlation_insight"] = f"Found {len(strong_correlations)} strong pattern correlations - investigate causal relationships"
            
            # Impact insights
            if impacts:
                high_impact = [i for i in impacts if i.impact_score > 0.8]
                if high_impact:
                    summary["impact_warning"] = f"{len(high_impact)} patterns have high impact - prioritize resolution"
            
            # Anomaly insights
            if anomalies:
                critical_anomalies = [a for a in anomalies if a.get("anomaly_score", 0) > 0.8]
                if critical_anomalies:
                    summary["anomaly_alert"] = f"{len(critical_anomalies)} critical pattern anomalies detected - investigate immediately"
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate insights summary: {e}")
            return {}
    
    def _generate_pattern_recommendations(
        self,
        trends: List[PatternTrend],
        correlations: List[PatternCorrelation],
        impacts: List[PatternImpact],
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate pattern-based recommendations."""
        try:
            recommendations = []
            
            # Trend-based recommendations
            increasing_patterns = [t for t in trends if t.trend_direction == TrendDirection.INCREASING]
            if increasing_patterns:
                high_confidence_increasing = [t for t in increasing_patterns if t.confidence > 0.7]
                if high_confidence_increasing:
                    recommendations.append({
                        "type": "trend_monitoring",
                        "priority": "medium",
                        "title": "Monitor Increasing Pattern Trends",
                        "description": f"{len(high_confidence_increasing)} patterns showing strong increasing trends",
                        "action": "Set up alerts and monitoring for these patterns to prevent issues"
                    })
            
            # Impact-based recommendations
            high_impact_patterns = [i for i in impacts if i.impact_score > 0.7]
            if high_impact_patterns:
                recommendations.append({
                    "type": "impact_mitigation",
                    "priority": "high",
                    "title": "Address High-Impact Patterns",
                    "description": f"{len(high_impact_patterns)} patterns have significant business impact",
                    "action": "Prioritize resolution of these patterns to minimize business disruption"
                })
            
            # Correlation-based recommendations
            strong_correlations = [c for c in correlations if abs(c.correlation_coefficient) > 0.8]
            if strong_correlations:
                recommendations.append({
                    "type": "correlation_investigation",
                    "priority": "medium",
                    "title": "Investigate Pattern Correlations",
                    "description": f"{len(strong_correlations)} strong pattern correlations found",
                    "action": "Analyze causal relationships to identify root causes"
                })
            
            # Anomaly-based recommendations
            if anomalies:
                critical_anomalies = [a for a in anomalies if a.get("anomaly_score", 0) > 0.8]
                if critical_anomalies:
                    recommendations.append({
                        "type": "anomaly_investigation",
                        "priority": "high",
                        "title": "Investigate Pattern Anomalies",
                        "description": f"{len(critical_anomalies)} critical pattern anomalies detected",
                        "action": "Immediate investigation required to understand anomaly causes"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []


# Global pattern analyzer instance
pattern_analyzer = PatternAnalyzer()