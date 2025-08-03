"""
Advanced Predictive Analytics Models for AI Features.

This module implements machine learning models for:
- Task prediction
- Performance forecasting
- Error likelihood estimation
- Resource usage prediction
- User behavior modeling
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

from shared.logging import setup_logging

logger = setup_logging("predictive_models")


@dataclass
class TaskPrediction:
    """Prediction for next likely tasks."""
    task_type: str
    task_description: str
    probability: float
    estimated_duration: float  # hours
    prerequisites: List[str]
    confidence: float


@dataclass
class PerformanceForecast:
    """Performance forecast for a specific metric."""
    metric_name: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    time_horizon: str  # "1h", "24h", "7d"
    factors: Dict[str, float]  # Contributing factors


@dataclass
class ErrorPrediction:
    """Prediction for potential errors."""
    error_type: str
    error_category: str
    likelihood: float
    time_window: str  # When error might occur
    prevention_steps: List[str]
    risk_factors: Dict[str, float]


@dataclass
class ResourcePrediction:
    """Resource usage prediction."""
    resource_type: str  # cpu, memory, storage, api_calls
    predicted_usage: float
    peak_time: datetime
    trend: str  # increasing, stable, decreasing
    recommendations: List[str]


@dataclass
class UserBehaviorPrediction:
    """User behavior prediction."""
    behavior_type: str
    likelihood: float
    next_actions: List[Dict[str, float]]  # action -> probability
    engagement_score: float
    churn_risk: float


class PredictiveModels:
    """
    Container for all predictive analytics models.
    
    Implements various ML models for different prediction tasks
    in the AI-enhanced development environment.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        self.model_metadata = {}
        
        # Model configurations
        self.configs = {
            "task_prediction": {
                "model_type": "gradient_boosting",
                "features": ["user_history", "project_context", "time_features", "session_state"],
                "target": "next_task_type"
            },
            "performance_forecast": {
                "model_type": "random_forest",
                "features": ["historical_metrics", "system_load", "code_complexity", "team_size"],
                "target": "performance_metric"
            },
            "error_likelihood": {
                "model_type": "gradient_boosting",
                "features": ["code_changes", "test_coverage", "complexity_metrics", "history"],
                "target": "error_occurred"
            },
            "resource_usage": {
                "model_type": "random_forest",
                "features": ["historical_usage", "active_users", "data_volume", "time_features"],
                "target": "resource_usage"
            },
            "user_behavior": {
                "model_type": "gradient_boosting",
                "features": ["interaction_history", "session_patterns", "project_activity", "time_since_last"],
                "target": "next_action"
            }
        }
        
        logger.info("Initialized PredictiveModels")
    
    def train_task_prediction_model(
        self,
        training_data: List[Dict[str, Any]],
        model_name: str = "task_prediction"
    ) -> Dict[str, Any]:
        """
        Train model to predict next likely tasks.
        
        Args:
            training_data: Historical task data
            model_name: Name for the model
            
        Returns:
            Training metrics and model info
        """
        try:
            logger.info(f"Training task prediction model with {len(training_data)} samples")
            
            # Extract features and targets
            X, y, feature_names = self._extract_task_features(training_data)
            
            if len(X) < 100:
                return {"error": "Insufficient training data"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Store model
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self.feature_extractors[model_name] = feature_names
            
            # Feature importance
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Metadata
            self.model_metadata[model_name] = {
                "trained_at": datetime.utcnow().isoformat(),
                "samples": len(training_data),
                "features": feature_names,
                "train_score": train_score,
                "test_score": test_score,
                "feature_importance": feature_importance
            }
            
            logger.info(f"Task prediction model trained: train={train_score:.3f}, test={test_score:.3f}")
            
            return {
                "model_name": model_name,
                "train_score": train_score,
                "test_score": test_score,
                "feature_importance": feature_importance,
                "samples_used": len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to train task prediction model: {e}")
            return {"error": str(e)}
    
    def predict_next_tasks(
        self,
        user_context: Dict[str, Any],
        n_predictions: int = 5
    ) -> List[TaskPrediction]:
        """
        Predict next likely tasks for a user.
        
        Args:
            user_context: Current user context
            n_predictions: Number of predictions to return
            
        Returns:
            List of task predictions
        """
        try:
            model_name = "task_prediction"
            
            if model_name not in self.models:
                logger.warning("Task prediction model not trained")
                return []
            
            # Extract features
            features = self._extract_single_task_features(user_context)
            
            # Scale features
            features_scaled = self.scalers[model_name].transform([features])
            
            # Get predictions
            probabilities = self.models[model_name].predict_proba(features_scaled)[0]
            classes = self.models[model_name].classes_
            
            # Sort by probability
            predictions = []
            for idx in np.argsort(probabilities)[::-1][:n_predictions]:
                task_type = classes[idx]
                
                prediction = TaskPrediction(
                    task_type=task_type,
                    task_description=self._generate_task_description(task_type, user_context),
                    probability=probabilities[idx],
                    estimated_duration=self._estimate_task_duration(task_type, user_context),
                    prerequisites=self._get_task_prerequisites(task_type, user_context),
                    confidence=min(0.95, probabilities[idx] * 1.2)  # Adjusted confidence
                )
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict next tasks: {e}")
            return []
    
    def train_performance_forecast_model(
        self,
        training_data: List[Dict[str, Any]],
        model_name: str = "performance_forecast"
    ) -> Dict[str, Any]:
        """
        Train model to forecast performance metrics.
        
        Args:
            training_data: Historical performance data
            model_name: Name for the model
            
        Returns:
            Training metrics
        """
        try:
            logger.info(f"Training performance forecast model with {len(training_data)} samples")
            
            # Extract features
            X, y, feature_names = self._extract_performance_features(training_data)
            
            if len(X) < 50:
                return {"error": "Insufficient training data"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Calculate MAE
            from sklearn.metrics import mean_absolute_error
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Store
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self.feature_extractors[model_name] = feature_names
            
            self.model_metadata[model_name] = {
                "trained_at": datetime.utcnow().isoformat(),
                "samples": len(training_data),
                "train_score": train_score,
                "test_score": test_score,
                "mae": mae
            }
            
            logger.info(f"Performance model trained: R²={test_score:.3f}, MAE={mae:.3f}")
            
            return {
                "model_name": model_name,
                "train_score": train_score,
                "test_score": test_score,
                "mae": mae,
                "samples_used": len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to train performance model: {e}")
            return {"error": str(e)}
    
    def forecast_performance(
        self,
        current_metrics: Dict[str, float],
        system_context: Dict[str, Any],
        time_horizon: str = "24h"
    ) -> PerformanceForecast:
        """
        Forecast performance metrics.
        
        Args:
            current_metrics: Current performance metrics
            system_context: System context
            time_horizon: Forecast horizon
            
        Returns:
            Performance forecast
        """
        try:
            model_name = "performance_forecast"
            
            if model_name not in self.models:
                logger.warning("Performance forecast model not trained")
                return PerformanceForecast(
                    metric_name="response_time",
                    predicted_value=0.0,
                    confidence_interval=(0.0, 0.0),
                    time_horizon=time_horizon,
                    factors={}
                )
            
            # Extract features
            features = self._extract_single_performance_features(
                current_metrics, system_context
            )
            
            # Scale
            features_scaled = self.scalers[model_name].transform([features])
            
            # Predict
            prediction = self.models[model_name].predict(features_scaled)[0]
            
            # Get prediction intervals using forest
            predictions = []
            for tree in self.models[model_name].estimators_:
                predictions.append(tree.predict(features_scaled)[0])
            
            std_dev = np.std(predictions)
            confidence_interval = (
                prediction - 1.96 * std_dev,
                prediction + 1.96 * std_dev
            )
            
            # Analyze factors
            feature_names = self.feature_extractors[model_name]
            feature_importance = self.models[model_name].feature_importances_
            
            factors = {}
            for name, importance, value in zip(feature_names, feature_importance, features):
                if importance > 0.05:  # Only significant factors
                    factors[name] = importance * value
            
            return PerformanceForecast(
                metric_name="response_time",
                predicted_value=prediction,
                confidence_interval=confidence_interval,
                time_horizon=time_horizon,
                factors=factors
            )
            
        except Exception as e:
            logger.error(f"Failed to forecast performance: {e}")
            raise
    
    def predict_error_likelihood(
        self,
        code_context: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> List[ErrorPrediction]:
        """
        Predict likelihood of errors occurring.
        
        Args:
            code_context: Current code context
            history: Recent error history
            
        Returns:
            List of error predictions
        """
        try:
            # Analyze code complexity
            complexity_score = self._calculate_complexity_score(code_context)
            
            # Analyze recent patterns
            recent_errors = self._analyze_recent_errors(history)
            
            # Risk factors
            risk_factors = {
                "code_complexity": complexity_score,
                "recent_error_rate": recent_errors.get("error_rate", 0.0),
                "test_coverage": code_context.get("test_coverage", 0.0),
                "change_velocity": code_context.get("change_velocity", 0.0)
            }
            
            predictions = []
            
            # High complexity risk
            if complexity_score > 0.7:
                predictions.append(ErrorPrediction(
                    error_type="ComplexityError",
                    error_category="runtime",
                    likelihood=complexity_score,
                    time_window="next_execution",
                    prevention_steps=[
                        "Refactor complex functions",
                        "Add more unit tests",
                        "Improve error handling"
                    ],
                    risk_factors=risk_factors
                ))
            
            # Low test coverage risk
            if code_context.get("test_coverage", 1.0) < 0.5:
                predictions.append(ErrorPrediction(
                    error_type="UncaughtException",
                    error_category="testing",
                    likelihood=0.8 - code_context.get("test_coverage", 0.0),
                    time_window="deployment",
                    prevention_steps=[
                        "Increase test coverage",
                        "Add integration tests",
                        "Test edge cases"
                    ],
                    risk_factors=risk_factors
                ))
            
            # Pattern-based predictions
            if recent_errors.get("recurring_patterns"):
                for pattern in recent_errors["recurring_patterns"][:3]:
                    predictions.append(ErrorPrediction(
                        error_type=pattern["type"],
                        error_category=pattern["category"],
                        likelihood=pattern["recurrence_rate"],
                        time_window="based_on_pattern",
                        prevention_steps=pattern.get("prevention", []),
                        risk_factors=risk_factors
                    ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict error likelihood: {e}")
            return []
    
    def predict_resource_usage(
        self,
        current_usage: Dict[str, float],
        activity_metrics: Dict[str, Any],
        time_horizon: str = "24h"
    ) -> List[ResourcePrediction]:
        """
        Predict resource usage.
        
        Args:
            current_usage: Current resource usage
            activity_metrics: Activity metrics
            time_horizon: Prediction horizon
            
        Returns:
            Resource usage predictions
        """
        try:
            predictions = []
            
            # CPU prediction
            cpu_trend = self._analyze_resource_trend(
                current_usage.get("cpu_history", []),
                activity_metrics
            )
            
            predictions.append(ResourcePrediction(
                resource_type="cpu",
                predicted_usage=cpu_trend["predicted_value"],
                peak_time=cpu_trend["peak_time"],
                trend=cpu_trend["trend"],
                recommendations=self._get_resource_recommendations("cpu", cpu_trend)
            ))
            
            # Memory prediction
            memory_trend = self._analyze_resource_trend(
                current_usage.get("memory_history", []),
                activity_metrics
            )
            
            predictions.append(ResourcePrediction(
                resource_type="memory",
                predicted_usage=memory_trend["predicted_value"],
                peak_time=memory_trend["peak_time"],
                trend=memory_trend["trend"],
                recommendations=self._get_resource_recommendations("memory", memory_trend)
            ))
            
            # API calls prediction
            api_trend = self._analyze_resource_trend(
                current_usage.get("api_calls_history", []),
                activity_metrics
            )
            
            predictions.append(ResourcePrediction(
                resource_type="api_calls",
                predicted_usage=api_trend["predicted_value"],
                peak_time=api_trend["peak_time"],
                trend=api_trend["trend"],
                recommendations=self._get_resource_recommendations("api_calls", api_trend)
            ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict resource usage: {e}")
            return []
    
    def model_user_behavior(
        self,
        user_history: List[Dict[str, Any]],
        current_session: Dict[str, Any]
    ) -> UserBehaviorPrediction:
        """
        Model and predict user behavior.
        
        Args:
            user_history: User interaction history
            current_session: Current session data
            
        Returns:
            User behavior prediction
        """
        try:
            # Analyze patterns
            behavior_patterns = self._analyze_behavior_patterns(user_history)
            
            # Calculate engagement
            engagement_score = self._calculate_engagement_score(
                user_history, current_session
            )
            
            # Predict next actions
            next_actions = self._predict_next_actions(
                behavior_patterns, current_session
            )
            
            # Calculate churn risk
            churn_risk = self._calculate_churn_risk(
                user_history, engagement_score
            )
            
            # Determine behavior type
            behavior_type = self._classify_behavior_type(behavior_patterns)
            
            return UserBehaviorPrediction(
                behavior_type=behavior_type,
                likelihood=0.8,  # Confidence in classification
                next_actions=next_actions,
                engagement_score=engagement_score,
                churn_risk=churn_risk
            )
            
        except Exception as e:
            logger.error(f"Failed to model user behavior: {e}")
            return UserBehaviorPrediction(
                behavior_type="unknown",
                likelihood=0.0,
                next_actions=[],
                engagement_score=0.5,
                churn_risk=0.0
            )
    
    def save_models(self, path: str):
        """Save all trained models to disk."""
        try:
            for model_name, model in self.models.items():
                model_path = f"{path}/{model_name}_model.pkl"
                scaler_path = f"{path}/{model_name}_scaler.pkl"
                metadata_path = f"{path}/{model_name}_metadata.json"
                
                # Save model
                joblib.dump(model, model_path)
                
                # Save scaler
                if model_name in self.scalers:
                    joblib.dump(self.scalers[model_name], scaler_path)
                
                # Save metadata
                if model_name in self.model_metadata:
                    with open(metadata_path, 'w') as f:
                        json.dump(self.model_metadata[model_name], f)
            
            logger.info(f"Saved {len(self.models)} models to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise
    
    def load_models(self, path: str):
        """Load models from disk."""
        try:
            import os
            
            for filename in os.listdir(path):
                if filename.endswith("_model.pkl"):
                    model_name = filename.replace("_model.pkl", "")
                    
                    # Load model
                    model_path = f"{path}/{filename}"
                    self.models[model_name] = joblib.load(model_path)
                    
                    # Load scaler
                    scaler_path = f"{path}/{model_name}_scaler.pkl"
                    if os.path.exists(scaler_path):
                        self.scalers[model_name] = joblib.load(scaler_path)
                    
                    # Load metadata
                    metadata_path = f"{path}/{model_name}_metadata.json"
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            self.model_metadata[model_name] = json.load(f)
            
            logger.info(f"Loaded {len(self.models)} models from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    # Feature extraction methods
    
    def _extract_task_features(
        self,
        training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features for task prediction."""
        features = []
        targets = []
        
        for record in training_data:
            # Extract features
            feature_vector = [
                record.get("hour_of_day", 0),
                record.get("day_of_week", 0),
                record.get("tasks_completed_today", 0),
                record.get("project_phase", 0),
                record.get("team_size", 1),
                record.get("sprint_progress", 0.5),
                record.get("last_task_duration", 0),
                record.get("context_switches", 0)
            ]
            
            features.append(feature_vector)
            targets.append(record.get("next_task_type", "unknown"))
        
        feature_names = [
            "hour_of_day", "day_of_week", "tasks_completed_today",
            "project_phase", "team_size", "sprint_progress",
            "last_task_duration", "context_switches"
        ]
        
        return np.array(features), np.array(targets), feature_names
    
    def _extract_single_task_features(
        self,
        context: Dict[str, Any]
    ) -> List[float]:
        """Extract features for a single task prediction."""
        now = datetime.utcnow()
        
        return [
            now.hour,
            now.weekday(),
            context.get("tasks_completed_today", 0),
            context.get("project_phase", 0),
            context.get("team_size", 1),
            context.get("sprint_progress", 0.5),
            context.get("last_task_duration", 0),
            context.get("context_switches", 0)
        ]
    
    def _extract_performance_features(
        self,
        training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features for performance prediction."""
        features = []
        targets = []
        
        for record in training_data:
            feature_vector = [
                record.get("current_load", 0),
                record.get("active_users", 0),
                record.get("data_volume", 0),
                record.get("cache_hit_rate", 0),
                record.get("error_rate", 0),
                record.get("queue_length", 0),
                record.get("db_connections", 0),
                record.get("memory_usage", 0)
            ]
            
            features.append(feature_vector)
            targets.append(record.get("response_time", 0))
        
        feature_names = [
            "current_load", "active_users", "data_volume",
            "cache_hit_rate", "error_rate", "queue_length",
            "db_connections", "memory_usage"
        ]
        
        return np.array(features), np.array(targets), feature_names
    
    def _extract_single_performance_features(
        self,
        metrics: Dict[str, float],
        context: Dict[str, Any]
    ) -> List[float]:
        """Extract features for single performance prediction."""
        return [
            metrics.get("current_load", 0),
            context.get("active_users", 0),
            context.get("data_volume", 0),
            metrics.get("cache_hit_rate", 0),
            metrics.get("error_rate", 0),
            metrics.get("queue_length", 0),
            metrics.get("db_connections", 0),
            metrics.get("memory_usage", 0)
        ]
    
    # Helper methods
    
    def _generate_task_description(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate human-readable task description."""
        descriptions = {
            "code_review": "Review pending pull requests",
            "testing": "Run test suite and fix failures",
            "documentation": "Update documentation for recent changes",
            "refactoring": "Refactor identified code smells",
            "feature_development": "Continue feature implementation",
            "bug_fixing": "Fix reported bugs",
            "deployment": "Deploy to staging/production",
            "planning": "Plan next sprint tasks"
        }
        
        return descriptions.get(task_type, f"Work on {task_type}")
    
    def _estimate_task_duration(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Estimate task duration in hours."""
        base_durations = {
            "code_review": 1.0,
            "testing": 2.0,
            "documentation": 1.5,
            "refactoring": 3.0,
            "feature_development": 4.0,
            "bug_fixing": 2.5,
            "deployment": 1.0,
            "planning": 2.0
        }
        
        base = base_durations.get(task_type, 2.0)
        
        # Adjust based on context
        complexity_factor = context.get("complexity_factor", 1.0)
        
        return base * complexity_factor
    
    def _get_task_prerequisites(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Get task prerequisites."""
        prerequisites = {
            "deployment": ["All tests passing", "Code review approved"],
            "testing": ["Code changes committed"],
            "code_review": ["Pull request created"],
            "documentation": ["Feature completed"]
        }
        
        return prerequisites.get(task_type, [])
    
    def _calculate_complexity_score(
        self,
        code_context: Dict[str, Any]
    ) -> float:
        """Calculate code complexity score."""
        factors = {
            "cyclomatic_complexity": code_context.get("cyclomatic_complexity", 0) / 20,
            "lines_of_code": min(code_context.get("lines_of_code", 0) / 1000, 1.0),
            "dependencies": min(code_context.get("dependencies", 0) / 50, 1.0),
            "nesting_depth": code_context.get("max_nesting", 0) / 10
        }
        
        # Weighted average
        weights = {"cyclomatic_complexity": 0.4, "lines_of_code": 0.2,
                  "dependencies": 0.2, "nesting_depth": 0.2}
        
        score = sum(factors[k] * weights[k] for k in factors)
        
        return min(score, 1.0)
    
    def _analyze_recent_errors(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze recent error patterns."""
        if not history:
            return {"error_rate": 0.0, "recurring_patterns": []}
        
        # Count errors by type
        error_counts = {}
        for record in history:
            error_type = record.get("error_type", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Find recurring patterns
        recurring = []
        total_errors = sum(error_counts.values())
        
        for error_type, count in error_counts.items():
            if count > 2:  # Recurring if appears more than twice
                recurring.append({
                    "type": error_type,
                    "category": "recurring",
                    "recurrence_rate": count / total_errors,
                    "prevention": ["Review similar past errors", "Add specific tests"]
                })
        
        return {
            "error_rate": len(history) / max(len(history), 100),  # Per 100 operations
            "recurring_patterns": recurring
        }
    
    def _analyze_resource_trend(
        self,
        history: List[float],
        activity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze resource usage trend."""
        if not history:
            return {
                "predicted_value": 0.0,
                "peak_time": datetime.utcnow(),
                "trend": "stable"
            }
        
        # Simple trend analysis
        recent = history[-10:] if len(history) > 10 else history
        avg_recent = np.mean(recent)
        
        if len(history) > 20:
            older = history[-20:-10]
            avg_older = np.mean(older)
            
            if avg_recent > avg_older * 1.1:
                trend = "increasing"
            elif avg_recent < avg_older * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Predict next value (simple moving average)
        predicted = avg_recent * (1 + activity.get("growth_rate", 0))
        
        # Estimate peak time based on patterns
        peak_hour = 14  # 2 PM default
        peak_time = datetime.utcnow().replace(hour=peak_hour, minute=0, second=0)
        
        return {
            "predicted_value": predicted,
            "peak_time": peak_time,
            "trend": trend
        }
    
    def _get_resource_recommendations(
        self,
        resource_type: str,
        trend: Dict[str, Any]
    ) -> List[str]:
        """Get resource optimization recommendations."""
        recommendations = []
        
        if trend["trend"] == "increasing":
            if resource_type == "cpu":
                recommendations.extend([
                    "Consider horizontal scaling",
                    "Optimize CPU-intensive operations",
                    "Review background job scheduling"
                ])
            elif resource_type == "memory":
                recommendations.extend([
                    "Review memory leaks",
                    "Optimize caching strategy",
                    "Consider memory limit increase"
                ])
            elif resource_type == "api_calls":
                recommendations.extend([
                    "Implement request batching",
                    "Add response caching",
                    "Review API rate limits"
                ])
        
        return recommendations
    
    def _analyze_behavior_patterns(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user behavior patterns."""
        if not history:
            return {}
        
        # Analyze action frequencies
        action_counts = {}
        for record in history:
            action = record.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Time patterns
        hour_counts = [0] * 24
        for record in history:
            timestamp = record.get("timestamp")
            if timestamp:
                hour = datetime.fromisoformat(timestamp).hour
                hour_counts[hour] += 1
        
        # Find peak hours
        peak_hours = sorted(
            range(24),
            key=lambda h: hour_counts[h],
            reverse=True
        )[:3]
        
        return {
            "frequent_actions": sorted(
                action_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "peak_hours": peak_hours,
            "total_actions": len(history)
        }
    
    def _calculate_engagement_score(
        self,
        history: List[Dict[str, Any]],
        session: Dict[str, Any]
    ) -> float:
        """Calculate user engagement score."""
        if not history:
            return 0.5
        
        # Factors
        factors = {
            "session_duration": min(session.get("duration_minutes", 0) / 60, 1.0),
            "action_frequency": min(len(history) / 100, 1.0),
            "feature_usage": len(set(r.get("feature") for r in history)) / 10,
            "return_rate": min(session.get("sessions_this_week", 0) / 7, 1.0)
        }
        
        # Weighted score
        weights = {
            "session_duration": 0.2,
            "action_frequency": 0.3,
            "feature_usage": 0.3,
            "return_rate": 0.2
        }
        
        score = sum(factors.get(k, 0) * weights[k] for k in weights)
        
        return min(score, 1.0)
    
    def _predict_next_actions(
        self,
        patterns: Dict[str, Any],
        session: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        """Predict next user actions."""
        predictions = []
        
        # Based on frequent actions
        frequent_actions = patterns.get("frequent_actions", [])
        total_actions = patterns.get("total_actions", 1)
        
        for action, count in frequent_actions[:5]:
            probability = count / total_actions
            
            # Adjust based on session context
            if session.get("last_action") == action:
                probability *= 0.7  # Less likely to repeat immediately
            
            predictions.append({
                "action": action,
                "probability": probability
            })
        
        # Normalize probabilities
        total_prob = sum(p["probability"] for p in predictions)
        if total_prob > 0:
            for p in predictions:
                p["probability"] /= total_prob
        
        return predictions
    
    def _calculate_churn_risk(
        self,
        history: List[Dict[str, Any]],
        engagement_score: float
    ) -> float:
        """Calculate user churn risk."""
        # Base churn risk
        base_risk = 1.0 - engagement_score
        
        # Adjust based on recent activity
        if history:
            recent_actions = [
                r for r in history
                if datetime.fromisoformat(r.get("timestamp", "2000-01-01"))
                > datetime.utcnow() - timedelta(days=7)
            ]
            
            if len(recent_actions) < 5:
                base_risk += 0.2
            elif len(recent_actions) > 20:
                base_risk -= 0.1
        
        return max(0.0, min(1.0, base_risk))
    
    def _classify_behavior_type(
        self,
        patterns: Dict[str, Any]
    ) -> str:
        """Classify user behavior type."""
        frequent_actions = patterns.get("frequent_actions", [])
        
        if not frequent_actions:
            return "new_user"
        
        # Analyze action distribution
        action_types = [action for action, _ in frequent_actions]
        
        if "search" in action_types[:3]:
            return "explorer"
        elif "create" in action_types[:3]:
            return "creator"
        elif "analyze" in action_types[:3]:
            return "analyst"
        elif "configure" in action_types[:3]:
            return "administrator"
        else:
            return "regular_user"


# Global instance
predictive_models = PredictiveModels()