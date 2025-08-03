"""
Advanced Prediction Service for AI-Enhanced Features.

This service provides unified access to all predictive analytics capabilities,
managing model lifecycle and serving predictions.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID
import json

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text, func

from ..models.base import get_db_context
from ..ml.predictive_models import (
    predictive_models, TaskPrediction, PerformanceForecast,
    ErrorPrediction, ResourcePrediction, UserBehaviorPrediction
)
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("prediction_service")


class PredictionService:
    """
    Unified service for all prediction capabilities.
    
    Features:
    - Task prediction
    - Performance forecasting
    - Error likelihood estimation
    - Resource usage prediction
    - User behavior modeling
    - Model management and training
    - Caching and optimization
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.models = predictive_models
        self.analytics_service = TimeSeriesAnalyticsService()
        self._initialized = False
        
        # Cache settings
        self.cache_ttl = {
            "task_prediction": 300,  # 5 minutes
            "performance_forecast": 600,  # 10 minutes
            "error_prediction": 300,  # 5 minutes
            "resource_prediction": 600,  # 10 minutes
            "user_behavior": 900  # 15 minutes
        }
        
        # Model training thresholds
        self.training_thresholds = {
            "min_samples": 100,
            "retrain_interval_days": 7,
            "performance_degradation": 0.1
        }
        
        logger.info("Initialized PredictionService")
    
    async def initialize(self):
        """Initialize the prediction service."""
        if self._initialized:
            return
        
        try:
            # Initialize dependencies
            await self.analytics_service.initialize()
            await redis_client.initialize()
            
            # Load existing models if available
            await self._load_models()
            
            # Schedule periodic model updates
            asyncio.create_task(self._periodic_model_update())
            
            self._initialized = True
            logger.info("PredictionService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PredictionService: {e}")
            raise
    
    async def predict_next_tasks(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        n_predictions: int = 5
    ) -> List[TaskPrediction]:
        """
        Predict next likely tasks for a user.
        
        Args:
            user_id: User ID
            session_id: Optional session ID
            project_id: Optional project ID
            n_predictions: Number of predictions
            
        Returns:
            List of task predictions
        """
        try:
            # Check cache
            cache_key = f"task_predictions:{user_id}:{project_id}"
            cached = await redis_client.get(cache_key)
            
            if cached:
                predictions_data = json.loads(cached)
                return [
                    TaskPrediction(**pred) for pred in predictions_data
                ]
            
            # Get user context
            user_context = await self._get_user_context(
                user_id, session_id, project_id
            )
            
            # Get predictions
            predictions = self.models.predict_next_tasks(
                user_context, n_predictions
            )
            
            # Cache results
            if predictions:
                predictions_data = [
                    {
                        "task_type": p.task_type,
                        "task_description": p.task_description,
                        "probability": p.probability,
                        "estimated_duration": p.estimated_duration,
                        "prerequisites": p.prerequisites,
                        "confidence": p.confidence
                    }
                    for p in predictions
                ]
                
                await redis_client.setex(
                    cache_key,
                    self.cache_ttl["task_prediction"],
                    json.dumps(predictions_data)
                )
            
            # Record analytics
            await self._record_prediction_analytics(
                "task_prediction", user_id, len(predictions)
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict next tasks: {e}")
            return []
    
    async def forecast_performance(
        self,
        metric_name: str = "response_time",
        time_horizon: str = "24h",
        system_id: Optional[str] = None
    ) -> PerformanceForecast:
        """
        Forecast performance metrics.
        
        Args:
            metric_name: Metric to forecast
            time_horizon: Forecast horizon
            system_id: Optional system ID
            
        Returns:
            Performance forecast
        """
        try:
            # Get current metrics
            current_metrics = await self._get_current_metrics(metric_name, system_id)
            
            # Get system context
            system_context = await self._get_system_context(system_id)
            
            # Get forecast
            forecast = self.models.forecast_performance(
                current_metrics, system_context, time_horizon
            )
            
            # Store forecast for validation
            await self._store_forecast(forecast, system_id)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Failed to forecast performance: {e}")
            return PerformanceForecast(
                metric_name=metric_name,
                predicted_value=0.0,
                confidence_interval=(0.0, 0.0),
                time_horizon=time_horizon,
                factors={}
            )
    
    async def predict_error_likelihood(
        self,
        code_file: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[ErrorPrediction]:
        """
        Predict likelihood of errors.
        
        Args:
            code_file: Optional specific file
            project_id: Optional project ID
            user_id: Optional user ID
            
        Returns:
            List of error predictions
        """
        try:
            # Check cache
            cache_key = f"error_predictions:{project_id}:{code_file}"
            cached = await redis_client.get(cache_key)
            
            if cached:
                predictions_data = json.loads(cached)
                return [
                    ErrorPrediction(**pred) for pred in predictions_data
                ]
            
            # Get code context
            code_context = await self._get_code_context(code_file, project_id)
            
            # Get error history
            error_history = await self._get_error_history(project_id, user_id)
            
            # Get predictions
            predictions = self.models.predict_error_likelihood(
                code_context, error_history
            )
            
            # Cache results
            if predictions:
                predictions_data = [
                    {
                        "error_type": p.error_type,
                        "error_category": p.error_category,
                        "likelihood": p.likelihood,
                        "time_window": p.time_window,
                        "prevention_steps": p.prevention_steps,
                        "risk_factors": p.risk_factors
                    }
                    for p in predictions
                ]
                
                await redis_client.setex(
                    cache_key,
                    self.cache_ttl["error_prediction"],
                    json.dumps(predictions_data)
                )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict error likelihood: {e}")
            return []
    
    async def predict_resource_usage(
        self,
        resource_types: List[str] = ["cpu", "memory", "api_calls"],
        time_horizon: str = "24h",
        system_id: Optional[str] = None
    ) -> List[ResourcePrediction]:
        """
        Predict resource usage.
        
        Args:
            resource_types: Resources to predict
            time_horizon: Prediction horizon
            system_id: Optional system ID
            
        Returns:
            Resource usage predictions
        """
        try:
            # Get current usage
            current_usage = await self._get_resource_usage(resource_types, system_id)
            
            # Get activity metrics
            activity_metrics = await self._get_activity_metrics(system_id)
            
            # Get predictions
            predictions = self.models.predict_resource_usage(
                current_usage, activity_metrics, time_horizon
            )
            
            # Alert if critical
            for prediction in predictions:
                if prediction.predicted_usage > 0.9:  # 90% threshold
                    await self._send_resource_alert(prediction, system_id)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict resource usage: {e}")
            return []
    
    async def model_user_behavior(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> UserBehaviorPrediction:
        """
        Model and predict user behavior.
        
        Args:
            user_id: User ID
            session_id: Optional session ID
            
        Returns:
            User behavior prediction
        """
        try:
            # Check cache
            cache_key = f"user_behavior:{user_id}"
            cached = await redis_client.get(cache_key)
            
            if cached:
                return UserBehaviorPrediction(**json.loads(cached))
            
            # Get user history
            user_history = await self._get_user_history(user_id)
            
            # Get current session
            current_session = await self._get_session_data(session_id or user_id)
            
            # Get prediction
            prediction = self.models.model_user_behavior(
                user_history, current_session
            )
            
            # Cache result
            await redis_client.setex(
                cache_key,
                self.cache_ttl["user_behavior"],
                json.dumps({
                    "behavior_type": prediction.behavior_type,
                    "likelihood": prediction.likelihood,
                    "next_actions": prediction.next_actions,
                    "engagement_score": prediction.engagement_score,
                    "churn_risk": prediction.churn_risk
                })
            )
            
            # Handle churn risk
            if prediction.churn_risk > 0.7:
                await self._handle_churn_risk(user_id, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to model user behavior: {e}")
            return UserBehaviorPrediction(
                behavior_type="unknown",
                likelihood=0.0,
                next_actions=[],
                engagement_score=0.5,
                churn_risk=0.0
            )
    
    async def train_models(
        self,
        model_types: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Train or retrain prediction models.
        
        Args:
            model_types: Specific models to train
            force: Force retraining even if not due
            
        Returns:
            Training results
        """
        try:
            results = {}
            
            if not model_types:
                model_types = [
                    "task_prediction",
                    "performance_forecast",
                    "error_likelihood",
                    "resource_usage",
                    "user_behavior"
                ]
            
            for model_type in model_types:
                # Check if training needed
                if not force and not await self._should_retrain(model_type):
                    results[model_type] = {"status": "skipped", "reason": "not due"}
                    continue
                
                # Get training data
                training_data = await self._get_training_data(model_type)
                
                if len(training_data) < self.training_thresholds["min_samples"]:
                    results[model_type] = {
                        "status": "skipped",
                        "reason": "insufficient data",
                        "samples": len(training_data)
                    }
                    continue
                
                # Train model
                if model_type == "task_prediction":
                    result = self.models.train_task_prediction_model(training_data)
                elif model_type == "performance_forecast":
                    result = self.models.train_performance_forecast_model(training_data)
                else:
                    result = {"status": "not_implemented"}
                
                results[model_type] = result
                
                # Update metadata
                await self._update_model_metadata(model_type, result)
            
            # Save models
            await self._save_models()
            
            logger.info(f"Model training completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            return {"error": str(e)}
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all prediction models."""
        try:
            status = {}
            
            for model_name, metadata in self.models.model_metadata.items():
                status[model_name] = {
                    "trained_at": metadata.get("trained_at"),
                    "samples": metadata.get("samples", 0),
                    "train_score": metadata.get("train_score", 0),
                    "test_score": metadata.get("test_score", 0),
                    "last_prediction": await self._get_last_prediction_time(model_name)
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            return {"error": str(e)}
    
    # Helper methods
    
    async def _get_user_context(
        self,
        user_id: str,
        session_id: Optional[str],
        project_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get user context for predictions."""
        context = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            with get_db_context() as db:
                # Get user stats
                from ..models.memory import MemoryItem
                
                today = datetime.utcnow().date()
                tasks_today = db.query(func.count(MemoryItem.id)).filter(
                    MemoryItem.user_id == user_id,
                    func.date(MemoryItem.created_at) == today,
                    MemoryItem.memory_type == "task"
                ).scalar() or 0
                
                context["tasks_completed_today"] = tasks_today
                
                # Get project context
                if project_id:
                    context["project_id"] = project_id
                    # Add project-specific context
                    
                # Get session context
                if session_id:
                    from ..models.session import Session
                    session = db.query(Session).filter_by(id=session_id).first()
                    if session:
                        context["session_duration"] = (
                            datetime.utcnow() - session.created_at
                        ).total_seconds() / 60
                        context["context_switches"] = len(session.context_history or [])
                
                return context
                
        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return context
    
    async def _get_current_metrics(
        self,
        metric_name: str,
        system_id: Optional[str]
    ) -> Dict[str, float]:
        """Get current system metrics."""
        try:
            # Get from analytics service
            recent_metrics = await self.analytics_service.get_recent_metrics(
                metric_type=metric_name,
                window_minutes=60
            )
            
            if recent_metrics:
                return {
                    "current_load": recent_metrics.get("avg_value", 0),
                    "error_rate": recent_metrics.get("error_rate", 0),
                    "cache_hit_rate": recent_metrics.get("cache_hit_rate", 0.8),
                    "queue_length": recent_metrics.get("queue_length", 0),
                    "db_connections": recent_metrics.get("db_connections", 10),
                    "memory_usage": recent_metrics.get("memory_usage", 0.5)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    async def _get_system_context(
        self,
        system_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get system context for predictions."""
        context = {
            "system_id": system_id or "default",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Get active users
            with get_db_context() as db:
                from ..models.session import Session
                
                active_sessions = db.query(func.count(Session.id)).filter(
                    Session.status == "active",
                    Session.last_activity > datetime.utcnow() - timedelta(minutes=30)
                ).scalar() or 0
                
                context["active_users"] = active_sessions
                
                # Get data volume estimate
                from ..models.memory import MemoryItem
                data_volume = db.query(func.count(MemoryItem.id)).scalar() or 0
                context["data_volume"] = data_volume
                
            return context
            
        except Exception as e:
            logger.error(f"Failed to get system context: {e}")
            return context
    
    async def _get_code_context(
        self,
        code_file: Optional[str],
        project_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get code context for error prediction."""
        context = {
            "file": code_file,
            "project_id": project_id
        }
        
        # Simplified - in production, analyze actual code
        context["cyclomatic_complexity"] = 10
        context["lines_of_code"] = 500
        context["dependencies"] = 20
        context["max_nesting"] = 4
        context["test_coverage"] = 0.75
        context["change_velocity"] = 0.2
        
        return context
    
    async def _get_error_history(
        self,
        project_id: Optional[str],
        user_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Get error history for predictions."""
        try:
            with get_db_context() as db:
                from ..models.error_pattern import ErrorOccurrence
                
                query = db.query(ErrorOccurrence)
                
                if project_id:
                    query = query.filter(ErrorOccurrence.execution_context["project_id"].astext == project_id)
                if user_id:
                    query = query.filter(ErrorOccurrence.user_id == user_id)
                
                recent_errors = query.order_by(
                    ErrorOccurrence.timestamp.desc()
                ).limit(100).all()
                
                return [
                    {
                        "error_type": error.pattern.error_type if error.pattern else "unknown",
                        "timestamp": error.timestamp.isoformat(),
                        "resolved": error.resolved
                    }
                    for error in recent_errors
                ]
                
        except Exception as e:
            logger.error(f"Failed to get error history: {e}")
            return []
    
    async def _get_resource_usage(
        self,
        resource_types: List[str],
        system_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get current resource usage."""
        usage = {}
        
        try:
            # Get from monitoring system
            for resource_type in resource_types:
                history = await self.analytics_service.get_metric_history(
                    metric_type=f"resource_{resource_type}",
                    hours=24
                )
                
                usage[f"{resource_type}_history"] = [
                    point["value"] for point in history
                ]
            
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    async def _get_activity_metrics(
        self,
        system_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get system activity metrics."""
        return {
            "growth_rate": 0.1,  # 10% growth
            "peak_hours": [9, 14, 16],  # 9 AM, 2 PM, 4 PM
            "avg_session_duration": 45  # minutes
        }
    
    async def _get_user_history(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get user interaction history."""
        try:
            with get_db_context() as db:
                from ..models.activity import UserActivity
                
                activities = db.query(UserActivity).filter(
                    UserActivity.user_id == user_id
                ).order_by(
                    UserActivity.timestamp.desc()
                ).limit(1000).all()
                
                return [
                    {
                        "action": activity.action,
                        "feature": activity.feature,
                        "timestamp": activity.timestamp.isoformat()
                    }
                    for activity in activities
                ]
                
        except Exception as e:
            logger.error(f"Failed to get user history: {e}")
            return []
    
    async def _get_session_data(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get current session data."""
        try:
            from ..models.session import Session
            with get_db_context() as db:
                session = db.query(Session).filter_by(id=session_id).first()
                
                if session:
                    return {
                        "duration_minutes": (
                            datetime.utcnow() - session.created_at
                        ).total_seconds() / 60,
                        "interaction_count": session.interaction_count,
                        "last_action": session.active_tasks[0] if session.active_tasks else None,
                        "sessions_this_week": db.query(func.count(Session.id)).filter(
                            Session.user_id == session.user_id,
                            Session.created_at > datetime.utcnow() - timedelta(days=7)
                        ).scalar() or 0
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get session data: {e}")
            return {}
    
    async def _record_prediction_analytics(
        self,
        prediction_type: str,
        user_id: str,
        prediction_count: int
    ):
        """Record prediction analytics."""
        try:
            await self.analytics_service.record_metric(
                metric_type="prediction_made",
                value=prediction_count,
                tags={
                    "prediction_type": prediction_type,
                    "user_id": user_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to record analytics: {e}")
    
    async def _store_forecast(
        self,
        forecast: PerformanceForecast,
        system_id: Optional[str]
    ):
        """Store forecast for later validation."""
        try:
            forecast_data = {
                "metric_name": forecast.metric_name,
                "predicted_value": forecast.predicted_value,
                "confidence_interval": forecast.confidence_interval,
                "time_horizon": forecast.time_horizon,
                "created_at": datetime.utcnow().isoformat(),
                "system_id": system_id
            }
            
            await redis_client.setex(
                f"forecast:{forecast.metric_name}:{system_id}",
                86400,  # 24 hours
                json.dumps(forecast_data)
            )
            
        except Exception as e:
            logger.warning(f"Failed to store forecast: {e}")
    
    async def _send_resource_alert(
        self,
        prediction: ResourcePrediction,
        system_id: Optional[str]
    ):
        """Send alert for critical resource prediction."""
        logger.warning(
            f"Resource alert: {prediction.resource_type} predicted to reach "
            f"{prediction.predicted_usage:.1%} at {prediction.peak_time}"
        )
        # In production, send actual alerts
    
    async def _handle_churn_risk(
        self,
        user_id: str,
        prediction: UserBehaviorPrediction
    ):
        """Handle high churn risk users."""
        logger.warning(
            f"High churn risk for user {user_id}: {prediction.churn_risk:.1%}"
        )
        # In production, trigger retention campaigns
    
    async def _should_retrain(
        self,
        model_type: str
    ) -> bool:
        """Check if model should be retrained."""
        try:
            metadata = self.models.model_metadata.get(model_type, {})
            
            if not metadata:
                return True
            
            # Check age
            trained_at = metadata.get("trained_at")
            if trained_at:
                age_days = (
                    datetime.utcnow() - datetime.fromisoformat(trained_at)
                ).days
                
                if age_days > self.training_thresholds["retrain_interval_days"]:
                    return True
            
            # Check performance degradation
            train_score = metadata.get("train_score", 0)
            test_score = metadata.get("test_score", 0)
            
            if train_score - test_score > self.training_thresholds["performance_degradation"]:
                return True
            
            return False
            
        except Exception:
            return True
    
    async def _get_training_data(
        self,
        model_type: str
    ) -> List[Dict[str, Any]]:
        """Get training data for specific model type."""
        # In production, retrieve from database
        # For now, return sample data
        
        if model_type == "task_prediction":
            return [
                {
                    "hour_of_day": i % 24,
                    "day_of_week": i % 7,
                    "tasks_completed_today": i % 10,
                    "project_phase": i % 4,
                    "team_size": 5,
                    "sprint_progress": (i % 100) / 100,
                    "last_task_duration": 2.5,
                    "context_switches": i % 5,
                    "next_task_type": ["coding", "testing", "review"][i % 3]
                }
                for i in range(200)
            ]
        
        elif model_type == "performance_forecast":
            return [
                {
                    "current_load": (i % 100) / 100,
                    "active_users": i % 50,
                    "data_volume": i * 1000,
                    "cache_hit_rate": 0.8 + (i % 20) / 100,
                    "error_rate": (i % 10) / 100,
                    "queue_length": i % 20,
                    "db_connections": 10 + i % 40,
                    "memory_usage": 0.3 + (i % 60) / 100,
                    "response_time": 100 + i % 400
                }
                for i in range(150)
            ]
        
        return []
    
    async def _update_model_metadata(
        self,
        model_type: str,
        result: Dict[str, Any]
    ):
        """Update model metadata after training."""
        try:
            metadata_key = f"model_metadata:{model_type}"
            metadata = {
                "model_type": model_type,
                "last_trained": datetime.utcnow().isoformat(),
                "training_result": result,
                "version": self.models.model_metadata.get(model_type, {}).get("version", 0) + 1
            }
            
            await redis_client.set(
                metadata_key,
                json.dumps(metadata)
            )
            
        except Exception as e:
            logger.warning(f"Failed to update model metadata: {e}")
    
    async def _save_models(self):
        """Save models to persistent storage."""
        try:
            import os
            
            model_dir = "/opt/projects/knowledgehub/models"
            os.makedirs(model_dir, exist_ok=True)
            
            self.models.save_models(model_dir)
            logger.info(f"Models saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    async def _load_models(self):
        """Load models from persistent storage."""
        try:
            import os
            
            model_dir = "/opt/projects/knowledgehub/models"
            if os.path.exists(model_dir):
                self.models.load_models(model_dir)
                logger.info(f"Models loaded from {model_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
    
    async def _get_last_prediction_time(
        self,
        model_name: str
    ) -> Optional[str]:
        """Get last prediction time for a model."""
        try:
            key = f"last_prediction:{model_name}"
            timestamp = await redis_client.get(key)
            return timestamp.decode() if timestamp else None
            
        except Exception:
            return None
    
    async def _periodic_model_update(self):
        """Periodically check and update models."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                # Check if any models need retraining
                models_to_train = []
                for model_type in self.models.configs.keys():
                    if await self._should_retrain(model_type):
                        models_to_train.append(model_type)
                
                if models_to_train:
                    logger.info(f"Retraining models: {models_to_train}")
                    await self.train_models(models_to_train)
                
            except Exception as e:
                logger.error(f"Periodic model update failed: {e}")
    
    async def cleanup(self):
        """Clean up service resources."""
        await self.analytics_service.cleanup()
        self._initialized = False
        logger.info("PredictionService cleaned up")


# Global prediction service instance
prediction_service = PredictionService()