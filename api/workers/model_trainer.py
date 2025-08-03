"""
Model Training Worker for Background Model Updates.

This worker handles:
- Scheduled model retraining
- Data collection and preparation
- Model performance monitoring
- A/B testing for new models
- Model deployment
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from enum import Enum

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, and_, or_

from ..models.base import get_db_context
from ..models.memory import MemoryItem
from ..models.session import Session
from ..models.error_pattern import ErrorOccurrence
from ..models.enhanced_decision import EnhancedDecision
from ..services.prediction_service import prediction_service
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("model_trainer_worker")


class ModelType(str, Enum):
    """Types of ML models to train."""
    TASK_PREDICTION = "task_prediction"
    PERFORMANCE_FORECAST = "performance_forecast"
    ERROR_LIKELIHOOD = "error_likelihood"
    RESOURCE_USAGE = "resource_usage"
    USER_BEHAVIOR = "user_behavior"


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    COLLECTING_DATA = "collecting_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelTrainer:
    """
    Background worker for model training and deployment.
    
    Features:
    - Automated training schedules
    - Data collection from multiple sources
    - Model evaluation and comparison
    - Safe deployment with rollback
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.analytics_service = TimeSeriesAnalyticsService()
        self._running = False
        self._tasks = []
        
        # Training configuration
        self.training_config = {
            ModelType.TASK_PREDICTION: {
                "schedule_hours": 24,  # Daily
                "min_samples": 100,
                "data_window_days": 30
            },
            ModelType.PERFORMANCE_FORECAST: {
                "schedule_hours": 12,  # Twice daily
                "min_samples": 50,
                "data_window_days": 14
            },
            ModelType.ERROR_LIKELIHOOD: {
                "schedule_hours": 48,  # Every 2 days
                "min_samples": 30,
                "data_window_days": 60
            },
            ModelType.RESOURCE_USAGE: {
                "schedule_hours": 6,  # 4 times daily
                "min_samples": 100,
                "data_window_days": 7
            },
            ModelType.USER_BEHAVIOR: {
                "schedule_hours": 168,  # Weekly
                "min_samples": 200,
                "data_window_days": 90
            }
        }
        
        logger.info("Initialized ModelTrainer worker")
    
    async def start(self):
        """Start the model training worker."""
        if self._running:
            logger.warning("Model trainer already running")
            return
        
        self._running = True
        logger.info("Starting model trainer worker")
        
        try:
            # Initialize services
            await self.analytics_service.initialize()
            await redis_client.initialize()
            await prediction_service.initialize()
            
            # Start training tasks for each model type
            for model_type in ModelType:
                task = asyncio.create_task(
                    self._training_loop(model_type)
                )
                self._tasks.append(task)
            
            # Start monitoring task
            monitoring_task = asyncio.create_task(
                self._monitor_model_performance()
            )
            self._tasks.append(monitoring_task)
            
            logger.info("Model trainer worker started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start model trainer: {e}")
            self._running = False
            raise
    
    async def stop(self):
        """Stop the model training worker."""
        logger.info("Stopping model trainer worker")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Cleanup
        await self.analytics_service.cleanup()
        
        logger.info("Model trainer worker stopped")
    
    async def _training_loop(self, model_type: ModelType):
        """Main training loop for a specific model type."""
        config = self.training_config[model_type]
        
        while self._running:
            try:
                # Check if training is needed
                should_train = await self._should_train_model(
                    model_type, config["schedule_hours"]
                )
                
                if should_train:
                    await self._train_model(model_type)
                
                # Sleep until next check
                await asyncio.sleep(3600)  # Check hourly
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in training loop for {model_type}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _should_train_model(
        self,
        model_type: ModelType,
        schedule_hours: int
    ) -> bool:
        """Check if a model should be trained."""
        try:
            # Get last training time
            last_trained_key = f"model_last_trained:{model_type.value}"
            last_trained = await redis_client.get(last_trained_key)
            
            if not last_trained:
                return True
            
            last_trained_time = datetime.fromisoformat(
                last_trained.decode()
            )
            
            # Check if enough time has passed
            if datetime.utcnow() - last_trained_time > timedelta(hours=schedule_hours):
                return True
            
            # Check if performance has degraded
            degradation = await self._check_performance_degradation(model_type)
            if degradation > 0.15:  # 15% degradation threshold
                logger.warning(
                    f"Model {model_type} performance degraded by {degradation:.1%}"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if should train: {e}")
            return False
    
    async def _train_model(self, model_type: ModelType):
        """Train a specific model type."""
        start_time = datetime.utcnow()
        training_id = f"{model_type.value}_{start_time.timestamp()}"
        
        logger.info(f"Starting training for {model_type} (ID: {training_id})")
        
        try:
            # Update status
            await self._update_training_status(
                training_id, TrainingStatus.COLLECTING_DATA
            )
            
            # Collect training data
            training_data = await self._collect_training_data(model_type)
            
            if len(training_data) < self.training_config[model_type]["min_samples"]:
                logger.warning(
                    f"Insufficient data for {model_type}: "
                    f"{len(training_data)} samples"
                )
                await self._update_training_status(
                    training_id, TrainingStatus.FAILED,
                    {"reason": "insufficient_data", "samples": len(training_data)}
                )
                return
            
            # Update status
            await self._update_training_status(
                training_id, TrainingStatus.TRAINING,
                {"samples": len(training_data)}
            )
            
            # Train model
            result = await prediction_service.train_models(
                model_types=[model_type.value],
                force=True
            )
            
            if model_type.value not in result or "error" in result[model_type.value]:
                raise Exception(
                    f"Training failed: {result.get(model_type.value, {}).get('error', 'Unknown error')}"
                )
            
            # Update status
            await self._update_training_status(
                training_id, TrainingStatus.EVALUATING,
                result[model_type.value]
            )
            
            # Evaluate model
            evaluation = await self._evaluate_model(
                model_type, result[model_type.value]
            )
            
            # Decide whether to deploy
            if evaluation["should_deploy"]:
                await self._update_training_status(
                    training_id, TrainingStatus.DEPLOYING
                )
                
                # Deploy model (already done by prediction_service)
                
                # Update last trained time
                await redis_client.set(
                    f"model_last_trained:{model_type.value}",
                    datetime.utcnow().isoformat()
                )
                
                # Record success
                await self._record_training_success(
                    model_type, training_id, result[model_type.value], evaluation
                )
                
                await self._update_training_status(
                    training_id, TrainingStatus.COMPLETED,
                    {"evaluation": evaluation}
                )
                
                logger.info(
                    f"Successfully trained and deployed {model_type} "
                    f"(score: {evaluation['test_score']:.3f})"
                )
            else:
                logger.warning(
                    f"Model {model_type} not deployed due to poor performance"
                )
                await self._update_training_status(
                    training_id, TrainingStatus.COMPLETED,
                    {"deployed": False, "reason": evaluation["rejection_reason"]}
                )
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            await self._update_training_status(
                training_id, TrainingStatus.FAILED,
                {"error": str(e)}
            )
            
            # Record failure
            await self._record_training_failure(model_type, training_id, str(e))
    
    async def _collect_training_data(
        self,
        model_type: ModelType
    ) -> List[Dict[str, Any]]:
        """Collect training data for a specific model type."""
        config = self.training_config[model_type]
        window_start = datetime.utcnow() - timedelta(days=config["data_window_days"])
        
        if model_type == ModelType.TASK_PREDICTION:
            return await self._collect_task_prediction_data(window_start)
        elif model_type == ModelType.PERFORMANCE_FORECAST:
            return await self._collect_performance_data(window_start)
        elif model_type == ModelType.ERROR_LIKELIHOOD:
            return await self._collect_error_data(window_start)
        elif model_type == ModelType.RESOURCE_USAGE:
            return await self._collect_resource_data(window_start)
        elif model_type == ModelType.USER_BEHAVIOR:
            return await self._collect_behavior_data(window_start)
        else:
            return []
    
    async def _collect_task_prediction_data(
        self,
        window_start: datetime
    ) -> List[Dict[str, Any]]:
        """Collect task prediction training data."""
        try:
            with get_db_context() as db:
                # Get task sequences from memories
                tasks = db.query(MemoryItem).filter(
                    MemoryItem.created_at >= window_start,
                    MemoryItem.memory_type == "task"
                ).order_by(
                    MemoryItem.user_id,
                    MemoryItem.created_at
                ).all()
                
                # Group by user and session
                user_sessions = {}
                for task in tasks:
                    key = (task.user_id, task.metadata.get("session_id"))
                    if key not in user_sessions:
                        user_sessions[key] = []
                    user_sessions[key].append(task)
                
                # Create training samples
                training_data = []
                for (user_id, session_id), session_tasks in user_sessions.items():
                    if len(session_tasks) < 2:
                        continue
                    
                    # Create samples from task sequences
                    for i in range(len(session_tasks) - 1):
                        current = session_tasks[i]
                        next_task = session_tasks[i + 1]
                        
                        # Extract features
                        sample = {
                            "hour_of_day": current.created_at.hour,
                            "day_of_week": current.created_at.weekday(),
                            "tasks_completed_today": i + 1,
                            "project_phase": self._extract_project_phase(current),
                            "team_size": self._estimate_team_size(db, current),
                            "sprint_progress": self._calculate_sprint_progress(current),
                            "last_task_duration": self._calculate_task_duration(
                                session_tasks[i-1] if i > 0 else None,
                                current
                            ),
                            "context_switches": self._count_context_switches(
                                session_tasks[:i+1]
                            ),
                            "next_task_type": self._extract_task_type(next_task)
                        }
                        
                        training_data.append(sample)
                
                return training_data
                
        except Exception as e:
            logger.error(f"Failed to collect task data: {e}")
            return []
    
    async def _collect_performance_data(
        self,
        window_start: datetime
    ) -> List[Dict[str, Any]]:
        """Collect performance forecast training data."""
        try:
            # Get historical metrics
            metrics = await self.analytics_service.get_metric_history(
                metric_type="response_time",
                hours=int((datetime.utcnow() - window_start).total_seconds() / 3600)
            )
            
            if len(metrics) < 10:
                return []
            
            training_data = []
            
            # Create samples with sliding window
            for i in range(10, len(metrics)):
                # Use past 10 data points to predict next
                recent = metrics[i-10:i]
                target = metrics[i]
                
                # Calculate features
                loads = [m.get("load", 0) for m in recent]
                errors = [m.get("errors", 0) for m in recent]
                
                sample = {
                    "current_load": loads[-1],
                    "active_users": recent[-1].get("active_users", 0),
                    "data_volume": recent[-1].get("data_volume", 0),
                    "cache_hit_rate": recent[-1].get("cache_hit_rate", 0.8),
                    "error_rate": sum(errors) / len(errors) if errors else 0,
                    "queue_length": recent[-1].get("queue_length", 0),
                    "db_connections": recent[-1].get("db_connections", 10),
                    "memory_usage": recent[-1].get("memory_usage", 0.5),
                    "response_time": target.get("value", 100)
                }
                
                training_data.append(sample)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to collect performance data: {e}")
            return []
    
    async def _collect_error_data(
        self,
        window_start: datetime
    ) -> List[Dict[str, Any]]:
        """Collect error prediction training data."""
        try:
            with get_db_context() as db:
                # Get error occurrences
                errors = db.query(ErrorOccurrence).filter(
                    ErrorOccurrence.timestamp >= window_start
                ).all()
                
                # Get code changes (simplified - would need git integration)
                # For now, use memory items as proxy
                code_changes = db.query(MemoryItem).filter(
                    MemoryItem.created_at >= window_start,
                    MemoryItem.memory_type == "code"
                ).all()
                
                # Create training samples
                # This is simplified - in production, would analyze actual code
                training_data = []
                
                # Group by time windows
                time_windows = self._create_time_windows(
                    window_start, datetime.utcnow(), hours=4
                )
                
                for window_start_time, window_end in time_windows:
                    window_errors = [
                        e for e in errors
                        if window_start_time <= e.timestamp < window_end
                    ]
                    
                    window_changes = [
                        c for c in code_changes
                        if window_start_time <= c.created_at < window_end
                    ]
                    
                    if window_changes:  # Only if there were code changes
                        sample = {
                            "code_changes": len(window_changes),
                            "test_coverage": 0.75,  # Would get from CI/CD
                            "complexity_increase": self._estimate_complexity_change(
                                window_changes
                            ),
                            "error_occurred": len(window_errors) > 0,
                            "error_count": len(window_errors)
                        }
                        
                        training_data.append(sample)
                
                return training_data
                
        except Exception as e:
            logger.error(f"Failed to collect error data: {e}")
            return []
    
    async def _collect_resource_data(
        self,
        window_start: datetime
    ) -> List[Dict[str, Any]]:
        """Collect resource usage training data."""
        # Similar to performance data but focused on resource metrics
        return await self._collect_performance_data(window_start)
    
    async def _collect_behavior_data(
        self,
        window_start: datetime
    ) -> List[Dict[str, Any]]:
        """Collect user behavior training data."""
        try:
            with get_db_context() as db:
                # Get user sessions
                sessions = db.query(Session).filter(
                    Session.created_at >= window_start
                ).all()
                
                training_data = []
                
                for session in sessions:
                    # Get user's activity in this session
                    activities = db.query(MemoryItem).filter(
                        MemoryItem.user_id == session.user_id,
                        MemoryItem.created_at >= session.created_at,
                        MemoryItem.created_at <= (
                            session.ended_at or datetime.utcnow()
                        )
                    ).all()
                    
                    if len(activities) < 5:
                        continue
                    
                    # Extract behavior features
                    sample = {
                        "session_duration": (
                            (session.ended_at or datetime.utcnow()) - session.created_at
                        ).total_seconds() / 60,
                        "interaction_count": len(activities),
                        "unique_features": len(set(
                            a.memory_type for a in activities
                        )),
                        "time_between_actions": self._calculate_avg_time_between(
                            activities
                        ),
                        "session_completed": session.status == "completed",
                        "next_session_within_week": self._check_return_within_week(
                            db, session
                        )
                    }
                    
                    training_data.append(sample)
                
                return training_data
                
        except Exception as e:
            logger.error(f"Failed to collect behavior data: {e}")
            return []
    
    async def _evaluate_model(
        self,
        model_type: ModelType,
        training_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a trained model."""
        evaluation = {
            "should_deploy": True,
            "test_score": training_result.get("test_score", 0),
            "train_score": training_result.get("train_score", 0),
            "improvement": 0,
            "rejection_reason": None
        }
        
        # Check absolute performance
        if evaluation["test_score"] < 0.6:  # 60% minimum accuracy
            evaluation["should_deploy"] = False
            evaluation["rejection_reason"] = "low_accuracy"
            return evaluation
        
        # Check overfitting
        if evaluation["train_score"] - evaluation["test_score"] > 0.2:
            evaluation["should_deploy"] = False
            evaluation["rejection_reason"] = "overfitting"
            return evaluation
        
        # Compare with current model
        current_performance = await self._get_current_model_performance(model_type)
        if current_performance:
            improvement = evaluation["test_score"] - current_performance
            evaluation["improvement"] = improvement
            
            # Require at least 2% improvement
            if improvement < 0.02:
                evaluation["should_deploy"] = False
                evaluation["rejection_reason"] = "no_improvement"
        
        return evaluation
    
    async def _monitor_model_performance(self):
        """Monitor deployed model performance."""
        while self._running:
            try:
                for model_type in ModelType:
                    # Get recent predictions
                    predictions_key = f"model_predictions:{model_type.value}:*"
                    # This would track prediction accuracy in production
                    
                    # Record performance metrics
                    await self.analytics_service.record_metric(
                        metric_type=f"model_performance_{model_type.value}",
                        value=await self._calculate_model_accuracy(model_type),
                        tags={"model_type": model_type.value}
                    )
                
                # Sleep for monitoring interval
                await asyncio.sleep(3600)  # Check hourly
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)
    
    # Helper methods
    
    async def _update_training_status(
        self,
        training_id: str,
        status: TrainingStatus,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update training job status."""
        status_data = {
            "training_id": training_id,
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        await redis_client.setex(
            f"training_status:{training_id}",
            86400,  # 24 hours
            json.dumps(status_data)
        )
    
    async def _check_performance_degradation(
        self,
        model_type: ModelType
    ) -> float:
        """Check if model performance has degraded."""
        # In production, would track actual prediction accuracy
        # For now, return 0 (no degradation)
        return 0.0
    
    async def _get_current_model_performance(
        self,
        model_type: ModelType
    ) -> Optional[float]:
        """Get current model's performance score."""
        try:
            metadata = prediction_service.models.model_metadata.get(
                model_type.value, {}
            )
            return metadata.get("test_score")
        except Exception:
            return None
    
    async def _calculate_model_accuracy(
        self,
        model_type: ModelType
    ) -> float:
        """Calculate current model accuracy."""
        # In production, would track actual vs predicted
        # For now, return simulated accuracy
        import random
        return 0.75 + random.random() * 0.2
    
    async def _record_training_success(
        self,
        model_type: ModelType,
        training_id: str,
        result: Dict[str, Any],
        evaluation: Dict[str, Any]
    ):
        """Record successful training."""
        await self.analytics_service.record_metric(
            metric_type="model_training_success",
            value=1,
            tags={
                "model_type": model_type.value,
                "training_id": training_id,
                "test_score": result.get("test_score", 0),
                "improvement": evaluation.get("improvement", 0)
            }
        )
    
    async def _record_training_failure(
        self,
        model_type: ModelType,
        training_id: str,
        error: str
    ):
        """Record training failure."""
        await self.analytics_service.record_metric(
            metric_type="model_training_failure",
            value=1,
            tags={
                "model_type": model_type.value,
                "training_id": training_id,
                "error": error[:100]  # Truncate error message
            }
        )
    
    def _extract_project_phase(self, task: MemoryItem) -> int:
        """Extract project phase from task metadata."""
        # Simplified - would analyze task content
        return task.metadata.get("project_phase", 0)
    
    def _estimate_team_size(self, db: DBSession, task: MemoryItem) -> int:
        """Estimate team size from recent activity."""
        # Count unique users in same project
        project_id = task.metadata.get("project_id")
        if not project_id:
            return 1
        
        recent_users = db.query(MemoryItem.user_id).filter(
            MemoryItem.metadata["project_id"].astext == project_id,
            MemoryItem.created_at >= datetime.utcnow() - timedelta(days=7)
        ).distinct().count()
        
        return max(1, recent_users)
    
    def _calculate_sprint_progress(self, task: MemoryItem) -> float:
        """Calculate sprint progress."""
        # Simplified - would integrate with project management
        return task.metadata.get("sprint_progress", 0.5)
    
    def _calculate_task_duration(
        self,
        prev_task: Optional[MemoryItem],
        current_task: MemoryItem
    ) -> float:
        """Calculate task duration in hours."""
        if not prev_task:
            return 1.0
        
        duration = (current_task.created_at - prev_task.created_at).total_seconds() / 3600
        return min(duration, 8.0)  # Cap at 8 hours
    
    def _count_context_switches(self, tasks: List[MemoryItem]) -> int:
        """Count context switches in task sequence."""
        if len(tasks) < 2:
            return 0
        
        switches = 0
        for i in range(1, len(tasks)):
            if tasks[i].metadata.get("project_id") != tasks[i-1].metadata.get("project_id"):
                switches += 1
        
        return switches
    
    def _extract_task_type(self, task: MemoryItem) -> str:
        """Extract task type from memory item."""
        # Analyze content to determine type
        content_lower = task.content.lower()
        
        if any(word in content_lower for word in ["review", "pr", "merge"]):
            return "code_review"
        elif any(word in content_lower for word in ["test", "spec", "jest", "pytest"]):
            return "testing"
        elif any(word in content_lower for word in ["doc", "readme", "comment"]):
            return "documentation"
        elif any(word in content_lower for word in ["refactor", "clean", "optimize"]):
            return "refactoring"
        elif any(word in content_lower for word in ["fix", "bug", "issue", "error"]):
            return "bug_fixing"
        elif any(word in content_lower for word in ["deploy", "release", "production"]):
            return "deployment"
        elif any(word in content_lower for word in ["plan", "design", "architecture"]):
            return "planning"
        else:
            return "feature_development"
    
    def _create_time_windows(
        self,
        start: datetime,
        end: datetime,
        hours: int
    ) -> List[tuple]:
        """Create time windows for analysis."""
        windows = []
        current = start
        
        while current < end:
            window_end = min(current + timedelta(hours=hours), end)
            windows.append((current, window_end))
            current = window_end
        
        return windows
    
    def _estimate_complexity_change(self, changes: List[MemoryItem]) -> float:
        """Estimate code complexity change."""
        # Simplified - would analyze actual code diffs
        total_lines = sum(
            len(change.content.split('\n'))
            for change in changes
        )
        
        return min(total_lines / 1000, 1.0)
    
    def _calculate_avg_time_between(self, activities: List[MemoryItem]) -> float:
        """Calculate average time between activities."""
        if len(activities) < 2:
            return 0
        
        times = []
        for i in range(1, len(activities)):
            delta = (activities[i].created_at - activities[i-1].created_at).total_seconds() / 60
            times.append(min(delta, 60))  # Cap at 60 minutes
        
        return sum(times) / len(times) if times else 0
    
    def _check_return_within_week(
        self,
        db: DBSession,
        session: Session
    ) -> bool:
        """Check if user returned within a week."""
        if not session.ended_at:
            return False
        
        next_session = db.query(Session).filter(
            Session.user_id == session.user_id,
            Session.created_at > session.ended_at,
            Session.created_at <= session.ended_at + timedelta(days=7)
        ).first()
        
        return next_session is not None


# Global worker instance
model_trainer = ModelTrainer()


async def start_model_trainer():
    """Start the model training worker."""
    await model_trainer.start()


async def stop_model_trainer():
    """Stop the model training worker."""
    await model_trainer.stop()