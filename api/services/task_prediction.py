"""
Task Prediction Service for Claude Code
Anticipates next actions and preloads relevant context
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, or_, func

from ..models.memory import Memory
from .memory_service import MemoryService


class TaskPredictionService:
    """Predicts likely next tasks and prepares context for Claude Code"""
    
    def __init__(self, db: Session, memory_service: MemoryService):
        self.db = db
        self.memory_service = memory_service
        
        # Common task sequences
        self.task_sequences = {
            "error_fix": ["identify_error", "search_solution", "apply_fix", "test_fix"],
            "feature_add": ["understand_requirements", "search_codebase", "implement", "test", "commit"],
            "refactor": ["identify_code", "plan_changes", "implement", "test", "update_tests"],
            "debug": ["reproduce_issue", "add_logging", "identify_cause", "fix", "verify"],
            "setup": ["check_requirements", "install_dependencies", "configure", "test_setup"],
            "deployment": ["build", "test", "package", "deploy", "verify"]
        }
        
        # Task indicators (keywords that suggest certain tasks)
        self.task_indicators = {
            "error": ["error", "exception", "failed", "broken", "fix", "issue", "bug"],
            "feature": ["add", "implement", "create", "new", "feature", "functionality"],
            "refactor": ["refactor", "clean", "improve", "optimize", "reorganize"],
            "test": ["test", "verify", "check", "ensure", "validate"],
            "document": ["document", "readme", "comment", "explain"],
            "deploy": ["deploy", "build", "release", "publish", "production"]
        }
    
    async def predict_next_tasks(
        self,
        current_context: Dict[str, Any],
        session_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict likely next tasks based on current context
        
        Args:
            current_context: Current session/project context
            session_id: Current session ID
            limit: Maximum number of predictions
            
        Returns:
            List of predicted tasks with confidence scores
        """
        predictions = []
        
        # 1. Analyze recent activities
        recent_activities = await self._get_recent_activities(session_id)
        current_task_type = self._identify_current_task_type(recent_activities)
        
        # 2. Check for unfinished tasks
        unfinished = await self._get_unfinished_tasks(session_id)
        for task in unfinished:
            predictions.append({
                "task": f"Continue: {task['content']}",
                "type": "continuation",
                "confidence": 0.9,
                "reason": "Unfinished task from previous session",
                "context_needed": task.get("context_needed", [])
            })
        
        # 3. Predict based on task sequences
        if current_task_type:
            sequence_predictions = self._predict_from_sequence(
                current_task_type,
                recent_activities
            )
            predictions.extend(sequence_predictions)
        
        # 4. Predict based on patterns
        pattern_predictions = await self._predict_from_patterns(
            current_context,
            recent_activities
        )
        predictions.extend(pattern_predictions)
        
        # 5. Check for scheduled or recurring tasks
        scheduled = await self._get_scheduled_tasks(current_context)
        predictions.extend(scheduled)
        
        # Remove duplicates and sort by confidence
        unique_predictions = self._deduplicate_predictions(predictions)
        unique_predictions.sort(key=lambda p: p["confidence"], reverse=True)
        
        # Record predictions for learning
        if session_id and unique_predictions:
            await self._record_predictions(session_id, unique_predictions[:limit])
        
        return unique_predictions[:limit]
    
    async def prepare_for_tasks(
        self,
        likely_tasks: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare context for likely tasks
        
        Args:
            likely_tasks: List of predicted tasks
            session_id: Current session ID
            
        Returns:
            Preloaded context organized by task
        """
        preloaded_context = {
            "prepared_at": datetime.utcnow().isoformat(),
            "tasks": {}
        }
        
        for task in likely_tasks:
            task_context = {
                "task": task["task"],
                "type": task["type"],
                "confidence": task["confidence"],
                "relevant_memories": [],
                "suggested_tools": [],
                "related_files": [],
                "similar_solutions": []
            }
            
            # Load context based on task type
            if task["type"] == "error_fix":
                # Load error-related context
                errors = await self._load_error_context(task.get("error_indicators", []))
                task_context["similar_solutions"] = errors
                task_context["suggested_tools"] = ["grep", "logs", "debugger"]
                
            elif task["type"] == "feature":
                # Load feature implementation context
                similar_features = await self._load_similar_features(task.get("feature_keywords", []))
                task_context["relevant_memories"] = similar_features
                task_context["suggested_tools"] = ["search", "scaffold", "test"]
                
            elif task["type"] == "continuation":
                # Load previous task context
                previous_context = await self._load_task_context(task.get("task_id"))
                task_context.update(previous_context)
            
            # Add common context needs
            task_context["context_needed"] = task.get("context_needed", [])
            
            preloaded_context["tasks"][task["task"]] = task_context
        
        return preloaded_context
    
    async def track_task_completion(
        self,
        task: str,
        completed: bool,
        session_id: str,
        time_taken: Optional[float] = None,
        obstacles: Optional[List[str]] = None
    ) -> None:
        """Track task completion for improving predictions"""
        metadata = {
            "task": task,
            "completed": completed,
            "session_id": session_id,
            "tracked_at": datetime.utcnow().isoformat()
        }
        
        if time_taken:
            metadata["time_taken"] = time_taken
        if obstacles:
            metadata["obstacles"] = obstacles
        
        # Record task completion
        content = f"TASK {'COMPLETED' if completed else 'ABANDONED'}: {task}"
        if obstacles:
            content += f"\nObstacles: {', '.join(obstacles)}"
        
        await self.memory_service.create_memory(
            session_id=session_id,
            content=content,
            memory_type="pattern",
            importance=0.6 if completed else 0.4,
            metadata=metadata
        )
    
    async def _get_recent_activities(
        self,
        session_id: Optional[str],
        limit: int = 20
    ) -> List[Memory]:
        """Get recent activities from current and previous sessions"""
        query = self.db.query(Memory).filter(
            Memory.created_at > datetime.utcnow() - timedelta(hours=24)
        )
        
        if session_id:
            # Include current session
            query = query.filter(Memory.session_id == session_id)
        
        return query.order_by(desc(Memory.created_at)).limit(limit).all()
    
    async def _get_unfinished_tasks(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        """Get unfinished tasks from previous sessions"""
        tasks = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"task_status": "in_progress"}),
                Memory.created_at > datetime.utcnow() - timedelta(days=7)
            )
        ).order_by(desc(Memory.importance)).limit(5).all()
        
        return [
            {
                "content": task.content,
                "created_at": task.created_at,
                "session_id": task.session_id,
                "context_needed": task.metadata.get("context_needed", [])
            }
            for task in tasks
        ]
    
    def _identify_current_task_type(self, activities: List[Memory]) -> Optional[str]:
        """Identify the type of task being performed"""
        if not activities:
            return None
        
        # Count task indicators in recent activities
        indicator_counts = defaultdict(int)
        
        for activity in activities[:10]:  # Look at last 10 activities
            content_lower = activity.content.lower()
            for task_type, indicators in self.task_indicators.items():
                for indicator in indicators:
                    if indicator in content_lower:
                        indicator_counts[task_type] += 1
        
        if not indicator_counts:
            return None
        
        # Return most common task type
        return max(indicator_counts.items(), key=lambda x: x[1])[0]
    
    def _predict_from_sequence(
        self,
        current_task_type: str,
        recent_activities: List[Memory]
    ) -> List[Dict[str, Any]]:
        """Predict next tasks based on common sequences"""
        predictions = []
        
        # Map task type to sequence
        sequence_map = {
            "error": "error_fix",
            "feature": "feature_add",
            "refactor": "refactor",
            "test": "debug",
            "deploy": "deployment"
        }
        
        sequence_name = sequence_map.get(current_task_type)
        if not sequence_name or sequence_name not in self.task_sequences:
            return predictions
        
        sequence = self.task_sequences[sequence_name]
        
        # Find current position in sequence
        current_position = self._find_sequence_position(sequence, recent_activities)
        
        if current_position < len(sequence) - 1:
            next_task = sequence[current_position + 1]
            predictions.append({
                "task": f"{next_task.replace('_', ' ').title()}",
                "type": current_task_type,
                "confidence": 0.7,
                "reason": f"Next step in {sequence_name} workflow",
                "context_needed": self._get_context_for_task(next_task)
            })
        
        return predictions
    
    async def _predict_from_patterns(
        self,
        current_context: Dict[str, Any],
        recent_activities: List[Memory]
    ) -> List[Dict[str, Any]]:
        """Predict tasks based on historical patterns"""
        predictions = []
        
        # Look for patterns in similar contexts
        similar_sessions = await self._find_similar_sessions(current_context)
        
        # Analyze what tasks followed in similar sessions
        task_frequencies = Counter()
        
        for session in similar_sessions:
            next_tasks = await self._get_session_tasks(session["session_id"])
            for task in next_tasks[:3]:  # Look at next 3 tasks
                task_frequencies[task.content] += 1
        
        # Convert to predictions
        for task_content, frequency in task_frequencies.most_common(3):
            confidence = min(frequency / len(similar_sessions), 0.8) if similar_sessions else 0
            if confidence > 0.3:
                predictions.append({
                    "task": task_content,
                    "type": "pattern",
                    "confidence": confidence,
                    "reason": f"Common in similar contexts ({frequency} times)",
                    "context_needed": []
                })
        
        return predictions
    
    async def _get_scheduled_tasks(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for scheduled or recurring tasks"""
        scheduled = []
        
        # Check for daily/weekly patterns
        day_of_week = datetime.utcnow().strftime("%A").lower()
        hour_of_day = datetime.utcnow().hour
        
        # Common scheduled tasks
        if day_of_week == "monday" and hour_of_day < 12:
            scheduled.append({
                "task": "Review weekend changes and system status",
                "type": "scheduled",
                "confidence": 0.5,
                "reason": "Monday morning routine",
                "context_needed": ["system_status", "recent_commits"]
            })
        
        # Check for deployment windows
        if day_of_week in ["tuesday", "thursday"] and 14 <= hour_of_day <= 16:
            scheduled.append({
                "task": "Prepare for deployment window",
                "type": "scheduled",
                "confidence": 0.4,
                "reason": "Common deployment time",
                "context_needed": ["pending_changes", "test_results"]
            })
        
        return scheduled
    
    def _deduplicate_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate predictions, keeping highest confidence"""
        seen = {}
        unique = []
        
        for pred in predictions:
            task_key = pred["task"].lower().strip()
            if task_key not in seen or pred["confidence"] > seen[task_key]["confidence"]:
                seen[task_key] = pred
        
        return list(seen.values())
    
    async def _record_predictions(self, session_id: str, predictions: List[Dict[str, Any]]) -> None:
        """Record predictions for later analysis"""
        content = "TASK PREDICTIONS:\n"
        for i, pred in enumerate(predictions, 1):
            content += f"{i}. {pred['task']} (confidence: {pred['confidence']:.2f})\n"
        
        await self.memory_service.create_memory(
            session_id=session_id,
            content=content,
            memory_type="pattern",
            importance=0.3,
            metadata={
                "prediction_type": "task",
                "predictions": predictions,
                "predicted_at": datetime.utcnow().isoformat()
            }
        )
    
    def _find_sequence_position(self, sequence: List[str], activities: List[Memory]) -> int:
        """Find current position in a task sequence"""
        # Simple matching - could be improved with better NLP
        for i, step in enumerate(sequence):
            step_keywords = step.split("_")
            for activity in activities[:5]:  # Check recent activities
                if all(keyword in activity.content.lower() for keyword in step_keywords):
                    return i
        return 0
    
    def _get_context_for_task(self, task: str) -> List[str]:
        """Determine what context is needed for a task"""
        context_map = {
            "identify_error": ["error_logs", "stack_trace", "recent_changes"],
            "search_solution": ["similar_errors", "documentation", "past_solutions"],
            "apply_fix": ["code_location", "dependencies", "test_files"],
            "test_fix": ["test_commands", "expected_output", "regression_tests"],
            "implement": ["requirements", "existing_code", "patterns"],
            "test": ["test_framework", "test_data", "coverage_requirements"]
        }
        
        return context_map.get(task, [])
    
    async def _find_similar_sessions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find sessions with similar context"""
        # This is simplified - in practice would use vector similarity
        similar = []
        
        # Look for sessions with similar project
        if "project_id" in context:
            sessions = self.db.query(Memory).filter(
                and_(
                    Memory.metadata.contains({"project_id": context["project_id"]}),
                    Memory.metadata.contains({"session_start": True})
                )
            ).limit(10).all()
            
            similar.extend([
                {"session_id": s.session_id, "similarity": 0.8}
                for s in sessions
            ])
        
        return similar
    
    async def _get_session_tasks(self, session_id: str) -> List[Memory]:
        """Get tasks from a specific session"""
        return self.db.query(Memory).filter(
            Memory.session_id == session_id
        ).order_by(Memory.created_at).all()
    
    async def _load_error_context(self, error_indicators: List[str]) -> List[Dict[str, Any]]:
        """Load context for error-related tasks"""
        # Simplified - would integrate with error_learning service
        return []
    
    async def _load_similar_features(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Load context for feature implementation"""
        # Simplified - would search for similar implementations
        return []
    
    async def _load_task_context(self, task_id: Optional[str]) -> Dict[str, Any]:
        """Load context for a specific task"""
        if not task_id:
            return {}
        
        # Simplified - would load full task context
        return {
            "previous_attempts": [],
            "related_files": [],
            "dependencies": []
        }