"""
Proactive Assistant - Anticipates needs and provides assistance before being asked
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import cast, String, desc, and_, func, or_

from ..models.memory import MemoryItem
from .project_context_manager import ProjectContextManager
from .mistake_learning_system import MistakeLearningSystem


class ProactiveAssistant:
    """Provides proactive assistance by anticipating needs"""
    
    def __init__(self):
        self.project_manager = ProjectContextManager()
        self.mistake_learner = MistakeLearningSystem()
        self.reminders_file = Path.home() / ".claude_reminders.json"
        self.predictions_file = Path.home() / ".claude_predictions.json"
        self.context_cache_file = Path.home() / ".claude_context_cache.json"
        
        # Patterns that indicate work states
        self.work_patterns = {
            "todo": [r"\bTODO\b", r"\bFIXME\b", r"\bHACK\b", r"\bNOTE\b"],
            "in_progress": [r"implementing", r"working on", r"fixing", r"debugging"],
            "blocked": [r"blocked by", r"waiting for", r"depends on", r"need to"],
            "questions": [r"\?$", r"how to", r"what is", r"why does"],
            "errors": [r"error", r"failed", r"exception", r"bug"]
        }
        
        # Task priority indicators
        self.priority_indicators = {
            "critical": ["CRITICAL", "URGENT", "ASAP", "immediately", "breaking"],
            "high": ["important", "priority", "needed", "required"],
            "medium": ["should", "would be nice", "consider"],
            "low": ["maybe", "eventually", "someday", "nice to have"]
        }
    
    def analyze_session_state(self, db: Session, session_id: str, 
                            project_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze current session state and provide proactive insights"""
        
        # Get recent session activities
        recent_memories = self._get_recent_session_memories(db, session_id, hours=4)
        
        # Analyze work patterns
        work_state = self._analyze_work_patterns(recent_memories)
        
        # Find incomplete tasks
        incomplete_tasks = self._find_incomplete_tasks(db, session_id, project_id)
        
        # Check for unresolved errors
        unresolved_errors = self._find_unresolved_errors(db, session_id, project_id)
        
        # Predict next actions
        predictions = self._predict_next_actions(
            work_state, incomplete_tasks, unresolved_errors, project_id
        )
        
        # Generate reminders
        reminders = self._generate_reminders(
            incomplete_tasks, unresolved_errors, work_state
        )
        
        # Preload relevant context
        preloaded_context = self._preload_relevant_context(
            db, predictions, project_id
        )
        
        return {
            "session_id": session_id,
            "work_state": work_state,
            "incomplete_tasks": incomplete_tasks,
            "unresolved_errors": unresolved_errors,
            "predictions": predictions,
            "reminders": reminders,
            "preloaded_context": preloaded_context,
            "proactive_suggestions": self._generate_suggestions(
                work_state, predictions, reminders
            )
        }
    
    def _get_recent_session_memories(self, db: Session, session_id: str, 
                                   hours: int = 4) -> List[MemoryItem]:
        """Get recent memories from current session"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        memories = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains(f'"session_id": "{session_id}"'),
                MemoryItem.created_at > cutoff_time
            )
        ).order_by(desc(MemoryItem.created_at)).limit(50).all()
        
        return memories
    
    def _analyze_work_patterns(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Analyze patterns in recent work"""
        patterns = {
            "todos_found": 0,
            "in_progress_items": 0,
            "blocked_items": 0,
            "questions_asked": 0,
            "errors_encountered": 0,
            "current_focus": None,
            "work_velocity": "normal"
        }
        
        # Analyze each memory
        for memory in memories:
            content = memory.content.lower()
            
            for pattern_type, regexes in self.work_patterns.items():
                for regex in regexes:
                    if re.search(regex, content, re.IGNORECASE):
                        patterns[f"{pattern_type}_found"] += 1
        
        # Determine current focus
        if patterns["errors_encountered"] > 2:
            patterns["current_focus"] = "debugging"
        elif patterns["blocked_items"] > 0:
            patterns["current_focus"] = "blocked"
        elif patterns["in_progress_items"] > 0:
            patterns["current_focus"] = "implementation"
        elif patterns["questions_asked"] > 2:
            patterns["current_focus"] = "research"
        
        # Calculate work velocity
        total_activities = len(memories)
        if total_activities > 20:
            patterns["work_velocity"] = "high"
        elif total_activities < 5:
            patterns["work_velocity"] = "low"
        
        return patterns
    
    def _find_incomplete_tasks(self, db: Session, session_id: str, 
                              project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find tasks that were started but not completed"""
        incomplete = []
        
        # Look for TODOs and tasks in memories
        query = db.query(MemoryItem).filter(
            or_(
                cast(MemoryItem.meta_data, String).contains('"todo"'),
                cast(MemoryItem.meta_data, String).contains('"task"'),
                MemoryItem.content.ilike('%TODO%'),
                MemoryItem.content.ilike('%FIXME%')
            )
        )
        
        if project_id:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            )
        
        task_memories = query.order_by(desc(MemoryItem.created_at)).limit(20).all()
        
        for memory in task_memories:
            # Extract task description
            task_match = re.search(r'TODO:?\s*(.+?)(?:\n|$)', memory.content, re.IGNORECASE)
            if task_match:
                task_desc = task_match.group(1)
                
                # Check if this task was marked complete
                if not self._is_task_completed(db, task_desc, memory.created_at):
                    priority = self._determine_task_priority(task_desc)
                    incomplete.append({
                        "task": task_desc,
                        "created": memory.created_at.isoformat(),
                        "age_hours": (datetime.now(timezone.utc) - memory.created_at).total_seconds() / 3600,
                        "priority": priority,
                        "source": "todo"
                    })
        
        # Also check handoff notes
        handoff_memories = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains('"handoff": true')
        ).order_by(desc(MemoryItem.created_at)).limit(5).all()
        
        for memory in handoff_memories:
            next_tasks = memory.meta_data.get("next_tasks", [])
            for task in next_tasks:
                if not self._is_task_completed(db, task, memory.created_at):
                    incomplete.append({
                        "task": task,
                        "created": memory.created_at.isoformat(),
                        "age_hours": (datetime.now(timezone.utc) - memory.created_at).total_seconds() / 3600,
                        "priority": "high",  # Handoff tasks are usually important
                        "source": "handoff"
                    })
        
        # Sort by priority and age
        incomplete.sort(key=lambda x: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["priority"], 4),
            -x["age_hours"]
        ))
        
        return incomplete[:10]  # Top 10 incomplete tasks
    
    def _is_task_completed(self, db: Session, task_desc: str, 
                          created_after: datetime) -> bool:
        """Check if a task has been marked as completed"""
        # Look for completion indicators
        completion_terms = ["completed", "done", "fixed", "implemented", "resolved"]
        
        # Search for memories mentioning this task with completion terms
        for term in completion_terms:
            completed = db.query(MemoryItem).filter(
                and_(
                    MemoryItem.content.ilike(f"%{task_desc[:30]}%"),
                    MemoryItem.content.ilike(f"%{term}%"),
                    MemoryItem.created_at > created_after
                )
            ).first()
            
            if completed:
                return True
        
        return False
    
    def _determine_task_priority(self, task_desc: str) -> str:
        """Determine priority level of a task"""
        task_lower = task_desc.lower()
        
        for priority, indicators in self.priority_indicators.items():
            for indicator in indicators:
                if indicator.lower() in task_lower:
                    return priority
        
        return "medium"
    
    def _find_unresolved_errors(self, db: Session, session_id: str,
                               project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find errors that haven't been resolved"""
        unresolved = []
        
        # Get recent errors
        error_memories = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "error"'),
                or_(
                    cast(MemoryItem.meta_data, String).contains('"success": false'),
                    cast(MemoryItem.meta_data, String).contains('"solution": null')
                )
            )
        ).order_by(desc(MemoryItem.created_at)).limit(10).all()
        
        for error in error_memories:
            meta = error.meta_data
            unresolved.append({
                "error_type": meta.get("error_type", "Unknown"),
                "error_message": meta.get("error_message", "")[:100],
                "attempts": meta.get("repetition_count", 1),
                "last_seen": error.accessed_at.isoformat() if error.accessed_at else error.created_at.isoformat(),
                "suggested_solution": self._suggest_error_solution(meta.get("error_type"), meta.get("error_message"))
            })
        
        return unresolved
    
    def _suggest_error_solution(self, error_type: str, error_message: str) -> Optional[str]:
        """Suggest solution for common errors"""
        # Common error solutions
        solutions = {
            "ImportError": "Check if module is installed: pip install <module>",
            "AttributeError": "Verify object exists and has the attribute",
            "TypeError": "Check data types match expected values",
            "TimeoutError": "Increase timeout or optimize performance",
            "PermissionError": "Check file permissions or run with appropriate privileges"
        }
        
        return solutions.get(error_type, "Check documentation for this error type")
    
    def _predict_next_actions(self, work_state: Dict[str, Any],
                             incomplete_tasks: List[Dict[str, Any]],
                             unresolved_errors: List[Dict[str, Any]],
                             project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Predict likely next actions based on current state"""
        predictions = []
        
        # If blocked, suggest unblocking actions
        if work_state["current_focus"] == "blocked":
            predictions.append({
                "action": "Resolve blocking issues",
                "confidence": 0.9,
                "reason": "Work appears to be blocked",
                "type": "unblock"
            })
        
        # If debugging, suggest fixing errors
        if work_state["current_focus"] == "debugging" or unresolved_errors:
            for error in unresolved_errors[:2]:
                predictions.append({
                    "action": f"Fix {error['error_type']}: {error['suggested_solution']}",
                    "confidence": 0.85,
                    "reason": f"Unresolved error with {error['attempts']} attempts",
                    "type": "error_fix"
                })
        
        # Suggest continuing incomplete tasks
        for task in incomplete_tasks[:3]:
            confidence = 0.8 if task["priority"] == "critical" else 0.7
            predictions.append({
                "action": f"Continue: {task['task'][:80]}",
                "confidence": confidence,
                "reason": f"{task['priority']} priority task from {task['source']}",
                "type": "task_continuation"
            })
        
        # If high velocity, suggest taking a break
        if work_state["work_velocity"] == "high":
            predictions.append({
                "action": "Consider taking a break or creating a checkpoint",
                "confidence": 0.6,
                "reason": "High activity detected - checkpoint recommended",
                "type": "maintenance"
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return predictions[:5]
    
    def _generate_reminders(self, incomplete_tasks: List[Dict[str, Any]],
                           unresolved_errors: List[Dict[str, Any]],
                           work_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate helpful reminders"""
        reminders = []
        
        # Remind about old incomplete tasks
        for task in incomplete_tasks:
            if task["age_hours"] > 24:
                reminders.append({
                    "type": "overdue_task",
                    "message": f"Task pending for {int(task['age_hours'])} hours: {task['task'][:60]}...",
                    "priority": task["priority"],
                    "action": "Consider completing or removing if no longer relevant"
                })
        
        # Remind about repeated errors
        if unresolved_errors:
            error_types = Counter(e["error_type"] for e in unresolved_errors)
            for error_type, count in error_types.items():
                if count > 1:
                    reminders.append({
                        "type": "repeated_error",
                        "message": f"{error_type} occurred {count} times",
                        "priority": "high",
                        "action": "Consider finding a permanent solution"
                    })
        
        # Remind to create checkpoint if lots of work
        if work_state.get("work_velocity") == "high" and len(incomplete_tasks) > 3:
            reminders.append({
                "type": "checkpoint",
                "message": "High activity with multiple tasks in progress",
                "priority": "medium",
                "action": "Create a checkpoint to save progress"
            })
        
        # Remind about questions
        if work_state.get("questions_asked", 0) > 2:
            reminders.append({
                "type": "unanswered_questions",
                "message": f"{work_state['questions_asked']} questions detected",
                "priority": "medium",
                "action": "Review and answer questions or research solutions"
            })
        
        return reminders
    
    def _preload_relevant_context(self, db: Session, 
                                 predictions: List[Dict[str, Any]],
                                 project_id: Optional[str] = None) -> Dict[str, Any]:
        """Preload context that might be needed"""
        context = {
            "relevant_files": [],
            "related_errors": [],
            "similar_solutions": [],
            "project_patterns": {}
        }
        
        # For each prediction, load relevant context
        for pred in predictions[:3]:
            if pred["type"] == "error_fix":
                # Load similar errors and solutions
                error_match = re.search(r'Fix (\w+):', pred["action"])
                if error_match:
                    error_type = error_match.group(1)
                    similar_errors = self.mistake_learner._find_similar_mistakes(
                        db, error_type, "", project_id
                    )
                    for err in similar_errors[:2]:
                        if err.meta_data.get("successful_solution"):
                            context["similar_solutions"].append({
                                "error": err.meta_data.get("error_message", "")[:100],
                                "solution": err.meta_data.get("successful_solution"),
                                "worked": True
                            })
            
            elif pred["type"] == "task_continuation":
                # Load project patterns if working on code
                if project_id and not context["project_patterns"]:
                    context["project_patterns"] = self.project_manager.get_project_conventions(
                        project_id
                    )
        
        return context
    
    def _generate_suggestions(self, work_state: Dict[str, Any],
                            predictions: List[Dict[str, Any]],
                            reminders: List[Dict[str, Any]]) -> List[str]:
        """Generate proactive suggestions"""
        suggestions = []
        
        # Top prediction
        if predictions:
            top_pred = predictions[0]
            suggestions.append(f"Next recommended action: {top_pred['action']} (confidence: {top_pred['confidence']:.0%})")
        
        # Critical reminders
        critical_reminders = [r for r in reminders if r.get("priority") == "critical"]
        if critical_reminders:
            suggestions.append(f"⚠️ Critical: {critical_reminders[0]['message']}")
        
        # Work state advice
        if work_state["current_focus"] == "blocked":
            suggestions.append("You appear blocked. Consider asking for help or working on a different task.")
        elif work_state["current_focus"] == "debugging":
            suggestions.append("Multiple errors detected. Consider systematic debugging or checking logs.")
        
        return suggestions[:3]
    
    def get_session_brief(self, db: Session, session_id: str,
                         project_id: Optional[str] = None) -> str:
        """Get a brief proactive summary for session start"""
        analysis = self.analyze_session_state(db, session_id, project_id)
        
        lines = ["🤖 Proactive Assistant Summary"]
        lines.append("=" * 40)
        
        # Incomplete tasks
        if analysis["incomplete_tasks"]:
            lines.append(f"\n📝 {len(analysis['incomplete_tasks'])} incomplete tasks:")
            for task in analysis["incomplete_tasks"][:3]:
                age = f"({int(task['age_hours'])}h ago)" if task['age_hours'] > 1 else "(recent)"
                lines.append(f"  - {task['task'][:60]}... {age}")
        
        # Unresolved errors
        if analysis["unresolved_errors"]:
            lines.append(f"\n⚠️  {len(analysis['unresolved_errors'])} unresolved errors:")
            for error in analysis["unresolved_errors"][:2]:
                lines.append(f"  - {error['error_type']}: {error['suggested_solution']}")
        
        # Suggestions
        if analysis["proactive_suggestions"]:
            lines.append("\n💡 Suggestions:")
            for suggestion in analysis["proactive_suggestions"]:
                lines.append(f"  - {suggestion}")
        
        # Preloaded context
        if analysis["preloaded_context"]["similar_solutions"]:
            lines.append("\n🔧 Relevant solutions from history:")
            for sol in analysis["preloaded_context"]["similar_solutions"][:2]:
                lines.append(f"  - {sol['solution']}")
        
        return "\n".join(lines)
    
    def should_interrupt(self, action: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine if we should proactively interrupt with assistance"""
        # Check if action might cause known issue
        prevention = self.mistake_learner.check_for_prevention(action, context)
        if prevention and prevention.get("confidence", 0) > 0.7:
            return {
                "interrupt": True,
                "reason": "Known issue detected",
                "message": prevention["suggestion"],
                "priority": "high"
            }
        
        # Check if action relates to incomplete critical task
        if "TODO" in action or "FIXME" in action:
            return {
                "interrupt": True,
                "reason": "TODO detected",
                "message": "Would you like me to track this as a task?",
                "priority": "low"
            }
        
        return None