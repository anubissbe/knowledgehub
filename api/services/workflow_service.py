"""
Workflow Automation Service.

This service provides comprehensive workflow management including:
- Pattern detection and learning
- Template management
- Execution orchestration
- Automation rule processing
- Analytics and monitoring
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import json

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, and_, or_, desc

from ..models.base import get_db_context
from ..models.workflow import (
    WorkflowPattern, WorkflowTemplate, WorkflowExecution, TaskExecution,
    AutomationRule, WorkflowStatus, ExecutionStatus, PatternType,
    WorkflowPatternCreate, WorkflowTemplateCreate, WorkflowExecutionCreate,
    AutomationRuleCreate, WorkflowAnalytics
)
from ..models.memory import MemoryItem
from ..models.session import Session
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("workflow_service")


class WorkflowService:
    """
    Comprehensive workflow automation service.
    
    Features:
    - Pattern detection from user behavior
    - Template creation and management
    - Workflow execution orchestration
    - Automation rule processing
    - Performance analytics
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.analytics_service = TimeSeriesAnalyticsService()
        self._initialized = False
        
        # Pattern detection configuration
        self.pattern_config = {
            "min_sequence_length": 3,
            "min_occurrences": 5,
            "confidence_threshold": 0.7,
            "time_window_hours": 168,  # 1 week
            "similarity_threshold": 0.8
        }
        
        logger.info("Initialized WorkflowService")
    
    async def initialize(self):
        """Initialize the workflow service."""
        if self._initialized:
            return
        
        try:
            await self.analytics_service.initialize()
            await redis_client.initialize()
            
            self._initialized = True
            logger.info("WorkflowService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowService: {e}")
            raise
    
    async def detect_patterns(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        force_redetection: bool = False
    ) -> List[WorkflowPattern]:
        """
        Detect workflow patterns from user behavior.
        
        Args:
            user_id: Filter by user ID
            project_id: Filter by project ID
            force_redetection: Force re-detection even if recent
            
        Returns:
            List of detected patterns
        """
        try:
            # Check if recent detection exists
            if not force_redetection:
                cache_key = f"pattern_detection:{user_id}:{project_id}"
                cached = await redis_client.get(cache_key)
                if cached:
                    logger.debug("Using cached pattern detection results")
                    return []  # Return empty to avoid re-processing
            
            with get_db_context() as db:
                # Get user activity sequences
                sequences = await self._extract_activity_sequences(
                    db, user_id, project_id
                )
                
                if not sequences:
                    return []
                
                # Detect sequential patterns
                sequential_patterns = self._detect_sequential_patterns(sequences)
                
                # Detect parallel patterns
                parallel_patterns = self._detect_parallel_patterns(sequences)
                
                # Detect conditional patterns
                conditional_patterns = self._detect_conditional_patterns(sequences)
                
                # Combine all patterns
                all_patterns = (
                    sequential_patterns + 
                    parallel_patterns + 
                    conditional_patterns
                )
                
                # Filter and save high-confidence patterns
                saved_patterns = []
                for pattern_data in all_patterns:
                    if pattern_data["confidence_score"] >= self.pattern_config["confidence_threshold"]:
                        pattern = await self._save_pattern(db, pattern_data)
                        if pattern:
                            saved_patterns.append(pattern)
                
                # Cache results
                await redis_client.setex(
                    f"pattern_detection:{user_id}:{project_id}",
                    3600,  # 1 hour
                    "completed"
                )
                
                # Record analytics
                await self.analytics_service.record_metric(
                    metric_type="patterns_detected",
                    value=len(saved_patterns),
                    tags={
                        "user_id": user_id or "all",
                        "project_id": project_id or "all"
                    }
                )
                
                logger.info(f"Detected {len(saved_patterns)} workflow patterns")
                return saved_patterns
                
        except Exception as e:
            logger.error(f"Failed to detect patterns: {e}")
            return []
    
    async def create_template(
        self,
        template_data: WorkflowTemplateCreate,
        created_by: str
    ) -> WorkflowTemplate:
        """Create a new workflow template."""
        try:
            with get_db_context() as db:
                template = WorkflowTemplate(
                    name=template_data.name,
                    description=template_data.description,
                    template_definition=template_data.template_definition,
                    input_schema=template_data.input_schema,
                    output_schema=template_data.output_schema,
                    trigger_type=template_data.trigger_type.value,
                    trigger_config=template_data.trigger_config,
                    timeout_minutes=template_data.timeout_minutes,
                    retry_count=template_data.retry_count,
                    auto_approval_required=template_data.auto_approval_required,
                    risk_level=template_data.risk_level,
                    estimated_duration=template_data.estimated_duration,
                    pattern_id=UUID(template_data.pattern_id) if template_data.pattern_id else None,
                    created_by=created_by,
                    tags=template_data.tags
                )
                
                db.add(template)
                db.commit()
                db.refresh(template)
                
                logger.info(f"Created workflow template {template.id}")
                return template
                
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            raise
    
    async def execute_workflow(
        self,
        execution_data: WorkflowExecutionCreate,
        executed_by: str
    ) -> WorkflowExecution:
        """
        Execute a workflow from template.
        
        Args:
            execution_data: Execution configuration
            executed_by: User executing the workflow
            
        Returns:
            Workflow execution instance
        """
        try:
            with get_db_context() as db:
                # Get template
                template = db.query(WorkflowTemplate).filter_by(
                    id=UUID(execution_data.template_id)
                ).first()
                
                if not template:
                    raise ValueError("Template not found")
                
                # Create execution
                execution = WorkflowExecution(
                    execution_name=execution_data.execution_name or f"Execution of {template.name}",
                    template_id=template.id,
                    pattern_id=template.pattern_id,
                    status=WorkflowStatus.DRAFT.value,
                    trigger_type=execution_data.trigger_type.value,
                    trigger_data=execution_data.trigger_data,
                    input_data=execution_data.input_data,
                    execution_context=execution_data.execution_context,
                    executed_by=executed_by,
                    project_id=execution_data.project_id,
                    session_id=execution_data.session_id
                )
                
                # Parse template definition
                template_def = template.template_definition
                execution.total_steps = len(template_def.get("steps", []))
                
                db.add(execution)
                db.commit()
                db.refresh(execution)
                
                # Start execution asynchronously
                asyncio.create_task(
                    self._execute_workflow_async(str(execution.id))
                )
                
                logger.info(f"Started workflow execution {execution.id}")
                return execution
                
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            raise
    
    async def create_automation_rule(
        self,
        rule_data: AutomationRuleCreate,
        created_by: str
    ) -> AutomationRule:
        """Create a new automation rule."""
        try:
            with get_db_context() as db:
                rule = AutomationRule(
                    rule_name=rule_data.rule_name,
                    description=rule_data.description,
                    trigger_conditions=rule_data.trigger_conditions,
                    action_template_id=UUID(rule_data.action_template_id),
                    is_active=rule_data.is_active,
                    priority=rule_data.priority,
                    cooldown_minutes=rule_data.cooldown_minutes,
                    requires_approval=rule_data.requires_approval,
                    auto_approve_conditions=rule_data.auto_approve_conditions,
                    approval_timeout_minutes=rule_data.approval_timeout_minutes,
                    max_executions_per_day=rule_data.max_executions_per_day,
                    max_concurrent_executions=rule_data.max_concurrent_executions,
                    user_filters=rule_data.user_filters,
                    project_filters=rule_data.project_filters,
                    time_constraints=rule_data.time_constraints,
                    created_by=created_by
                )
                
                db.add(rule)
                db.commit()
                db.refresh(rule)
                
                logger.info(f"Created automation rule {rule.id}")
                return rule
                
        except Exception as e:
            logger.error(f"Failed to create automation rule: {e}")
            raise
    
    async def check_automation_triggers(
        self,
        context: Dict[str, Any]
    ) -> List[Tuple[AutomationRule, WorkflowTemplate]]:
        """
        Check for automation rules that should be triggered.
        
        Args:
            context: Current context data
            
        Returns:
            List of (rule, template) tuples to execute
        """
        try:
            with get_db_context() as db:
                # Get active automation rules ordered by priority
                rules = db.query(AutomationRule).filter(
                    AutomationRule.is_active == True
                ).order_by(AutomationRule.priority).all()
                
                triggered_rules = []
                
                for rule in rules:
                    # Check if rule can execute
                    can_execute, reason = rule.can_execute(context)
                    if not can_execute:
                        continue
                    
                    # Check trigger conditions
                    if self._evaluate_trigger_conditions(
                        rule.trigger_conditions, context
                    ):
                        # Get template
                        template = db.query(WorkflowTemplate).filter_by(
                            id=rule.action_template_id
                        ).first()
                        
                        if template and template.is_active:
                            triggered_rules.append((rule, template))
                            
                            # Check max concurrent executions
                            if len(triggered_rules) >= rule.max_concurrent_executions:
                                break
                
                return triggered_rules
                
        except Exception as e:
            logger.error(f"Failed to check automation triggers: {e}")
            return []
    
    async def get_workflow_analytics(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        time_window_days: int = 30
    ) -> WorkflowAnalytics:
        """Get comprehensive workflow analytics."""
        try:
            with get_db_context() as db:
                window_start = datetime.utcnow() - timedelta(days=time_window_days)
                
                # Base queries
                patterns_query = db.query(WorkflowPattern)
                templates_query = db.query(WorkflowTemplate)
                executions_query = db.query(WorkflowExecution)
                
                # Apply filters
                if user_id:
                    patterns_query = patterns_query.filter(WorkflowPattern.user_id == user_id)
                    executions_query = executions_query.filter(WorkflowExecution.executed_by == user_id)
                
                if project_id:
                    patterns_query = patterns_query.filter(WorkflowPattern.project_id == project_id)
                    executions_query = executions_query.filter(WorkflowExecution.project_id == project_id)
                
                # Get counts
                total_patterns = patterns_query.count()
                active_templates = templates_query.filter(WorkflowTemplate.is_active == True).count()
                
                # Get executions in time window
                recent_executions = executions_query.filter(
                    WorkflowExecution.created_at >= window_start
                ).all()
                
                total_executions = len(recent_executions)
                successful_executions = len([e for e in recent_executions if e.success])
                success_rate = successful_executions / total_executions if total_executions > 0 else 0
                
                # Calculate average execution time
                completed_executions = [e for e in recent_executions if e.completed_at]
                avg_execution_time = 0
                if completed_executions:
                    total_time = sum(
                        (e.completed_at - e.started_at).total_seconds() / 3600
                        for e in completed_executions
                        if e.started_at
                    )
                    avg_execution_time = total_time / len(completed_executions)
                
                # Calculate time saved (from pattern efficiency)
                total_time_saved = 0
                for pattern in patterns_query.all():
                    if pattern.avg_effort_saved:
                        total_time_saved += pattern.avg_effort_saved * pattern.occurrence_count
                
                # Get top patterns
                top_patterns = patterns_query.order_by(
                    desc(WorkflowPattern.occurrence_count)
                ).limit(10).all()
                
                top_patterns_data = [
                    {
                        "id": str(p.id),
                        "name": p.pattern_name,
                        "type": p.pattern_type,
                        "occurrences": p.occurrence_count,
                        "success_rate": p.success_rate,
                        "automation_potential": p.automation_potential
                    }
                    for p in top_patterns
                ]
                
                # Get execution trends (simplified)
                execution_trends = self._calculate_execution_trends(recent_executions)
                
                # Calculate automation adoption rate
                total_possible_automations = sum(
                    p.occurrence_count for p in patterns_query.all()
                    if p.automation_potential > 0.7
                )
                actual_automations = total_executions
                automation_adoption_rate = (
                    actual_automations / total_possible_automations
                    if total_possible_automations > 0 else 0
                )
                
                # Efficiency metrics
                efficiency_metrics = {
                    "avg_pattern_confidence": patterns_query.with_entities(
                        func.avg(WorkflowPattern.confidence_score)
                    ).scalar() or 0,
                    "avg_automation_potential": patterns_query.with_entities(
                        func.avg(WorkflowPattern.automation_potential)
                    ).scalar() or 0,
                    "workflow_adoption_rate": automation_adoption_rate,
                    "error_rate": 1 - success_rate,
                    "time_savings_per_execution": (
                        total_time_saved / total_executions if total_executions > 0 else 0
                    )
                }
                
                return WorkflowAnalytics(
                    total_patterns=total_patterns,
                    active_templates=active_templates,
                    total_executions=total_executions,
                    success_rate=success_rate,
                    avg_execution_time=avg_execution_time,
                    total_time_saved=total_time_saved,
                    automation_adoption_rate=automation_adoption_rate,
                    top_patterns=top_patterns_data,
                    execution_trends=execution_trends,
                    efficiency_metrics=efficiency_metrics
                )
                
        except Exception as e:
            logger.error(f"Failed to get workflow analytics: {e}")
            raise
    
    # Internal methods
    
    async def _extract_activity_sequences(
        self,
        db: DBSession,
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[List[Dict[str, Any]]]:
        """Extract activity sequences from user behavior."""
        try:
            # Get recent user activities
            query = db.query(MemoryItem)
            
            if user_id:
                query = query.filter(MemoryItem.user_id == user_id)
            
            if project_id:
                query = query.filter(
                    MemoryItem.metadata["project_id"].astext == project_id
                )
            
            # Get activities from last week
            week_ago = datetime.utcnow() - timedelta(
                hours=self.pattern_config["time_window_hours"]
            )
            
            activities = query.filter(
                MemoryItem.created_at >= week_ago
            ).order_by(
                MemoryItem.user_id,
                MemoryItem.created_at
            ).all()
            
            # Group by user and session
            sequences = []
            current_sequence = []
            last_user = None
            last_session = None
            
            for activity in activities:
                current_user = activity.user_id
                current_session = activity.metadata.get("session_id")
                
                # Start new sequence if user or session changed
                if (current_user != last_user or 
                    current_session != last_session):
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = []
                
                # Add activity to sequence
                current_sequence.append({
                    "type": activity.memory_type,
                    "content": activity.content,
                    "timestamp": activity.created_at,
                    "metadata": activity.metadata,
                    "user_id": activity.user_id
                })
                
                last_user = current_user
                last_session = current_session
            
            # Add final sequence
            if current_sequence:
                sequences.append(current_sequence)
            
            # Filter sequences by minimum length
            min_length = self.pattern_config["min_sequence_length"]
            return [seq for seq in sequences if len(seq) >= min_length]
            
        except Exception as e:
            logger.error(f"Failed to extract activity sequences: {e}")
            return []
    
    def _detect_sequential_patterns(
        self,
        sequences: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect sequential workflow patterns."""
        patterns = []
        
        try:
            # Find common subsequences
            subsequences = {}
            
            for sequence in sequences:
                # Generate all subsequences of minimum length
                min_length = self.pattern_config["min_sequence_length"]
                
                for i in range(len(sequence) - min_length + 1):
                    for j in range(i + min_length, len(sequence) + 1):
                        subseq = sequence[i:j]
                        
                        # Create pattern key
                        pattern_key = tuple(
                            (item["type"], self._normalize_content(item["content"]))
                            for item in subseq
                        )
                        
                        if pattern_key not in subsequences:
                            subsequences[pattern_key] = {
                                "occurrences": 0,
                                "sequences": [],
                                "users": set(),
                                "execution_times": []
                            }
                        
                        subsequences[pattern_key]["occurrences"] += 1
                        subsequences[pattern_key]["sequences"].append(subseq)
                        subsequences[pattern_key]["users"].add(subseq[0]["user_id"])
                        
                        # Calculate execution time
                        if len(subseq) > 1:
                            exec_time = (
                                subseq[-1]["timestamp"] - subseq[0]["timestamp"]
                            ).total_seconds() / 3600
                            subsequences[pattern_key]["execution_times"].append(exec_time)
            
            # Filter by minimum occurrences
            min_occurrences = self.pattern_config["min_occurrences"]
            
            for pattern_key, data in subsequences.items():
                if data["occurrences"] >= min_occurrences:
                    # Calculate pattern metrics
                    confidence = min(1.0, data["occurrences"] / len(sequences))
                    automation_potential = self._calculate_automation_potential(
                        pattern_key, data
                    )
                    
                    avg_execution_time = (
                        sum(data["execution_times"]) / len(data["execution_times"])
                        if data["execution_times"] else 0
                    )
                    
                    pattern = {
                        "pattern_name": self._generate_pattern_name(pattern_key),
                        "pattern_type": PatternType.SEQUENTIAL.value,
                        "description": self._generate_pattern_description(pattern_key),
                        "tasks_sequence": [
                            {"type": item[0], "content_template": item[1]}
                            for item in pattern_key
                        ],
                        "occurrence_count": data["occurrences"],
                        "success_rate": 0.9,  # Would calculate from outcomes
                        "confidence_score": confidence,
                        "automation_potential": automation_potential,
                        "avg_execution_time": avg_execution_time,
                        "avg_effort_saved": avg_execution_time * 0.5,  # Estimated
                        "user_count": len(data["users"])
                    }
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect sequential patterns: {e}")
            return []
    
    def _detect_parallel_patterns(
        self,
        sequences: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect parallel workflow patterns."""
        # Simplified - in production would analyze concurrent activities
        return []
    
    def _detect_conditional_patterns(
        self,
        sequences: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect conditional workflow patterns."""
        # Simplified - in production would analyze branching behavior
        return []
    
    async def _save_pattern(
        self,
        db: DBSession,
        pattern_data: Dict[str, Any]
    ) -> Optional[WorkflowPattern]:
        """Save a detected pattern to database."""
        try:
            # Check if similar pattern exists
            existing = db.query(WorkflowPattern).filter(
                WorkflowPattern.pattern_name == pattern_data["pattern_name"],
                WorkflowPattern.pattern_type == pattern_data["pattern_type"]
            ).first()
            
            if existing:
                # Update existing pattern
                existing.occurrence_count = pattern_data["occurrence_count"]
                existing.confidence_score = pattern_data["confidence_score"]
                existing.automation_potential = pattern_data["automation_potential"]
                existing.last_detected = datetime.utcnow()
                
                db.commit()
                return existing
            else:
                # Create new pattern
                pattern = WorkflowPattern(
                    pattern_name=pattern_data["pattern_name"],
                    pattern_type=pattern_data["pattern_type"],
                    description=pattern_data["description"],
                    tasks_sequence=pattern_data["tasks_sequence"],
                    occurrence_count=pattern_data["occurrence_count"],
                    success_rate=pattern_data["success_rate"],
                    confidence_score=pattern_data["confidence_score"],
                    automation_potential=pattern_data["automation_potential"],
                    avg_execution_time=pattern_data["avg_execution_time"],
                    avg_effort_saved=pattern_data["avg_effort_saved"],
                    first_detected=datetime.utcnow(),
                    last_detected=datetime.utcnow(),
                    detection_algorithm="sequence_mining_v1"
                )
                
                db.add(pattern)
                db.commit()
                db.refresh(pattern)
                
                return pattern
                
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
            return None
    
    async def _execute_workflow_async(self, execution_id: str):
        """Execute workflow asynchronously."""
        try:
            with get_db_context() as db:
                execution = db.query(WorkflowExecution).filter_by(
                    id=UUID(execution_id)
                ).first()
                
                if not execution:
                    return
                
                # Update status to running
                execution.status = WorkflowStatus.ACTIVE.value
                execution.started_at = datetime.utcnow()
                db.commit()
                
                # Get template
                template = db.query(WorkflowTemplate).filter_by(
                    id=execution.template_id
                ).first()
                
                if not template:
                    execution.status = WorkflowStatus.FAILED.value
                    execution.error_message = "Template not found"
                    db.commit()
                    return
                
                # Execute steps
                steps = template.template_definition.get("steps", [])
                success = True
                
                for i, step in enumerate(steps):
                    try:
                        # Create task execution
                        task = TaskExecution(
                            task_name=step.get("name", f"Step {i+1}"),
                            task_type=step.get("type", "generic"),
                            workflow_id=execution.id,
                            step_number=i + 1,
                            task_config=step,
                            status=ExecutionStatus.RUNNING.value,
                            started_at=datetime.utcnow()
                        )
                        
                        db.add(task)
                        db.commit()
                        
                        # Execute step (simplified)
                        step_success = await self._execute_step(step, execution.input_data)
                        
                        # Update task
                        task.status = (
                            ExecutionStatus.COMPLETED.value if step_success 
                            else ExecutionStatus.FAILED.value
                        )
                        task.success = step_success
                        task.completed_at = datetime.utcnow()
                        
                        # Update execution progress
                        execution.completed_steps = i + 1
                        execution.progress_percentage = (i + 1) / len(steps) * 100
                        execution.current_step = task.task_name
                        
                        db.commit()
                        
                        if not step_success:
                            success = False
                            break
                            
                    except Exception as e:
                        logger.error(f"Step {i+1} failed: {e}")
                        success = False
                        break
                
                # Update final status
                execution.status = (
                    WorkflowStatus.COMPLETED.value if success 
                    else WorkflowStatus.FAILED.value
                )
                execution.success = success
                execution.completed_at = datetime.utcnow()
                
                db.commit()
                
                # Update template usage
                template.usage_count += 1
                if success:
                    template.success_count += 1
                
                db.commit()
                
                logger.info(f"Workflow execution {execution_id} completed: {success}")
                
        except Exception as e:
            logger.error(f"Failed to execute workflow {execution_id}: {e}")
    
    async def _execute_step(
        self,
        step: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> bool:
        """Execute a single workflow step."""
        # Simplified step execution
        # In production, would have specific executors for different step types
        
        step_type = step.get("type", "generic")
        
        if step_type == "delay":
            # Sleep for specified duration
            duration = step.get("duration_seconds", 1)
            await asyncio.sleep(duration)
            return True
        
        elif step_type == "api_call":
            # Make API call (simplified)
            return True
        
        elif step_type == "condition":
            # Evaluate condition
            condition = step.get("condition", True)
            return bool(condition)
        
        else:
            # Generic step - assume success
            await asyncio.sleep(1)  # Simulate work
            return True
    
    def _evaluate_trigger_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate if trigger conditions are met."""
        # Simplified condition evaluation
        # In production, would have sophisticated rule engine
        
        for key, expected_value in conditions.items():
            if key not in context:
                return False
            
            actual_value = context[key]
            
            if isinstance(expected_value, dict):
                # Handle operators like {"gt": 10}
                for operator, value in expected_value.items():
                    if operator == "gt" and not (actual_value > value):
                        return False
                    elif operator == "lt" and not (actual_value < value):
                        return False
                    elif operator == "eq" and not (actual_value == value):
                        return False
            else:
                # Direct comparison
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _calculate_execution_trends(
        self,
        executions: List[WorkflowExecution]
    ) -> List[Dict[str, Any]]:
        """Calculate execution trends over time."""
        # Group by day and calculate metrics
        daily_stats = {}
        
        for execution in executions:
            date_key = execution.created_at.date().isoformat()
            
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "date": date_key,
                    "total_executions": 0,
                    "successful_executions": 0,
                    "avg_duration": 0,
                    "durations": []
                }
            
            daily_stats[date_key]["total_executions"] += 1
            
            if execution.success:
                daily_stats[date_key]["successful_executions"] += 1
            
            if execution.started_at and execution.completed_at:
                duration = (execution.completed_at - execution.started_at).total_seconds() / 3600
                daily_stats[date_key]["durations"].append(duration)
        
        # Calculate averages
        trends = []
        for stats in daily_stats.values():
            if stats["durations"]:
                stats["avg_duration"] = sum(stats["durations"]) / len(stats["durations"])
            
            trends.append({
                "date": stats["date"],
                "total_executions": stats["total_executions"],
                "success_rate": (
                    stats["successful_executions"] / stats["total_executions"]
                    if stats["total_executions"] > 0 else 0
                ),
                "avg_duration_hours": stats["avg_duration"]
            })
        
        return sorted(trends, key=lambda x: x["date"])
    
    # Helper methods
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for pattern matching."""
        # Remove specific details, keep general structure
        words = content.lower().split()
        
        # Keep only action words and remove specifics
        normalized_words = []
        for word in words:
            if len(word) > 3 and not word.isdigit():
                normalized_words.append(word)
        
        return " ".join(normalized_words[:5])  # Limit to 5 words
    
    def _calculate_automation_potential(
        self,
        pattern_key: tuple,
        data: Dict[str, Any]
    ) -> float:
        """Calculate automation potential for a pattern."""
        base_score = 0.5
        
        # More occurrences = higher potential
        if data["occurrences"] > 10:
            base_score += 0.2
        
        # Multiple users = higher potential
        if len(data["users"]) > 1:
            base_score += 0.2
        
        # Consistent timing = higher potential
        if data["execution_times"]:
            avg_time = sum(data["execution_times"]) / len(data["execution_times"])
            if avg_time < 2:  # Less than 2 hours
                base_score += 0.1
        
        return min(1.0, base_score)
    
    def _generate_pattern_name(self, pattern_key: tuple) -> str:
        """Generate human-readable pattern name."""
        task_types = [item[0] for item in pattern_key]
        return f"Pattern: {' → '.join(task_types)}"
    
    def _generate_pattern_description(self, pattern_key: tuple) -> str:
        """Generate pattern description."""
        steps = [f"{item[0]}: {item[1][:50]}..." for item in pattern_key]
        return f"Sequential workflow with steps: {'; '.join(steps)}"


# Global service instance
workflow_service = WorkflowService()