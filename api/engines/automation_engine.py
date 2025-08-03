"""
Automation Engine for Intelligent Workflow Execution.

This engine provides:
- Real-time event monitoring
- Intelligent rule evaluation
- Automated workflow triggering
- Smart execution management
- Performance optimization
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import and_, or_

from ..models.base import get_db_context
from ..models.workflow import (
    AutomationRule, WorkflowTemplate, WorkflowExecution,
    WorkflowStatus, WorkflowTriggerType
)
from ..models.memory import MemoryItem
from ..models.session import Session
from ..services.workflow_service import workflow_service
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("automation_engine")


class EventType(str, Enum):
    """Types of automation events."""
    MEMORY_CREATED = "memory_created"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    ERROR_OCCURRED = "error_occurred"
    DECISION_MADE = "decision_made"
    PATTERN_DETECTED = "pattern_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    TIME_TRIGGER = "time_trigger"
    USER_ACTION = "user_action"


class TriggerResult(str, Enum):
    """Results of trigger evaluation."""
    TRIGGERED = "triggered"
    BLOCKED = "blocked"
    DELAYED = "delayed"
    IGNORED = "ignored"


@dataclass
class AutomationEvent:
    """Automation event data."""
    event_type: EventType
    data: Dict[str, Any]
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class TriggerEvaluation:
    """Result of trigger evaluation."""
    rule_id: str
    result: TriggerResult
    confidence: float
    reason: str
    execution_delay: Optional[int] = None  # seconds
    context: Dict[str, Any] = None


class AutomationEngine:
    """
    Intelligent automation engine for workflow management.
    
    Features:
    - Real-time event processing
    - Smart rule evaluation
    - Context-aware triggering
    - Performance optimization
    - Learning from outcomes
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.analytics_service = TimeSeriesAnalyticsService()
        
        # Engine state
        self._running = False
        self._event_queue = asyncio.Queue()
        self._active_executions = {}
        self._rule_cache = {}
        self._performance_stats = {}
        
        # Event handlers
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        
        # Configuration
        self.max_concurrent_executions = 10
        self.rule_cache_ttl = 300  # 5 minutes
        self.event_batch_size = 50
        self.evaluation_timeout = 30  # seconds
        
        logger.info("Initialized AutomationEngine")
    
    async def start(self):
        """Start the automation engine."""
        if self._running:
            logger.warning("Automation engine already running")
            return
        
        self._running = True
        logger.info("Starting automation engine")
        
        try:
            # Initialize services
            await self.analytics_service.initialize()
            await redis_client.initialize()
            await workflow_service.initialize()
            
            # Load automation rules
            await self._load_automation_rules()
            
            # Start event processing loop
            asyncio.create_task(self._event_processing_loop())
            
            # Start periodic tasks
            asyncio.create_task(self._periodic_rule_refresh())
            asyncio.create_task(self._periodic_cleanup())
            asyncio.create_task(self._time_based_triggers())
            
            logger.info("Automation engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start automation engine: {e}")
            self._running = False
            raise
    
    async def stop(self):
        """Stop the automation engine."""
        logger.info("Stopping automation engine")
        self._running = False
        
        # Wait for pending executions to complete
        if self._active_executions:
            logger.info(f"Waiting for {len(self._active_executions)} executions to complete")
            await asyncio.sleep(5)  # Grace period
        
        # Cleanup
        await self.analytics_service.cleanup()
        
        logger.info("Automation engine stopped")
    
    async def trigger_event(self, event: AutomationEvent):
        """Trigger an automation event."""
        try:
            # Add to event queue
            await self._event_queue.put(event)
            
            # Record event metrics
            await self.analytics_service.record_metric(
                metric_type="automation_event",
                value=1,
                tags={
                    "event_type": event.event_type.value,
                    "user_id": event.user_id or "unknown",
                    "project_id": event.project_id or "unknown"
                }
            )
            
            logger.debug(f"Triggered automation event: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Failed to trigger event: {e}")
    
    async def register_event_handler(
        self,
        event_type: EventType,
        handler: Callable[[AutomationEvent], None]
    ):
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
    
    async def evaluate_rules(
        self,
        event: AutomationEvent
    ) -> List[TriggerEvaluation]:
        """Evaluate automation rules for an event."""
        try:
            evaluations = []
            
            # Get applicable rules
            rules = await self._get_applicable_rules(event)
            
            for rule in rules:
                # Evaluate rule
                evaluation = await self._evaluate_single_rule(rule, event)
                evaluations.append(evaluation)
                
                # Record evaluation metrics
                await self.analytics_service.record_metric(
                    metric_type="rule_evaluation",
                    value=1,
                    tags={
                        "rule_id": str(rule.id),
                        "result": evaluation.result.value,
                        "confidence": evaluation.confidence
                    }
                )
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Failed to evaluate rules: {e}")
            return []
    
    async def execute_triggered_workflows(
        self,
        evaluations: List[TriggerEvaluation],
        event: AutomationEvent
    ) -> List[str]:
        """Execute workflows from trigger evaluations."""
        executed_workflows = []
        
        try:
            # Sort by priority and confidence
            triggered_evaluations = [
                e for e in evaluations 
                if e.result == TriggerResult.TRIGGERED
            ]
            
            triggered_evaluations.sort(
                key=lambda x: (-x.confidence, x.rule_id)
            )
            
            for evaluation in triggered_evaluations:
                # Check execution limits
                if len(self._active_executions) >= self.max_concurrent_executions:
                    logger.warning("Max concurrent executions reached")
                    break
                
                # Execute workflow
                execution_id = await self._execute_workflow_from_rule(
                    evaluation, event
                )
                
                if execution_id:
                    executed_workflows.append(execution_id)
                    self._active_executions[execution_id] = {
                        "started_at": datetime.utcnow(),
                        "rule_id": evaluation.rule_id,
                        "event_type": event.event_type.value
                    }
            
            return executed_workflows
            
        except Exception as e:
            logger.error(f"Failed to execute triggered workflows: {e}")
            return []
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get automation engine status."""
        try:
            # Get recent performance stats
            recent_events = await redis_client.get("automation_events_count") or 0
            recent_executions = len(self._active_executions)
            
            # Calculate success rate
            with get_db_context() as db:
                total_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                successful_executions = db.query(WorkflowExecution).filter(
                    WorkflowExecution.created_at >= datetime.utcnow() - timedelta(hours=24),
                    WorkflowExecution.success == True
                ).count()
                
                success_rate = (
                    successful_executions / total_executions 
                    if total_executions > 0 else 0
                )
            
            return {
                "status": "running" if self._running else "stopped",
                "active_executions": recent_executions,
                "events_processed_24h": recent_events,
                "success_rate_24h": success_rate,
                "rule_cache_size": len(self._rule_cache),
                "queue_size": self._event_queue.qsize(),
                "performance_stats": self._performance_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get engine status: {e}")
            return {"status": "error", "error": str(e)}
    
    # Internal methods
    
    async def _event_processing_loop(self):
        """Main event processing loop."""
        while self._running:
            try:
                # Get events in batches
                events = []
                deadline = asyncio.get_event_loop().time() + 1.0  # 1 second timeout
                
                while (len(events) < self.event_batch_size and 
                       asyncio.get_event_loop().time() < deadline):
                    try:
                        event = await asyncio.wait_for(
                            self._event_queue.get(), 
                            timeout=0.1
                        )
                        events.append(event)
                    except asyncio.TimeoutError:
                        break
                
                if not events:
                    continue
                
                # Process events
                for event in events:
                    await self._process_single_event(event)
                
                # Update performance stats
                self._performance_stats["events_processed"] = (
                    self._performance_stats.get("events_processed", 0) + len(events)
                )
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_event(self, event: AutomationEvent):
        """Process a single automation event."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Call registered handlers
            handlers = self._event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
            
            # Evaluate automation rules
            evaluations = await self._evaluate_rules_with_timeout(event)
            
            # Execute triggered workflows
            if evaluations:
                executed_workflows = await self.execute_triggered_workflows(
                    evaluations, event
                )
                
                if executed_workflows:
                    logger.info(
                        f"Executed {len(executed_workflows)} workflows "
                        f"for event {event.event_type}"
                    )
            
            # Record processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            self._performance_stats["avg_processing_time"] = (
                (self._performance_stats.get("avg_processing_time", 0) * 0.9) +
                (processing_time * 0.1)
            )
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_type}: {e}")
    
    async def _evaluate_rules_with_timeout(
        self,
        event: AutomationEvent
    ) -> List[TriggerEvaluation]:
        """Evaluate rules with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.evaluate_rules(event),
                timeout=self.evaluation_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Rule evaluation timed out for {event.event_type}")
            return []
    
    async def _load_automation_rules(self):
        """Load automation rules into cache."""
        try:
            with get_db_context() as db:
                rules = db.query(AutomationRule).filter(
                    AutomationRule.is_active == True
                ).all()
                
                self._rule_cache = {
                    str(rule.id): {
                        "rule": rule,
                        "cached_at": datetime.utcnow()
                    }
                    for rule in rules
                }
                
                logger.info(f"Loaded {len(rules)} automation rules")
                
        except Exception as e:
            logger.error(f"Failed to load automation rules: {e}")
    
    async def _get_applicable_rules(
        self,
        event: AutomationEvent
    ) -> List[AutomationRule]:
        """Get rules applicable to an event."""
        applicable_rules = []
        
        for rule_data in self._rule_cache.values():
            rule = rule_data["rule"]
            
            # Check if rule matches event context
            if self._rule_matches_event(rule, event):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_matches_event(
        self,
        rule: AutomationRule,
        event: AutomationEvent
    ) -> bool:
        """Check if a rule matches an event."""
        # Check user filters
        if rule.user_filters and event.user_id not in rule.user_filters:
            return False
        
        # Check project filters
        if rule.project_filters and event.project_id not in rule.project_filters:
            return False
        
        # Check time constraints
        if rule.time_constraints:
            if not self._check_time_constraints(rule.time_constraints):
                return False
        
        # Check trigger conditions
        return self._check_trigger_conditions(
            rule.trigger_conditions, event
        )
    
    def _check_time_constraints(
        self,
        constraints: Dict[str, Any]
    ) -> bool:
        """Check if current time matches constraints."""
        now = datetime.utcnow()
        
        # Check allowed hours
        if "allowed_hours" in constraints:
            if now.hour not in constraints["allowed_hours"]:
                return False
        
        # Check allowed days
        if "allowed_days" in constraints:
            if now.weekday() not in constraints["allowed_days"]:
                return False
        
        return True
    
    def _check_trigger_conditions(
        self,
        conditions: Dict[str, Any],
        event: AutomationEvent
    ) -> bool:
        """Check if trigger conditions are met."""
        # Event type condition
        if "event_type" in conditions:
            if event.event_type.value != conditions["event_type"]:
                return False
        
        # Data conditions
        if "data" in conditions:
            data_conditions = conditions["data"]
            for key, expected in data_conditions.items():
                if key not in event.data:
                    return False
                
                actual = event.data[key]
                if not self._compare_values(actual, expected):
                    return False
        
        return True
    
    def _compare_values(self, actual: Any, expected: Any) -> bool:
        """Compare actual vs expected values with operators."""
        if isinstance(expected, dict):
            # Handle operators
            for op, value in expected.items():
                if op == "eq" and actual != value:
                    return False
                elif op == "gt" and not (actual > value):
                    return False
                elif op == "lt" and not (actual < value):
                    return False
                elif op == "in" and actual not in value:
                    return False
                elif op == "contains" and value not in str(actual):
                    return False
        else:
            # Direct comparison
            return actual == expected
        
        return True
    
    async def _evaluate_single_rule(
        self,
        rule: AutomationRule,
        event: AutomationEvent
    ) -> TriggerEvaluation:
        """Evaluate a single automation rule."""
        try:
            # Check if rule can execute
            can_execute, reason = rule.can_execute({
                "user_id": event.user_id,
                "project_id": event.project_id,
                "timestamp": event.timestamp
            })
            
            if not can_execute:
                return TriggerEvaluation(
                    rule_id=str(rule.id),
                    result=TriggerResult.BLOCKED,
                    confidence=0.0,
                    reason=reason
                )
            
            # Calculate confidence based on rule performance
            confidence = await self._calculate_rule_confidence(rule, event)
            
            # Determine if should trigger
            if confidence > 0.7:  # High confidence threshold
                result = TriggerResult.TRIGGERED
            elif confidence > 0.4:  # Medium confidence - delay
                result = TriggerResult.DELAYED
                execution_delay = 300  # 5 minutes
            else:
                result = TriggerResult.IGNORED
            
            return TriggerEvaluation(
                rule_id=str(rule.id),
                result=result,
                confidence=confidence,
                reason=f"Confidence: {confidence:.2f}",
                execution_delay=execution_delay if result == TriggerResult.DELAYED else None,
                context={
                    "event_type": event.event_type.value,
                    "rule_priority": rule.priority
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.id}: {e}")
            return TriggerEvaluation(
                rule_id=str(rule.id),
                result=TriggerResult.IGNORED,
                confidence=0.0,
                reason=f"Evaluation error: {str(e)}"
            )
    
    async def _calculate_rule_confidence(
        self,
        rule: AutomationRule,
        event: AutomationEvent
    ) -> float:
        """Calculate confidence score for rule execution."""
        base_confidence = 0.5
        
        # Historical success rate
        if rule.execution_count > 0:
            success_rate = rule.success_count / rule.execution_count
            base_confidence += (success_rate - 0.5) * 0.4
        
        # Event type alignment
        if "event_type" in rule.trigger_conditions:
            if rule.trigger_conditions["event_type"] == event.event_type.value:
                base_confidence += 0.2
        
        # Time since last execution (prefer some spacing)
        if rule.last_execution:
            hours_since = (datetime.utcnow() - rule.last_execution).total_seconds() / 3600
            if hours_since > rule.cooldown_minutes / 60:
                base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    async def _execute_workflow_from_rule(
        self,
        evaluation: TriggerEvaluation,
        event: AutomationEvent
    ) -> Optional[str]:
        """Execute workflow from rule evaluation."""
        try:
            with get_db_context() as db:
                rule = db.query(AutomationRule).filter_by(
                    id=UUID(evaluation.rule_id)
                ).first()
                
                if not rule:
                    return None
                
                template = db.query(WorkflowTemplate).filter_by(
                    id=rule.action_template_id
                ).first()
                
                if not template:
                    return None
                
                # Create execution data
                from ..models.workflow import WorkflowExecutionCreate
                
                execution_data = WorkflowExecutionCreate(
                    execution_name=f"Auto: {template.name}",
                    template_id=str(template.id),
                    trigger_type=WorkflowTriggerType.EVENT_BASED,
                    trigger_data={
                        "event_type": event.event_type.value,
                        "event_data": event.data,
                        "rule_id": evaluation.rule_id,
                        "confidence": evaluation.confidence
                    },
                    input_data=event.data,
                    execution_context={
                        "automated": True,
                        "rule_triggered": True,
                        "event_timestamp": event.timestamp.isoformat()
                    },
                    project_id=event.project_id,
                    session_id=event.session_id
                )
                
                # Execute workflow
                execution = await workflow_service.execute_workflow(
                    execution_data,
                    executed_by=event.user_id or "automation_engine"
                )
                
                # Update rule statistics
                rule.execution_count += 1
                rule.last_execution = datetime.utcnow()
                db.commit()
                
                logger.info(
                    f"Executed workflow {execution.id} from rule {rule.id}"
                )
                
                return str(execution.id)
                
        except Exception as e:
            logger.error(f"Failed to execute workflow from rule: {e}")
            return None
    
    async def _periodic_rule_refresh(self):
        """Periodically refresh automation rules."""
        while self._running:
            try:
                await asyncio.sleep(self.rule_cache_ttl)
                await self._load_automation_rules()
                
            except Exception as e:
                logger.error(f"Failed to refresh rules: {e}")
    
    async def _periodic_cleanup(self):
        """Periodically clean up completed executions."""
        while self._running:
            try:
                await asyncio.sleep(600)  # 10 minutes
                
                # Remove completed executions from active list
                completed_executions = []
                for exec_id, exec_data in self._active_executions.items():
                    # Check if execution completed (simplified)
                    if (datetime.utcnow() - exec_data["started_at"]).total_seconds() > 3600:
                        completed_executions.append(exec_id)
                
                for exec_id in completed_executions:
                    del self._active_executions[exec_id]
                
                logger.debug(f"Cleaned up {len(completed_executions)} executions")
                
            except Exception as e:
                logger.error(f"Failed to cleanup executions: {e}")
    
    async def _time_based_triggers(self):
        """Handle time-based workflow triggers."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for time-based rules
                with get_db_context() as db:
                    time_rules = db.query(AutomationRule).filter(
                        AutomationRule.is_active == True,
                        AutomationRule.trigger_conditions.contains({"event_type": "time_trigger"})
                    ).all()
                    
                    for rule in time_rules:
                        # Check if it's time to trigger
                        if self._should_trigger_time_rule(rule):
                            # Create time trigger event
                            event = AutomationEvent(
                                event_type=EventType.TIME_TRIGGER,
                                data={"rule_id": str(rule.id)},
                                timestamp=datetime.utcnow()
                            )
                            
                            await self.trigger_event(event)
                
            except Exception as e:
                logger.error(f"Failed to process time triggers: {e}")
    
    def _should_trigger_time_rule(self, rule: AutomationRule) -> bool:
        """Check if time-based rule should trigger."""
        # Simplified time rule evaluation
        # In production, would support cron-like expressions
        
        if not rule.last_execution:
            return True
        
        # Check if enough time has passed
        cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
        return datetime.utcnow() - rule.last_execution >= cooldown_delta


# Global automation engine instance
automation_engine = AutomationEngine()


# Convenience functions for common events

async def trigger_memory_created(
    memory_item: MemoryItem,
    user_id: str,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Trigger automation for memory creation."""
    event = AutomationEvent(
        event_type=EventType.MEMORY_CREATED,
        data={
            "memory_id": str(memory_item.id),
            "memory_type": memory_item.memory_type,
            "content_length": len(memory_item.content),
            "has_metadata": bool(memory_item.metadata)
        },
        user_id=user_id,
        project_id=project_id,
        session_id=session_id
    )
    
    await automation_engine.trigger_event(event)


async def trigger_session_started(
    session: Session,
    user_id: str,
    project_id: Optional[str] = None
):
    """Trigger automation for session start."""
    event = AutomationEvent(
        event_type=EventType.SESSION_STARTED,
        data={
            "session_id": str(session.id),
            "session_type": session.session_type,
            "context_size": len(session.context_history) if session.context_history else 0
        },
        user_id=user_id,
        project_id=project_id,
        session_id=str(session.id)
    )
    
    await automation_engine.trigger_event(event)


async def trigger_pattern_detected(
    pattern_data: Dict[str, Any],
    user_id: Optional[str] = None,
    project_id: Optional[str] = None
):
    """Trigger automation for pattern detection."""
    event = AutomationEvent(
        event_type=EventType.PATTERN_DETECTED,
        data=pattern_data,
        user_id=user_id,
        project_id=project_id
    )
    
    await automation_engine.trigger_event(event)


async def start_automation_engine():
    """Start the automation engine."""
    await automation_engine.start()


async def stop_automation_engine():
    """Stop the automation engine."""
    await automation_engine.stop()