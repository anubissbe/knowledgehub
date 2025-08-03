"""
Multi-Agent Orchestrator
Implements the orchestrator-worker pattern for complex query decomposition
"""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

import logging
logger = logging.getLogger(__name__)
from ..rag.simple_rag_service import SimpleRAGService as LlamaIndexRAGService
from ..zep_memory import ZepMemoryService
from .agents import (
    BaseAgent,
    DocumentationAgent,
    CodebaseAgent,
    PerformanceAgent,
    StyleGuideAgent,
    TestingAgent,
    SynthesisAgent
)
from .query_decomposer import QueryDecomposer
from .task_planner import TaskPlanner


class TaskType(str, Enum):
    """Types of tasks the orchestrator can handle"""
    DOCUMENTATION = "documentation"
    CODE_SEARCH = "code_search"
    PERFORMANCE = "performance"
    STYLE_CHECK = "style_check"
    TESTING = "testing"
    SYNTHESIS = "synthesis"
    GENERAL = "general"


@dataclass
class AgentTask:
    """Represents a task assigned to an agent"""
    id: str
    type: TaskType
    description: str
    dependencies: List[str] = None
    priority: int = 1
    context: Dict[str, Any] = None
    status: str = "pending"
    result: Any = None
    error: str = None
    assigned_agent: str = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class QueryPlan:
    """Execution plan for a complex query"""
    query: str
    tasks: List[AgentTask]
    dependencies: Dict[str, List[str]]
    estimated_time: float
    complexity_score: float


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents to handle complex queries
    Uses orchestrator-worker pattern with task decomposition
    """
    
    def __init__(
        self,
        rag_service: LlamaIndexRAGService,
        zep_service: ZepMemoryService,
        max_concurrent_agents: int = 5
    ):
        self.logger = logger
        self.rag_service = rag_service
        self.zep_service = zep_service
        self.max_concurrent_agents = max_concurrent_agents
        
        # Initialize specialized agents
        self.agents: Dict[TaskType, BaseAgent] = {
            TaskType.DOCUMENTATION: DocumentationAgent(rag_service),
            TaskType.CODE_SEARCH: CodebaseAgent(rag_service),
            TaskType.PERFORMANCE: PerformanceAgent(rag_service),
            TaskType.STYLE_CHECK: StyleGuideAgent(rag_service),
            TaskType.TESTING: TestingAgent(rag_service),
            TaskType.SYNTHESIS: SynthesisAgent(rag_service, zep_service)
        }
        
        # Initialize query decomposer and task planner
        self.query_decomposer = QueryDecomposer()
        self.task_planner = TaskPlanner()
        
        # Task management
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, AgentTask] = {}
        self.task_lock = asyncio.Lock()
        
        self.logger.info("Multi-agent orchestrator initialized")
    
    async def process_query(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a complex query using multiple agents
        
        Args:
            query: User's query
            user_id: User identifier
            session_id: Optional session for memory context
            context: Additional context
            
        Returns:
            Synthesized response from all agents
        """
        try:
            # Step 1: Decompose query into sub-tasks
            decomposed = await self.query_decomposer.decompose(query, context)
            
            # Step 2: Create execution plan
            plan = await self.task_planner.create_plan(
                query=query,
                sub_queries=decomposed["sub_queries"],
                complexity=decomposed["complexity"]
            )
            
            self.logger.info(
                f"Created execution plan with {len(plan.tasks)} tasks",
                extra={
                    "query": query,
                    "task_count": len(plan.tasks),
                    "complexity": plan.complexity_score
                }
            )
            
            # Step 3: Execute tasks according to plan
            results = await self._execute_plan(plan, user_id, session_id)
            
            # Step 4: Synthesize final response
            synthesis_task = AgentTask(
                id="synthesis-final",
                type=TaskType.SYNTHESIS,
                description="Synthesize all results into coherent response",
                context={
                    "original_query": query,
                    "agent_results": results,
                    "plan": plan
                }
            )
            
            final_response = await self.agents[TaskType.SYNTHESIS].execute(
                synthesis_task
            )
            
            # Store in memory if session exists
            if session_id:
                await self._store_in_memory(
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                    response=final_response,
                    plan=plan,
                    results=results
                )
            
            return {
                "response": final_response,
                "plan": self._serialize_plan(plan),
                "agent_results": results,
                "metadata": {
                    "total_tasks": len(plan.tasks),
                    "complexity_score": plan.complexity_score,
                    "execution_time": sum(
                        t.get("execution_time", 0) for t in results.values()
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(
                f"Error processing multi-agent query: {str(e)}",
                exc_info=True
            )
            
            # Fallback to simple RAG query
            fallback_result = await self.rag_service.query(
                query_text=query,
                metadata={"fallback": True, "error": str(e)}
            )
            
            return {
                "response": fallback_result,
                "fallback": True,
                "error": str(e)
            }
    
    async def _execute_plan(
        self,
        plan: QueryPlan,
        user_id: str,
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Execute tasks according to the plan"""
        results = {}
        completed_task_ids: Set[str] = set()
        
        # Group tasks by priority
        priority_groups = {}
        for task in plan.tasks:
            priority = task.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(task)
        
        # Execute tasks by priority (highest first)
        for priority in sorted(priority_groups.keys(), reverse=True):
            tasks = priority_groups[priority]
            
            # Execute tasks in parallel within same priority
            batch_results = await self._execute_task_batch(
                tasks=tasks,
                completed_task_ids=completed_task_ids,
                plan_dependencies=plan.dependencies,
                user_id=user_id,
                session_id=session_id
            )
            
            results.update(batch_results)
            completed_task_ids.update(batch_results.keys())
        
        return results
    
    async def _execute_task_batch(
        self,
        tasks: List[AgentTask],
        completed_task_ids: Set[str],
        plan_dependencies: Dict[str, List[str]],
        user_id: str,
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Execute a batch of tasks in parallel"""
        # Filter tasks whose dependencies are met
        ready_tasks = []
        for task in tasks:
            deps = plan_dependencies.get(task.id, [])
            if all(dep_id in completed_task_ids for dep_id in deps):
                ready_tasks.append(task)
        
        # Limit concurrent execution
        results = {}
        for i in range(0, len(ready_tasks), self.max_concurrent_agents):
            batch = ready_tasks[i:i + self.max_concurrent_agents]
            
            # Create coroutines for parallel execution
            coroutines = []
            for task in batch:
                coroutines.append(
                    self._execute_single_task(task, user_id, session_id)
                )
            
            # Execute in parallel
            batch_results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # Process results
            for task, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results[task.id] = {
                        "error": str(result),
                        "task": task.description
                    }
                else:
                    results[task.id] = result
        
        return results
    
    async def _execute_single_task(
        self,
        task: AgentTask,
        user_id: str,
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Execute a single task with an appropriate agent"""
        start_time = datetime.utcnow()
        
        async with self.task_lock:
            task.status = "in_progress"
            task.started_at = start_time
            self.active_tasks[task.id] = task
        
        try:
            # Select appropriate agent
            agent = self.agents.get(task.type)
            if not agent:
                # Use synthesis agent as general fallback
                agent = self.agents[TaskType.SYNTHESIS]
            
            # Add user context
            if not task.context:
                task.context = {}
            task.context["user_id"] = user_id
            task.context["session_id"] = session_id
            
            # Execute task
            result = await agent.execute(task)
            
            # Update task status
            async with self.task_lock:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                task.result = result
                self.completed_tasks[task.id] = task
                self.active_tasks.pop(task.id, None)
            
            execution_time = (task.completed_at - start_time).total_seconds()
            
            return {
                "result": result,
                "task": task.description,
                "type": task.type,
                "execution_time": execution_time,
                "agent": agent.__class__.__name__
            }
            
        except Exception as e:
            # Update task with error
            async with self.task_lock:
                task.status = "failed"
                task.completed_at = datetime.utcnow()
                task.error = str(e)
                self.completed_tasks[task.id] = task
                self.active_tasks.pop(task.id, None)
            
            self.logger.error(
                f"Task {task.id} failed: {str(e)}",
                extra={"task": task.description, "type": task.type}
            )
            
            raise
    
    async def _store_in_memory(
        self,
        session_id: str,
        user_id: str,
        query: str,
        response: Any,
        plan: QueryPlan,
        results: Dict[str, Any]
    ):
        """Store the multi-agent interaction in memory"""
        try:
            # Create structured metadata
            metadata = {
                "type": "multi_agent_query",
                "query": query,
                "plan": {
                    "tasks": len(plan.tasks),
                    "complexity": plan.complexity_score,
                    "task_types": list(set(t.type for t in plan.tasks))
                },
                "agents_used": list(set(
                    r.get("agent", "Unknown") for r in results.values()
                )),
                "execution_time": sum(
                    r.get("execution_time", 0) for r in results.values()
                )
            }
            
            # Store the interaction
            await self.zep_service.add_message(
                session_id=session_id,
                role="user",
                content=query,
                user_id=user_id,
                metadata={"multi_agent": True}
            )
            
            await self.zep_service.add_message(
                session_id=session_id,
                role="assistant",
                content=json.dumps(response) if not isinstance(response, str) else response,
                user_id=user_id,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to store multi-agent interaction in memory: {str(e)}"
            )
    
    def _serialize_plan(self, plan: QueryPlan) -> Dict[str, Any]:
        """Serialize QueryPlan for API response"""
        return {
            "query": plan.query,
            "tasks": [
                {
                    "id": task.id,
                    "type": task.type,
                    "description": task.description,
                    "priority": task.priority,
                    "status": task.status
                }
                for task in plan.tasks
            ],
            "dependencies": plan.dependencies,
            "estimated_time": plan.estimated_time,
            "complexity_score": plan.complexity_score
        }
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents and active tasks"""
        async with self.task_lock:
            active_tasks_data = [
                {
                    "id": task.id,
                    "type": task.type,
                    "description": task.description,
                    "status": task.status,
                    "started_at": task.started_at.isoformat() if task.started_at else None
                }
                for task in self.active_tasks.values()
            ]
        
        agent_status = {}
        for task_type, agent in self.agents.items():
            agent_status[task_type] = {
                "name": agent.__class__.__name__,
                "ready": True,  # Could add health checks
                "capabilities": agent.get_capabilities()
            }
        
        return {
            "agents": agent_status,
            "active_tasks": active_tasks_data,
            "max_concurrent": self.max_concurrent_agents,
            "total_completed": len(self.completed_tasks)
        }