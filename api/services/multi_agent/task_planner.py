"""
Task Planner for Multi-Agent System
Creates execution plans from decomposed queries
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import uuid
from datetime import datetime

import logging
logger = logging.getLogger(__name__)
from .orchestrator import AgentTask, TaskType, QueryPlan


class TaskPlanner:
    """
    Creates optimized execution plans for multi-agent processing
    Handles task ordering, parallelization, and resource allocation
    """
    
    def __init__(self):
        self.logger = logger
        
        # Task execution time estimates (seconds)
        self.task_time_estimates = {
            TaskType.DOCUMENTATION: 2.0,
            TaskType.CODE_SEARCH: 3.0,
            TaskType.PERFORMANCE: 2.5,
            TaskType.STYLE_CHECK: 1.5,
            TaskType.TESTING: 2.0,
            TaskType.SYNTHESIS: 1.0,
            TaskType.GENERAL: 2.0
        }
        
        # Task type mappings from query types
        self.type_mapping = {
            "documentation": TaskType.DOCUMENTATION,
            "code": TaskType.CODE_SEARCH,
            "performance": TaskType.PERFORMANCE,
            "style": TaskType.STYLE_CHECK,
            "testing": TaskType.TESTING,
            "synthesis": TaskType.SYNTHESIS,
            "general": TaskType.GENERAL
        }
    
    async def create_plan(
        self,
        query: str,
        sub_queries: List[Dict[str, Any]],
        complexity: float,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Create an execution plan from decomposed queries
        
        Args:
            query: Original query
            sub_queries: Decomposed sub-queries
            complexity: Query complexity score
            context: Additional context
            
        Returns:
            QueryPlan with tasks and dependencies
        """
        try:
            # Convert sub-queries to tasks
            tasks = self._create_tasks_from_queries(sub_queries, context)
            
            # Optimize task order
            tasks, dependencies = self._optimize_task_order(tasks, sub_queries)
            
            # Assign priorities
            tasks = self._assign_priorities(tasks, dependencies)
            
            # Estimate execution time
            estimated_time = self._estimate_execution_time(tasks, dependencies)
            
            # Create plan
            plan = QueryPlan(
                query=query,
                tasks=tasks,
                dependencies=dependencies,
                estimated_time=estimated_time,
                complexity_score=complexity
            )
            
            self.logger.info(
                f"Created execution plan with {len(tasks)} tasks",
                extra={
                    "task_count": len(tasks),
                    "estimated_time": estimated_time,
                    "complexity": complexity
                }
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating plan: {str(e)}")
            # Return simple plan on error
            return self._create_fallback_plan(query, sub_queries, complexity)
    
    def _create_tasks_from_queries(
        self,
        sub_queries: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> List[AgentTask]:
        """Convert sub-queries to executable tasks"""
        tasks = []
        
        for sq in sub_queries:
            # Map query type to task type
            task_type = self.type_mapping.get(
                sq.get("type", "general"),
                TaskType.GENERAL
            )
            
            # Create task
            task = AgentTask(
                id=sq.get("id", str(uuid.uuid4())),
                type=task_type,
                description=sq.get("text", ""),
                context={
                    "query": sq.get("text", ""),
                    "keywords": sq.get("keywords", []),
                    "original_context": context or {}
                },
                priority=sq.get("priority", 1)
            )
            
            tasks.append(task)
        
        # Add enrichment tasks if needed
        tasks.extend(self._add_enrichment_tasks(tasks))
        
        return tasks
    
    def _add_enrichment_tasks(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Add additional tasks for better results"""
        enrichment_tasks = []
        
        # Check if we have code tasks without documentation
        has_code = any(t.type == TaskType.CODE_SEARCH for t in tasks)
        has_docs = any(t.type == TaskType.DOCUMENTATION for t in tasks)
        
        if has_code and not has_docs:
            # Add documentation search for context
            doc_task = AgentTask(
                id=f"enrich_doc_{uuid.uuid4().hex[:8]}",
                type=TaskType.DOCUMENTATION,
                description="Search documentation for implementation context",
                context={
                    "query": " ".join([
                        t.context.get("query", "") 
                        for t in tasks 
                        if t.type == TaskType.CODE_SEARCH
                    ]),
                    "enrichment": True
                },
                priority=tasks[0].priority + 1  # Higher priority
            )
            enrichment_tasks.append(doc_task)
        
        # Check if we need style checking
        if has_code and not any(t.type == TaskType.STYLE_CHECK for t in tasks):
            style_task = AgentTask(
                id=f"enrich_style_{uuid.uuid4().hex[:8]}",
                type=TaskType.STYLE_CHECK,
                description="Check code style and best practices",
                context={
                    "query": "style guide best practices",
                    "enrichment": True
                },
                priority=1  # Low priority
            )
            enrichment_tasks.append(style_task)
        
        return enrichment_tasks
    
    def _optimize_task_order(
        self,
        tasks: List[AgentTask],
        sub_queries: List[Dict[str, Any]]
    ) -> tuple[List[AgentTask], Dict[str, List[str]]]:
        """Optimize task execution order"""
        # Build dependency map from sub-queries
        dependency_map = {}
        sq_id_to_task = {t.id: t for t in tasks if not t.id.startswith("enrich_")}
        
        for sq in sub_queries:
            sq_id = sq.get("id")
            deps = sq.get("dependencies", [])
            
            if sq_id in sq_id_to_task:
                dependency_map[sq_id] = deps
        
        # Add dependencies for enrichment tasks
        for task in tasks:
            if task.id.startswith("enrich_"):
                # Enrichment tasks have no dependencies
                dependency_map[task.id] = []
        
        # Topological sort for optimal order
        ordered_tasks = self._topological_sort(tasks, dependency_map)
        
        return ordered_tasks, dependency_map
    
    def _topological_sort(
        self,
        tasks: List[AgentTask],
        dependencies: Dict[str, List[str]]
    ) -> List[AgentTask]:
        """Perform topological sort on tasks"""
        # Create task lookup
        task_map = {t.id: t for t in tasks}
        
        # Calculate in-degrees
        in_degree = {t.id: 0 for t in tasks}
        for task_id, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Find tasks with no dependencies
        queue = [t for t in tasks if in_degree[t.id] == 0]
        ordered = []
        
        while queue:
            # Sort queue by priority for consistent ordering
            queue.sort(key=lambda t: (-t.priority, t.id))
            task = queue.pop(0)
            ordered.append(task)
            
            # Update in-degrees
            for other_id, deps in dependencies.items():
                if task.id in deps and other_id in in_degree:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(task_map[other_id])
        
        # Add any remaining tasks (in case of cycles)
        remaining = [t for t in tasks if t not in ordered]
        ordered.extend(remaining)
        
        return ordered
    
    def _assign_priorities(
        self,
        tasks: List[AgentTask],
        dependencies: Dict[str, List[str]]
    ) -> List[AgentTask]:
        """Assign execution priorities to tasks"""
        # Calculate levels based on dependencies
        levels = self._calculate_dependency_levels(tasks, dependencies)
        
        # Assign priorities based on levels and type
        for task in tasks:
            level = levels.get(task.id, 0)
            
            # Higher level = lower priority (executes later)
            base_priority = 10 - level
            
            # Adjust based on task type
            if task.type == TaskType.DOCUMENTATION:
                base_priority += 2  # Documentation often needed first
            elif task.type == TaskType.SYNTHESIS:
                base_priority -= 5  # Synthesis always last
            elif task.type == TaskType.STYLE_CHECK:
                base_priority -= 1  # Style checking is lower priority
            
            # Enrichment tasks get slight boost
            if task.context.get("enrichment"):
                base_priority += 1
            
            task.priority = max(1, min(10, base_priority))
        
        return tasks
    
    def _calculate_dependency_levels(
        self,
        tasks: List[AgentTask],
        dependencies: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """Calculate dependency levels for tasks"""
        levels = {}
        task_map = {t.id: t for t in tasks}
        
        def get_level(task_id: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            
            if task_id in levels:
                return levels[task_id]
            
            if task_id in visited:
                # Cycle detected
                return 0
            
            visited.add(task_id)
            
            # Find maximum level of dependencies
            max_dep_level = -1
            for dep_id in dependencies.get(task_id, []):
                if dep_id in task_map:
                    dep_level = get_level(dep_id, visited.copy())
                    max_dep_level = max(max_dep_level, dep_level)
            
            levels[task_id] = max_dep_level + 1
            return levels[task_id]
        
        # Calculate levels for all tasks
        for task in tasks:
            get_level(task.id)
        
        return levels
    
    def _estimate_execution_time(
        self,
        tasks: List[AgentTask],
        dependencies: Dict[str, List[str]]
    ) -> float:
        """Estimate total execution time considering parallelization"""
        # Group tasks by level for parallel execution
        levels = self._calculate_dependency_levels(tasks, dependencies)
        
        level_times = {}
        for task in tasks:
            level = levels.get(task.id, 0)
            task_time = self.task_time_estimates.get(
                task.type,
                self.task_time_estimates[TaskType.GENERAL]
            )
            
            if level not in level_times:
                level_times[level] = []
            level_times[level].append(task_time)
        
        # Total time is sum of max time at each level
        total_time = 0.0
        for level, times in level_times.items():
            # Assume tasks at same level run in parallel
            # But limited by max concurrent agents (5)
            if len(times) <= 5:
                level_time = max(times)
            else:
                # Some tasks must wait
                times.sort(reverse=True)
                level_time = max(times[:5]) + sum(times[5:]) / 5
            
            total_time += level_time
        
        # Add overhead for coordination
        overhead = len(tasks) * 0.1
        
        return total_time + overhead
    
    def _create_fallback_plan(
        self,
        query: str,
        sub_queries: List[Dict[str, Any]],
        complexity: float
    ) -> QueryPlan:
        """Create a simple fallback plan"""
        # Create a single general task
        task = AgentTask(
            id="fallback_main",
            type=TaskType.GENERAL,
            description=query,
            context={"sub_queries": sub_queries},
            priority=5
        )
        
        return QueryPlan(
            query=query,
            tasks=[task],
            dependencies={},
            estimated_time=5.0,
            complexity_score=complexity
        )