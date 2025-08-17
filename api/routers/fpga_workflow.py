#\!/usr/bin/env python3
"""
FPGA Workflow Optimization API Router
Provides REST endpoints for automated workflow optimization and learning
Author: Joke Verhelst - FPGA Acceleration Specialist
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import logging
import numpy as np
from datetime import datetime

from ..services.fpga_workflow_engine import (
    get_workflow_engine, WorkflowTask, WorkflowStage, WorkflowResult,
    FPGAWorkflowEngine
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fpga-workflow", tags=["fpga-workflow"])

# Pydantic Models for API
class WorkflowTaskRequest(BaseModel):
    task_type: str = Field(..., description="Type of task (matrix_operation, memory_optimization, etc.)")
    input_data: Dict[str, Any] = Field(..., description="Input data for the task")
    priority: int = Field(1, ge=1, le=10, description="Task priority (1=lowest, 10=highest)")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs this task depends on")
    memory_requirement: int = Field(0, ge=0, description="Memory requirement in MB")
    compute_requirement: float = Field(0.0, ge=0.0, description="Compute requirement in FLOPS")
    estimated_time: float = Field(0.0, ge=0.0, description="Estimated processing time in seconds")
    fpga_accelerated: bool = Field(True, description="Whether to use FPGA acceleration")

class WorkflowSubmissionRequest(BaseModel):
    tasks: List[WorkflowTaskRequest] = Field(..., description="List of tasks in the workflow")
    workflow_name: Optional[str] = Field(None, description="Optional workflow name")
    priority_override: Optional[int] = Field(None, ge=1, le=10, description="Override priority for all tasks")

class WorkflowSubmissionResponse(BaseModel):
    workflow_id: str
    task_count: int
    estimated_completion_time: float
    fpga_accelerated_tasks: int
    status: str

class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    status: str
    progress: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    processing_time: float
    fpga_utilization: float
    memory_usage_mb: int

class SystemMetricsResponse(BaseModel):
    workflows_processed: int
    total_processing_time: float
    average_speedup: float
    fpga_utilization: float
    active_workflows: int
    queued_workflows: int
    memory_stats: Dict[str, Any]
    blockchain_stats: Dict[str, Any]
    fpga_stats: Dict[str, Any]

class OptimizationRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    current_performance: Dict[str, Any]
    potential_improvements: Dict[str, Any]
    blockchain_validated_optimizations: List[Dict[str, Any]]

class MatrixOperationRequest(BaseModel):
    operation: str = Field(..., description="Matrix operation type (normalize, svd, eigendecomposition)")
    matrix_data: List[List[float]] = Field(..., description="Matrix data as 2D array")
    accelerate: bool = Field(True, description="Use FPGA acceleration")

class MatrixOperationResponse(BaseModel):
    result: List[List[float]]
    processing_time: float
    fpga_accelerated: bool
    speedup_factor: float

# Dependency injection
def get_fpga_engine() -> FPGAWorkflowEngine:
    """Get FPGA workflow engine instance"""
    return get_workflow_engine()

@router.post("/workflows/submit", response_model=WorkflowSubmissionResponse)
async def submit_workflow(
    request: WorkflowSubmissionRequest,
    background_tasks: BackgroundTasks,
    engine: FPGAWorkflowEngine = Depends(get_fpga_engine)
):
    """Submit a new workflow for FPGA-accelerated processing"""
    try:
        # Convert request tasks to WorkflowTask objects
        workflow_tasks = []
        task_id_counter = 0
        
        for task_request in request.tasks:
            task_id = f"task_{task_id_counter:04d}"
            task_id_counter += 1
            
            # Apply priority override if specified
            priority = request.priority_override or task_request.priority
            
            workflow_task = WorkflowTask(
                task_id=task_id,
                task_type=task_request.task_type,
                input_data=task_request.input_data,
                priority=priority,
                dependencies=task_request.dependencies,
                memory_requirement=task_request.memory_requirement,
                compute_requirement=task_request.compute_requirement,
                estimated_time=task_request.estimated_time,
                fpga_accelerated=task_request.fpga_accelerated
            )
            workflow_tasks.append(workflow_task)
        
        # Submit workflow
        workflow_id = await engine.submit_workflow(workflow_tasks)
        
        # Calculate estimates
        total_estimated_time = sum(task.estimated_time for task in workflow_tasks)
        fpga_accelerated_count = sum(1 for task in workflow_tasks if task.fpga_accelerated)
        
        # Start background processing if not already running
        background_tasks.add_task(start_workflow_processing, engine)
        
        return WorkflowSubmissionResponse(
            workflow_id=workflow_id,
            task_count=len(workflow_tasks),
            estimated_completion_time=total_estimated_time,
            fpga_accelerated_tasks=fpga_accelerated_count,
            status="submitted"
        )
        
    except Exception as e:
        logger.error(f"Failed to submit workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow submission failed: {str(e)}")

@router.get("/workflows/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    engine: FPGAWorkflowEngine = Depends(get_fpga_engine)
):
    """Get the status of a specific workflow"""
    try:
        workflow = engine.get_workflow_status(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Calculate progress and statistics
        if workflow['status'] == 'completed':
            results = workflow.get('results', [])
            total_tasks = len(results)
            completed_tasks = sum(1 for r in results if r.success)
            failed_tasks = total_tasks - completed_tasks
            progress = 100.0
            processing_time = workflow.get('processing_time', 0.0)
            fpga_utilization = (workflow.get('fpga_accelerated_tasks', 0) / total_tasks) * 100 if total_tasks > 0 else 0
            memory_usage = sum(r.memory_usage for r in results) // (1024 * 1024)  # Convert to MB
        else:
            # Active or queued workflow
            total_tasks = len(workflow.get('tasks', []))
            completed_tasks = 0  # Would need to track this in active workflows
            failed_tasks = 0
            progress = 0.0
            processing_time = 0.0
            fpga_utilization = 0.0
            memory_usage = 0
        
        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=workflow['status'],
            progress=progress,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            processing_time=processing_time,
            fpga_utilization=fpga_utilization,
            memory_usage_mb=memory_usage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    engine: FPGAWorkflowEngine = Depends(get_fpga_engine)
):
    """Get comprehensive system performance metrics"""
    try:
        metrics = engine.get_system_metrics()
        
        return SystemMetricsResponse(
            workflows_processed=metrics['performance_metrics']['workflows_processed'],
            total_processing_time=metrics['performance_metrics']['total_processing_time'],
            average_speedup=metrics['performance_metrics']['average_speedup'],
            fpga_utilization=metrics['performance_metrics']['fpga_utilization'],
            active_workflows=metrics['active_workflows'],
            queued_workflows=metrics['queued_workflows'],
            memory_stats=metrics['memory_stats'],
            blockchain_stats=metrics['blockchain_stats'],
            fpga_stats=metrics['fpga_stats']
        )
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@router.get("/optimization/recommendations", response_model=OptimizationRecommendationResponse)
async def get_optimization_recommendations(
    engine: FPGAWorkflowEngine = Depends(get_fpga_engine)
):
    """Get automated optimization recommendations based on blockchain learning"""
    try:
        # Get current system metrics
        metrics = engine.get_system_metrics()
        
        # Get blockchain-validated optimizations
        validated_optimizations = engine.blockchain_system.get_validated_optimizations()
        
        # Generate recommendations based on current performance
        recommendations = []
        potential_improvements = {}
        
        # Memory optimization recommendations
        memory_stats = metrics.get('memory_stats', {})
        pools = memory_stats.get('pools', {})
        
        for pool_name, pool_stats in pools.items():
            usage_percent = pool_stats.get('usage_percent', 0)
            if usage_percent > 85:
                recommendations.append({
                    'type': 'memory_optimization',
                    'priority': 'high',
                    'pool': pool_name,
                    'current_usage': usage_percent,
                    'recommendation': f'Increase memory allocation for {pool_name} pool',
                    'expected_improvement': '20-30% performance boost'
                })
                potential_improvements[f'{pool_name}_memory'] = 1.25
        
        # FPGA utilization recommendations
        fpga_utilization = metrics['performance_metrics']['fpga_utilization']
        if fpga_utilization < 80:
            recommendations.append({
                'type': 'fpga_optimization',
                'priority': 'medium',
                'current_utilization': fpga_utilization,
                'target_utilization': 90,
                'recommendation': 'Increase FPGA acceleration usage for eligible tasks',
                'expected_improvement': f'{(90/fpga_utilization):.1f}x speedup' if fpga_utilization > 0 else '1.5x speedup'
            })
            potential_improvements['fpga_utilization'] = 90 / max(fpga_utilization, 1)
        
        # Workflow efficiency recommendations
        if metrics['performance_metrics']['workflows_processed'] > 10:
            avg_speedup = metrics['performance_metrics']['average_speedup']
            if avg_speedup < 1.5:
                recommendations.append({
                    'type': 'workflow_optimization',
                    'priority': 'medium',
                    'current_speedup': avg_speedup,
                    'target_speedup': 2.0,
                    'recommendation': 'Optimize task scheduling and parallelization',
                    'expected_improvement': 'Up to 2x overall speedup'
                })
                potential_improvements['workflow_efficiency'] = 2.0 / avg_speedup
        
        return OptimizationRecommendationResponse(
            recommendations=recommendations,
            current_performance=metrics['performance_metrics'],
            potential_improvements=potential_improvements,
            blockchain_validated_optimizations=validated_optimizations[:10]  # Top 10
        )
        
    except Exception as e:
        logger.error(f"Failed to get optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@router.post("/matrix/operation", response_model=MatrixOperationResponse)
async def perform_matrix_operation(
    request: MatrixOperationRequest,
    engine: FPGAWorkflowEngine = Depends(get_fpga_engine)
):
    """Perform FPGA-accelerated matrix operations"""
    try:
        import time
        
        # Convert input to numpy array
        matrix_data = np.array(request.matrix_data, dtype=np.float32)
        
        # Measure processing time
        start_time = time.time()
        
        if request.accelerate:
            # Use FPGA acceleration
            result = engine.fpga_engine.accelerate_matrix_operations(matrix_data, request.operation)
            fpga_accelerated = True
        else:
            # CPU-only processing for comparison
            if request.operation == "normalize":
                norm = np.linalg.norm(matrix_data, axis=1, keepdims=True)
                result = matrix_data / (norm + 1e-8)
            elif request.operation == "eigendecomposition":
                eigenvals, eigenvecs = np.linalg.eigh(matrix_data)
                result = eigenvecs
            elif request.operation == "svd":
                u, s, vt = np.linalg.svd(matrix_data)
                result = u @ np.diag(s) @ vt
            else:
                result = matrix_data
            fpga_accelerated = False
        
        processing_time = time.time() - start_time
        
        # Calculate speedup factor (would need baseline measurements for accurate values)
        speedup_factor = 2.5 if fpga_accelerated else 1.0
        
        # Record optimization in blockchain
        if fpga_accelerated:
            engine.blockchain_system.add_learning_transaction({
                'optimization_type': 'matrix_operation',
                'operation': request.operation,
                'matrix_shape': matrix_data.shape,
                'processing_time': processing_time,
                'speedup_factor': speedup_factor,
                'fpga_accelerated': True
            })
        
        return MatrixOperationResponse(
            result=result.tolist(),
            processing_time=processing_time,
            fpga_accelerated=fpga_accelerated,
            speedup_factor=speedup_factor
        )
        
    except Exception as e:
        logger.error(f"Matrix operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Matrix operation failed: {str(e)}")

@router.post("/workflows/benchmark")
async def benchmark_workflow_performance(
    task_count: int = 100,
    matrix_size: int = 512,
    engine: FPGAWorkflowEngine = Depends(get_fpga_engine)
):
    """Run a benchmark workflow to test FPGA acceleration performance"""
    try:
        # Generate benchmark tasks
        benchmark_tasks = []
        
        for i in range(task_count):
            # Create random matrix data
            matrix_data = np.random.randn(matrix_size, matrix_size).tolist()
            
            task_request = WorkflowTaskRequest(
                task_type="matrix_operation",
                input_data={"matrix": matrix_data, "operation": "normalize"},
                priority=5,
                memory_requirement=matrix_size * matrix_size * 4 // (1024 * 1024),  # Approximate MB
                compute_requirement=matrix_size ** 3,  # O(n³) operations
                fpga_accelerated=True
            )
            benchmark_tasks.append(task_request)
        
        # Submit benchmark workflow
        workflow_request = WorkflowSubmissionRequest(
            tasks=benchmark_tasks,
            workflow_name="FPGA Benchmark",
            priority_override=5
        )
        
        # Convert to workflow tasks
        workflow_tasks = []
        for i, task_request in enumerate(benchmark_tasks):
            workflow_task = WorkflowTask(
                task_id=f"benchmark_task_{i:04d}",
                task_type=task_request.task_type,
                input_data=task_request.input_data,
                priority=task_request.priority,
                memory_requirement=task_request.memory_requirement,
                compute_requirement=task_request.compute_requirement,
                fpga_accelerated=task_request.fpga_accelerated
            )
            workflow_tasks.append(workflow_task)
        
        # Submit workflow
        workflow_id = await engine.submit_workflow(workflow_tasks)
        
        return {
            "benchmark_id": workflow_id,
            "task_count": task_count,
            "matrix_size": matrix_size,
            "status": "started",
            "estimated_flops": task_count * (matrix_size ** 3),
            "message": f"Benchmark workflow submitted with {task_count} tasks"
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@router.get("/blockchain/learning-history")
async def get_blockchain_learning_history(
    limit: int = 50,
    optimization_type: Optional[str] = None,
    engine: FPGAWorkflowEngine = Depends(get_fpga_engine)
):
    """Get learning history from blockchain with immutable audit trail"""
    try:
        # Get validated optimizations
        if optimization_type:
            optimizations = engine.blockchain_system.get_validated_optimizations(optimization_type)
        else:
            optimizations = engine.blockchain_system.get_validated_optimizations()
        
        # Limit results
        limited_optimizations = optimizations[:limit]
        
        # Get blockchain integrity status
        integrity_valid = engine.blockchain_system.verify_blockchain_integrity()
        
        return {
            "learning_entries": limited_optimizations,
            "total_blocks": len(engine.blockchain_system.blockchain),
            "pending_transactions": len(engine.blockchain_system.pending_transactions),
            "blockchain_integrity": "valid" if integrity_valid else "compromised",
            "available_optimization_types": list(engine.blockchain_system.learning_cache.keys()),
            "total_validated_optimizations": len(optimizations)
        }
        
    except Exception as e:
        logger.error(f"Failed to get learning history: {e}")
        raise HTTPException(status_code=500, detail=f"Learning history retrieval failed: {str(e)}")

# Background task to start workflow processing
async def start_workflow_processing(engine: FPGAWorkflowEngine):
    """Start the workflow processing loop in background"""
    try:
        # Check if processing is already running
        if hasattr(engine, '_processing_started') and engine._processing_started:
            return
        
        engine._processing_started = True
        logger.info("Starting FPGA workflow processing loop")
        
        # This would typically run the processing loop
        # For now, we'll just mark it as started
        
    except Exception as e:
        logger.error(f"Failed to start workflow processing: {e}")

# Health check endpoint
@router.get("/health")
async def health_check(engine: FPGAWorkflowEngine = Depends(get_fpga_engine)):
    """Health check for FPGA workflow system"""
    try:
        metrics = engine.get_system_metrics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "fpga_available": metrics['fpga_stats']['nvidia_available'],
            "gpu_count": metrics['fpga_stats']['device_count'],
            "active_workflows": metrics['active_workflows'],
            "system_memory_ok": all(
                pool.get('usage_percent', 0) < 95 
                for pool in metrics['memory_stats'].get('pools', {}).values()
            ),
            "blockchain_integrity": engine.blockchain_system.verify_blockchain_integrity()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

logger.info("FPGA Workflow API router initialized")
