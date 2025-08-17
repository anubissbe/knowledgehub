"""
Enhanced RAG API endpoints with hybrid retrieval and agent orchestration
Implements advanced multi-modal RAG with LangGraph agent workflows
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json
import asyncio

from ..services.hybrid_rag_service import (
    HybridRAGService, 
    RetrievalMode, 
    RerankingModel,
    get_hybrid_rag_service
)
from ..services.agent_orchestrator import (
    AgentOrchestrator,
    WorkflowType,
    get_agent_orchestrator
)
from ..services.auth import get_current_user
from ..models.user import User

router = APIRouter(prefix="/api/rag/enhanced", tags=["Enhanced RAG"])


# Request/Response Models
class HybridRAGRequest(BaseModel):
    """Request model for hybrid RAG queries"""
    query: str = Field(..., description="User query")
    retrieval_mode: RetrievalMode = Field(
        RetrievalMode.HYBRID_ALL, 
        description="Retrieval strategy to use"
    )
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    include_reasoning: bool = Field(False, description="Include reasoning steps")
    enable_reranking: bool = Field(True, description="Enable cross-encoder reranking")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class AgentWorkflowRequest(BaseModel):
    """Request model for agent workflow execution"""
    query: str = Field(..., description="User query")
    workflow_type: WorkflowType = Field(
        WorkflowType.SIMPLE_QA,
        description="Type of agent workflow to execute"
    )
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    max_execution_time: int = Field(300, ge=30, le=600, description="Max execution time in seconds")


class DocumentIngestionRequest(BaseModel):
    """Request model for enhanced document ingestion"""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    doc_id: Optional[str] = Field(None, description="Optional document ID")
    extract_entities: bool = Field(True, description="Extract entities for graph")
    enable_chunking: bool = Field(True, description="Enable intelligent chunking")
    chunk_strategy: str = Field("semantic", description="Chunking strategy")


class ConfigurationRequest(BaseModel):
    """Request model for RAG configuration updates"""
    dense_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    sparse_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    graph_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    rerank_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    dense_top_k: Optional[int] = Field(None, ge=1, le=100)
    sparse_top_k: Optional[int] = Field(None, ge=1, le=100)
    graph_top_k: Optional[int] = Field(None, ge=1, le=100)


# Initialize services (will be set in lifespan)
hybrid_rag_service: Optional[HybridRAGService] = None
agent_orchestrator: Optional[AgentOrchestrator] = None


@router.on_event("startup")
async def initialize_services():
    """Initialize enhanced RAG services"""
    global hybrid_rag_service, agent_orchestrator
    try:
        hybrid_rag_service = await get_hybrid_rag_service()
        agent_orchestrator = await get_agent_orchestrator()
    except Exception as e:
        # Services will be initialized on first use
        pass


@router.post("/query", response_model=Dict[str, Any])
async def hybrid_rag_query(
    request: HybridRAGRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Execute enhanced hybrid RAG query with multiple retrieval strategies
    """
    try:
        service = hybrid_rag_service or await get_hybrid_rag_service()
        
        result = await service.query(
            query=request.query,
            user_id=str(current_user.id),
            session_id=request.session_id,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            include_reasoning=request.include_reasoning
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Hybrid RAG query failed: {str(e)}"
        )


@router.post("/agent/workflow", response_model=Dict[str, Any])
async def execute_agent_workflow(
    request: AgentWorkflowRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Execute multi-agent workflow for complex queries
    """
    try:
        orchestrator = agent_orchestrator or await get_agent_orchestrator()
        
        # Execute workflow with timeout
        result = await asyncio.wait_for(
            orchestrator.execute_workflow(
                query=request.query,
                user_id=str(current_user.id),
                workflow_type=request.workflow_type,
                session_id=request.session_id,
                context=request.context
            ),
            timeout=request.max_execution_time
        )
        
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Workflow execution timed out after {request.max_execution_time} seconds"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent workflow execution failed: {str(e)}"
        )


@router.post("/ingest", response_model=Dict[str, Any])
async def ingest_document_enhanced(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Ingest document with enhanced processing for all retrieval modes
    """
    try:
        service = hybrid_rag_service or await get_hybrid_rag_service()
        
        # Add user context to metadata
        enhanced_metadata = {
            **request.metadata,
            "ingested_by": str(current_user.id),
            "ingested_at": datetime.utcnow().isoformat(),
            "extract_entities": request.extract_entities,
            "chunk_strategy": request.chunk_strategy
        }
        
        # Ingest document
        result = await service.ingest_document(
            content=request.content,
            metadata=enhanced_metadata,
            doc_id=request.doc_id
        )
        
        # Update search indexes in background
        background_tasks.add_task(
            _update_search_indexes,
            service,
            request.doc_id or result.get("doc_id")
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced document ingestion failed: {str(e)}"
        )


@router.get("/retrieval-modes", response_model=List[Dict[str, Any]])
async def get_retrieval_modes():
    """
    Get available retrieval modes and their descriptions
    """
    modes = []
    for mode in RetrievalMode:
        description = _get_mode_description(mode)
        modes.append({
            "mode": mode.value,
            "name": mode.value.replace("_", " ").title(),
            "description": description,
            "use_cases": _get_mode_use_cases(mode)
        })
    
    return modes


@router.get("/workflows", response_model=List[Dict[str, Any]])
async def get_available_workflows():
    """
    Get available agent workflows
    """
    try:
        orchestrator = agent_orchestrator or await get_agent_orchestrator()
        workflows = await orchestrator.get_available_workflows()
        return workflows
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflows: {str(e)}"
        )


class CompareModesRequest(BaseModel):
    """Request model for retrieval mode comparison"""
    query: str = Field(..., description="Query to compare")
    modes: List[RetrievalMode] = Field(..., description="Modes to compare")

@router.post("/compare", response_model=Dict[str, Any])
async def compare_retrieval_modes(
    request: CompareModesRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Compare different retrieval modes for the same query
    """
    try:
        service = hybrid_rag_service or await get_hybrid_rag_service()
        
        comparison_results = {}
        
        for mode in modes:
            result = await service.query(
                query=query,
                user_id=str(current_user.id),
                retrieval_mode=mode,
                top_k=5,
                include_reasoning=True
            )
            
            comparison_results[mode.value] = {
                "results_count": len(result.get("results", [])),
                "processing_time": result.get("metadata", {}).get("processing_time", 0),
                "performance_metrics": result.get("performance_metrics", {}),
                "top_result": result.get("results", [{}])[0] if result.get("results") else None
            }
        
        return {
            "query": query,
            "comparison": comparison_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval mode comparison failed: {str(e)}"
        )


@router.post("/config", response_model=Dict[str, Any])
async def update_rag_configuration(
    request: ConfigurationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update RAG configuration (admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        service = hybrid_rag_service or await get_hybrid_rag_service()
        
        # Update configuration
        config_updates = {}
        for field, value in request.dict(exclude_none=True).items():
            if hasattr(service.config, field):
                setattr(service.config, field, value)
                config_updates[field] = value
        
        return {
            "success": True,
            "updated_config": config_updates,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Configuration update failed: {str(e)}"
        )


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    current_user: User = Depends(get_current_user)
):
    """
    Get performance metrics for RAG services
    """
    try:
        metrics = {}
        
        # Hybrid RAG metrics
        if hybrid_rag_service:
            metrics["hybrid_rag"] = await hybrid_rag_service.get_performance_stats()
        
        # Agent orchestrator metrics
        if agent_orchestrator:
            metrics["agent_orchestrator"] = await agent_orchestrator.get_performance_stats()
        
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def enhanced_rag_health_check():
    """
    Check health of enhanced RAG services
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    try:
        # Check hybrid RAG service
        if hybrid_rag_service:
            rag_health = await hybrid_rag_service.health_check()
            health["services"]["hybrid_rag"] = rag_health
        else:
            health["services"]["hybrid_rag"] = {"status": "not_initialized"}
        
        # Check agent orchestrator
        if agent_orchestrator:
            orchestrator_health = await agent_orchestrator.health_check()
            health["services"]["agent_orchestrator"] = orchestrator_health
        else:
            health["services"]["agent_orchestrator"] = {"status": "not_initialized"}
        
        # Overall status
        service_statuses = [
            service.get("status", "unknown") 
            for service in health["services"].values()
        ]
        
        if any(status in ["unhealthy", "error", "not_initialized"] for status in service_statuses):
            health["status"] = "degraded"
        
    except Exception as e:
        health["status"] = "error"
        health["error"] = str(e)
    
    return health


class BenchmarkRequest(BaseModel):
    """Request model for RAG benchmark testing"""
    queries: List[str] = Field(..., description="Test queries")
    modes: Optional[List[RetrievalMode]] = Field(None, description="Modes to benchmark")

@router.post("/benchmark", response_model=Dict[str, Any])
async def run_rag_benchmark(
    request: BenchmarkRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Run benchmark test on RAG system (admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        service = hybrid_rag_service or await get_hybrid_rag_service()
        
        modes_to_test = request.modes or [RetrievalMode.HYBRID_ALL]
        benchmark_results = {}
        
        for mode in modes_to_test:
            mode_results = []
            
            for query in request.queries:
                start_time = datetime.utcnow()
                
                result = await service.query(
                    query=query,
                    user_id=str(current_user.id),
                    retrieval_mode=mode,
                    top_k=10
                )
                
                end_time = datetime.utcnow()
                processing_time = (end_time - start_time).total_seconds()
                
                mode_results.append({
                    "query": query,
                    "results_count": len(result.get("results", [])),
                    "processing_time": processing_time,
                    "has_results": len(result.get("results", [])) > 0
                })
            
            # Calculate aggregate metrics
            total_queries = len(mode_results)
            avg_time = sum(r["processing_time"] for r in mode_results) / total_queries
            success_rate = sum(1 for r in mode_results if r["has_results"]) / total_queries
            
            benchmark_results[mode.value] = {
                "individual_results": mode_results,
                "aggregate_metrics": {
                    "total_queries": total_queries,
                    "average_processing_time": avg_time,
                    "success_rate": success_rate,
                    "queries_with_results": sum(1 for r in mode_results if r["has_results"])
                }
            }
        
        return {
            "benchmark_results": benchmark_results,
            "test_timestamp": datetime.utcnow().isoformat(),
            "test_queries_count": len(request.queries)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark execution failed: {str(e)}"
        )


# Helper functions
async def _update_search_indexes(service: HybridRAGService, doc_id: str):
    """Background task to update search indexes after ingestion"""
    try:
        # This would trigger any necessary index updates
        # For now, just log the completion
        pass
    except Exception as e:
        logger.error(f"Failed to update search indexes for doc {doc_id}: {e}")


def _get_mode_description(mode: RetrievalMode) -> str:
    """Get description for retrieval mode"""
    descriptions = {
        RetrievalMode.DENSE_ONLY: "Pure vector similarity search using dense embeddings",
        RetrievalMode.SPARSE_ONLY: "Keyword-based search using BM25 sparse retrieval",
        RetrievalMode.GRAPH_ONLY: "Entity and relationship-based graph traversal",
        RetrievalMode.DENSE_SPARSE: "Combined dense vector and sparse keyword search",
        RetrievalMode.DENSE_GRAPH: "Vector search enhanced with graph relationships",
        RetrievalMode.SPARSE_GRAPH: "Keyword search enhanced with graph context",
        RetrievalMode.HYBRID_ALL: "Full hybrid approach combining all retrieval methods"
    }
    return descriptions.get(mode, "Unknown retrieval mode")


def _get_mode_use_cases(mode: RetrievalMode) -> List[str]:
    """Get use cases for retrieval mode"""
    use_cases = {
        RetrievalMode.DENSE_ONLY: [
            "Semantic similarity queries",
            "Conceptual questions",
            "When exact keywords don't match"
        ],
        RetrievalMode.SPARSE_ONLY: [
            "Exact keyword matching",
            "Specific term searches",
            "When terminology is important"
        ],
        RetrievalMode.GRAPH_ONLY: [
            "Relationship exploration",
            "Entity-based queries",
            "Connection discovery"
        ],
        RetrievalMode.DENSE_SPARSE: [
            "Balanced semantic and keyword search",
            "General purpose queries",
            "Improved recall"
        ],
        RetrievalMode.DENSE_GRAPH: [
            "Semantic search with context",
            "Entity-aware semantic matching",
            "Rich contextual queries"
        ],
        RetrievalMode.SPARSE_GRAPH: [
            "Keyword search with relationships",
            "Term-based exploration",
            "Structured content search"
        ],
        RetrievalMode.HYBRID_ALL: [
            "Maximum coverage queries",
            "Complex multi-faceted questions",
            "Best overall performance"
        ]
    }
    return use_cases.get(mode, [])