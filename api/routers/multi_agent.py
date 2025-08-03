"""
Multi-Agent System API Router
Endpoints for complex query processing with multiple specialized agents
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from ..services.multi_agent import MultiAgentOrchestrator
from ..services.rag.simple_rag_service import SimpleRAGService as LlamaIndexRAGService
from ..services.zep_memory import ZepMemoryService
from ..services.auth import get_current_user
from ..services.rbac_service import Permission
from ..middleware.rbac_middleware import require_permission
from ..models.user import User
import logging
logger = logging.getLogger(__name__)


# Request/Response models
class MultiAgentQueryRequest(BaseModel):
    """Request model for multi-agent queries"""
    query: str = Field(..., description="The complex query to process")
    session_id: Optional[str] = Field(None, description="Session ID for memory context")
    output_format: str = Field("structured", description="Output format: structured, narrative, or recommendations")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    max_agents: Optional[int] = Field(5, description="Maximum concurrent agents")


class TaskStatus(BaseModel):
    """Task status information"""
    id: str
    type: str
    description: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


class AgentStatus(BaseModel):
    """Agent status information"""
    name: str
    type: str
    ready: bool
    capabilities: Dict[str, Any]


class MultiAgentResponse(BaseModel):
    """Response from multi-agent processing"""
    response: Any
    plan: Optional[Dict[str, Any]]
    agent_results: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    fallback: bool = False
    error: Optional[str]


class SystemStatusResponse(BaseModel):
    """System status response"""
    agents: Dict[str, AgentStatus]
    active_tasks: List[TaskStatus]
    max_concurrent: int
    total_completed: int


# Create router
router = APIRouter(
    prefix="/api/multi-agent",
    tags=["multi-agent"],
    responses={404: {"description": "Not found"}}
)


# Dependency to get orchestrator
async def get_orchestrator(
    rag_service: LlamaIndexRAGService = Depends(LlamaIndexRAGService),
    zep_service: ZepMemoryService = Depends(ZepMemoryService)
) -> MultiAgentOrchestrator:
    """Get or create orchestrator instance"""
    return MultiAgentOrchestrator(
        rag_service=rag_service,
        zep_service=zep_service
    )


@router.post("/query", response_model=MultiAgentResponse)
@require_permission(Permission.RAG_QUERY)
async def process_multi_agent_query(
    request: MultiAgentQueryRequest,
    background_tasks: BackgroundTasks,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
    current_user: User = Depends(get_current_user)
) -> MultiAgentResponse:
    """
    Process a complex query using multiple specialized agents
    
    This endpoint:
    1. Decomposes the query into sub-tasks
    2. Assigns tasks to specialized agents
    3. Executes tasks in parallel where possible
    4. Synthesizes results into a coherent response
    
    The system uses these specialized agents:
    - DocumentationAgent: Searches technical documentation
    - CodebaseAgent: Analyzes code patterns and implementations
    - PerformanceAgent: Provides performance insights
    - StyleGuideAgent: Checks code style and best practices
    - TestingAgent: Suggests testing strategies
    - SynthesisAgent: Combines results from other agents
    """
    try:
        logger.info(
            f"Processing multi-agent query for user {current_user.id}",
            extra={
                "user_id": current_user.id,
                "query_length": len(request.query),
                "session_id": request.session_id
            }
        )
        
        # Update orchestrator settings if provided
        if request.max_agents:
            orchestrator.max_concurrent_agents = request.max_agents
        
        # Process query
        result = await orchestrator.process_query(
            query=request.query,
            user_id=current_user.id,
            session_id=request.session_id,
            context={
                **(request.context or {}),
                "output_format": request.output_format,
                "user_role": current_user.role
            }
        )
        
        # Log successful processing
        background_tasks.add_task(
            log_query_success,
            user_id=current_user.id,
            query=request.query,
            agent_count=result["metadata"].get("total_tasks", 0)
        )
        
        return MultiAgentResponse(
            response=result["response"],
            plan=result.get("plan"),
            agent_results=result.get("agent_results"),
            metadata=result.get("metadata", {}),
            fallback=result.get("fallback", False),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(
            f"Error in multi-agent query processing: {str(e)}",
            exc_info=True
        )
        
        # Return error response
        return MultiAgentResponse(
            response="An error occurred processing your query. Falling back to simple search.",
            plan=None,
            agent_results=None,
            metadata={"error": True},
            fallback=True,
            error=str(e)
        )


@router.get("/status", response_model=SystemStatusResponse)
@require_permission(Permission.SYSTEM_MONITOR)
async def get_system_status(
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
    current_user: User = Depends(get_current_user)
) -> SystemStatusResponse:
    """
    Get the current status of the multi-agent system
    
    Returns:
    - Status of each specialized agent
    - Currently active tasks
    - System configuration
    """
    try:
        status = await orchestrator.get_agent_status()
        
        # Convert to response model
        agents = {}
        for agent_type, agent_info in status["agents"].items():
            agents[agent_type] = AgentStatus(
                name=agent_info["name"],
                type=agent_type,
                ready=agent_info["ready"],
                capabilities=agent_info["capabilities"]
            )
        
        active_tasks = []
        for task in status["active_tasks"]:
            active_tasks.append(TaskStatus(
                id=task["id"],
                type=task["type"],
                description=task["description"],
                status=task["status"],
                started_at=task.get("started_at"),
                completed_at=None,
                error=None
            ))
        
        return SystemStatusResponse(
            agents=agents,
            active_tasks=active_tasks,
            max_concurrent=status["max_concurrent"],
            total_completed=status["total_completed"]
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.post("/decompose")
@require_permission(Permission.RAG_QUERY)
async def decompose_query(
    query: str,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Decompose a complex query without executing it
    
    Useful for:
    - Understanding how the system will process a query
    - Debugging query decomposition
    - Educational purposes
    """
    try:
        # Just decompose, don't execute
        decomposed = await orchestrator.query_decomposer.decompose(query)
        
        return {
            "original_query": query,
            "decomposition": decomposed,
            "estimated_agents": decomposed.get("estimated_agents", 1),
            "complexity": decomposed.get("complexity", 1.0)
        }
        
    except Exception as e:
        logger.error(f"Error decomposing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to decompose query")


@router.get("/capabilities")
async def get_agent_capabilities(
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Get capabilities of all available agents
    
    Public endpoint to help users understand what each agent can do
    """
    try:
        capabilities = {}
        
        for agent_type, agent in orchestrator.agents.items():
            capabilities[agent_type] = {
                "name": agent.__class__.__name__,
                "capabilities": agent.get_capabilities()
            }
        
        return {
            "agents": capabilities,
            "query_types": [
                {
                    "type": "documentation",
                    "description": "Questions about how to use something",
                    "examples": ["How do I implement OAuth2?", "What is the syntax for async functions?"]
                },
                {
                    "type": "code",
                    "description": "Requests for code examples or implementations",
                    "examples": ["Show me how to create a REST API", "Implement a binary search"]
                },
                {
                    "type": "performance",
                    "description": "Performance optimization queries",
                    "examples": ["How to optimize database queries?", "Improve API response time"]
                },
                {
                    "type": "style",
                    "description": "Code style and best practices",
                    "examples": ["Python naming conventions", "Clean code principles"]
                },
                {
                    "type": "testing",
                    "description": "Testing strategies and examples",
                    "examples": ["How to test async functions?", "Unit testing best practices"]
                }
            ],
            "complex_query_examples": [
                "Implement a REST API with authentication and write tests for it",
                "Show me how to optimize database queries and follow best practices",
                "Create a React component with TypeScript and add unit tests"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get capabilities")


# Background tasks
async def log_query_success(user_id: str, query: str, agent_count: int):
    """Log successful query processing"""
    logger.info(
        "Multi-agent query processed successfully",
        extra={
            "user_id": user_id,
            "query_preview": query[:100],
            "agent_count": agent_count
        }
    )