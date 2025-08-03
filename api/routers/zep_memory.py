"""
Zep Memory Router
API endpoints for conversational memory with temporal knowledge graphs
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..services.zep_memory import get_zep_service
from ..services.auth import get_current_user
from ..models.user import User

router = APIRouter(prefix="/api/zep", tags=["Zep Memory"])


# Request/Response Models
class AddMessageRequest(BaseModel):
    """Request to add a message to memory"""
    session_id: str = Field(..., description="Unique session identifier")
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class GetMemoryResponse(BaseModel):
    """Response containing conversation memory"""
    session_id: str
    messages: List[Dict[str, Any]]
    summary: Optional[str] = None
    facts: List[str] = []
    entities: List[Dict[str, Any]] = []


class SearchMemoryRequest(BaseModel):
    """Request to search conversation memory"""
    query: str = Field(..., description="Search query")
    session_ids: Optional[List[str]] = Field(None, description="Filter by sessions")
    limit: int = Field(5, ge=1, le=20, description="Maximum results")


class HybridSearchRequest(BaseModel):
    """Request for hybrid search (RAG + Memory)"""
    query: str = Field(..., description="Search query")
    rag_results: List[Dict[str, Any]] = Field(..., description="Results from RAG system")
    weight_memory: float = Field(0.3, ge=0, le=1, description="Weight for memory results")
    weight_rag: float = Field(0.7, ge=0, le=1, description="Weight for RAG results")


# Initialize service
zep_service = get_zep_service()


@router.post("/messages", response_model=Dict[str, Any])
async def add_message(
    request: AddMessageRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Add a message to conversation memory
    """
    try:
        success = await zep_service.add_message(
            session_id=request.session_id,
            role=request.role,
            content=request.content,
            user_id=str(current_user.id),
            metadata=request.metadata
        )
        
        if success:
            return {
                "status": "success",
                "message": "Message added to memory",
                "session_id": request.session_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add message")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding message: {str(e)}")


@router.get("/memory/{session_id}", response_model=GetMemoryResponse)
async def get_memory(
    session_id: str,
    limit: int = Query(10, ge=1, le=100, description="Maximum messages to return"),
    include_summary: bool = Query(True, description="Include conversation summary"),
    current_user: User = Depends(get_current_user)
):
    """
    Get conversation memory for a session
    """
    try:
        memory = await zep_service.get_memory(
            session_id=session_id,
            limit=limit,
            include_summary=include_summary
        )
        
        # Verify user has access to this session
        # In production, implement proper access control
        
        return GetMemoryResponse(**memory)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memory: {str(e)}")


@router.post("/search", response_model=List[Dict[str, Any]])
async def search_memory(
    request: SearchMemoryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Search across user's conversation memories
    """
    try:
        results = await zep_service.search_memory(
            query=request.query,
            user_id=str(current_user.id),
            session_ids=request.session_ids,
            limit=request.limit
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching memory: {str(e)}")


@router.get("/sessions", response_model=List[str])
async def get_user_sessions(
    limit: int = Query(10, ge=1, le=50, description="Maximum sessions"),
    current_user: User = Depends(get_current_user)
):
    """
    Get all sessions for the current user
    """
    try:
        sessions = await zep_service.get_user_sessions(
            user_id=str(current_user.id),
            limit=limit
        )
        
        return sessions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sessions: {str(e)}")


@router.post("/hybrid-search", response_model=List[Dict[str, Any]])
async def hybrid_search(
    request: HybridSearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Perform hybrid search combining RAG and memory results
    """
    try:
        results = await zep_service.hybrid_retrieval(
            query=request.query,
            user_id=str(current_user.id),
            rag_results=request.rag_results,
            weight_memory=request.weight_memory,
            weight_rag=request.weight_rag
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in hybrid search: {str(e)}")


@router.put("/users/{user_id}", response_model=Dict[str, Any])
async def update_user_profile(
    user_id: str,
    metadata: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Update user profile in Zep (admin only)
    """
    if current_user.role != "admin" and str(current_user.id) != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    try:
        success = await zep_service.create_or_update_user(
            user_id=user_id,
            metadata=metadata
        )
        
        if success:
            return {
                "status": "success",
                "message": "User profile updated",
                "user_id": user_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update user profile")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def zep_health_check():
    """
    Check Zep memory system health
    """
    health = {
        "status": "healthy" if zep_service.enabled else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "zep_enabled": zep_service.enabled,
        "fallback_mode": not zep_service.enabled
    }
    
    if not zep_service.enabled:
        health["message"] = "Zep not available, using cache fallback"
        
    return health