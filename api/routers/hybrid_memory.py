#!/usr/bin/env python3
"""
Hybrid Memory API Routes - Fast local memory with distributed sync
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

from ..services.hybrid_memory_service import HybridMemoryService
from ..services.memory_sync_service import MemorySyncService
from ..utils.token_optimizer import TokenOptimizer
from ..dependencies import get_db

# Mock get_current_user for now
async def get_current_user():
    return "claude"

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton services
hybrid_service = HybridMemoryService()
sync_service = None
token_optimizer = TokenOptimizer()

# Initialize sync service on import
import asyncio

async def _init_sync_service():
    global sync_service
    if sync_service is None:
        await hybrid_service.initialize()
        sync_service = MemorySyncService(
            local_db_path=str(hybrid_service.local_db_path),
            redis_url="redis://redis:6379"
        )
        await sync_service.start()
        logger.info("Hybrid memory sync service initialized")

# Try to initialize immediately
try:
    asyncio.create_task(_init_sync_service())
except RuntimeError:
    # No event loop yet, will initialize on first request
    pass


# Request/Response models
class QuickStoreRequest(BaseModel):
    content: str
    type: str = "general"
    project: Optional[str] = None
    tags: Optional[List[str]] = None
    priority: str = "normal"


class QuickStoreResponse(BaseModel):
    memory_id: str
    tokens_saved: int
    sync_status: str


class MemoryItem(BaseModel):
    id: str
    content: str
    type: str
    created_at: str
    accessed_at: str
    access_count: int
    metadata: Optional[Dict[str, Any]] = None


class OptimizeRequest(BaseModel):
    content: str
    target_reduction: float = 50.0
    preserve_code: bool = True
    context: Optional[Dict[str, Any]] = None


class OptimizeResponse(BaseModel):
    original_tokens: int
    optimized_tokens: int
    savings_percentage: float
    optimized_content: str
    strategies_applied: List[str]


class WorkflowRequest(BaseModel):
    project: str
    phase: str
    context: Optional[str] = None


class BoardUpdateRequest(BaseModel):
    project: str
    task_id: str
    status: str
    notes: Optional[str] = None


class RelationshipRequest(BaseModel):
    source: str
    target: str
    relationship: str
    strength: float = 1.0


@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global sync_service
    
    await hybrid_service.initialize()
    
    sync_service = MemorySyncService(
        local_db_path=str(hybrid_service.local_db_path),
        redis_url="redis://redis:6379"
    )
    await sync_service.start()
    
    logger.info("Hybrid memory services initialized")


@router.post("/quick-store", response_model=QuickStoreResponse)
async def quick_store(
    request: QuickStoreRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
):
    """Store memory locally with automatic background sync"""
    try:
        # Ensure sync service is initialized
        if sync_service is None:
            await _init_sync_service()
        metadata = {}
        if request.project:
            metadata["project"] = request.project
        if request.tags:
            metadata["tags"] = request.tags
        metadata["priority"] = request.priority
        
        # Store in hybrid system
        memory_id = await hybrid_service.store(
            user_id=user_id,
            content=request.content,
            memory_type=request.type,
            metadata=metadata
        )
        
        # Get token savings
        original_tokens = token_optimizer.count_tokens(request.content)
        optimized, opt_meta = token_optimizer.optimize(request.content)
        
        # Queue background sync if service is available
        if sync_service:
            background_tasks.add_task(sync_service.force_sync, [memory_id])
        
        return QuickStoreResponse(
            memory_id=memory_id,
            tokens_saved=opt_meta["token_savings"],
            sync_status="pending"
        )
        
    except Exception as e:
        logger.error(f"Quick store failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick-recall", response_model=List[MemoryItem])
async def quick_recall(
    query: str,
    type: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    user_id: str = Depends(get_current_user)
):
    """Ultra-fast memory recall with cascade search"""
    try:
        results = await hybrid_service.recall(
            query=query,
            user_id=user_id,
            memory_type=type,
            limit=limit
        )
        
        # Filter by project if specified
        if project:
            results = [
                r for r in results 
                if r.get("metadata", {}).get("project") == project
            ]
        
        # Convert to response model
        return [
            MemoryItem(
                id=r["id"],
                content=r["content"],
                type=r.get("type", "general"),
                created_at=r["created_at"],
                accessed_at=r["accessed_at"],
                access_count=r["access_count"],
                metadata=r.get("metadata")
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Quick recall failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_context(request: OptimizeRequest):
    """Optimize content to reduce token usage"""
    try:
        optimized, metadata = token_optimizer.optimize(
            request.content,
            context=request.context
        )
        
        return OptimizeResponse(
            original_tokens=metadata["original_tokens"],
            optimized_tokens=metadata["optimized_tokens"],
            savings_percentage=metadata["savings_percentage"],
            optimized_content=optimized,
            strategies_applied=metadata["strategies_applied"]
        )
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/track")
async def track_workflow(
    request: WorkflowRequest,
    user_id: str = Depends(get_current_user)
):
    """Track project workflow phases"""
    try:
        await hybrid_service.track_workflow(
            project_id=request.project,
            phase=request.phase,
            context=request.context
        )
        
        return {"status": "success", "message": f"Workflow updated: {request.project} → {request.phase}"}
        
    except Exception as e:
        logger.error(f"Workflow tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/board/update")
async def update_board(
    request: BoardUpdateRequest,
    user_id: str = Depends(get_current_user)
):
    """Update task board"""
    try:
        # Store as special board memory
        board_content = f"BOARD|{request.project}|{request.task_id}|{request.status}"
        if request.notes:
            board_content += f"|{request.notes}"
        
        memory_id = await hybrid_service.store(
            user_id=user_id,
            content=board_content,
            memory_type="board_update",
            metadata={
                "project": request.project,
                "task_id": request.task_id,
                "status": request.status,
                "notes": request.notes or ""
            }
        )
        
        return {
            "status": "success", 
            "memory_id": memory_id,
            "message": f"Board updated: {request.task_id} → {request.status}"
        }
        
    except Exception as e:
        logger.error(f"Board update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationships")
async def create_relationship(
    request: RelationshipRequest,
    user_id: str = Depends(get_current_user)
):
    """Create relationship between memories"""
    try:
        await hybrid_service.map_relationships(
            source_id=request.source,
            target_id=request.target,
            relationship=request.relationship,
            strength=request.strength
        )
        
        return {
            "status": "success",
            "message": f"Relationship created: {request.source} -{request.relationship}→ {request.target}"
        }
        
    except Exception as e:
        logger.error(f"Relationship creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync/status")
async def get_sync_status(detailed: bool = Query(False)):
    """Get memory synchronization status"""
    try:
        # Ensure sync service is initialized
        if sync_service is None:
            await _init_sync_service()
            
        if not sync_service:
            # Return default status if sync service is still not available
            return {
                "is_syncing": False,
                "last_sync": None,
                "pending_count": 0,
                "failed_count": 0
            }
        
        status = await sync_service.get_sync_status()
        
        if not detailed:
            # Simplified status
            return {
                "is_syncing": status["is_syncing"],
                "last_sync": status.get("last_sync"),
                "pending_count": status.get("status_counts", {}).get("pending", 0),
                "failed_count": status.get("status_counts", {}).get("failed", 0)
            }
        
        return status
        
    except Exception as e:
        logger.error(f"Sync status failed: {e}", exc_info=True)
        # Return safe default status on error
        return {
            "is_syncing": False,
            "last_sync": None,
            "pending_count": 0,
            "failed_count": 0
        }


@router.post("/sync/force")
async def force_sync(
    memory_ids: Optional[List[str]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Force synchronization of specific memories or all pending"""
    try:
        if not sync_service:
            raise HTTPException(status_code=503, detail="Sync service not initialized")
        
        background_tasks.add_task(sync_service.force_sync, memory_ids)
        
        return {
            "status": "success",
            "message": f"Sync queued for {len(memory_ids) if memory_ids else 'all pending'} memories"
        }
        
    except Exception as e:
        logger.error(f"Force sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache performance statistics"""
    try:
        metrics = await hybrid_service.get_metrics()
        
        return {
            "cache_hit_rate": f"{metrics['cache_hit_rate']*100:.1f}%",
            "local_hit_rate": f"{metrics['local_hit_rate']*100:.1f}%",
            "total_queries": metrics["total_queries"],
            "token_savings": metrics["token_savings_total"],
            "performance": {
                "local_hits": metrics["local_hits"],
                "cache_hits": metrics["cache_hits"],
                "remote_hits": metrics["remote_hits"]
            }
        }
        
    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{project}")
async def analyze_project_memory(
    project: str,
    timeframe: str = Query("7d", regex="^\\d+d$"),
    include_suggestions: bool = Query(True)
):
    """Analyze memory patterns for a project"""
    try:
        # TODO: Implement project-specific analysis
        return {
            "project": project,
            "timeframe": timeframe,
            "analysis": "Not yet implemented",
            "suggestions": [] if include_suggestions else None
        }
        
    except Exception as e:
        logger.error(f"Memory analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))