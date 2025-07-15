"""
Memory Sync Router - API endpoints for memory synchronization
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..services.memory_sync import memory_sync_service

router = APIRouter(tags=["memory-sync"])
logger = logging.getLogger(__name__)


@router.get("/sync/health")
async def sync_health():
    """Health check for memory sync service"""
    return {
        "status": "healthy",
        "service": "memory_sync",
        "local_memory_path": str(memory_sync_service.local_memory_path),
        "capabilities": [
            "sync_local_to_database",
            "unified_memory_access",
            "cross_system_search"
        ]
    }


@router.post("/sync/run")
async def run_memory_sync(
    batch_size: int = Query(50, description="Batch size for sync operations")
):
    """Run memory synchronization from local system to database"""
    try:
        logger.info(f"Starting memory sync with batch size {batch_size}")
        
        # Run the sync
        sync_result = await memory_sync_service.sync_local_to_database(batch_size=batch_size)
        
        return {
            "success": True,
            "message": "Memory sync completed",
            "stats": sync_result,
            "batch_size": batch_size
        }
    
    except Exception as e:
        logger.error(f"Memory sync failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Memory sync failed: {str(e)}"
        )


@router.get("/sync/status")
async def get_sync_status():
    """Get current sync status and statistics"""
    return {
        "status": "available",
        "stats": memory_sync_service.sync_stats,
        "local_memory_path": str(memory_sync_service.local_memory_path),
        "synced_entries_file": str(memory_sync_service.synced_entries_file)
    }


@router.get("/sync/unified")
async def get_unified_memories(
    limit: int = Query(100, description="Maximum number of memories to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get memories from both local system and database"""
    try:
        result = await memory_sync_service.get_unified_memories(limit=limit, offset=offset)
        return result
    
    except Exception as e:
        logger.error(f"Error getting unified memories: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get unified memories: {str(e)}"
        )


@router.get("/sync/search")
async def search_unified_memories(
    query: str = Query(..., description="Search query"),
    limit: int = Query(50, description="Maximum number of results to return")
):
    """Search memories across both local system and database"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        result = await memory_sync_service.search_unified_memories(query=query, limit=limit)
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching unified memories: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/sync/count")
async def get_memory_counts():
    """Get memory counts from both systems"""
    try:
        # Get local count
        local_memories = memory_sync_service._get_local_memories()
        local_count = len(local_memories)
        
        # Get database count  
        from ..models import get_db
        from ..models.memory import MemoryItem
        
        db = next(get_db())
        try:
            db_count = db.query(MemoryItem).count()
        finally:
            db.close()
        
        return {
            "local_system": local_count,
            "database": db_count,
            "total": local_count + db_count,
            "sync_ratio": db_count / local_count if local_count > 0 else 0
        }
    
    except Exception as e:
        logger.error(f"Error getting memory counts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory counts: {str(e)}"
        )


@router.delete("/sync/reset")
async def reset_sync_state():
    """Reset sync state (clear synced entries tracking)"""
    try:
        # Remove synced entries file
        if memory_sync_service.synced_entries_file.exists():
            memory_sync_service.synced_entries_file.unlink()
        
        # Reset stats
        memory_sync_service.sync_stats = {
            "total_local": 0,
            "total_synced": 0,
            "errors": [],
            "last_sync": None
        }
        
        return {
            "success": True,
            "message": "Sync state reset successfully"
        }
    
    except Exception as e:
        logger.error(f"Error resetting sync state: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset sync state: {str(e)}"
        )