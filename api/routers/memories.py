"""Memory management router"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging

from ..dependencies import get_memory_service, get_db
from ..schemas.memory import MemoryCreate
from ..models.memory import MemoryResponse
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=List[MemoryResponse])
async def get_memories(
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    source: Optional[str] = Query(None, description="Filter by source"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    db: Session = Depends(get_db),
    memory_service=Depends(get_memory_service)
):
    """Get all memories with optional filtering"""
    try:
        memories = memory_service.get_memories(
            db=db,
            memory_type=memory_type,
            source=source,
            skip=skip,
            limit=limit
        )
        # Convert SQLAlchemy models to MemoryResponse
        return [memory_service._memory_to_response(memory) for memory in memories]
        
    except Exception as e:
        logger.error(f"Error fetching memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch memories")


@router.post("/", response_model=MemoryResponse, status_code=201)
async def create_memory(
    memory: MemoryCreate,
    db: Session = Depends(get_db),
    memory_service=Depends(get_memory_service)
):
    """Store a new memory item using the proper memory service"""
    try:
        # Convert from router schema to service schema
        from ..models.memory import MemoryCreate as ServiceMemoryCreate, MemoryType, MemoryImportance
        
        service_memory = ServiceMemoryCreate(
            user_id="api_user",  # Default user for API calls
            session_id="api_session",  # Default session for API calls
            content=memory.content,
            memory_type=MemoryType.CONVERSATION,  # Default type
            context={},
            metadata=memory.metadata or {},
            tags=memory.tags or [],
            importance=MemoryImportance.MEDIUM
        )
        
        # Use the proper memory service WITH embeddings and AI features
        result = await memory_service.create_memory(
            service_memory,
            generate_embeddings=True,
            auto_cluster=True
        )
        return result
        
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {str(e)}")


@router.get("/stats")
async def get_memory_stats(
    db: Session = Depends(get_db),
    memory_service=Depends(get_memory_service)
):
    """Get memory statistics"""
    try:
        from sqlalchemy import func
        from ..models.memory import Memory
        
        # Get total count
        total = db.query(func.count(Memory.id)).scalar() or 0
        
        # Get count by type
        type_counts = db.query(
            Memory.memory_type,
            func.count(Memory.id)
        ).group_by(Memory.memory_type).all()
        
        # Get count by source
        source_counts = db.query(
            Memory.source,
            func.count(Memory.id)
        ).group_by(Memory.source).all()
        
        # Get recent activity
        recent = db.query(func.count(Memory.id)).filter(
            Memory.created_at >= func.now() - text("INTERVAL '1 day'")
        ).scalar() or 0
        
        return {
            "total_memories": total,
            "memories_by_type": {t: c for t, c in type_counts if t},
            "memories_by_source": {s: c for s, c in source_counts if s},
            "recent_24h": recent,
            "storage_used_mb": round(total * 0.1, 2),  # Estimate ~100KB per memory
            "active_sessions": 1  # Placeholder
        }
        
    except Exception as e:
        logger.error(f"Error fetching memory stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch memory statistics")


@router.get("/recent", response_model=List[MemoryResponse])
async def get_recent_memories(
    limit: int = Query(100, ge=1, le=1000, description="Number of recent memories to retrieve"),
    db: Session = Depends(get_db),
    memory_service=Depends(get_memory_service)
):
    """Get recent memories"""
    try:
        memories = await memory_service.get_memories(
            db,
            skip=0,
            limit=limit,
            memory_type=None,
            source=None
        )
        return memories
        
    except Exception as e:
        logger.error(f"Error fetching recent memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent memories")


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    db: Session = Depends(get_db),
    memory_service=Depends(get_memory_service)
):
    """Get a specific memory by ID"""
    try:
        from uuid import UUID
        memory = memory_service.get_memory(db, UUID(memory_id))
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch memory")


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    db: Session = Depends(get_db),
    memory_service=Depends(get_memory_service)
):
    """Delete a memory by ID"""
    try:
        from uuid import UUID
        from ..models.memory import Memory
        
        # Convert string to UUID
        try:
            memory_uuid = UUID(memory_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid memory ID format")
        
        # Find the memory directly in the database
        memory = db.query(Memory).filter(Memory.id == memory_uuid).first()
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        # Delete the memory
        db.delete(memory)
        db.commit()
        
        logger.info(f"Deleted memory: {memory_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")


