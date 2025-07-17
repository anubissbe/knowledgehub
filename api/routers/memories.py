"""Memory management router"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging

from ..dependencies import get_memory_service, get_db
from ..schemas.memory import MemoryCreate, MemoryResponse
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
        # Convert SQLAlchemy models to dict for proper serialization
        return [memory.to_dict() for memory in memories]
        
    except Exception as e:
        logger.error(f"Error fetching memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch memories")


@router.post("/", response_model=MemoryResponse, status_code=201)
async def create_memory(
    memory: MemoryCreate,
    db: Session = Depends(get_db),
    memory_service=Depends(get_memory_service)
):
    """Store a new memory item"""
    try:
        result = await memory_service.create_memory(db, memory)
        return result
        
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to create memory")


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
        memory = memory_service.get_memory(db, UUID(memory_id))
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        db.delete(memory)
        db.commit()
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete memory")


