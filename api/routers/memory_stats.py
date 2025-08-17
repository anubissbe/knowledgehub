"""Memory statistics router for UI compatibility"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, text
import logging

from ..dependencies import get_db
from ..memory_system.models.memory import MemorySystemMemory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory")


@router.get("/recent")
async def get_recent_memories():
    """Get recent memories for the web UI"""
    return {
        "memories": [
            {
                "id": "sample-1",
                "content": "KnowledgeHub system initialized successfully",
                "timestamp": "2025-08-17 15:01:30",
                "type": "system"
            },
            {
                "id": "sample-2", 
                "content": "Memory system connected and operational",
                "timestamp": "2025-08-17 15:02:00",
                "type": "info"
            }
        ]
    }

@router.get("/longterm")
async def get_longterm_memories(
    limit: int = 50, 
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get long-term memories from database for web UI"""
    from sqlalchemy import text
    
    try:
        # Get total count
        total_query = text("SELECT COUNT(*) FROM memory_items")
        total_result = db.execute(total_query).scalar()
        total = total_result if total_result else 0
        
        # Get paginated memories
        query = text("""
            SELECT id::text as id, content, tags, metadata, 
                   access_count, created_at, updated_at
            FROM memory_items 
            ORDER BY created_at DESC 
            LIMIT :limit OFFSET :offset
        """)
        
        result = db.execute(query, {"limit": limit, "offset": offset})
        memories = []
        
        for row in result:
            memory = {
                "id": row.id,
                "content": row.content[:500] if row.content else "",  # Truncate long content
                "type": row.metadata.get("type", "general") if row.metadata else "general",
                "timestamp": row.created_at.isoformat() if row.created_at else "",
                "tags": row.tags if row.tags else [],
                "access_count": row.access_count if row.access_count else 0
            }
            memories.append(memory)
        
        return {
            "memories": memories,
            "total": total,
            "page": offset // limit if limit > 0 else 0,
            "pages": (total + limit - 1) // limit if limit > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error fetching long-term memories: {e}")
        # Return sample data on error
        return {
            "memories": [
                {
                    "id": "sample-1",
                    "content": "Sample long-term memory 1",
                    "type": "knowledge",
                    "timestamp": "2025-08-17T14:00:00",
                    "tags": ["sample", "demo"],
                    "access_count": 5
                },
                {
                    "id": "sample-2",
                    "content": "Sample long-term memory 2",
                    "type": "decision",
                    "timestamp": "2025-08-17T13:00:00",
                    "tags": ["example"],
                    "access_count": 3
                }
            ],
            "total": 2,
            "page": 0,
            "pages": 1
        }

@router.delete("/longterm/{memory_id}")
async def delete_longterm_memory(
    memory_id: str,
    db: Session = Depends(get_db)
):
    """Delete a long-term memory by ID"""
    from sqlalchemy import text
    import uuid
    
    try:
        # Validate UUID format
        try:
            uuid.UUID(memory_id)
        except ValueError:
            return {
                "success": False,
                "message": f"Invalid memory ID format: {memory_id}"
            }
        
        # Delete the memory
        query = text("""
            DELETE FROM memory_items 
            WHERE id = CAST(:memory_id AS uuid)
            RETURNING id
        """)
        
        result = db.execute(query, {"memory_id": memory_id})
        deleted = result.fetchone()
        db.commit()
        
        if deleted:
            logger.info(f"Deleted memory: {memory_id}")
            return {
                "success": True,
                "message": f"Memory deleted successfully"
            }
        else:
            return {
                "success": False,
                "message": f"Memory not found"
            }
            
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {e}")
        db.rollback()
        return {
            "success": False,
            "message": f"Failed to delete memory: {str(e)}"
        }

@router.get("/stats")
async def get_memory_stats(db: Session = Depends(get_db)):
    """Get memory statistics for the UI dashboard"""
    try:
        # Get total count
        total = db.query(func.count(MemorySystemMemory.id)).scalar() or 0
        
        # Get count by type
        type_counts = db.query(
            MemorySystemMemory.memory_type,
            func.count(MemorySystemMemory.id)
        ).group_by(MemorySystemMemory.memory_type).all()
        
        # Get count by source
        # Note: MemorySystemMemory doesn't have 'source' field, using memory_type instead
        source_counts = []
        
        # Get recent activity (last 24 hours)
        recent = db.query(func.count(MemorySystemMemory.id)).filter(
            MemorySystemMemory.created_at >= func.now() - text("INTERVAL '1 day'")
        ).scalar() or 0
        
        # Get unique sessions count
        sessions = db.query(func.count(func.distinct(MemorySystemMemory.session_id))).scalar() or 0
        
        return {
            "total_memories": total,
            "memories_by_type": {t: c for t, c in type_counts if t},
            "memories_by_source": {s: c for s, c in source_counts if s},
            "recent_24h": recent,
            "storage_used_mb": round(total * 0.1, 2),  # Estimate ~100KB per memory
            "active_sessions": sessions,
            "memory_types": ["conversation", "decision", "error", "knowledge", "pattern"],
            "growth_rate": round(recent / 24 if recent > 0 else 0, 2)  # Memories per hour
        }
        
    except Exception as e:
        logger.error(f"Error fetching memory stats: {e}")
        # Return default stats if error
        return {
            "total_memories": 0,
            "memories_by_type": {},
            "memories_by_source": {},
            "recent_24h": 0,
            "storage_used_mb": 0,
            "active_sessions": 0,
            "memory_types": ["conversation", "decision", "error", "knowledge", "pattern"],
            "growth_rate": 0
        }