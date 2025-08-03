"""Memory statistics router for UI compatibility"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, text
import logging

from ..dependencies import get_db
from ..memory_system.models.memory import MemorySystemMemory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory")


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