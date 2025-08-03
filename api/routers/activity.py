"""
Activity Tracking Router
Provides recent activity data for the dashboard
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..dependencies import get_db
from ..models import MemoryItem, MistakeTracking

router = APIRouter(prefix="/api/activity")


@router.get("/recent")
async def get_recent_activity(
    limit: int = 10,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get recent activity across the system
    
    Returns:
    - Recent memories created
    - Recent errors tracked
    - Recent sessions
    - Activity timeline
    """
    
    # Get recent memories
    recent_memories = db.query(MemoryItem)\
        .order_by(desc(MemoryItem.created_at))\
        .limit(limit)\
        .all()
    
    # Get recent errors
    recent_errors = db.query(MistakeTracking)\
        .order_by(desc(MistakeTracking.created_at))\
        .limit(5)\
        .all()
    
    # Build activity timeline
    activities = []
    
    for memory in recent_memories:
        # Extract type from tags or metadata
        memory_type = 'general'
        if memory.tags and len(memory.tags) > 0:
            memory_type = memory.tags[0]
        elif memory.meta_data and 'type' in memory.meta_data:
            memory_type = memory.meta_data['type']
            
        activities.append({
            "type": "memory",
            "title": f"Memory: {memory_type}",
            "description": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
            "timestamp": memory.created_at.isoformat() if memory.created_at else datetime.utcnow().isoformat(),
            "metadata": {
                "memory_id": str(memory.id),
                "tags": memory.tags or [],
                "type": memory_type,
                "access_count": memory.access_count
            }
        })
    
    for error in recent_errors:
        activities.append({
            "type": "error",
            "title": f"Error: {error.error_type}",
            "description": error.error_message[:100] + "..." if len(error.error_message) > 100 else error.error_message,
            "timestamp": error.created_at.isoformat() if error.created_at else datetime.utcnow().isoformat(),
            "metadata": {
                "error_id": str(error.id),
                "resolved": error.resolved,
                "occurrences": getattr(error, 'occurrence_count', 1)
            }
        })
    
    # Sort activities by timestamp
    activities.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Get activity stats
    now = datetime.utcnow()
    last_hour = now - timedelta(hours=1)
    last_day = now - timedelta(days=1)
    
    hourly_memories = db.query(MemoryItem)\
        .filter(MemoryItem.created_at >= last_hour)\
        .count()
    
    daily_memories = db.query(MemoryItem)\
        .filter(MemoryItem.created_at >= last_day)\
        .count()
    
    return {
        "activities": activities[:limit],
        "stats": {
            "last_hour": hourly_memories,
            "last_day": daily_memories,
            "total_memories": db.query(MemoryItem).count(),
            "total_errors": db.query(MistakeTracking).count(),
            "active_sessions": 1  # Simplified for now
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/stats")
async def get_activity_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get activity statistics"""
    
    now = datetime.utcnow()
    stats = {
        "hourly": {},
        "daily": {},
        "weekly": {}
    }
    
    # Hourly stats (last 24 hours)
    for i in range(24):
        hour_start = now - timedelta(hours=i+1)
        hour_end = now - timedelta(hours=i)
        
        count = db.query(MemoryItem)\
            .filter(MemoryItem.created_at >= hour_start)\
            .filter(MemoryItem.created_at < hour_end)\
            .count()
        
        stats["hourly"][hour_start.strftime("%Y-%m-%d %H:00")] = count
    
    # Daily stats (last 7 days)
    for i in range(7):
        day_start = now.date() - timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        
        count = db.query(MemoryItem)\
            .filter(MemoryItem.created_at >= day_start)\
            .filter(MemoryItem.created_at < day_end)\
            .count()
        
        stats["daily"][str(day_start)] = count
    
    return {
        "stats": stats,
        "timestamp": datetime.utcnow().isoformat()
    }