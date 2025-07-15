"""
Adapter for Claude Code enhancements to work with existing memory system
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
from sqlalchemy.orm import Session

from ..models.memory import MemoryItem
from ..schemas.memory import MemoryCreate


class ClaudeMemoryAdapter:
    """Adapts Claude enhancement services to work with existing memory system"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        """Create memory with Claude-specific metadata"""
        
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            "memory_type": memory_type,
            "importance": importance,
            "session_id": session_id,
            "project_id": project_id,
            "created_by": "claude-code"
        })
        
        # Create memory data
        memory_data = MemoryCreate(
            content=content,
            tags=tags or [],
            metadata=meta
        )
        
        # Use existing memory service pattern
        from ..services.memory_service import MemoryService
        memory_service = MemoryService()
        
        return await memory_service.create_memory(self.db, memory_data)
    
    def get_memories_by_session(self, session_id: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories for a specific session"""
        from sqlalchemy import cast, String
        return self.db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"session_id": "{session_id}"')
        ).order_by(MemoryItem.created_at.desc()).limit(limit).all()
    
    def get_memories_by_project(self, project_id: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories for a specific project"""
        from sqlalchemy import cast, String
        return self.db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
        ).order_by(MemoryItem.created_at.desc()).limit(limit).all()
    
    def get_memories_by_type(self, memory_type: str, limit: int = 100) -> List[MemoryItem]:
        """Get memories of a specific type"""
        from sqlalchemy import cast, String
        return self.db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"memory_type": "{memory_type}"')
        ).order_by(MemoryItem.created_at.desc()).limit(limit).all()
    
    def get_error_memories(self, limit: int = 50) -> List[MemoryItem]:
        """Get error memories"""
        return self.get_memories_by_type("error", limit)
    
    def search_memories(self, query: str, limit: int = 20) -> List[MemoryItem]:
        """Search memories by content"""
        return self.db.query(MemoryItem).filter(
            MemoryItem.content.ilike(f"%{query}%")
        ).order_by(MemoryItem.access_count.desc()).limit(limit).all()
    
    def update_memory_metadata(self, memory_id: str, metadata_updates: Dict[str, Any]) -> Optional[MemoryItem]:
        """Update memory metadata"""
        memory = self.db.query(MemoryItem).filter(MemoryItem.id == memory_id).first()
        if memory:
            current_meta = memory.meta_data or {}
            current_meta.update(metadata_updates)
            memory.meta_data = current_meta
            memory.access_count += 1
            memory.accessed_at = datetime.utcnow()
            self.db.commit()
        return memory