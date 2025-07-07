"""Session management for memory system"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ..models import MemorySession, Memory
from ..api.schemas import SessionCreate, SessionUpdate, SessionResponse
from ...api.services.cache import redis_client

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages Claude-Code conversation sessions"""
    
    def __init__(self, db: Session):
        self.db = db
        self._cache_ttl = 3600  # 1 hour cache
    
    async def create_session(self, session_data: SessionCreate) -> MemorySession:
        """Create a new session"""
        try:
            # Check for recent active sessions to link
            if not session_data.parent_session_id:
                recent_session = await self._find_recent_session(
                    session_data.user_id,
                    session_data.project_id
                )
                if recent_session:
                    session_data.parent_session_id = recent_session.id
                    logger.info(f"Linking to recent session: {recent_session.id}")
            
            # Create new session
            session = MemorySession(
                user_id=session_data.user_id,
                project_id=session_data.project_id,
                parent_session_id=session_data.parent_session_id,
                metadata=session_data.metadata or {},
                tags=session_data.tags or []
            )
            
            # Add automatic tags
            if session_data.project_id:
                session.add_tag("project")
            if session_data.parent_session_id:
                session.add_tag("continued")
            
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            
            # Cache session
            await self._cache_session(session)
            
            logger.info(f"Created session {session.id} for user {session.user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            self.db.rollback()
            raise
    
    async def get_session(self, session_id: UUID) -> Optional[MemorySession]:
        """Get session by ID"""
        # Check cache first
        cached = await self._get_cached_session(session_id)
        if cached:
            return cached
        
        # Query database
        session = self.db.query(MemorySession).filter_by(id=session_id).first()
        
        if session:
            await self._cache_session(session)
        
        return session
    
    async def update_session(self, session_id: UUID, 
                             update_data: SessionUpdate) -> Optional[MemorySession]:
        """Update session"""
        session = await self.get_session(session_id)
        if not session:
            return None
        
        try:
            # Update fields
            if update_data.metadata is not None:
                session.metadata.update(update_data.metadata)
            
            if update_data.tags is not None:
                for tag in update_data.tags:
                    session.add_tag(tag)
            
            if update_data.ended_at is not None:
                session.ended_at = update_data.ended_at
            
            self.db.commit()
            self.db.refresh(session)
            
            # Update cache
            await self._cache_session(session)
            
            logger.info(f"Updated session {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            self.db.rollback()
            raise
    
    async def end_session(self, session_id: UUID) -> Optional[MemorySession]:
        """End a session"""
        session = await self.get_session(session_id)
        if not session:
            return None
        
        try:
            session.end_session()
            self.db.commit()
            
            # Process session for insights
            await self._process_session_end(session)
            
            # Update cache
            await self._cache_session(session)
            
            logger.info(f"Ended session {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            self.db.rollback()
            raise
    
    async def get_user_sessions(self, user_id: str, 
                                project_id: Optional[UUID] = None,
                                active_only: bool = False,
                                limit: int = 10) -> List[MemorySession]:
        """Get sessions for a user"""
        query = self.db.query(MemorySession).filter_by(user_id=user_id)
        
        if project_id:
            query = query.filter_by(project_id=project_id)
        
        if active_only:
            query = query.filter(MemorySession.ended_at.is_(None))
        
        sessions = query.order_by(desc(MemorySession.started_at)).limit(limit).all()
        
        return sessions
    
    async def get_session_chain(self, session_id: UUID) -> List[MemorySession]:
        """Get all sessions in a conversation chain"""
        session = await self.get_session(session_id)
        if not session:
            return []
        
        chain = []
        
        # Walk up to root
        current = session
        while current.parent_session_id:
            parent = await self.get_session(current.parent_session_id)
            if parent:
                chain.insert(0, parent)
                current = parent
            else:
                break
        
        # Add current session
        chain.append(session)
        
        # Walk down to children
        children = self.db.query(MemorySession).filter_by(
            parent_session_id=session_id
        ).order_by(MemorySession.started_at).all()
        
        chain.extend(children)
        
        return chain
    
    async def cleanup_stale_sessions(self, hours: int = 24):
        """Clean up sessions that have been inactive for too long"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        stale_sessions = self.db.query(MemorySession).filter(
            and_(
                MemorySession.ended_at.is_(None),
                MemorySession.updated_at < cutoff
            )
        ).all()
        
        for session in stale_sessions:
            try:
                session.end_session()
                session.add_tag("auto-closed")
                logger.info(f"Auto-closed stale session {session.id}")
            except Exception as e:
                logger.error(f"Failed to close session {session.id}: {e}")
        
        self.db.commit()
        return len(stale_sessions)
    
    # Private methods
    async def _find_recent_session(self, user_id: str, 
                                   project_id: Optional[UUID],
                                   window_minutes: int = 30) -> Optional[MemorySession]:
        """Find recent session to potentially link to"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        query = self.db.query(MemorySession).filter(
            and_(
                MemorySession.user_id == user_id,
                MemorySession.ended_at.is_(None),
                MemorySession.updated_at > cutoff
            )
        )
        
        if project_id:
            query = query.filter_by(project_id=project_id)
        
        return query.order_by(desc(MemorySession.updated_at)).first()
    
    async def _process_session_end(self, session: MemorySession):
        """Process session end for insights and cleanup"""
        # Extract session summary
        summary = {
            'duration': session.duration,
            'memory_count': session.memory_count,
            'important_memories': len(session.important_memories),
            'entities': self._extract_session_entities(session)
        }
        
        session.add_metadata('summary', summary)
        
        # Log analytics
        logger.info(f"Session {session.id} summary: {summary}")
    
    def _extract_session_entities(self, session: MemorySession) -> List[str]:
        """Extract unique entities from session memories"""
        entities = set()
        for memory in session.memories:
            if memory.entities:
                entities.update(memory.entities)
        return list(entities)
    
    async def _cache_session(self, session: MemorySession):
        """Cache session in Redis"""
        try:
            key = f"session:{session.id}"
            await redis_client.setex(
                key,
                self._cache_ttl,
                session.to_dict()
            )
        except Exception as e:
            logger.warning(f"Failed to cache session: {e}")
    
    async def _get_cached_session(self, session_id: UUID) -> Optional[MemorySession]:
        """Get session from cache"""
        try:
            key = f"session:{session_id}"
            data = await redis_client.get(key)
            if data:
                # Reconstruct session from cached data
                # Note: This is simplified, full implementation would deserialize properly
                return None  # For now, skip cache reconstruction
        except Exception as e:
            logger.warning(f"Failed to get cached session: {e}")
        
        return None