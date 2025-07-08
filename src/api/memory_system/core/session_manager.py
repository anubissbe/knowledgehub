"""Session management for memory system"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from ..models import MemorySession, Memory
from ..api.schemas import SessionCreate, SessionUpdate, SessionResponse
from ...services.cache import redis_client

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
                session_metadata=session_data.metadata or {},
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
            # Ensure session is attached to database session (in case it came from cache)
            if session not in self.db:
                # If session came from cache, merge it back into the DB session
                session = self.db.merge(session)
            # Update fields
            if update_data.metadata is not None:
                # Create new dict and assign to ensure SQLAlchemy detects change
                new_metadata = dict(session.session_metadata or {})
                new_metadata.update(update_data.metadata)
                session.session_metadata = new_metadata
            
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
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
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
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        
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
            # Convert session to dict for caching
            session_data = {
                'id': str(session.id),
                'user_id': session.user_id,
                'project_id': str(session.project_id) if session.project_id else None,
                'parent_session_id': str(session.parent_session_id) if session.parent_session_id else None,
                'started_at': session.started_at.isoformat() if session.started_at else None,
                'ended_at': session.ended_at.isoformat() if session.ended_at else None,
                'session_metadata': session.session_metadata,
                'tags': session.tags,
                'created_at': session.created_at.isoformat() if session.created_at else None,
                'updated_at': session.updated_at.isoformat() if session.updated_at else None
            }
            await redis_client.set(key, session_data, self._cache_ttl)
        except Exception as e:
            logger.warning(f"Failed to cache session: {e}")
    
    async def _get_cached_session(self, session_id: UUID) -> Optional[MemorySession]:
        """Get session from cache"""
        try:
            key = f"session:{session_id}"
            data = await redis_client.get(key)
            if data:
                # Reconstruct session object from cached data
                return self._reconstruct_session_from_cache(data)
        except Exception as e:
            logger.warning(f"Failed to get cached session: {e}")
        
        return None
    
    def _reconstruct_session_from_cache(self, cached_data: Dict[str, Any]) -> MemorySession:
        """Reconstruct MemorySession object from cached data"""
        
        # Create a new MemorySession instance without going through __init__
        session = MemorySession.__new__(MemorySession)
        
        # Set the basic attributes
        session.id = UUID(cached_data['id'])
        session.user_id = cached_data['user_id']
        session.project_id = UUID(cached_data['project_id']) if cached_data['project_id'] else None
        session.parent_session_id = UUID(cached_data['parent_session_id']) if cached_data['parent_session_id'] else None
        session.session_metadata = cached_data['session_metadata'] or {}
        session.tags = cached_data['tags'] or []
        
        # Parse datetime fields
        session.started_at = datetime.fromisoformat(cached_data['started_at']) if cached_data['started_at'] else None
        session.ended_at = datetime.fromisoformat(cached_data['ended_at']) if cached_data['ended_at'] else None
        session.created_at = datetime.fromisoformat(cached_data['created_at']) if cached_data['created_at'] else None
        session.updated_at = datetime.fromisoformat(cached_data['updated_at']) if cached_data['updated_at'] else None
        
        # Set SQLAlchemy state attributes to make it behave like a fresh DB object
        # This is important for proper session tracking and updates
        session._sa_class_manager = MemorySession.__mapper__.class_manager
        session._sa_instance_state = None  # Will be set when added to session
        
        logger.debug(f"Reconstructed session {session.id} from cache")
        return session