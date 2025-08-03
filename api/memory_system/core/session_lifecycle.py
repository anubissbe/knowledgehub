"""Session Lifecycle Management

This module handles the complete lifecycle of memory sessions including:
- Session initialization with proper defaults
- State transition management
- Session finalization and cleanup
- Error recovery and resilience
- Analytics and insights extraction
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from uuid import UUID
import asyncio

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..models import MemorySession, MemorySystemMemory, MemoryType
from ..api.schemas import SessionCreate
from ...services.cache import redis_client
from ..services.context_service import ContextService

logger = logging.getLogger(__name__)


class SessionLifecycleManager:
    """Manages the complete lifecycle of memory sessions"""
    
    def __init__(self, db: Session):
        self.db = db
        self.context_service = ContextService()
        
    async def start_session(
        self, 
        session_data: SessionCreate,
        parent_session_id: Optional[UUID] = None
    ) -> MemorySession:
        """Start a new session with proper initialization
        
        Args:
            session_data: Session creation data
            parent_session_id: Optional parent session for conversation chains
            
        Returns:
            Initialized MemorySession instance
        """
        try:
            # Check for existing active session
            existing_active = await self._check_active_session(
                session_data.user_id, 
                session_data.project_id
            )
            
            if existing_active:
                logger.warning(
                    f"User {session_data.user_id} has active session {existing_active.id}. "
                    "Creating linked session."
                )
                parent_session_id = existing_active.id
            
            # Create session
            session = MemorySession(
                user_id=session_data.user_id,
                project_id=session_data.project_id,
                parent_session_id=parent_session_id,
                session_metadata=session_data.metadata or {},
                tags=session_data.tags or []
            )
            
            # Initialize session state
            await self._initialize_session_state(session)
            
            # Add to database
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            
            # Post-creation tasks
            await self._post_session_creation(session)
            
            logger.info(
                f"Started session {session.id} for user {session.user_id} "
                f"(parent: {parent_session_id})"
            )
            
            return session
            
        except SQLAlchemyError as e:
            logger.error(f"Database error starting session: {e}")
            self.db.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error starting session: {e}")
            raise
    
    async def end_session(
        self, 
        session_id: UUID,
        reason: str = "normal",
        final_summary: Optional[str] = None
    ) -> MemorySession:
        """End a session with proper cleanup and finalization
        
        Args:
            session_id: Session to end
            reason: Reason for ending (normal, timeout, error, user_requested)
            final_summary: Optional final summary of the session
            
        Returns:
            Ended MemorySession instance
        """
        session = self.db.query(MemorySession).filter_by(id=session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.ended_at:
            logger.warning(f"Session {session_id} already ended")
            return session
        
        try:
            # Pre-end tasks
            await self._pre_session_end(session)
            
            # End the session
            session.end_session()
            session.add_metadata('end_reason', reason)
            
            if final_summary:
                session.add_metadata('final_summary', final_summary)
            
            # Extract insights
            insights = await self._extract_session_insights(session)
            session.add_metadata('insights', insights)
            
            # Generate session summary
            summary = await self._generate_session_summary(session)
            session.add_metadata('summary', summary)
            
            self.db.commit()
            
            # Post-end tasks
            await self._post_session_end(session)
            
            logger.info(
                f"Ended session {session_id} (reason: {reason}, "
                f"duration: {session.duration}, memories: {session.memory_count})"
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            self.db.rollback()
            raise
    
    async def _check_active_session(
        self, 
        user_id: str, 
        project_id: Optional[UUID]
    ) -> Optional[MemorySession]:
        """Check for existing active session"""
        query = self.db.query(MemorySession).filter_by(
            user_id=user_id,
            ended_at=None
        )
        
        if project_id:
            query = query.filter_by(project_id=project_id)
        
        # Get most recent active session
        return query.order_by(MemorySession.started_at.desc()).first()
    
    async def _initialize_session_state(self, session: MemorySession):
        """Initialize session state and metadata"""
        # Set initial state
        session.add_metadata('state', 'active')
        session.add_metadata('initialized_at', datetime.now(timezone.utc).isoformat())
        
        # Initialize context if parent exists
        if session.parent_session_id:
            parent_context = await self._get_parent_context(session.parent_session_id)
            if parent_context:
                session.add_metadata('parent_context', parent_context)
        
        # Set session parameters
        session.add_metadata('parameters', {
            'max_memories': 1000,
            'context_window': 50,
            'importance_threshold': 0.7,
            'auto_cleanup': True
        })
    
    async def _post_session_creation(self, session: MemorySession):
        """Post-creation tasks"""
        # Cache session
        await self._cache_session(session)
        
        # Initialize welcome memory if first session
        if not session.parent_session_id:
            await self._create_welcome_memory(session)
        
        # Schedule cleanup task
        asyncio.create_task(self._schedule_session_cleanup(session.id))
    
    async def _pre_session_end(self, session: MemorySession):
        """Pre-end cleanup and validation"""
        # Ensure all pending memories are saved
        await self._flush_pending_memories(session)
        
        # Update final statistics
        session.add_metadata('final_stats', {
            'total_memories': session.memory_count,
            'important_memories': len(session.important_memories),
            'avg_importance': self._calculate_avg_importance(session),
            'unique_entities': len(self._extract_unique_entities(session))
        })
    
    async def _post_session_end(self, session: MemorySession):
        """Post-end tasks"""
        # Update cache
        await self._cache_session(session)
        
        # Trigger analytics event
        await self._trigger_analytics_event('session_ended', {
            'session_id': str(session.id),
            'user_id': session.user_id,
            'duration': session.duration,
            'memory_count': session.memory_count
        })
        
        # Clean up temporary data
        await self._cleanup_session_temp_data(session.id)
    
    async def _extract_session_insights(self, session: MemorySession) -> Dict[str, Any]:
        """Extract insights from the session"""
        insights = {
            'topics': await self._extract_topics(session),
            'key_decisions': await self._extract_decisions(session),
            'errors_encountered': await self._extract_errors(session),
            'learning_points': await self._extract_learning_points(session),
            'action_items': await self._extract_action_items(session)
        }
        
        return insights
    
    async def _generate_session_summary(self, session: MemorySession) -> str:
        """Generate a natural language summary of the session"""
        # Get most important memories
        important_memories = sorted(
            session.memories,
            key=lambda m: m.importance,
            reverse=True
        )[:5]
        
        # Build summary
        summary_parts = []
        
        if session.session_metadata.get('final_summary'):
            summary_parts.append(session.session_metadata['final_summary'])
        
        summary_parts.append(
            f"Session lasted {session.duration} with {session.memory_count} memories."
        )
        
        if important_memories:
            summary_parts.append("Key points discussed:")
            for memory in important_memories:
                summary_parts.append(f"- {memory.summary or memory.content[:100]}...")
        
        return " ".join(summary_parts)
    
    async def _get_parent_context(self, parent_id: UUID) -> Optional[Dict[str, Any]]:
        """Get context from parent session"""
        parent = self.db.query(MemorySession).filter_by(id=parent_id).first()
        if not parent:
            return None
        
        return {
            'session_id': str(parent.id),
            'summary': parent.session_metadata.get('summary', ''),
            'key_entities': self._extract_unique_entities(parent),
            'important_facts': [m.content for m in parent.important_memories[:3]]
        }
    
    async def _create_welcome_memory(self, session: MemorySession):
        """Create initial welcome memory for new sessions"""
        welcome_memory = MemorySystemMemory(
            session_id=session.id,
            content="New session started. I'm ready to help and will remember our conversation.",
            summary="Session initialization",
            memory_type=MemoryType.FACT.value,
            importance=0.5,
            confidence=1.0,
            entities=["Claude-Code", "session", "initialization"]
        )
        
        self.db.add(welcome_memory)
        self.db.commit()
    
    async def _flush_pending_memories(self, session: MemorySession):
        """Ensure all pending memories are saved"""
        # This would handle any in-memory buffers or pending operations
        pass
    
    def _calculate_avg_importance(self, session: MemorySession) -> float:
        """Calculate average importance of session memories"""
        if not session.memories:
            return 0.0
        
        total_importance = sum(m.importance for m in session.memories)
        return total_importance / len(session.memories)
    
    def _extract_unique_entities(self, session: MemorySession) -> List[str]:
        """Extract unique entities from session"""
        entities = set()
        for memory in session.memories:
            if memory.entities:
                entities.update(memory.entities)
        return list(entities)
    
    async def _extract_topics(self, session: MemorySession) -> List[str]:
        """Extract main topics discussed"""
        # Simple entity-based topic extraction
        entity_counts = {}
        for memory in session.memories:
            if memory.entities:
                for entity in memory.entities:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        # Return top entities as topics
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        return [entity for entity, _ in sorted_entities[:5]]
    
    async def _extract_decisions(self, session: MemorySession) -> List[str]:
        """Extract decisions made during session"""
        decisions = []
        for memory in session.memories:
            if memory.memory_type == MemoryType.DECISION.value:
                decisions.append(memory.summary or memory.content[:100])
        return decisions
    
    async def _extract_errors(self, session: MemorySession) -> List[str]:
        """Extract errors encountered"""
        errors = []
        for memory in session.memories:
            if memory.memory_type == MemoryType.ERROR.value:
                errors.append(memory.summary or memory.content[:100])
        return errors
    
    async def _extract_learning_points(self, session: MemorySession) -> List[str]:
        """Extract learning points from session"""
        learning = []
        for memory in session.memories:
            if memory.memory_type in [MemoryType.FACT.value, MemoryType.PATTERN.value]:
                if memory.importance >= 0.8:
                    learning.append(memory.summary or memory.content[:100])
        return learning
    
    async def _extract_action_items(self, session: MemorySession) -> List[str]:
        """Extract action items from session"""
        actions = []
        for memory in session.memories:
            # Use DECISION type for action items since there's no task type
            if memory.memory_type == MemoryType.DECISION.value and "TODO" in memory.content.upper():
                actions.append(memory.summary or memory.content[:100])
        return actions
    
    async def _cache_session(self, session: MemorySession):
        """Cache session data"""
        try:
            if redis_client.client:
                key = f"session:{session.id}"
                await redis_client.set(key, session.to_dict(), expiry=3600)  # 1 hour
        except Exception as e:
            logger.warning(f"Failed to cache session: {e}")
    
    async def _trigger_analytics_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger analytics event"""
        # This would integrate with an analytics service
        logger.info(f"Analytics event: {event_type} - {data}")
    
    async def _cleanup_session_temp_data(self, session_id: UUID):
        """Clean up temporary session data"""
        # Clean up any temporary caches or buffers
        if redis_client.client:
            temp_keys = [
                f"session:buffer:{session_id}",
                f"session:pending:{session_id}"
            ]
            for key in temp_keys:
                await redis_client.delete(key)
    
    async def _schedule_session_cleanup(self, session_id: UUID):
        """Schedule automatic session cleanup"""
        # Wait for inactivity timeout (24 hours)
        await asyncio.sleep(24 * 60 * 60)
        
        # Check if session is still active
        session = self.db.query(MemorySession).filter_by(id=session_id).first()
        if session and not session.ended_at:
            logger.info(f"Auto-ending inactive session {session_id}")
            await self.end_session(session_id, reason="timeout")
