"""Learning Session Persistence Service

Manages the persistence and continuity of learning sessions across conversation sessions,
enabling true cross-session learning capabilities.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Set
from uuid import UUID, uuid4

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, and_, or_, desc, func

from ..models.learning_session import LearningSession, LearningSessionType, LearningSessionStatus
from ..models.knowledge_transfer import KnowledgeTransfer, TransferType, TransferStatus
from ..models.user_learning_profile import UserLearningProfile
from ..models.pattern_evolution import PatternEvolution
from ..models.learning_pattern import LearningPattern

logger = logging.getLogger(__name__)


class LearningSessionPersistence:
    """Service for managing learning session persistence across conversation sessions"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_learning_session(
        self,
        user_id: str,
        session_type: str,
        session_name: Optional[str] = None,
        conversation_session_id: Optional[UUID] = None,
        learning_objectives: Optional[Dict[str, Any]] = None,
        learning_context: Optional[Dict[str, Any]] = None,
        parent_session_id: Optional[UUID] = None
    ) -> LearningSession:
        """Create a new learning session"""
        
        session = LearningSession(
            session_type=session_type,
            session_name=session_name,
            user_id=user_id,
            conversation_session_id=conversation_session_id,
            learning_objectives=learning_objectives or {},
            learning_context=learning_context or {},
            parent_learning_session_id=parent_session_id,
            status=LearningSessionStatus.ACTIVE.value
        )
        
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        
        logger.info(f"Created learning session {session.id} for user {user_id}")
        return session
    
    async def get_active_learning_sessions(
        self,
        user_id: str,
        session_types: Optional[List[str]] = None
    ) -> List[LearningSession]:
        """Get all active learning sessions for a user"""
        
        query = select(LearningSession).where(
            and_(
                LearningSession.user_id == user_id,
                LearningSession.status == LearningSessionStatus.ACTIVE.value
            )
        ).order_by(desc(LearningSession.last_activity_at))
        
        if session_types:
            query = query.where(LearningSession.session_type.in_(session_types))
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_learning_session_by_id(self, session_id: UUID) -> Optional[LearningSession]:
        """Get a learning session by ID with all relationships loaded"""
        
        query = select(LearningSession).where(
            LearningSession.id == session_id
        ).options(
            selectinload(LearningSession.incoming_transfers),
            selectinload(LearningSession.outgoing_transfers)
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def continue_learning_session(
        self,
        session_id: UUID,
        new_conversation_session_id: Optional[UUID] = None
    ) -> Optional[LearningSession]:
        """Continue an existing learning session in a new conversation"""
        
        session = await self.get_learning_session_by_id(session_id)
        if not session:
            return None
        
        # Update session to active if paused
        if session.status == LearningSessionStatus.PAUSED.value:
            session.resume_session()
        
        # Update conversation session if provided
        if new_conversation_session_id:
            session.conversation_session_id = new_conversation_session_id
        
        session.update_activity()
        await self.db.commit()
        
        logger.info(f"Continued learning session {session_id}")
        return session
    
    async def pause_learning_session(self, session_id: UUID) -> bool:
        """Pause a learning session"""
        
        session = await self.get_learning_session_by_id(session_id)
        if not session:
            return False
        
        session.pause_session()
        await self.db.commit()
        
        logger.info(f"Paused learning session {session_id}")
        return True
    
    async def complete_learning_session(
        self,
        session_id: UUID,
        success_rate: Optional[float] = None,
        final_summary: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Complete a learning session"""
        
        session = await self.get_learning_session_by_id(session_id)
        if not session:
            return False
        
        session.complete_session(success_rate)
        
        if final_summary:
            if session.session_metadata is None:
                session.session_metadata = {}
            session.session_metadata['final_summary'] = final_summary
        
        await self.db.commit()
        
        # Update user learning profile
        await self._update_user_profile_from_session(session)
        
        logger.info(f"Completed learning session {session_id}")
        return True
    
    async def find_related_learning_sessions(
        self,
        user_id: str,
        session_type: Optional[str] = None,
        learning_context: Optional[Dict[str, Any]] = None,
        max_age_days: int = 30,
        max_results: int = 10
    ) -> List[LearningSession]:
        """Find related learning sessions for knowledge transfer"""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        query = select(LearningSession).where(
            and_(
                LearningSession.user_id == user_id,
                LearningSession.last_activity_at >= cutoff_date,
                LearningSession.status.in_([
                    LearningSessionStatus.COMPLETED.value,
                    LearningSessionStatus.ACTIVE.value
                ])
            )
        ).order_by(desc(LearningSession.learning_effectiveness))
        
        if session_type:
            query = query.where(LearningSession.session_type == session_type)
        
        query = query.limit(max_results)
        
        result = await self.db.execute(query)
        sessions = result.scalars().all()
        
        # If learning context provided, score sessions by relevance
        if learning_context and sessions:
            scored_sessions = []
            for session in sessions:
                relevance_score = self._calculate_context_relevance(
                    session.learning_context, learning_context
                )
                scored_sessions.append((session, relevance_score))
            
            # Sort by relevance score descending
            scored_sessions.sort(key=lambda x: x[1], reverse=True)
            sessions = [session for session, _ in scored_sessions]
        
        return sessions
    
    async def transfer_knowledge_between_sessions(
        self,
        source_session_id: UUID,
        destination_session_id: UUID,
        transfer_type: str,
        transfer_data: Dict[str, Any],
        transfer_name: Optional[str] = None
    ) -> Optional[KnowledgeTransfer]:
        """Create a knowledge transfer between learning sessions"""
        
        # Validate sessions exist
        source_session = await self.get_learning_session_by_id(source_session_id)
        destination_session = await self.get_learning_session_by_id(destination_session_id)
        
        if not source_session or not destination_session:
            logger.error(f"Invalid session IDs for knowledge transfer")
            return None
        
        # Create knowledge transfer
        transfer = KnowledgeTransfer(
            transfer_type=transfer_type,
            transfer_name=transfer_name,
            source_learning_session_id=source_session_id,
            destination_learning_session_id=destination_session_id,
            transferred_data=transfer_data,
            user_id=destination_session.user_id,
            knowledge_units_transferred=len(transfer_data.get('patterns', [])),
            patterns_transferred=len(transfer_data.get('patterns', []))
        )
        
        self.db.add(transfer)
        await self.db.commit()
        await self.db.refresh(transfer)
        
        # Update destination session knowledge count
        destination_session.transferred_knowledge_count += transfer.knowledge_units_transferred
        await self.db.commit()
        
        logger.info(f"Created knowledge transfer {transfer.id} from {source_session_id} to {destination_session_id}")
        return transfer
    
    async def get_session_learning_history(
        self,
        user_id: str,
        max_sessions: int = 20,
        session_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get learning history for a user across sessions"""
        
        query = select(LearningSession).where(
            LearningSession.user_id == user_id
        ).order_by(desc(LearningSession.started_at))
        
        if session_types:
            query = query.where(LearningSession.session_type.in_(session_types))
        
        query = query.limit(max_sessions)
        
        result = await self.db.execute(query)
        sessions = result.scalars().all()
        
        history = []
        for session in sessions:
            session_summary = session.get_learning_summary()
            
            # Add knowledge transfer information
            transfers_query = select(KnowledgeTransfer).where(
                or_(
                    KnowledgeTransfer.source_learning_session_id == session.id,
                    KnowledgeTransfer.destination_learning_session_id == session.id
                )
            )
            transfers_result = await self.db.execute(transfers_query)
            transfers = transfers_result.scalars().all()
            
            session_summary['knowledge_transfers'] = {
                'incoming': len([t for t in transfers if t.destination_learning_session_id == session.id]),
                'outgoing': len([t for t in transfers if t.source_learning_session_id == session.id])
            }
            
            history.append(session_summary)
        
        return history
    
    async def get_cross_session_patterns(
        self,
        user_id: str,
        pattern_types: Optional[List[str]] = None,
        min_sessions: int = 2
    ) -> List[Dict[str, Any]]:
        """Get patterns that appear across multiple learning sessions"""
        
        # Get patterns that have been reinforced across sessions
        query = select(LearningPattern).join(
            PatternEvolution, LearningPattern.id == PatternEvolution.pattern_id
        ).where(
            and_(
                PatternEvolution.user_id == user_id,
                LearningPattern.usage_count >= min_sessions
            )
        ).distinct()
        
        if pattern_types:
            query = query.where(LearningPattern.pattern_type.in_(pattern_types))
        
        result = await self.db.execute(query)
        patterns = result.scalars().all()
        
        cross_session_patterns = []
        for pattern in patterns:
            # Get evolution history across sessions
            evolutions_query = select(PatternEvolution).where(
                PatternEvolution.pattern_id == pattern.id
            ).order_by(PatternEvolution.evolved_at)
            
            evolutions_result = await self.db.execute(evolutions_query)
            evolutions = evolutions_result.scalars().all()
            
            # Count unique sessions
            unique_sessions = set(
                e.learning_session_id for e in evolutions 
                if e.learning_session_id is not None
            )
            
            if len(unique_sessions) >= min_sessions:
                pattern_info = {
                    'pattern_id': str(pattern.id),
                    'pattern_type': pattern.pattern_type,
                    'confidence_score': pattern.confidence_score,
                    'usage_count': pattern.usage_count,
                    'sessions_count': len(unique_sessions),
                    'evolution_count': len(evolutions),
                    'pattern_data': pattern.pattern_data
                }
                cross_session_patterns.append(pattern_info)
        
        return cross_session_patterns
    
    async def cleanup_old_sessions(
        self,
        max_age_days: int = 90,
        keep_successful_sessions: bool = True
    ) -> int:
        """Clean up old learning sessions"""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        query = select(LearningSession).where(
            and_(
                LearningSession.last_activity_at < cutoff_date,
                LearningSession.status.in_([
                    LearningSessionStatus.COMPLETED.value,
                    LearningSessionStatus.FAILED.value
                ])
            )
        )
        
        if keep_successful_sessions:
            # Only delete sessions with low effectiveness
            query = query.where(
                or_(
                    LearningSession.learning_effectiveness < 0.3,
                    LearningSession.learning_effectiveness.is_(None)
                )
            )
        
        result = await self.db.execute(query)
        sessions_to_delete = result.scalars().all()
        
        deleted_count = 0
        for session in sessions_to_delete:
            await self.db.delete(session)
            deleted_count += 1
        
        await self.db.commit()
        
        logger.info(f"Cleaned up {deleted_count} old learning sessions")
        return deleted_count
    
    def _calculate_context_relevance(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Calculate relevance score between two learning contexts"""
        
        if not context1 or not context2:
            return 0.0
        
        score = 0.0
        max_score = 0.0
        
        # Compare common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        for key in common_keys:
            max_score += 1.0
            if context1[key] == context2[key]:
                score += 1.0
            elif isinstance(context1[key], str) and isinstance(context2[key], str):
                # Partial string matching
                similarity = len(set(context1[key].lower().split()) & 
                               set(context2[key].lower().split()))
                if similarity > 0:
                    score += 0.5
        
        # Factor in total key overlap
        all_keys = set(context1.keys()) | set(context2.keys())
        if all_keys:
            key_overlap = len(common_keys) / len(all_keys)
            score += key_overlap
            max_score += 1.0
        
        return score / max_score if max_score > 0 else 0.0
    
    async def _update_user_profile_from_session(self, session: LearningSession):
        """Update user learning profile based on completed session"""
        
        try:
            # Get or create user learning profile
            profile_query = select(UserLearningProfile).where(
                UserLearningProfile.user_id == session.user_id
            )
            profile_result = await self.db.execute(profile_query)
            profile = profile_result.scalar_one_or_none()
            
            if not profile:
                profile = UserLearningProfile(user_id=session.user_id)
                self.db.add(profile)
            
            # Update session counts
            profile.total_learning_sessions += 1
            profile.total_patterns_learned += session.patterns_learned
            
            # Update effectiveness
            if session.learning_effectiveness is not None:
                profile.update_learning_effectiveness(session.learning_effectiveness)
            
            # Update last session reference
            profile.last_updated_by_session = session.id
            profile.update_interaction_timestamp()
            
            # Recalculate profile completeness
            profile.calculate_profile_completeness()
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating user profile from session {session.id}: {e}")
            await self.db.rollback()