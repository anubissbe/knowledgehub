"""Advanced session linking service"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from ..models import MemorySession, Memory
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


class SessionLinkingService:
    """Service for intelligent session linking and chain management"""
    
    def __init__(self):
        self.max_chain_length = 10  # Maximum sessions in a chain
        self.similarity_threshold = 0.7  # Threshold for context similarity
        self.recent_window_minutes = 30  # Window for "recent" sessions
        self.related_window_hours = 24  # Window for finding related sessions
    
    async def find_linkable_sessions(
        self, 
        db: Session, 
        user_id: str,
        project_id: Optional[UUID] = None,
        context_hint: Optional[str] = None,
        exclude_session_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """Find sessions that could be linked to a new session"""
        
        candidates = []
        
        # 1. Find recent active sessions (highest priority)
        recent_sessions = await self._find_recent_active_sessions(
            db, user_id, project_id, exclude_session_id
        )
        
        for session in recent_sessions:
            candidates.append({
                'session': session,
                'link_type': 'recent_active',
                'priority': 1.0,
                'reason': f'Active session from {session.started_at.strftime("%H:%M")}',
                'confidence': 0.9
            })
        
        # 2. Find contextually similar sessions (if context hint provided)
        if context_hint:
            similar_sessions = await self._find_contextually_similar_sessions(
                db, user_id, context_hint, project_id, exclude_session_id
            )
            
            for session, similarity_score in similar_sessions:
                candidates.append({
                    'session': session,
                    'link_type': 'contextual_similarity',
                    'priority': 0.8,
                    'reason': f'Similar context (score: {similarity_score:.2f})',
                    'confidence': similarity_score
                })
        
        # 3. Find project-related sessions
        if project_id:
            project_sessions = await self._find_project_related_sessions(
                db, user_id, project_id, exclude_session_id
            )
            
            for session in project_sessions:
                candidates.append({
                    'session': session,
                    'link_type': 'project_related',
                    'priority': 0.6,
                    'reason': 'Same project context',
                    'confidence': 0.7
                })
        
        # 4. Find sessions with shared entities/topics
        topic_sessions = await self._find_topic_related_sessions(
            db, user_id, context_hint, exclude_session_id
        )
        
        for session, shared_entities in topic_sessions:
            candidates.append({
                'session': session,
                'link_type': 'topic_related',
                'priority': 0.4,
                'reason': f'Shared topics: {", ".join(shared_entities[:3])}',
                'confidence': min(0.8, len(shared_entities) / 5)
            })
        
        # Sort by priority and confidence
        candidates.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)
        
        # Remove duplicates (same session with different link types)
        seen_sessions = set()
        unique_candidates = []
        
        for candidate in candidates:
            session_id = candidate['session'].id
            if session_id not in seen_sessions:
                seen_sessions.add(session_id)
                unique_candidates.append(candidate)
        
        return unique_candidates[:5]  # Return top 5 candidates
    
    async def suggest_session_link(
        self,
        db: Session,
        user_id: str,
        project_id: Optional[UUID] = None,
        context_hint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the best session linking suggestion"""
        
        candidates = await self.find_linkable_sessions(
            db, user_id, project_id, context_hint
        )
        
        if candidates:
            best_candidate = candidates[0]
            
            # Check if the suggested session chain isn't too long
            chain_length = await self._get_chain_length(db, best_candidate['session'].id)
            
            if chain_length >= self.max_chain_length:
                logger.info(f"Session chain too long ({chain_length}), suggesting new chain")
                return None
            
            return best_candidate
        
        return None
    
    async def get_session_chain_analysis(
        self, 
        db: Session, 
        session_id: UUID
    ) -> Dict[str, Any]:
        """Analyze a session chain for insights"""
        
        # Get the full chain
        chain_sessions = await self._get_full_session_chain(db, session_id)
        
        if not chain_sessions:
            return {'error': 'Session not found'}
        
        # Analyze the chain
        analysis = {
            'chain_length': len(chain_sessions),
            'total_duration': self._calculate_total_duration(chain_sessions),
            'total_memories': sum(s.memory_count for s in chain_sessions),
            'unique_topics': await self._extract_chain_topics(chain_sessions),
            'session_summary': [],
            'continuation_patterns': await self._analyze_continuation_patterns(chain_sessions)
        }
        
        # Create session summaries
        for i, session in enumerate(chain_sessions):
            summary = {
                'session_id': session.id,
                'position': i + 1,
                'started_at': session.started_at,
                'ended_at': session.ended_at,
                'duration_minutes': session.duration.total_seconds() / 60 if session.duration else None,
                'memory_count': session.memory_count,
                'tags': session.tags,
                'is_root': session.parent_session_id is None,
                'has_children': i < len(chain_sessions) - 1
            }
            analysis['session_summary'].append(summary)
        
        return analysis
    
    async def merge_session_chains(
        self,
        db: Session,
        primary_chain_session_id: UUID,
        secondary_chain_session_id: UUID,
        merge_reason: str = "Manual merge"
    ) -> Dict[str, Any]:
        """Merge two session chains"""
        
        try:
            # Get both chains
            primary_chain = await self._get_full_session_chain(db, primary_chain_session_id)
            secondary_chain = await self._get_full_session_chain(db, secondary_chain_session_id)
            
            if not primary_chain or not secondary_chain:
                return {'success': False, 'error': 'One or both session chains not found'}
            
            # Find the last session in the primary chain
            last_primary = max(primary_chain, key=lambda s: s.started_at)
            
            # Find the root session in the secondary chain
            secondary_root = min(secondary_chain, key=lambda s: s.started_at)
            
            # Link the secondary root to the last primary session
            secondary_root.parent_session_id = last_primary.id
            secondary_root.add_tag('merged-chain')
            secondary_root.add_metadata('merged_from', str(secondary_chain_session_id))
            secondary_root.add_metadata('merge_reason', merge_reason)
            secondary_root.add_metadata('merged_at', datetime.now(timezone.utc).isoformat())
            
            db.commit()
            
            logger.info(f"Merged session chains: {primary_chain_session_id} + {secondary_chain_session_id}")
            
            return {
                'success': True,
                'merged_chain_root': primary_chain[0].id,
                'total_sessions': len(primary_chain) + len(secondary_chain),
                'merge_point': last_primary.id
            }
            
        except Exception as e:
            logger.error(f"Failed to merge session chains: {e}")
            db.rollback()
            return {'success': False, 'error': str(e)}
    
    # Private helper methods
    
    async def _find_recent_active_sessions(
        self,
        db: Session,
        user_id: str,
        project_id: Optional[UUID],
        exclude_session_id: Optional[UUID]
    ) -> List[MemorySession]:
        """Find recent active sessions"""
        
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.recent_window_minutes)
        
        query = db.query(MemorySession).filter(
            and_(
                MemorySession.user_id == user_id,
                MemorySession.ended_at.is_(None),  # Active sessions
                MemorySession.started_at > cutoff
            )
        )
        
        if project_id:
            query = query.filter(MemorySession.project_id == project_id)
        
        if exclude_session_id:
            query = query.filter(MemorySession.id != exclude_session_id)
        
        return query.order_by(desc(MemorySession.started_at)).limit(3).all()
    
    async def _find_contextually_similar_sessions(
        self,
        db: Session,
        user_id: str,
        context_hint: str,
        project_id: Optional[UUID],
        exclude_session_id: Optional[UUID]
    ) -> List[Tuple[MemorySession, float]]:
        """Find sessions with similar context using simple text matching"""
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.related_window_hours)
        
        # Extract keywords from context hint
        keywords = [word.lower().strip('.,!?()[]{}') 
                   for word in context_hint.split() 
                   if len(word) > 3][:5]  # Top 5 keywords
        
        if not keywords:
            return []
        
        # Find sessions with memories containing these keywords
        query = db.query(MemorySession, func.count(Memory.id).label('matches')).join(
            Memory, MemorySession.id == Memory.session_id
        ).filter(
            and_(
                MemorySession.user_id == user_id,
                MemorySession.started_at > cutoff,
                or_(*[Memory.content.ilike(f'%{keyword}%') for keyword in keywords])
            )
        )
        
        if project_id:
            query = query.filter(MemorySession.project_id == project_id)
        
        if exclude_session_id:
            query = query.filter(MemorySession.id != exclude_session_id)
        
        results = query.group_by(MemorySession.id).order_by(desc('matches')).limit(3).all()
        
        # Calculate similarity scores based on keyword matches
        similar_sessions = []
        for session, match_count in results:
            # Simple similarity score based on keyword matches
            similarity_score = min(1.0, match_count / len(keywords))
            if similarity_score >= self.similarity_threshold:
                similar_sessions.append((session, similarity_score))
        
        return similar_sessions
    
    async def _find_project_related_sessions(
        self,
        db: Session,
        user_id: str,
        project_id: UUID,
        exclude_session_id: Optional[UUID]
    ) -> List[MemorySession]:
        """Find sessions related to the same project"""
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.related_window_hours)
        
        query = db.query(MemorySession).filter(
            and_(
                MemorySession.user_id == user_id,
                MemorySession.project_id == project_id,
                MemorySession.started_at > cutoff
            )
        )
        
        if exclude_session_id:
            query = query.filter(MemorySession.id != exclude_session_id)
        
        return query.order_by(desc(MemorySession.started_at)).limit(3).all()
    
    async def _find_topic_related_sessions(
        self,
        db: Session,
        user_id: str,
        context_hint: Optional[str],
        exclude_session_id: Optional[UUID]
    ) -> List[Tuple[MemorySession, List[str]]]:
        """Find sessions with shared topics/entities"""
        
        if not context_hint:
            return []
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.related_window_hours)
        
        # Extract potential entities from context hint
        potential_entities = [word.lower() for word in context_hint.split() if len(word) > 3]
        
        # Find sessions with memories containing these entities
        sessions_with_entities = []
        
        sessions = db.query(MemorySession).filter(
            and_(
                MemorySession.user_id == user_id,
                MemorySession.started_at > cutoff
            )
        )
        
        if exclude_session_id:
            sessions = sessions.filter(MemorySession.id != exclude_session_id)
        
        for session in sessions.all():
            shared_entities = []
            
            # Check session tags for entity matches
            for tag in session.tags or []:
                if tag.lower() in potential_entities:
                    shared_entities.append(tag)
            
            # Check memory entities
            for memory in session.memories:
                if memory.entities:
                    for entity in memory.entities:
                        if entity.lower() in potential_entities and entity not in shared_entities:
                            shared_entities.append(entity)
            
            if shared_entities:
                sessions_with_entities.append((session, shared_entities))
        
        # Sort by number of shared entities
        sessions_with_entities.sort(key=lambda x: len(x[1]), reverse=True)
        
        return sessions_with_entities[:3]
    
    async def _get_full_session_chain(self, db: Session, session_id: UUID) -> List[MemorySession]:
        """Get all sessions in a complete chain"""
        
        session = db.query(MemorySession).filter_by(id=session_id).first()
        if not session:
            return []
        
        chain = []
        
        # Walk up to root
        current = session
        while current.parent_session_id:
            parent = db.query(MemorySession).filter_by(id=current.parent_session_id).first()
            if parent:
                chain.insert(0, parent)
                current = parent
            else:
                break
        
        # Add current session
        chain.append(session)
        
        # Walk down to all children recursively
        def add_children(parent_session):
            children = db.query(MemorySession).filter_by(
                parent_session_id=parent_session.id
            ).order_by(MemorySession.started_at).all()
            
            for child in children:
                if child not in chain:  # Avoid cycles
                    chain.append(child)
                    add_children(child)
        
        add_children(session)
        
        return chain
    
    async def _get_chain_length(self, db: Session, session_id: UUID) -> int:
        """Get the length of the session chain starting from root"""
        chain = await self._get_full_session_chain(db, session_id)
        return len(chain)
    
    def _calculate_total_duration(self, sessions: List[MemorySession]) -> timedelta:
        """Calculate total duration of all sessions in chain"""
        total_seconds = 0
        
        for session in sessions:
            if session.duration:
                total_seconds += session.duration.total_seconds()
        
        return timedelta(seconds=total_seconds)
    
    async def _extract_chain_topics(self, sessions: List[MemorySession]) -> List[str]:
        """Extract unique topics/entities from all sessions in chain"""
        topics = set()
        
        for session in sessions:
            # Add session tags
            if session.tags:
                topics.update(session.tags)
            
            # Add memory entities
            for memory in session.memories:
                if memory.entities:
                    topics.update(memory.entities)
        
        return list(topics)
    
    async def _analyze_continuation_patterns(self, sessions: List[MemorySession]) -> Dict[str, Any]:
        """Analyze patterns in session continuations"""
        
        if len(sessions) < 2:
            return {}
        
        gaps = []
        session_durations = []
        
        for i in range(1, len(sessions)):
            prev_session = sessions[i-1]
            current_session = sessions[i]
            
            if prev_session.ended_at and current_session.started_at:
                gap = current_session.started_at - prev_session.ended_at
                gaps.append(gap.total_seconds() / 60)  # Gap in minutes
            
            if current_session.duration:
                session_durations.append(current_session.duration.total_seconds() / 60)
        
        patterns = {}
        
        if gaps:
            patterns['average_gap_minutes'] = sum(gaps) / len(gaps)
            patterns['max_gap_minutes'] = max(gaps)
            patterns['min_gap_minutes'] = min(gaps)
        
        if session_durations:
            patterns['average_session_duration_minutes'] = sum(session_durations) / len(session_durations)
            patterns['total_active_time_minutes'] = sum(session_durations)
        
        return patterns


# Global instance
session_linking_service = SessionLinkingService()