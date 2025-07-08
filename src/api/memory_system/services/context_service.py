"""Context injection service for Claude-Code integration"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, text

from ..models import Memory, MemorySession
from ..api.context_schemas import (
    ContextRequest, ContextResponse, ContextMemory, ContextSection,
    ContextTypeEnum, ContextRelevanceEnum, ContextStats
)
from .embedding_service import memory_embedding_service

logger = logging.getLogger(__name__)


class ContextService:
    """Service for retrieving and formatting context for Claude-Code"""
    
    def __init__(self):
        self.max_context_tokens = 8000  # Conservative limit for context
        self.token_per_char_ratio = 0.25  # Rough estimate: 4 chars per token
    
    async def retrieve_context(
        self, 
        db: Session, 
        request: ContextRequest
    ) -> ContextResponse:
        """Retrieve and format context for Claude-Code"""
        start_time = time.time()
        
        try:
            # Get memories by context type
            context_memories = await self._get_context_memories(db, request)
            
            # Apply relevance scoring and filtering
            scored_memories = await self._score_and_filter_memories(
                db, context_memories, request
            )
            
            # Organize into sections
            sections = self._organize_into_sections(scored_memories, request)
            
            # Apply token limits and optimize
            optimized_sections = self._optimize_for_tokens(sections, request.max_tokens)
            
            # Format for LLM consumption
            formatted_context, context_summary = self._format_for_llm(
                optimized_sections, request
            )
            
            # Calculate metrics
            total_memories = sum(len(section.memories) for section in optimized_sections)
            total_tokens = sum(section.token_count for section in optimized_sections)
            max_relevance = max(
                (section.relevance_score for section in optimized_sections), 
                default=0.0
            )
            
            retrieval_time_ms = (time.time() - start_time) * 1000
            
            return ContextResponse(
                user_id=request.user_id,
                session_id=request.session_id,
                query=request.query,
                sections=optimized_sections,
                total_memories=total_memories,
                total_tokens=total_tokens,
                max_relevance=max_relevance,
                retrieval_time_ms=retrieval_time_ms,
                formatted_context=formatted_context,
                context_summary=context_summary
            )
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            raise
    
    async def _get_context_memories(
        self, 
        db: Session, 
        request: ContextRequest
    ) -> Dict[ContextTypeEnum, List[Memory]]:
        """Retrieve memories by context type"""
        context_memories = {}
        
        for context_type in request.context_types:
            memories = []
            
            if context_type == ContextTypeEnum.recent:
                memories = await self._get_recent_memories(db, request)
            elif context_type == ContextTypeEnum.similar:
                memories = await self._get_similar_memories(db, request)
            elif context_type == ContextTypeEnum.entities:
                memories = await self._get_entity_memories(db, request)
            elif context_type == ContextTypeEnum.decisions:
                memories = await self._get_decision_memories(db, request)
            elif context_type == ContextTypeEnum.errors:
                memories = await self._get_error_memories(db, request)
            elif context_type == ContextTypeEnum.patterns:
                memories = await self._get_pattern_memories(db, request)
            elif context_type == ContextTypeEnum.preferences:
                memories = await self._get_preference_memories(db, request)
            
            context_memories[context_type] = memories
        
        return context_memories
    
    async def _get_recent_memories(self, db: Session, request: ContextRequest) -> List[Memory]:
        """Get recent memories from current or related sessions"""
        query = db.query(Memory).join(MemorySession)
        
        # Filter by user
        query = query.filter(MemorySession.user_id == request.user_id)
        
        # Filter by session if specified
        if request.session_id:
            query = query.filter(Memory.session_id == request.session_id)
        
        # Apply time window
        if request.time_window_hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=request.time_window_hours)
            query = query.filter(Memory.created_at >= cutoff)
        
        # Apply memory type filter
        if request.memory_types:
            query = query.filter(Memory.memory_type.in_(request.memory_types))
        
        # Order by creation time (most recent first)
        memories = query.order_by(desc(Memory.created_at)).limit(10).all()
        
        logger.info(f"Retrieved {len(memories)} recent memories")
        return memories
    
    async def _get_similar_memories(self, db: Session, request: ContextRequest) -> List[Memory]:
        """Get semantically similar memories"""
        if not request.query:
            return []
        
        try:
            # Generate embedding for query
            embeddings_client = memory_embedding_service.embeddings_client
            query_embedding = await embeddings_client.generate_embedding(
                request.query, normalize=True
            )
            
            if not query_embedding:
                logger.warning("Failed to generate query embedding for similarity search")
                return []
            
            # Use vector similarity search
            similar_memories = await memory_embedding_service.find_similar_memories(
                db=db,
                query_embedding=query_embedding,
                limit=10,
                min_similarity=request.min_relevance,
                user_id=request.user_id,
                session_id=None  # Search across all sessions
            )
            
            memories = [memory for memory, _ in similar_memories]
            logger.info(f"Retrieved {len(memories)} similar memories")
            return memories
            
        except Exception as e:
            logger.error(f"Similar memory search failed: {e}")
            return []
    
    async def _get_entity_memories(self, db: Session, request: ContextRequest) -> List[Memory]:
        """Get memories related to entities mentioned in query"""
        if not request.query:
            return []
        
        # Extract potential entities from query (simple word extraction)
        # In production, this could use NLP for better entity extraction
        query_words = [word.lower().strip('.,!?()[]{}') 
                      for word in request.query.split() 
                      if len(word) > 3]
        
        if not query_words:
            return []
        
        query = db.query(Memory).join(MemorySession)
        query = query.filter(MemorySession.user_id == request.user_id)
        
        # Find memories that contain any of the query entities
        entity_filters = []
        for word in query_words[:5]:  # Limit to 5 words to avoid too complex query
            entity_filters.append(Memory.entities.any(text(f"'{word}'::text")))
        
        if entity_filters:
            query = query.filter(or_(*entity_filters))
        
        memories = query.order_by(desc(Memory.importance)).limit(5).all()
        logger.info(f"Retrieved {len(memories)} entity-related memories")
        return memories
    
    async def _get_decision_memories(self, db: Session, request: ContextRequest) -> List[Memory]:
        """Get important decision memories"""
        query = db.query(Memory).join(MemorySession)
        query = query.filter(
            MemorySession.user_id == request.user_id,
            Memory.memory_type == 'decision',
            Memory.importance >= 0.7  # High importance decisions only
        )
        
        memories = query.order_by(desc(Memory.importance)).limit(5).all()
        logger.info(f"Retrieved {len(memories)} decision memories")
        return memories
    
    async def _get_error_memories(self, db: Session, request: ContextRequest) -> List[Memory]:
        """Get error and solution memories"""
        query = db.query(Memory).join(MemorySession)
        query = query.filter(
            MemorySession.user_id == request.user_id,
            Memory.memory_type == 'error'
        )
        
        # Apply time window for errors (more recent errors are more relevant)
        if request.time_window_hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=request.time_window_hours * 2)
            query = query.filter(Memory.created_at >= cutoff)
        
        memories = query.order_by(desc(Memory.created_at)).limit(3).all()
        logger.info(f"Retrieved {len(memories)} error memories")
        return memories
    
    async def _get_pattern_memories(self, db: Session, request: ContextRequest) -> List[Memory]:
        """Get pattern recognition memories"""
        query = db.query(Memory).join(MemorySession)
        query = query.filter(
            MemorySession.user_id == request.user_id,
            Memory.memory_type == 'pattern',
            Memory.importance >= 0.6
        )
        
        memories = query.order_by(desc(Memory.importance)).limit(3).all()
        logger.info(f"Retrieved {len(memories)} pattern memories")
        return memories
    
    async def _get_preference_memories(self, db: Session, request: ContextRequest) -> List[Memory]:
        """Get user preference memories"""
        query = db.query(Memory).join(MemorySession)
        query = query.filter(
            MemorySession.user_id == request.user_id,
            Memory.memory_type == 'preference'
        )
        
        memories = query.order_by(desc(Memory.importance)).limit(5).all()
        logger.info(f"Retrieved {len(memories)} preference memories")
        return memories
    
    async def _score_and_filter_memories(
        self, 
        db: Session,
        context_memories: Dict[ContextTypeEnum, List[Memory]],
        request: ContextRequest
    ) -> List[Tuple[Memory, ContextTypeEnum, float, str]]:
        """Score memories for relevance and filter"""
        scored_memories = []
        
        for context_type, memories in context_memories.items():
            for memory in memories:
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(
                    memory, context_type, request
                )
                
                if relevance_score >= request.min_relevance:
                    # Generate relevance reason
                    reason = self._generate_relevance_reason(
                        memory, context_type, relevance_score
                    )
                    
                    scored_memories.append((memory, context_type, relevance_score, reason))
        
        # Sort by relevance score (highest first)
        scored_memories.sort(key=lambda x: x[2], reverse=True)
        
        # Limit to max_memories
        return scored_memories[:request.max_memories]
    
    def _calculate_relevance_score(
        self, 
        memory: Memory, 
        context_type: ContextTypeEnum,
        request: ContextRequest
    ) -> float:
        """Calculate relevance score for a memory"""
        base_score = memory.importance * memory.confidence
        
        # Boost based on context type
        type_multipliers = {
            ContextTypeEnum.recent: 1.2,
            ContextTypeEnum.similar: 1.0,
            ContextTypeEnum.decisions: 1.1,
            ContextTypeEnum.errors: 1.0,
            ContextTypeEnum.patterns: 1.1,
            ContextTypeEnum.preferences: 0.9,
            ContextTypeEnum.entities: 0.8
        }
        
        score = base_score * type_multipliers.get(context_type, 1.0)
        
        # Apply recency boost for recent memories
        if hasattr(memory, 'age_days'):
            if memory.age_days < 1:  # Less than 1 day old
                score *= 1.2
            elif memory.age_days < 7:  # Less than 1 week old
                score *= 1.1
        
        # Boost high-confidence memories
        if memory.confidence >= 0.9:
            score *= 1.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_relevance_reason(
        self, 
        memory: Memory, 
        context_type: ContextTypeEnum,
        relevance_score: float
    ) -> str:
        """Generate human-readable reason for relevance"""
        reasons = []
        
        if context_type == ContextTypeEnum.recent:
            reasons.append("recent activity")
        elif context_type == ContextTypeEnum.similar:
            reasons.append("semantic similarity")
        elif context_type == ContextTypeEnum.decisions:
            reasons.append("important decision")
        elif context_type == ContextTypeEnum.errors:
            reasons.append("error/solution context")
        elif context_type == ContextTypeEnum.patterns:
            reasons.append("recognized pattern")
        elif context_type == ContextTypeEnum.preferences:
            reasons.append("user preference")
        elif context_type == ContextTypeEnum.entities:
            reasons.append("related entities")
        
        if memory.importance >= 0.8:
            reasons.append("high importance")
        
        if memory.confidence >= 0.9:
            reasons.append("high confidence")
        
        if hasattr(memory, 'age_days') and memory.age_days < 1:
            reasons.append("very recent")
        
        return ", ".join(reasons) if reasons else "contextually relevant"
    
    def _organize_into_sections(
        self, 
        scored_memories: List[Tuple[Memory, ContextTypeEnum, float, str]],
        request: ContextRequest
    ) -> List[ContextSection]:
        """Organize memories into logical sections"""
        sections_dict = {}
        
        for memory, context_type, relevance_score, reason in scored_memories:
            if context_type not in sections_dict:
                sections_dict[context_type] = []
            
            context_memory = ContextMemory(
                id=memory.id,
                content=memory.content,
                summary=memory.summary,
                memory_type=memory.memory_type,
                importance=memory.importance,
                confidence=memory.confidence,
                entities=memory.entities or [],
                relevance_score=relevance_score,
                relevance_reason=reason,
                context_type=context_type,
                created_at=memory.created_at,
                session_id=memory.session_id,
                age_days=getattr(memory, 'age_days', 0)
            )
            
            sections_dict[context_type].append(context_memory)
        
        # Convert to sections with metadata
        sections = []
        section_titles = {
            ContextTypeEnum.recent: "Recent Activity",
            ContextTypeEnum.similar: "Related Context", 
            ContextTypeEnum.decisions: "Key Decisions",
            ContextTypeEnum.errors: "Errors & Solutions",
            ContextTypeEnum.patterns: "Patterns & Insights",
            ContextTypeEnum.preferences: "User Preferences",
            ContextTypeEnum.entities: "Related Information"
        }
        
        for context_type, memories in sections_dict.items():
            if memories:
                # Calculate section token count
                token_count = sum(
                    self._estimate_memory_tokens(mem) for mem in memories
                )
                
                # Calculate average relevance
                avg_relevance = sum(mem.relevance_score for mem in memories) / len(memories)
                
                section = ContextSection(
                    title=section_titles.get(context_type, context_type.value.title()),
                    context_type=context_type,
                    memories=memories,
                    token_count=token_count,
                    relevance_score=avg_relevance
                )
                
                sections.append(section)
        
        # Sort sections by relevance
        sections.sort(key=lambda s: s.relevance_score, reverse=True)
        
        return sections
    
    def _optimize_for_tokens(
        self, 
        sections: List[ContextSection], 
        max_tokens: Optional[int]
    ) -> List[ContextSection]:
        """Optimize sections to fit within token limits"""
        if not max_tokens:
            return sections
        
        optimized_sections = []
        total_tokens = 0
        
        for section in sections:
            if total_tokens + section.token_count <= max_tokens:
                # Section fits completely
                optimized_sections.append(section)
                total_tokens += section.token_count
            else:
                # Need to trim section
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only include if we have meaningful space
                    trimmed_memories = []
                    section_tokens = 0
                    
                    for memory in section.memories:
                        memory_tokens = self._estimate_memory_tokens(memory)
                        if section_tokens + memory_tokens <= remaining_tokens:
                            trimmed_memories.append(memory)
                            section_tokens += memory_tokens
                        else:
                            break
                    
                    if trimmed_memories:
                        trimmed_section = ContextSection(
                            title=section.title,
                            context_type=section.context_type,
                            memories=trimmed_memories,
                            token_count=section_tokens,
                            relevance_score=section.relevance_score
                        )
                        optimized_sections.append(trimmed_section)
                        total_tokens += section_tokens
                
                break  # No more room for additional sections
        
        return optimized_sections
    
    def _estimate_memory_tokens(self, memory: ContextMemory) -> int:
        """Estimate token count for a memory"""
        text = memory.content
        if memory.summary:
            text = memory.summary  # Use summary if available (shorter)
        
        # Rough estimation: 4 characters per token
        return int(len(text) * self.token_per_char_ratio) + 10  # +10 for formatting
    
    def _format_for_llm(
        self, 
        sections: List[ContextSection], 
        request: ContextRequest
    ) -> Tuple[str, str]:
        """Format context for LLM consumption"""
        if not sections:
            return "", "No relevant context found"
        
        # Build formatted context
        formatted_parts = ["# Memory Context\n"]
        
        if request.query:
            formatted_parts.append(f"Query: {request.query}\n")
        
        summary_parts = []
        
        for section in sections:
            formatted_parts.append(f"\n## {section.title}\n")
            summary_parts.append(f"{len(section.memories)} {section.title.lower()}")
            
            for i, memory in enumerate(section.memories, 1):
                # Use summary if available, otherwise truncate content
                content = memory.summary if memory.summary else memory.content
                if len(content) > 200:
                    content = content[:197] + "..."
                
                formatted_parts.append(
                    f"{i}. **{memory.memory_type.title()}** "
                    f"(relevance: {memory.relevance_score:.2f}): {content}\n"
                    f"   *{memory.relevance_reason}*\n"
                )
        
        formatted_context = "".join(formatted_parts)
        
        # Create summary
        total_memories = sum(len(section.memories) for section in sections)
        context_summary = (
            f"Retrieved {total_memories} relevant memories: " + 
            ", ".join(summary_parts)
        )
        
        return formatted_context, context_summary


# Global instance
context_service = ContextService()


async def build_context_for_request(
    session_id: str, 
    request_path: str, 
    request_method: str,
    max_tokens: int = 4000,
    db: Session = None
) -> Dict[str, Any]:
    """
    Build context for a request using the global context service.
    
    This is a convenience function for middleware and other components
    that need to inject context without managing the service directly.
    """
    try:
        if not db:
            # Import here to avoid circular imports
            from ...models import get_db
            db = next(get_db())
        
        # Build context request
        from ..api.context_schemas import ContextRequest, ContextTypeEnum
        
        context_request = ContextRequest(
            session_id=UUID(session_id),
            query=f"{request_method} {request_path}",
            user_id="claude-code",
            max_tokens=max_tokens,
            context_types=[
                ContextTypeEnum.recent,
                ContextTypeEnum.similar,
                ContextTypeEnum.preferences,
                ContextTypeEnum.decisions,
                ContextTypeEnum.errors
            ]
        )
        
        # Get context from service
        context_response = await context_service.retrieve_context(db, context_request)
        
        # Convert to simple dict format for middleware use
        return {
            'session_id': session_id,
            'memories': [
                {
                    'id': str(mem.id),
                    'content': mem.content,
                    'memory_type': mem.memory_type,
                    'importance': mem.importance,
                    'created_at': mem.created_at.isoformat() if mem.created_at else None
                }
                for section in context_response.sections
                for mem in section.memories
            ],
            'sections': [
                {
                    'type': section.context_type.value,
                    'title': section.title,
                    'memory_count': len(section.memories),
                    'relevance': section.relevance_score
                }
                for section in context_response.sections
            ],
            'stats': {
                'total_memories': context_response.total_memories,
                'total_tokens': context_response.total_tokens,
                'max_relevance': context_response.max_relevance
            },
            'request_info': {
                'path': request_path,
                'method': request_method,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to build context for request {request_method} {request_path}: {e}")
        return {
            'session_id': session_id,
            'memories': [],
            'sections': [],
            'stats': {'total_memories': 0, 'selected_memories': 0, 'token_estimate': 0},
            'request_info': {
                'path': request_path,
                'method': request_method,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            'error': str(e)
        }