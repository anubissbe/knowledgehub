"""Memory Search Service integrating with main search infrastructure

This service bridges the memory system with the existing search service,
providing unified search capabilities across memories with vector similarity,
keyword matching, and hybrid search approaches.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from uuid import UUID
import asyncio

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from ...services.search_service import SearchService
from ...services.vector_store import vector_store
from ...services.embeddings_client import get_embeddings_client
from ...services.cache import redis_client
from ...schemas.search import SearchQuery, SearchType, SearchResponse
from ..models import Memory, MemorySession, MemoryType
from ..api.schemas import MemorySearchRequest, MemorySearchResponse, MemoryResponse
from .embedding_service import memory_embedding_service

logger = logging.getLogger(__name__)


class MemorySearchService(SearchService):
    """Extended search service for memory-specific search operations
    
    This service extends the base SearchService to provide memory-specific
    search capabilities including:
    - Vector similarity search on memory embeddings
    - Keyword search on memory content and summaries
    - Session-aware search with context
    - Memory type filtering
    - Importance-based ranking
    """
    
    def __init__(self):
        """Initialize memory search service"""
        super().__init__()
        self.memory_collection_name = "memory_embeddings"
    
    async def search_memories(
        self, 
        db: Session, 
        request: MemorySearchRequest
    ) -> MemorySearchResponse:
        """Search memories using hybrid search approach
        
        Args:
            db: Database session
            request: Memory search request with filters
            
        Returns:
            MemorySearchResponse with results and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        # Check cache first
        cache_key = self._get_memory_cache_key(request)
        if redis_client.client:
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for memory search: {cache_key}")
                return MemorySearchResponse(**cached_result)
        
        # Determine search type based on query
        if request.query and request.use_vector_search:
            # Use hybrid search for text queries with vector search enabled
            results = await self._hybrid_memory_search(db, request)
        elif request.query and not request.use_vector_search:
            # Use keyword search only
            results = await self._keyword_memory_search(db, request)
        else:
            # No query text - use filter-based search
            results = await self._filter_memory_search(db, request)
        
        # Format response
        response = MemorySearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            limit=request.limit,
            offset=request.offset,
            search_time_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        )
        
        # Cache the result
        if redis_client.client and len(results) > 0:
            await redis_client.set(cache_key, response.dict(), expiry=300)  # 5 min cache
        
        return response
    
    async def _hybrid_memory_search(
        self, 
        db: Session, 
        request: MemorySearchRequest
    ) -> List[MemoryResponse]:
        """Execute hybrid search combining vector and keyword approaches"""
        # Run both searches in parallel
        vector_task = asyncio.create_task(self._vector_memory_search(db, request))
        keyword_task = asyncio.create_task(self._keyword_memory_search(db, request))
        
        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task
        )
        
        # Merge and deduplicate results
        merged = self._merge_memory_results(vector_results, keyword_results)
        
        # Apply limit and offset
        start_idx = request.offset
        end_idx = start_idx + request.limit
        
        return merged[start_idx:end_idx]
    
    async def _vector_memory_search(
        self, 
        db: Session, 
        request: MemorySearchRequest
    ) -> List[MemoryResponse]:
        """Execute vector similarity search on memories"""
        if not request.query or not vector_store.client:
            return []
        
        try:
            # Generate embedding for query
            embeddings_client = get_embeddings_client()
            query_embedding = await embeddings_client.generate_embedding(
                request.query, 
                normalize=True
            )
            
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Build filters for vector search
            vector_filters = {}
            if request.user_id:
                vector_filters["user_id"] = request.user_id
            if request.project_id:
                vector_filters["project_id"] = str(request.project_id)
            if request.memory_types:
                vector_filters["memory_type"] = [t.value for t in request.memory_types]
            
            # Search in vector store
            similar_results = await memory_embedding_service.find_similar_memories(
                db=db,
                query_embedding=query_embedding,
                limit=request.limit * 2,  # Get more for merging
                min_similarity=0.7,
                user_id=request.user_id,
                memory_types=[t.value for t in request.memory_types] if request.memory_types else None
            )
            
            # Convert to MemoryResponse objects
            results = []
            for memory, similarity_score in similar_results:
                if request.min_importance and memory.importance < request.min_importance:
                    continue
                
                response = self._memory_to_response(memory)
                response.relevance_score = similarity_score
                results.append(response)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector memory search failed: {e}")
            return []
    
    async def _keyword_memory_search(
        self, 
        db: Session, 
        request: MemorySearchRequest
    ) -> List[MemoryResponse]:
        """Execute keyword search on memory content"""
        query = db.query(Memory).join(MemorySession)
        
        # Apply user/project filters
        if request.user_id:
            query = query.filter(MemorySession.user_id == request.user_id)
        if request.project_id:
            query = query.filter(MemorySession.project_id == request.project_id)
        
        # Apply memory type filters
        if request.memory_types:
            type_values = [t.value for t in request.memory_types]
            query = query.filter(Memory.memory_type.in_(type_values))
        
        # Apply importance filter
        if request.min_importance:
            query = query.filter(Memory.importance >= request.min_importance)
        
        # Apply text search
        if request.query:
            # Use PostgreSQL full-text search if available
            if db.bind.dialect.name == 'postgresql':
                search_query = func.plainto_tsquery('english', request.query)
                query = query.filter(
                    or_(
                        func.to_tsvector('english', Memory.content).op('@@')(search_query),
                        func.to_tsvector('english', Memory.summary).op('@@')(search_query)
                    )
                )
            else:
                # Fallback to ILIKE for other databases
                search_term = f"%{request.query}%"
                query = query.filter(
                    or_(
                        Memory.content.ilike(search_term),
                        Memory.summary.ilike(search_term)
                    )
                )
        
        # Order by importance and recency
        memories = query.order_by(
            desc(Memory.importance),
            desc(Memory.created_at)
        ).limit(request.limit * 2).all()  # Get more for merging
        
        # Convert to response objects
        results = []
        for memory in memories:
            response = self._memory_to_response(memory)
            # Keyword search gets base relevance from importance
            response.relevance_score = memory.importance
            results.append(response)
        
        return results
    
    async def _filter_memory_search(
        self, 
        db: Session, 
        request: MemorySearchRequest
    ) -> List[MemoryResponse]:
        """Search memories using only filters (no text query)"""
        query = db.query(Memory).join(MemorySession)
        
        # Apply all filters
        if request.user_id:
            query = query.filter(MemorySession.user_id == request.user_id)
        if request.project_id:
            query = query.filter(MemorySession.project_id == request.project_id)
        if request.memory_types:
            type_values = [t.value for t in request.memory_types]
            query = query.filter(Memory.memory_type.in_(type_values))
        if request.min_importance:
            query = query.filter(Memory.importance >= request.min_importance)
        
        # Order by importance and recency
        memories = query.order_by(
            desc(Memory.importance),
            desc(Memory.created_at)
        ).offset(request.offset).limit(request.limit).all()
        
        return [self._memory_to_response(memory) for memory in memories]
    
    def _merge_memory_results(
        self,
        vector_results: List[MemoryResponse],
        keyword_results: List[MemoryResponse]
    ) -> List[MemoryResponse]:
        """Merge vector and keyword search results with deduplication"""
        seen_ids = set()
        merged = []
        
        # Add vector results first (higher priority)
        for result in vector_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                merged.append(result)
        
        # Add keyword-only results with score penalty
        for result in keyword_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                # Apply score penalty for keyword-only results
                result.relevance_score *= 0.8
                merged.append(result)
        
        # Sort by relevance score and importance
        merged.sort(
            key=lambda x: (x.relevance_score, x.importance), 
            reverse=True
        )
        
        return merged
    
    def _memory_to_response(self, memory: Memory) -> MemoryResponse:
        """Convert Memory model to MemoryResponse"""
        return MemoryResponse(
            id=memory.id,
            session_id=memory.session_id,
            content=memory.content,
            summary=memory.summary,
            memory_type=memory.memory_type,
            importance=memory.importance,
            confidence=memory.confidence,
            entities=memory.entities or [],
            related_memories=memory.related_memories or [],
            metadata=memory.memory_metadata or {},
            access_count=memory.access_count,
            last_accessed=memory.last_accessed,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            has_embedding=memory.embedding is not None,
            age_days=memory.age_days,
            relevance_score=memory.relevance_score,
            is_recent=memory.is_recent,
            is_high_importance=memory.is_high_importance
        )
    
    def _get_memory_cache_key(self, request: MemorySearchRequest) -> str:
        """Generate cache key for memory search request"""
        import hashlib
        import json
        
        # Create deterministic string representation
        request_dict = {
            "query": request.query,
            "user_id": request.user_id,
            "project_id": str(request.project_id) if request.project_id else None,
            "memory_types": [t.value for t in request.memory_types] if request.memory_types else None,
            "min_importance": request.min_importance,
            "limit": request.limit,
            "offset": request.offset,
            "use_vector_search": request.use_vector_search
        }
        
        request_str = json.dumps(request_dict, sort_keys=True)
        return f"memory_search:{hashlib.md5(request_str.encode()).hexdigest()}"
    
    async def search_across_sessions(
        self,
        db: Session,
        user_id: str,
        query: str,
        session_ids: Optional[List[UUID]] = None,
        limit: int = 20
    ) -> List[Tuple[Memory, float, MemorySession]]:
        """Search memories across multiple sessions with session context
        
        This method is useful for finding memories across a user's entire
        conversation history, optionally filtered by specific sessions.
        
        Args:
            db: Database session
            user_id: User ID to search for
            query: Search query text
            session_ids: Optional list of session IDs to search within
            limit: Maximum number of results
            
        Returns:
            List of tuples containing (Memory, relevance_score, MemorySession)
        """
        # Build base query
        base_query = db.query(Memory, MemorySession).join(MemorySession)
        base_query = base_query.filter(MemorySession.user_id == user_id)
        
        if session_ids:
            base_query = base_query.filter(MemorySession.id.in_(session_ids))
        
        # Execute vector search if available
        results = []
        
        if query and vector_store.client:
            try:
                # Generate query embedding
                embeddings_client = get_embeddings_client()
                query_embedding = await embeddings_client.generate_embedding(query)
                
                if query_embedding:
                    # Get similar memories
                    similar_memories = await memory_embedding_service.find_similar_memories(
                        db=db,
                        query_embedding=query_embedding,
                        limit=limit,
                        user_id=user_id,
                        session_id=None  # Search across all sessions
                    )
                    
                    # Enhance with session information
                    for memory, score in similar_memories:
                        session = db.query(MemorySession).filter_by(
                            id=memory.session_id
                        ).first()
                        if session:
                            results.append((memory, score, session))
                
            except Exception as e:
                logger.error(f"Vector search across sessions failed: {e}")
        
        # Fallback to keyword search if no vector results
        if not results:
            memories_with_sessions = base_query.filter(
                or_(
                    Memory.content.ilike(f"%{query}%"),
                    Memory.summary.ilike(f"%{query}%")
                )
            ).order_by(
                desc(Memory.importance),
                desc(Memory.created_at)
            ).limit(limit).all()
            
            results = [(m, m.importance, s) for m, s in memories_with_sessions]
        
        return results


# Global instance
memory_search_service = MemorySearchService()