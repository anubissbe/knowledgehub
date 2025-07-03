"""Search service for hybrid search functionality"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import time

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..schemas.search import SearchQuery, SearchResult, SearchType
from ..models.document import DocumentChunk, ChunkType
from ..models.source import KnowledgeSource as Source


class SearchService:
    """Service for performing hybrid search across knowledge base"""
    
    def __init__(self):
        """Initialize without dependencies - they'll be injected per request"""
        pass
    
    async def search(self, db: Session, query: SearchQuery) -> Dict[str, Any]:
        """Perform hybrid search"""
        start_time = time.time()
        
        # Check cache first
        from .cache import redis_client
        cache_key = self._get_cache_key(query)
        if redis_client.client:
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                return cached_result
        
        # Perform search based on type
        if query.search_type == SearchType.HYBRID:
            results = await self._hybrid_search(db, query)
        elif query.search_type == SearchType.VECTOR:
            results = await self._vector_search(db, query)
        else:  # KEYWORD
            results = await self._keyword_search(db, query)
        
        # Format response
        response = {
            "query": query.query,
            "search_type": query.search_type.value,
            "results": results,
            "total": len(results),
            "search_time_ms": int((time.time() - start_time) * 1000),
            "filters": query.filters
        }
        
        # Cache the result
        if redis_client.client:
            await redis_client.set(cache_key, response, expiry=300)  # 5 min cache
        
        return response
    
    async def _hybrid_search(self, db: Session, query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform hybrid vector + keyword search"""
        # Run both searches in parallel
        vector_task = asyncio.create_task(self._vector_search(db, query))
        keyword_task = asyncio.create_task(self._keyword_search(db, query))
        
        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task
        )
        
        # Merge and re-rank results
        merged = self._merge_results(vector_results, keyword_results)
        
        return merged[:query.limit]
    
    async def _vector_search(self, db: Session, query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        from .vector_store import vector_store
        
        # Get embedding for query
        # In production, this would call an embedding service
        # For now, return empty if vector store not available
        if not vector_store.client:
            return []
        
        # Get embedding from embeddings service or use mock as fallback
        from .embeddings_client import get_embeddings_client
        
        client = get_embeddings_client()
        query_vector = await client.generate_embedding(query.query)
        
        # Search vector store
        results = await vector_store.search(
            query_vector=query_vector,
            limit=query.limit,
            filters=self._build_vector_filters(query)
        )
        
        # Enhance with database metadata
        enhanced_results = []
        for result in results:
            chunk = db.query(DocumentChunk).filter(
                DocumentChunk.id == result.get("chunk_id")
            ).first()
            
            if chunk:
                # Get source info from document
                if hasattr(chunk, 'document') and chunk.document:
                    source_name = chunk.document.source.name if chunk.document.source else "Unknown"
                    url = chunk.document.url if hasattr(chunk.document, 'url') else ""
                else:
                    source_name = "Unknown"
                    url = ""
                
                enhanced_results.append({
                    "content": chunk.content,
                    "source_name": source_name,
                    "url": url,
                    "score": result.get("score", 0),
                    "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
                    "metadata": chunk.chunk_metadata or {}
                })
        
        return enhanced_results
    
    async def _keyword_search(self, db: Session, query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform keyword-based search using PostgreSQL full-text search"""
        # Build base query
        chunks_query = db.query(DocumentChunk)
        
        # Apply keyword search
        if query.query:
            # Use PostgreSQL full-text search
            chunks_query = chunks_query.filter(
                func.to_tsvector('english', DocumentChunk.content).op('@@')(
                    func.plainto_tsquery('english', query.query)
                )
            )
        
        # Apply filters
        if query.filters:
            if query.filters.get("source_ids"):
                from ..models.document import Document
                chunks_query = chunks_query.join(Document).filter(
                    Document.source_id.in_(query.filters["source_ids"])
                )
            
            if query.filters.get("chunk_types"):
                chunks_query = chunks_query.filter(
                    DocumentChunk.chunk_type.in_(query.filters["chunk_types"])
                )
            
            if query.filters.get("date_from"):
                chunks_query = chunks_query.filter(
                    DocumentChunk.created_at >= query.filters["date_from"]
                )
            
            if query.filters.get("date_to"):
                chunks_query = chunks_query.filter(
                    DocumentChunk.created_at <= query.filters["date_to"]
                )
        
        # Order by relevance and limit
        chunks = chunks_query.limit(query.limit).all()
        
        # Format results according to SearchResult schema
        results = []
        for chunk in chunks:
            # Get source info from document
            if hasattr(chunk, 'document') and chunk.document:
                source_name = chunk.document.source.name if chunk.document.source else "Unknown"
                url = chunk.document.url if hasattr(chunk.document, 'url') else ""
            else:
                source_name = "Unknown"
                url = ""
            
            results.append({
                "content": chunk.content,
                "source_name": source_name,
                "url": url,
                "score": 1.0,  # Keyword search doesn't provide scores
                "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
                "metadata": chunk.chunk_metadata or {}
            })
        
        return results
    
    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge and re-rank results from multiple sources"""
        # Simple merge strategy: combine and deduplicate
        seen_chunks = set()
        merged = []
        
        # Add vector results first (they have better ranking)
        for result in vector_results:
            # Use content as unique identifier since we don't have chunk_id anymore
            content_hash = result["content"][:100]  # Use first 100 chars as identifier
            if content_hash not in seen_chunks:
                seen_chunks.add(content_hash)
                merged.append(result)
        
        # Add keyword results
        for result in keyword_results:
            content_hash = result["content"][:100]
            if content_hash not in seen_chunks:
                seen_chunks.add(content_hash)
                # Adjust score for keyword results
                result["score"] = result["score"] * 0.8
                merged.append(result)
        
        # Sort by score
        merged.sort(key=lambda x: x["score"], reverse=True)
        
        return merged
    
    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query"""
        import hashlib
        import json
        
        # Create a deterministic string representation
        query_dict = {
            "query": query.query,
            "search_type": query.search_type.value,
            "limit": query.limit,
            "filters": query.filters
        }
        
        query_str = json.dumps(query_dict, sort_keys=True)
        return f"search:{hashlib.md5(query_str.encode()).hexdigest()}"
    
    def _build_vector_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Build filters for vector store query"""
        filters = {}
        
        if query.filters:
            if query.filters.get("source_ids"):
                filters["source_id"] = query.filters["source_ids"]
            
            if query.filters.get("chunk_types"):
                filters["chunk_type"] = query.filters["chunk_types"]
        
        return filters