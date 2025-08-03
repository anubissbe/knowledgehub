"""
Traced Memory Service

Memory service with comprehensive OpenTelemetry tracing for performance analysis
and distributed debugging capabilities.
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ..middleware.tracing_middleware import trace_memory_search, trace_db_query, trace_ai_embedding
from ..services.opentelemetry_tracing import otel_tracing
from ..services.prometheus_metrics import prometheus_metrics

logger = logging.getLogger(__name__)

class TracedMemoryService:
    """
    Memory service with comprehensive tracing for:
    - Memory search operations (target: <50ms)
    - AI embedding generation
    - Database operations
    - Cross-service calls
    - Performance analysis
    """
    
    def __init__(self):
        self.service_name = "memory_service"
        self.performance_targets = {
            "memory_search": 0.05,  # 50ms
            "memory_create": 0.1,   # 100ms
            "embedding_generation": 2.0,  # 2s
            "database_query": 0.1   # 100ms
        }
        
    @trace_memory_search("search", "system", "semantic")
    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        memory_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search memories with semantic similarity.
        Target performance: <50ms
        """
        
        search_start = time.time()
        
        # Add span attributes for detailed analysis
        otel_tracing.set_span_attribute("memory.query_length", len(query))
        otel_tracing.set_span_attribute("memory.result_limit", limit)
        otel_tracing.set_span_attribute("memory.search_type", memory_type or "all")
        
        try:
            # Step 1: Generate query embedding (traced separately)
            with otel_tracing.start_span(
                "memory.generate_query_embedding",
                attributes={
                    "ai.model": "sentence-transformers",
                    "ai.input_text_length": len(query)
                }
            ) as embedding_span:
                
                embedding_start = time.time()
                query_embedding = await self._generate_embedding(query)
                embedding_duration = time.time() - embedding_start
                
                embedding_span.set_attribute("ai.embedding_dimensions", len(query_embedding) if query_embedding else 0)
                embedding_span.set_attribute("ai.generation_time_ms", embedding_duration * 1000)
                
                # Record performance metrics
                prometheus_metrics.record_ai_operation(
                    "embedding", "sentence-transformers", embedding_duration, True
                )
            
            # Step 2: Vector similarity search (traced)
            with otel_tracing.start_span(
                "memory.vector_search",
                attributes={
                    "db.operation": "similarity_search",
                    "db.type": "weaviate",
                    "vector.dimensions": len(query_embedding) if query_embedding else 0
                }
            ) as search_span:
                
                search_start_time = time.time()
                similar_memories = await self._vector_search(
                    query_embedding, limit, memory_type
                )
                search_duration = time.time() - search_start_time
                
                search_span.set_attribute("memory.results_found", len(similar_memories))
                search_span.set_attribute("memory.search_duration_ms", search_duration * 1000)
                
                # Performance analysis
                if search_duration > self.performance_targets["memory_search"]:
                    search_span.add_event("slow_search_detected", {
                        "duration_ms": search_duration * 1000,
                        "target_ms": self.performance_targets["memory_search"] * 1000,
                        "query_length": len(query),
                        "results_count": len(similar_memories)
                    })
            
            # Step 3: Enrich results with metadata (traced)
            with otel_tracing.start_span(
                "memory.enrich_results",
                attributes={
                    "memory.enrichment_count": len(similar_memories)
                }
            ) as enrich_span:
                
                enriched_memories = await self._enrich_memory_results(similar_memories)
                enrich_span.set_attribute("memory.enriched_count", len(enriched_memories))
            
            # Calculate total search time
            total_duration = time.time() - search_start
            
            # Record comprehensive performance metrics
            prometheus_metrics.record_memory_operation("search", user_id, total_duration)
            
            # Performance classification
            performance_status = "fast" if total_duration < 0.05 else "slow"
            otel_tracing.set_span_attribute("memory.performance_classification", performance_status)
            
            # Add search quality metrics
            if enriched_memories:
                avg_score = sum(m.get("score", 0) for m in enriched_memories) / len(enriched_memories)
                otel_tracing.set_span_attribute("memory.avg_relevance_score", avg_score)
                
                top_score = max(m.get("score", 0) for m in enriched_memories)
                otel_tracing.set_span_attribute("memory.top_relevance_score", top_score)
            
            return {
                "memories": enriched_memories,
                "query": query,
                "total_results": len(enriched_memories),
                "search_duration_ms": total_duration * 1000,
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            otel_tracing.record_exception(e)
            otel_tracing.add_span_event("memory_search_error", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "query_length": len(query),
                "user_id": user_id
            })
            
            # Record error metrics
            prometheus_metrics.record_error("memory_search", "memory_service")
            raise

    @trace_db_query("create_memory", "memories", "insert")
    async def create_memory(
        self,
        content: str,
        user_id: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new memory with embedding generation.
        Target performance: <100ms
        """
        
        create_start = time.time()
        
        # Add creation context to span
        otel_tracing.set_span_attribute("memory.content_length", len(content))
        otel_tracing.set_span_attribute("memory.type", memory_type)
        otel_tracing.set_span_attribute("memory.has_metadata", metadata is not None)
        
        try:
            # Step 1: Generate embedding for content
            with otel_tracing.start_span(
                "memory.generate_content_embedding",
                attributes={
                    "ai.model": "sentence-transformers",
                    "ai.content_length": len(content)
                }
            ) as embedding_span:
                
                embedding = await self._generate_embedding(content)
                embedding_span.set_attribute("ai.embedding_success", embedding is not None)
                
                if embedding:
                    embedding_span.set_attribute("ai.embedding_dimensions", len(embedding))
            
            # Step 2: Store in database
            with otel_tracing.start_span(
                "memory.database_insert",
                attributes={
                    "db.table": "memories",
                    "db.operation": "insert"
                }
            ) as db_span:
                
                memory_id = await self._store_memory(
                    content, user_id, memory_type, embedding, metadata
                )
                db_span.set_attribute("memory.created_id", memory_id)
            
            # Step 3: Update search index
            with otel_tracing.start_span(
                "memory.update_search_index",
                attributes={
                    "index.type": "weaviate",
                    "index.operation": "upsert"
                }
            ) as index_span:
                
                await self._update_search_index(memory_id, content, embedding, metadata)
                index_span.set_attribute("index.updated", True)
            
            # Calculate total creation time
            total_duration = time.time() - create_start
            
            # Record performance metrics
            prometheus_metrics.record_memory_operation("create", user_id, total_duration)
            
            # Performance analysis
            if total_duration > self.performance_targets["memory_create"]:
                otel_tracing.add_span_event("slow_memory_creation", {
                    "duration_ms": total_duration * 1000,
                    "target_ms": self.performance_targets["memory_create"] * 1000,
                    "content_length": len(content)
                })
            
            return {
                "memory_id": memory_id,
                "content": content,
                "user_id": user_id,
                "memory_type": memory_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "creation_duration_ms": total_duration * 1000
            }
            
        except Exception as e:
            otel_tracing.record_exception(e)
            otel_tracing.add_span_event("memory_creation_error", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "content_length": len(content),
                "user_id": user_id
            })
            
            prometheus_metrics.record_error("memory_create", "memory_service")
            raise

    @trace_db_query("get_memories", "memories", "select")
    async def get_user_memories(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        memory_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve user memories with pagination.
        Target performance: <100ms
        """
        
        # Add query context
        otel_tracing.set_span_attribute("memory.query_user_id", user_id)
        otel_tracing.set_span_attribute("memory.query_limit", limit)
        otel_tracing.set_span_attribute("memory.query_offset", offset)
        
        try:
            query_start = time.time()
            
            # Query database with tracing
            memories = await self._query_user_memories(user_id, limit, offset, memory_type)
            
            query_duration = time.time() - query_start
            
            # Add query performance attributes
            otel_tracing.set_span_attribute("memory.query_duration_ms", query_duration * 1000)
            otel_tracing.set_span_attribute("memory.memories_retrieved", len(memories))
            
            # Record metrics
            prometheus_metrics.record_memory_operation("retrieve", user_id, query_duration)
            
            return {
                "memories": memories,
                "user_id": user_id,
                "total_count": len(memories),
                "limit": limit,
                "offset": offset,
                "query_duration_ms": query_duration * 1000
            }
            
        except Exception as e:
            otel_tracing.record_exception(e)
            prometheus_metrics.record_error("memory_retrieve", "memory_service")
            raise

    # Private helper methods (also traced)

    @trace_ai_embedding("sentence-transformers")
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using AI model"""
        try:
            # Use the real embeddings client
            from .embeddings_client import get_embeddings_client
            
            client = get_embeddings_client()
            embedding = await client.generate_embedding(text)
            
            # Set tracing attributes
            otel_tracing.set_span_attribute("ai.embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            otel_tracing.set_span_attribute("ai.input_tokens", len(text.split()))
            otel_tracing.set_span_attribute("ai.embedding_dimensions", len(embedding))
            
            return embedding
            
        except Exception as e:
            otel_tracing.add_span_event("embedding_generation_failed", {
                "error": str(e),
                "text_length": len(text)
            })
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def _vector_search(
        self, 
        query_embedding: List[float], 
        limit: int, 
        memory_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        try:
            # Use the real vector store
            from .vector_store import vector_store
            
            # Build filters for vector search
            filters = {}
            if memory_type:
                filters["memory_type"] = memory_type
            
            # Perform actual vector search
            results = await vector_store.search(
                query_vector=query_embedding,
                limit=limit,
                filters=filters
            )
            
            # Transform results to expected format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "memory_type": result.get("memory_type", memory_type or "conversation"),
                    "metadata": result.get("metadata", {})
                })
            
            # Set tracing attributes
            otel_tracing.set_span_attribute("vector.search_algorithm", "cosine_similarity")
            otel_tracing.set_span_attribute("vector.results_returned", len(formatted_results))
            otel_tracing.set_span_attribute("vector.search_filters", str(filters))
            
            return formatted_results
            
        except Exception as e:
            otel_tracing.add_span_event("vector_search_failed", {
                "error": str(e),
                "limit": limit
            })
            return []

    async def _enrich_memory_results(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich memory results with additional metadata"""
        try:
            # Enrich memories with additional context and categorization
            for memory in memories:
                # Add enrichment timestamp
                memory["enriched_at"] = datetime.now(timezone.utc).isoformat()
                
                # Categorize by relevance score
                score = memory.get("score", 0)
                if score > 0.9:
                    memory["relevance_category"] = "very_high"
                elif score > 0.8:
                    memory["relevance_category"] = "high"
                elif score > 0.6:
                    memory["relevance_category"] = "medium"
                else:
                    memory["relevance_category"] = "low"
                
                # Add content length for UI display
                memory["content_length"] = len(memory.get("content", ""))
                
                # Add summary if content is long
                content = memory.get("content", "")
                if len(content) > 200:
                    memory["summary"] = content[:197] + "..."
                else:
                    memory["summary"] = content
            
            return memories
            
        except Exception as e:
            otel_tracing.add_span_event("memory_enrichment_failed", {
                "error": str(e),
                "memory_count": len(memories)
            })
            return memories

    async def _store_memory(
        self,
        content: str,
        user_id: str,
        memory_type: str,
        embedding: Optional[List[float]],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Store memory in database"""
        try:
            # Use the real memory service to store in database
            from ..memory_system.core.memory_manager import MemoryManager
            from ..database import get_db_session
            
            async with get_db_session() as db:
                memory_manager = MemoryManager(db)
                
                # Create memory data
                memory_data = {
                    "user_id": user_id,
                    "content": content,
                    "memory_type": memory_type,
                    "metadata": metadata or {},
                    "embedding": embedding
                }
                
                # Store the memory
                memory = await memory_manager.create_memory(memory_data)
                memory_id = str(memory.id)
                
                otel_tracing.set_span_attribute("db.memory_id", memory_id)
                otel_tracing.set_span_attribute("db.embedding_stored", embedding is not None)
                
                return memory_id
            
        except Exception as e:
            otel_tracing.add_span_event("memory_storage_failed", {
                "error": str(e),
                "user_id": user_id
            })
            raise

    async def _update_search_index(
        self,
        memory_id: str,
        content: str,
        embedding: Optional[List[float]],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Update search index with new memory"""
        try:
            # Update the vector store index with the new memory
            from .vector_store import vector_store
            
            if embedding and vector_store.client:
                # Index the memory in the vector store
                await vector_store.add_document(
                    doc_id=memory_id,
                    content=content,
                    embedding=embedding,
                    metadata={
                        **(metadata or {}),
                        "memory_id": memory_id,
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                )
                
                otel_tracing.set_span_attribute("index.memory_id", memory_id)
                otel_tracing.set_span_attribute("index.content_indexed", True)
                otel_tracing.set_span_attribute("index.vector_dimensions", len(embedding))
            else:
                otel_tracing.set_span_attribute("index.memory_id", memory_id)
                otel_tracing.set_span_attribute("index.content_indexed", False)
                otel_tracing.add_span_event("index_skipped", {
                    "reason": "No embedding or vector store unavailable"
                })
            
        except Exception as e:
            otel_tracing.add_span_event("index_update_failed", {
                "error": str(e),
                "memory_id": memory_id
            })
            logger.warning(f"Failed to update search index for memory {memory_id}: {e}")
            # Don't raise - index update failure shouldn't fail memory creation

    async def _query_user_memories(
        self,
        user_id: str,
        limit: int,
        offset: int,
        memory_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Query user memories from database"""
        try:
            # Use the real memory manager to query from database
            from ..memory_system.core.memory_manager import MemoryManager
            from ..database import get_db_session
            
            async with get_db_session() as db:
                memory_manager = MemoryManager(db)
                
                # Build filters
                filters = {"user_id": user_id}
                if memory_type:
                    filters["memory_type"] = memory_type
                
                # Query memories
                memories_data = await memory_manager.get_memories(
                    filters=filters,
                    limit=limit,
                    offset=offset
                )
                
                # Format results
                memories = []
                for memory in memories_data:
                    memories.append({
                        "id": str(memory.id),
                        "content": memory.content,
                        "memory_type": memory.memory_type,
                        "created_at": memory.created_at.isoformat() if memory.created_at else datetime.now(timezone.utc).isoformat(),
                        "user_id": memory.user_id,
                        "metadata": memory.metadata or {}
                    })
                
                otel_tracing.set_span_attribute("db.query_type", "user_memories")
                otel_tracing.set_span_attribute("db.results_returned", len(memories))
                
                return memories
            
        except Exception as e:
            otel_tracing.add_span_event("memory_query_failed", {
                "error": str(e),
                "user_id": user_id
            })
            raise

# Global traced memory service instance
traced_memory_service = TracedMemoryService()