"""
Advanced AI Memory Service Implementation.

This service provides intelligent memory storage, retrieval, clustering, and decay
management for the AI memory system with context-aware operations.
"""

import logging
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from ..models.memory import (
    Memory, MemoryCluster, MemoryAssociation, MemoryAccess,
    MemoryType, MemoryImportance, MemoryCreate, MemoryUpdate,
    MemoryRetrievalQuery, MemoryResponse, MemoryAnalytics,
    MemoryItem  # Keep backward compatibility
)
from ..models.base import get_db_context
from ..services.real_embeddings_service import real_embeddings_service
from ..services.real_websocket_events import real_websocket_events
from ..services.cache import redis_client
from ..services.time_series_analytics import TimeSeriesAnalyticsService, MetricType
from shared.config import Config
from shared.logging import setup_logging

logger = setup_logging("memory_service")


@dataclass
class MemorySearchResult:
    """Result of memory search operation."""
    memories: List[MemoryResponse]
    total_count: int
    search_time_ms: float
    clusters_found: int
    similarity_scores: List[float]
    metadata: Dict[str, Any]


@dataclass
class ClusteringResult:
    """Result of memory clustering operation."""
    cluster_id: str
    cluster_name: str
    memory_count: int
    centroid_embedding: List[float]
    topic_keywords: List[str]
    avg_similarity: float


class MemoryService:
    """
    Advanced AI Memory Service with intelligent storage, retrieval, and management.
    
    Features:
    - Context-aware memory storage with embeddings
    - Semantic similarity search
    - Automatic memory clustering
    - Memory decay and archival
    - Association discovery
    - Performance analytics
    - Cross-session memory linking
    - Backward compatibility with old MemoryItem system
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.embedding_service = real_embeddings_service
        self.analytics_service = TimeSeriesAnalyticsService()
        self._initialized = False
        
        # Configuration
        self.similarity_threshold = 0.7
        self.clustering_threshold = 0.8
        self.max_cluster_size = 50
        self.decay_update_interval = 3600  # 1 hour
        self.association_strength_threshold = 0.6
        
        logger.info("Initialized MemoryService")
    
    async def initialize(self):
        """Initialize the memory service."""
        if self._initialized:
            return
        
        try:
            # Initialize dependencies
            await self.embedding_service.initialize()
            await self.analytics_service.initialize()
            await redis_client.initialize()
            
            self._initialized = True
            logger.info("MemoryService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MemoryService: {e}")
            raise
    
    async def create_memory(
        self,
        memory_data: MemoryCreate,
        generate_embeddings: bool = True,
        auto_cluster: bool = True
    ) -> MemoryResponse:
        """
        Create a new memory with embeddings and clustering.
        
        Args:
            memory_data: Memory creation data
            generate_embeddings: Whether to generate embeddings
            auto_cluster: Whether to automatically assign to clusters
            
        Returns:
            Created memory response
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        try:
            with get_db_context() as db:
                # Generate content hash for deduplication
                content_hash = hashlib.sha256(memory_data.content.encode()).hexdigest()
                
                # Check for duplicates
                existing = db.query(Memory).filter_by(
                    content_hash=content_hash,
                    user_id=memory_data.user_id
                ).first()
                
                if existing:
                    # Update access count and return existing memory
                    existing.increment_access()
                    db.commit()
                    logger.info(f"Found duplicate memory, updated access count: {existing.id}")
                    return self._memory_to_response(existing)
                
                # Generate embeddings
                embeddings = None
                if generate_embeddings:
                    try:
                        embedding_result = await self.embedding_service.generate_text_embedding(
                            text=memory_data.content,
                            model="default"
                        )
                        embeddings = embedding_result.embedding
                        logger.debug(f"Generated embeddings: {len(embeddings)} dimensions")
                    except Exception as e:
                        logger.warning(f"Failed to generate embeddings: {e}")
                
                # Create memory record
                memory = Memory(
                    user_id=memory_data.user_id,
                    session_id=memory_data.session_id,
                    content=memory_data.content,
                    content_hash=content_hash,
                    memory_type=memory_data.memory_type.value,
                    context=memory_data.context,
                    meta_data=memory_data.metadata,
                    tags=memory_data.tags,
                    embeddings=embeddings,
                    importance=memory_data.importance.value,
                    expires_at=memory_data.expires_at
                )
                
                db.add(memory)
                db.flush()  # Get the ID
                
                # Auto-cluster if requested and embeddings available
                if auto_cluster and embeddings:
                    try:
                        cluster_id = await self._assign_to_cluster(db, memory, embeddings)
                        if cluster_id:
                            memory.cluster_id = cluster_id
                    except Exception as e:
                        logger.warning(f"Auto-clustering failed: {e}")
                
                # Find and create associations
                try:
                    await self._create_associations(db, memory, embeddings)
                except Exception as e:
                    logger.warning(f"Association creation failed: {e}")
                
                db.commit()
                
                # Record analytics
                processing_time = (time.time() - start_time) * 1000
                await self._record_memory_analytics(
                    memory_type=memory_data.memory_type,
                    processing_time=processing_time,
                    user_id=memory_data.user_id,
                    session_id=memory_data.session_id
                )
                
                # Publish WebSocket event
                try:
                    await real_websocket_events.publish_memory_created(
                        memory_id=str(memory.id),
                        user_id=memory.user_id,
                        session_id=memory.session_id,
                        content=memory.content,
                        memory_type=memory.memory_type,
                        relevance_score=memory.relevance_score,
                        project_id=memory_data.context.get("project_id")
                    )
                except Exception as e:
                    logger.warning(f"Failed to publish memory created event: {e}")
                
                logger.info(f"Created memory: {memory.id} for user {memory.user_id}")
                return self._memory_to_response(memory)
                
        except Exception as e:
            logger.error(f"Failed to create memory: {e}")
            raise
    
    async def retrieve_memories(
        self,
        query: MemoryRetrievalQuery,
        use_semantic_search: bool = True
    ) -> MemorySearchResult:
        """
        Retrieve memories with advanced search and filtering.
        
        Args:
            query: Memory retrieval query
            use_semantic_search: Whether to use semantic similarity
            
        Returns:
            Memory search results
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        try:
            with get_db_context() as db:
                # Start with base query
                base_query = db.query(Memory).filter(
                    Memory.user_id == query.user_id
                )
                
                # Add filters
                if not query.include_archived:
                    base_query = base_query.filter(Memory.is_archived == False)
                
                if query.session_id:
                    base_query = base_query.filter(Memory.session_id == query.session_id)
                
                if query.memory_types:
                    base_query = base_query.filter(
                        Memory.memory_type.in_([t.value for t in query.memory_types])
                    )
                
                if query.min_relevance > 0:
                    base_query = base_query.filter(
                        Memory.relevance_score >= query.min_relevance
                    )
                
                if query.time_window_hours:
                    time_threshold = datetime.utcnow() - timedelta(hours=query.time_window_hours)
                    base_query = base_query.filter(Memory.created_at >= time_threshold)
                
                # Semantic search if requested and embedding service available
                similarity_scores = []
                if use_semantic_search and self.embedding_service:
                    try:
                        memories_with_similarity = await self._semantic_search(
                            db, query.query, base_query, query.limit * 2  # Get more for reranking
                        )
                        
                        if memories_with_similarity:
                            memory_ids, similarity_scores = zip(*memories_with_similarity)
                            base_query = base_query.filter(Memory.id.in_(memory_ids))
                            
                            # Order by similarity
                            memory_order = {mid: i for i, mid in enumerate(memory_ids)}
                            memories = base_query.all()
                            memories.sort(key=lambda m: memory_order.get(m.id, float('inf')))
                        else:
                            memories = []
                    
                    except Exception as e:
                        logger.warning(f"Semantic search failed, falling back to text search: {e}")
                        use_semantic_search = False
                
                # Fallback to text search
                if not use_semantic_search:
                    # Text-based search
                    search_filter = or_(
                        Memory.content.ilike(f"%{query.query}%"),
                        Memory.tags.op("&&")(func.string_to_array(query.query.lower(), " "))
                    )
                    
                    # Context-based search
                    if query.context:
                        for key, value in query.context.items():
                            search_filter = or_(
                                search_filter,
                                Memory.context[key].astext.ilike(f"%{value}%")
                            )
                    
                    memories = base_query.filter(search_filter).order_by(
                        desc(Memory.relevance_score),
                        desc(Memory.last_accessed)
                    ).limit(query.limit).all()
                
                # Update access patterns for retrieved memories
                memory_ids = [m.id for m in memories[:query.limit]]
                if memory_ids:
                    await self._record_memory_access(
                        memory_ids, query, similarity_scores[:len(memory_ids)]
                    )
                
                # Get total count for pagination
                total_count = base_query.count()
                
                # Calculate clusters found
                cluster_ids = {m.cluster_id for m in memories if m.cluster_id}
                clusters_found = len(cluster_ids)
                
                search_time = (time.time() - start_time) * 1000
                
                # Convert to response format
                memory_responses = [
                    self._memory_to_response(memory) 
                    for memory in memories[:query.limit]
                ]
                
                logger.info(f"Retrieved {len(memory_responses)} memories in {search_time:.2f}ms")
                
                return MemorySearchResult(
                    memories=memory_responses,
                    total_count=total_count,
                    search_time_ms=search_time,
                    clusters_found=clusters_found,
                    similarity_scores=similarity_scores[:len(memory_responses)],
                    metadata={
                        "semantic_search_used": use_semantic_search,
                        "query_context": query.context,
                        "filters_applied": {
                            "session_id": query.session_id,
                            "memory_types": [t.value for t in query.memory_types] if query.memory_types else None,
                            "min_relevance": query.min_relevance,
                            "time_window_hours": query.time_window_hours
                        }
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise
    
    # Legacy method for backward compatibility
    async def create_memory_legacy(self, db: Session, memory_data) -> MemoryItem:
        """Create a memory using the legacy MemoryItem system."""
        try:
            # Generate content hash
            content_hash = hashlib.sha256(memory_data.content.encode()).hexdigest()[:16]
            
            # Check for duplicate
            existing = db.query(MemoryItem).filter(
                MemoryItem.content_hash == content_hash
            ).first()
            
            if existing:
                # Update access count and timestamp
                existing.access_count += 1
                existing.accessed_at = datetime.utcnow()
                db.commit()
                return existing
            
            # Create new memory
            from uuid import uuid4
            memory = MemoryItem(
                id=uuid4(),
                content=memory_data.content,
                content_hash=content_hash,
                tags=getattr(memory_data, 'tags', []) or [],
                meta_data=getattr(memory_data, 'metadata', {}) or {},
                embedding_id=None,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                access_count=1
            )
            
            db.add(memory)
            db.commit()
            db.refresh(memory)
            
            logger.info(f"Created legacy memory: {memory.id}")
            return memory
            
        except Exception as e:
            logger.error(f"Failed to create legacy memory: {e}")
            raise
    
    async def _semantic_search(
        self,
        db: Session,
        query_text: str,
        base_query,
        limit: int
    ) -> List[Tuple[str, float]]:
        """Perform semantic search using embeddings."""
        
        try:
            # Generate query embedding
            query_embedding_result = await self.embedding_service.generate_embedding(query_text)
            query_embedding = query_embedding_result.embeddings
            
            # Get memories with embeddings
            memories_with_embeddings = base_query.filter(
                Memory.embeddings.isnot(None)
            ).all()
            
            if not memories_with_embeddings:
                return []
            
            # Calculate similarities
            similarities = []
            for memory in memories_with_embeddings:
                try:
                    similarity = await self.embedding_service.calculate_similarity(
                        query_embedding,
                        memory.embeddings,
                        method="cosine"
                    )
                    similarities.append((str(memory.id), similarity))
                except Exception as e:
                    logger.warning(f"Similarity calculation failed for memory {memory.id}: {e}")
            
            # Sort by similarity and apply threshold
            similarities.sort(key=lambda x: x[1], reverse=True)
            filtered_similarities = [
                (mem_id, score) for mem_id, score in similarities
                if score >= self.similarity_threshold
            ]
            
            return filtered_similarities[:limit]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _assign_to_cluster(
        self,
        db: Session,
        memory: Memory,
        embeddings: List[float]
    ) -> Optional[str]:
        """Assign memory to an appropriate cluster or create new one."""
        
        try:
            # Find existing clusters for the user
            existing_clusters = db.query(MemoryCluster).filter(
                MemoryCluster.cluster_type == memory.memory_type
            ).all()
            
            best_cluster = None
            best_similarity = 0.0
            
            # Find best matching cluster
            for cluster in existing_clusters:
                if cluster.centroid_embedding:
                    similarity = await self.embedding_service.calculate_similarity(
                        embeddings,
                        cluster.centroid_embedding,
                        method="cosine"
                    )
                    
                    if similarity > best_similarity and similarity >= self.clustering_threshold:
                        # Check cluster size
                        cluster_memory_count = db.query(Memory).filter_by(
                            cluster_id=cluster.id
                        ).count()
                        
                        if cluster_memory_count < self.max_cluster_size:
                            best_cluster = cluster
                            best_similarity = similarity
            
            if best_cluster:
                # Update cluster statistics
                await self._update_cluster_stats(db, best_cluster.id)
                return str(best_cluster.id)
            
            # Create new cluster if no suitable one found
            return await self._create_memory_cluster(db, memory, embeddings)
            
        except Exception as e:
            logger.error(f"Cluster assignment failed: {e}")
            return None
    
    async def _create_memory_cluster(
        self,
        db: Session,
        memory: Memory,
        embeddings: List[float]
    ) -> str:
        """Create a new memory cluster."""
        
        try:
            # Generate cluster name based on content
            cluster_name = await self._generate_cluster_name(memory.content, memory.tags)
            
            cluster = MemoryCluster(
                name=cluster_name,
                description=f"Auto-generated cluster for {memory.memory_type} memories",
                cluster_type=memory.memory_type,
                centroid_embedding=embeddings,
                topic_keywords=memory.tags[:5] if memory.tags else [],
                memory_count=1,
                avg_relevance=memory.relevance_score or 1.0,
                last_accessed=datetime.utcnow()
            )
            
            db.add(cluster)
            db.flush()
            
            logger.info(f"Created new cluster: {cluster.id} for memory type {memory.memory_type}")
            return str(cluster.id)
            
        except Exception as e:
            logger.error(f"Cluster creation failed: {e}")
            return None
    
    async def _create_associations(
        self,
        db: Session,
        memory: Memory,
        embeddings: Optional[List[float]]
    ):
        """Create associations between memories."""
        
        if not embeddings:
            return
        
        try:
            # Find similar memories from same user
            similar_memories = db.query(Memory).filter(
                Memory.user_id == memory.user_id,
                Memory.id != memory.id,
                Memory.embeddings.isnot(None),
                Memory.is_archived == False
            ).limit(20).all()  # Limit for performance
            
            associations_created = 0
            
            for similar_memory in similar_memories:
                try:
                    similarity = await self.embedding_service.calculate_similarity(
                        embeddings,
                        similar_memory.embeddings,
                        method="cosine"
                    )
                    
                    if similarity >= self.association_strength_threshold:
                        # Determine association type
                        association_type = self._determine_association_type(memory, similar_memory)
                        
                        # Create association
                        association = MemoryAssociation(
                            source_memory_id=memory.id,
                            target_memory_id=similar_memory.id,
                            association_type=association_type,
                            strength=similarity,
                            confidence=min(memory.confidence_score, similar_memory.confidence_score),
                            context={
                                "similarity_score": similarity,
                                "created_by": "auto_association"
                            }
                        )
                        
                        db.add(association)
                        associations_created += 1
                        
                        if associations_created >= 5:  # Limit associations per memory
                            break
                            
                except Exception as e:
                    logger.warning(f"Association creation failed for memories {memory.id}-{similar_memory.id}: {e}")
            
            if associations_created > 0:
                logger.info(f"Created {associations_created} associations for memory {memory.id}")
                
        except Exception as e:
            logger.error(f"Association creation process failed: {e}")
    
    def _determine_association_type(self, memory1: Memory, memory2: Memory) -> str:
        """Determine the type of association between two memories."""
        
        # Same session = temporal association
        if memory1.session_id == memory2.session_id:
            return "temporal"
        
        # Same memory type = categorical association
        if memory1.memory_type == memory2.memory_type:
            return "categorical"
        
        # Overlapping tags = topical association
        if memory1.tags and memory2.tags:
            common_tags = set(memory1.tags) & set(memory2.tags)
            if common_tags:
                return "topical"
        
        # Default to similarity
        return "similar"
    
    async def _record_memory_analytics(
        self,
        memory_type: MemoryType,
        processing_time: float,
        user_id: str,
        session_id: str
    ):
        """Record memory operation analytics."""
        
        try:
            # Record in time-series analytics
            await self.analytics_service.record_metric(
                metric_type=MetricType.KNOWLEDGE_CREATION,
                value=1.0,
                tags={"memory_type": memory_type.value, "user_id": user_id},
                metadata={"processing_time_ms": processing_time, "session_id": session_id}
            )
            
            # Record performance metrics
            await self.analytics_service.record_performance(
                endpoint="memory_creation",
                response_time=processing_time / 1000.0,  # Convert to seconds
                error_count=0,
                request_count=1
            )
            
        except Exception as e:
            logger.warning(f"Analytics recording failed: {e}")
    
    async def _record_memory_access(
        self,
        memory_ids: List[str],
        query: MemoryRetrievalQuery,
        similarity_scores: List[float]
    ):
        """Record memory access patterns."""
        
        try:
            with get_db_context() as db:
                for i, memory_id in enumerate(memory_ids):
                    similarity_score = similarity_scores[i] if i < len(similarity_scores) else 0.0
                    
                    access_log = MemoryAccess(
                        memory_id=memory_id,
                        user_id=query.user_id,
                        session_id=query.session_id or "unknown",
                        access_type="retrieval",
                        context_similarity=similarity_score,
                        retrieval_method="semantic" if similarity_score > 0 else "text",
                        query_context=query.context,
                        result_rank=i + 1
                    )
                    
                    db.add(access_log)
                    
                    # Update memory access count
                    memory = db.query(Memory).filter_by(id=memory_id).first()
                    if memory:
                        memory.increment_access()
                
                db.commit()
                
        except Exception as e:
            logger.warning(f"Access logging failed: {e}")
    
    async def _generate_cluster_name(self, content: str, tags: List[str]) -> str:
        """Generate a descriptive name for a memory cluster."""
        
        # Use tags if available
        if tags:
            return f"{tags[0].title()} Cluster"
        
        # Extract key words from content
        words = content.lower().split()
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        key_words = [word for word in words[:10] if word not in common_words and len(word) > 3]
        
        if key_words:
            return f"{key_words[0].title()} Cluster"
        
        return "Memory Cluster"
    
    async def _update_cluster_stats(self, db: Session, cluster_id: str):
        """Update cluster statistics."""
        
        try:
            cluster = db.query(MemoryCluster).filter_by(id=cluster_id).first()
            if not cluster:
                return
            
            # Count memories in cluster
            memory_count = db.query(Memory).filter_by(cluster_id=cluster_id).count()
            
            # Calculate average relevance
            avg_relevance = db.query(func.avg(Memory.relevance_score)).filter_by(
                cluster_id=cluster_id
            ).scalar() or 0.0
            
            # Update cluster
            cluster.memory_count = memory_count
            cluster.avg_relevance = float(avg_relevance)
            cluster.last_accessed = datetime.utcnow()
            cluster.updated_at = datetime.utcnow()
            
        except Exception as e:
            logger.warning(f"Cluster stats update failed: {e}")
    
    def _memory_to_response(self, memory: Memory) -> MemoryResponse:
        """Convert Memory model to MemoryResponse."""
        
        return MemoryResponse(
            id=str(memory.id),
            user_id=memory.user_id,
            session_id=memory.session_id,
            content=memory.content,
            memory_type=MemoryType(memory.memory_type),
            context=memory.context or {},
            metadata=memory.meta_data or {},
            tags=memory.tags or [],
            relevance_score=memory.relevance_score,
            importance=MemoryImportance(memory.importance),
            confidence_score=memory.confidence_score,
            access_count=memory.access_count,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            last_accessed=memory.last_accessed,
            cluster_id=str(memory.cluster_id) if memory.cluster_id else None
        )
    
    def get_memories(self, db, memory_type=None, source=None, skip=0, limit=100):
        """
        Get memories with simple filtering - direct database query.
        
        This method provides basic memory retrieval without AI features.
        """
        try:
            query = db.query(Memory)
            
            if memory_type:
                query = query.filter(Memory.memory_type == memory_type)
            
            # Note: source filter not implemented since Memory model doesn't have source field
            
            memories = query.offset(skip).limit(limit).all()
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            raise

    async def create_memory_simple(self, memory_data: MemoryCreate) -> MemoryResponse:
        """
        Create a new memory without any AI features - direct database storage.
        
        This method bypasses embeddings, clustering, analytics, and WebSocket events
        to provide a simple, reliable memory creation that works without external dependencies.
        
        Args:
            memory_data: Memory creation data
            
        Returns:
            Created memory response
        """
        try:
            with get_db_context() as db:
                # Create memory directly
                memory = Memory(
                    user_id=memory_data.user_id,
                    session_id=memory_data.session_id,
                    content=memory_data.content,
                    memory_type=memory_data.memory_type.value,
                    context=memory_data.context,
                    meta_data=memory_data.metadata,
                    tags=memory_data.tags,
                    importance=memory_data.importance.value,
                    content_hash=hashlib.sha256(memory_data.content.encode()).hexdigest(),
                    relevance_score=0.5,  # Default relevance
                    confidence_score=0.5,  # Default confidence
                    access_count=1,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow()
                )
                
                db.add(memory)
                db.commit()
                db.refresh(memory)
                
                logger.info(f"Created simple memory: {memory.id} for user {memory.user_id}")
                return self._memory_to_response(memory)
                
        except Exception as e:
            logger.error(f"Failed to create simple memory: {e}")
            raise

    async def cleanup(self):
        """Clean up service resources."""
        await self.embedding_service.cleanup()
        await self.analytics_service.cleanup()
        self._initialized = False
        logger.info("MemoryService cleaned up")


# Global memory service instance
memory_service = MemoryService()