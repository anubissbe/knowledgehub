"""
Persistent Context Architecture

This module implements a comprehensive persistent context system that maintains
long-term memory and context across multiple sessions and extended time periods.
It builds on the existing memory system to provide true persistent context
capabilities for Claude-Code interactions.
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func, text
from sqlalchemy.dialects.postgresql import insert

from ..models import Memory, MemorySession, MemoryType
from ..services.embedding_service import EmbeddingService
from ...services.cache import redis_client
from ...services.vector_store import vector_store

logger = logging.getLogger(__name__)


class ContextScope(Enum):
    """Scope of context persistence"""
    SESSION = "session"           # Current session only
    PROJECT = "project"           # Project-specific context
    USER = "user"                 # User-specific context  
    GLOBAL = "global"             # Global context patterns
    DOMAIN = "domain"             # Domain-specific knowledge


class ContextType(Enum):
    """Types of persistent context"""
    CONVERSATION_FLOW = "conversation_flow"    # Conversation patterns
    TECHNICAL_KNOWLEDGE = "technical_knowledge" # Code patterns, solutions
    PREFERENCES = "preferences"                # User preferences
    DECISIONS = "decisions"                    # Past decisions and rationale
    PATTERNS = "patterns"                      # Recurring patterns
    RELATIONSHIPS = "relationships"            # Entity relationships
    WORKFLOWS = "workflows"                    # Process workflows
    LEARNINGS = "learnings"                    # Accumulated learnings


@dataclass
class ContextVector:
    """Vector representation of context for similarity matching"""
    id: UUID
    content: str
    embedding: List[float]
    context_type: ContextType
    scope: ContextScope
    importance: float
    last_accessed: datetime
    access_count: int
    related_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class ContextCluster:
    """Cluster of related context vectors"""
    id: UUID
    name: str
    description: str
    vectors: List[ContextVector]
    centroid: List[float]
    coherence_score: float
    last_updated: datetime
    access_pattern: Dict[str, int]


@dataclass
class ContextGraph:
    """Graph representation of context relationships"""
    nodes: Dict[UUID, ContextVector]
    edges: Dict[UUID, List[Tuple[UUID, float]]]  # node_id -> [(connected_node_id, weight)]
    clusters: Dict[UUID, ContextCluster]
    global_patterns: Dict[str, Any]


class PersistentContextManager:
    """
    Manages persistent context across sessions and time periods
    
    This system maintains long-term memory and context that persists
    across multiple Claude-Code interactions, providing continuity
    and accumulated knowledge.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EmbeddingService()
        self.context_graph = ContextGraph(
            nodes={},
            edges={},
            clusters={},
            global_patterns={}
        )
        
        # Configuration
        self.max_context_vectors = 10000
        self.similarity_threshold = 0.75
        self.importance_decay_rate = 0.95  # Per week
        self.access_boost_factor = 1.1
        self.clustering_interval = 3600  # 1 hour
        
        # Cache keys
        self.cache_prefix = "persistent_context"
        self.graph_cache_key = f"{self.cache_prefix}:graph"
        
        # Initialize on startup
        asyncio.create_task(self._initialize_context_graph())
    
    async def _initialize_context_graph(self):
        """Initialize the context graph from existing data"""
        try:
            # Load existing context graph from cache
            cached_graph = await self._load_graph_from_cache()
            if cached_graph:
                self.context_graph = cached_graph
                logger.info("Loaded context graph from cache")
                return
            
            # Build graph from database
            await self._build_graph_from_database()
            
            # Save to cache
            await self._save_graph_to_cache()
            
            logger.info(f"Initialized context graph with {len(self.context_graph.nodes)} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing context graph: {e}")
    
    async def _build_graph_from_database(self):
        """Build context graph from database memories"""
        try:
            # Get all memories with high importance
            memories = self.db.query(Memory).filter(
                Memory.importance_score > 0.5
            ).order_by(desc(Memory.importance_score)).limit(self.max_context_vectors).all()
            
            vectors = []
            for memory in memories:
                try:
                    # Generate embedding for memory content
                    embedding = await self.embedding_service.generate_embedding(
                        memory.content + " " + (memory.summary or "")
                    )
                    
                    # Create context vector
                    vector = ContextVector(
                        id=memory.id,
                        content=memory.content,
                        embedding=embedding,
                        context_type=self._map_memory_type_to_context_type(memory.memory_type),
                        scope=ContextScope.SESSION,  # Default scope
                        importance=memory.importance_score,
                        last_accessed=memory.created_at,
                        access_count=1,
                        related_entities=memory.entities or [],
                        metadata={
                            "session_id": str(memory.session_id),
                            "memory_type": memory.memory_type.value,
                            "created_at": memory.created_at.isoformat(),
                            "facts": memory.facts or []
                        }
                    )
                    
                    vectors.append(vector)
                    
                except Exception as e:
                    logger.warning(f"Error processing memory {memory.id}: {e}")
                    continue
            
            # Build graph structure
            await self._build_graph_structure(vectors)
            
        except Exception as e:
            logger.error(f"Error building graph from database: {e}")
    
    async def _build_graph_structure(self, vectors: List[ContextVector]):
        """Build graph structure from context vectors"""
        try:
            # Add nodes
            for vector in vectors:
                self.context_graph.nodes[vector.id] = vector
            
            # Build edges based on similarity
            for i, vector1 in enumerate(vectors):
                edges = []
                for j, vector2 in enumerate(vectors):
                    if i != j:
                        similarity = self._calculate_similarity(
                            vector1.embedding, vector2.embedding
                        )
                        if similarity > self.similarity_threshold:
                            edges.append((vector2.id, similarity))
                
                # Sort by similarity and keep top connections
                edges.sort(key=lambda x: x[1], reverse=True)
                self.context_graph.edges[vector1.id] = edges[:10]  # Top 10 connections
            
            # Create clusters
            await self._create_clusters()
            
        except Exception as e:
            logger.error(f"Error building graph structure: {e}")
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Cosine similarity calculation
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    async def _create_clusters(self):
        """Create clusters of related context vectors"""
        try:
            # Simple clustering based on context type and similarity
            clusters_by_type = defaultdict(list)
            
            for vector in self.context_graph.nodes.values():
                clusters_by_type[vector.context_type].append(vector)
            
            # Create clusters for each type
            for context_type, vectors in clusters_by_type.items():
                if len(vectors) >= 3:  # Minimum cluster size
                    cluster = await self._create_cluster(context_type, vectors)
                    self.context_graph.clusters[cluster.id] = cluster
            
            logger.info(f"Created {len(self.context_graph.clusters)} clusters")
            
        except Exception as e:
            logger.error(f"Error creating clusters: {e}")
    
    async def _create_cluster(self, context_type: ContextType, vectors: List[ContextVector]) -> ContextCluster:
        """Create a cluster from related vectors"""
        try:
            cluster_id = uuid4()
            
            # Calculate centroid
            centroid = await self._calculate_centroid([v.embedding for v in vectors])
            
            # Calculate coherence score
            coherence_score = await self._calculate_coherence_score(vectors)
            
            # Create cluster
            cluster = ContextCluster(
                id=cluster_id,
                name=f"{context_type.value.replace('_', ' ').title()} Cluster",
                description=f"Cluster of {len(vectors)} {context_type.value} contexts",
                vectors=vectors,
                centroid=centroid,
                coherence_score=coherence_score,
                last_updated=datetime.now(timezone.utc),
                access_pattern={}
            )
            
            return cluster
            
        except Exception as e:
            logger.error(f"Error creating cluster: {e}")
            return None
    
    async def _calculate_centroid(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate centroid of embeddings"""
        try:
            if not embeddings:
                return []
            
            # Calculate average of all embeddings
            centroid = [0.0] * len(embeddings[0])
            for embedding in embeddings:
                for i, value in enumerate(embedding):
                    centroid[i] += value
            
            # Normalize
            for i in range(len(centroid)):
                centroid[i] /= len(embeddings)
            
            return centroid
            
        except Exception as e:
            logger.error(f"Error calculating centroid: {e}")
            return []
    
    async def _calculate_coherence_score(self, vectors: List[ContextVector]) -> float:
        """Calculate coherence score for a cluster"""
        try:
            if len(vectors) < 2:
                return 1.0
            
            # Calculate average pairwise similarity
            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    similarity = self._calculate_similarity(
                        vectors[i].embedding, vectors[j].embedding
                    )
                    similarities.append(similarity)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating coherence score: {e}")
            return 0.0
    
    def _map_memory_type_to_context_type(self, memory_type: MemoryType) -> ContextType:
        """Map memory type to context type"""
        mapping = {
            MemoryType.FACT: ContextType.TECHNICAL_KNOWLEDGE,
            MemoryType.PREFERENCE: ContextType.PREFERENCES,
            MemoryType.CODE: ContextType.TECHNICAL_KNOWLEDGE,
            MemoryType.DECISION: ContextType.DECISIONS,
            MemoryType.ERROR: ContextType.LEARNINGS,
            MemoryType.PATTERN: ContextType.PATTERNS,
            MemoryType.ENTITY: ContextType.RELATIONSHIPS
        }
        return mapping.get(memory_type, ContextType.CONVERSATION_FLOW)
    
    async def add_context(self, content: str, context_type: ContextType, 
                         scope: ContextScope, importance: float = 0.5,
                         related_entities: List[str] = None,
                         metadata: Dict[str, Any] = None) -> UUID:
        """Add new context to the persistent system"""
        try:
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(content)
            
            # Create context vector
            vector = ContextVector(
                id=uuid4(),
                content=content,
                embedding=embedding,
                context_type=context_type,
                scope=scope,
                importance=importance,
                last_accessed=datetime.now(timezone.utc),
                access_count=1,
                related_entities=related_entities or [],
                metadata=metadata or {}
            )
            
            # Add to graph
            self.context_graph.nodes[vector.id] = vector
            
            # Update connections
            await self._update_connections(vector)
            
            # Update clusters
            await self._update_clusters(vector)
            
            # Save to cache
            await self._save_graph_to_cache()
            
            logger.info(f"Added context vector {vector.id} of type {context_type.value}")
            return vector.id
            
        except Exception as e:
            logger.error(f"Error adding context: {e}")
            return None
    
    async def _update_connections(self, new_vector: ContextVector):
        """Update graph connections for new vector"""
        try:
            edges = []
            
            # Find similar vectors
            for vector in self.context_graph.nodes.values():
                if vector.id != new_vector.id:
                    similarity = self._calculate_similarity(
                        new_vector.embedding, vector.embedding
                    )
                    if similarity > self.similarity_threshold:
                        edges.append((vector.id, similarity))
            
            # Sort and limit connections
            edges.sort(key=lambda x: x[1], reverse=True)
            self.context_graph.edges[new_vector.id] = edges[:10]
            
            # Update existing connections to include new vector
            for vector_id, existing_edges in self.context_graph.edges.items():
                if vector_id != new_vector.id:
                    # Check if new vector should connect to this one
                    existing_vector = self.context_graph.nodes[vector_id]
                    similarity = self._calculate_similarity(
                        new_vector.embedding, existing_vector.embedding
                    )
                    if similarity > self.similarity_threshold:
                        # Add connection and re-sort
                        existing_edges.append((new_vector.id, similarity))
                        existing_edges.sort(key=lambda x: x[1], reverse=True)
                        self.context_graph.edges[vector_id] = existing_edges[:10]
            
        except Exception as e:
            logger.error(f"Error updating connections: {e}")
    
    async def _update_clusters(self, new_vector: ContextVector):
        """Update clusters with new vector"""
        try:
            # Find best cluster for new vector
            best_cluster = None
            best_similarity = 0.0
            
            for cluster in self.context_graph.clusters.values():
                if cluster.centroid:
                    similarity = self._calculate_similarity(
                        new_vector.embedding, cluster.centroid
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = cluster
            
            # Add to cluster if similarity is high enough
            if best_cluster and best_similarity > self.similarity_threshold:
                best_cluster.vectors.append(new_vector)
                best_cluster.last_updated = datetime.now(timezone.utc)
                
                # Recalculate centroid
                embeddings = [v.embedding for v in best_cluster.vectors]
                best_cluster.centroid = await self._calculate_centroid(embeddings)
                
                # Recalculate coherence
                best_cluster.coherence_score = await self._calculate_coherence_score(
                    best_cluster.vectors
                )
            
        except Exception as e:
            logger.error(f"Error updating clusters: {e}")
    
    async def retrieve_context(self, query: str, context_type: Optional[ContextType] = None,
                             scope: Optional[ContextScope] = None,
                             limit: int = 10) -> List[ContextVector]:
        """Retrieve relevant context for a query"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Find similar vectors
            candidates = []
            for vector in self.context_graph.nodes.values():
                # Filter by type and scope if specified
                if context_type and vector.context_type != context_type:
                    continue
                if scope and vector.scope != scope:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(query_embedding, vector.embedding)
                
                # Boost by importance and access count
                boosted_score = similarity * (1 + vector.importance) * (1 + vector.access_count * 0.1)
                
                candidates.append((vector, boosted_score))
            
            # Sort by boosted score
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Update access patterns
            results = []
            for vector, score in candidates[:limit]:
                vector.access_count += 1
                vector.last_accessed = datetime.now(timezone.utc)
                results.append(vector)
            
            # Save updated access patterns
            await self._save_graph_to_cache()
            
            logger.info(f"Retrieved {len(results)} context vectors for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    async def get_context_summary(self, session_id: Optional[UUID] = None,
                                project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of available context"""
        try:
            summary = {
                "total_vectors": len(self.context_graph.nodes),
                "total_clusters": len(self.context_graph.clusters),
                "context_types": {},
                "scopes": {},
                "top_importance": [],
                "recent_access": []
            }
            
            # Analyze context types
            for vector in self.context_graph.nodes.values():
                context_type = vector.context_type.value
                if context_type not in summary["context_types"]:
                    summary["context_types"][context_type] = 0
                summary["context_types"][context_type] += 1
                
                scope = vector.scope.value
                if scope not in summary["scopes"]:
                    summary["scopes"][scope] = 0
                summary["scopes"][scope] += 1
            
            # Top importance vectors
            sorted_vectors = sorted(
                self.context_graph.nodes.values(),
                key=lambda v: v.importance,
                reverse=True
            )
            summary["top_importance"] = [
                {
                    "id": str(v.id),
                    "content": v.content[:100] + "..." if len(v.content) > 100 else v.content,
                    "importance": v.importance,
                    "type": v.context_type.value
                }
                for v in sorted_vectors[:5]
            ]
            
            # Recent access
            recently_accessed = sorted(
                self.context_graph.nodes.values(),
                key=lambda v: v.last_accessed,
                reverse=True
            )
            summary["recent_access"] = [
                {
                    "id": str(v.id),
                    "content": v.content[:100] + "..." if len(v.content) > 100 else v.content,
                    "last_accessed": v.last_accessed.isoformat(),
                    "access_count": v.access_count
                }
                for v in recently_accessed[:5]
            ]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting context summary: {e}")
            return {}
    
    async def _save_graph_to_cache(self):
        """Save context graph to Redis cache"""
        try:
            if not redis_client:
                return
            
            # Serialize graph (simplified version)
            graph_data = {
                "nodes": {
                    str(node_id): {
                        "id": str(vector.id),
                        "content": vector.content,
                        "context_type": vector.context_type.value,
                        "scope": vector.scope.value,
                        "importance": vector.importance,
                        "last_accessed": vector.last_accessed.isoformat(),
                        "access_count": vector.access_count,
                        "related_entities": vector.related_entities,
                        "metadata": vector.metadata
                    }
                    for node_id, vector in self.context_graph.nodes.items()
                },
                "clusters": {
                    str(cluster_id): {
                        "id": str(cluster.id),
                        "name": cluster.name,
                        "description": cluster.description,
                        "coherence_score": cluster.coherence_score,
                        "last_updated": cluster.last_updated.isoformat(),
                        "vector_count": len(cluster.vectors)
                    }
                    for cluster_id, cluster in self.context_graph.clusters.items()
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Save to Redis with expiration
            await redis_client.setex(
                self.graph_cache_key,
                86400,  # 24 hours
                json.dumps(graph_data)
            )
            
        except Exception as e:
            logger.warning(f"Error saving graph to cache: {e}")
    
    async def _load_graph_from_cache(self) -> Optional[ContextGraph]:
        """Load context graph from Redis cache"""
        try:
            if not redis_client:
                return None
            
            cached_data = await redis_client.get(self.graph_cache_key)
            if not cached_data:
                return None
            
            # Note: This is a simplified version that loads metadata only
            # Full implementation would need to rebuild embeddings and connections
            logger.info("Context graph cache found but full restoration not implemented")
            return None
            
        except Exception as e:
            logger.warning(f"Error loading graph from cache: {e}")
            return None
    
    async def decay_importance(self):
        """Apply importance decay over time"""
        try:
            current_time = datetime.now(timezone.utc)
            
            for vector in self.context_graph.nodes.values():
                # Calculate time since last access
                time_diff = current_time - vector.last_accessed
                weeks = time_diff.days / 7
                
                # Apply decay
                vector.importance *= (self.importance_decay_rate ** weeks)
                
                # Minimum importance threshold
                if vector.importance < 0.1:
                    vector.importance = 0.1
            
            logger.info("Applied importance decay to all vectors")
            
        except Exception as e:
            logger.error(f"Error applying importance decay: {e}")
    
    async def cleanup_old_context(self, max_age_days: int = 90):
        """Clean up old, unused context"""
        try:
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time - timedelta(days=max_age_days)
            
            to_remove = []
            for vector_id, vector in self.context_graph.nodes.items():
                if (vector.last_accessed < cutoff_time and 
                    vector.importance < 0.2 and 
                    vector.access_count < 3):
                    to_remove.append(vector_id)
            
            # Remove old vectors
            for vector_id in to_remove:
                del self.context_graph.nodes[vector_id]
                if vector_id in self.context_graph.edges:
                    del self.context_graph.edges[vector_id]
            
            # Clean up edges pointing to removed vectors
            for edges in self.context_graph.edges.values():
                edges[:] = [(node_id, weight) for node_id, weight in edges 
                           if node_id not in to_remove]
            
            logger.info(f"Cleaned up {len(to_remove)} old context vectors")
            
        except Exception as e:
            logger.error(f"Error cleaning up old context: {e}")
    
    async def get_context_analytics(self) -> Dict[str, Any]:
        """Get analytics about context usage and patterns"""
        try:
            analytics = {
                "total_vectors": len(self.context_graph.nodes),
                "total_clusters": len(self.context_graph.clusters),
                "avg_importance": 0.0,
                "avg_access_count": 0.0,
                "context_type_distribution": {},
                "scope_distribution": {},
                "cluster_health": {},
                "access_patterns": {},
                "memory_usage": 0
            }
            
            if not self.context_graph.nodes:
                return analytics
            
            # Calculate averages
            total_importance = sum(v.importance for v in self.context_graph.nodes.values())
            total_access = sum(v.access_count for v in self.context_graph.nodes.values())
            
            analytics["avg_importance"] = total_importance / len(self.context_graph.nodes)
            analytics["avg_access_count"] = total_access / len(self.context_graph.nodes)
            
            # Distribution analysis
            for vector in self.context_graph.nodes.values():
                # Context type distribution
                context_type = vector.context_type.value
                if context_type not in analytics["context_type_distribution"]:
                    analytics["context_type_distribution"][context_type] = 0
                analytics["context_type_distribution"][context_type] += 1
                
                # Scope distribution
                scope = vector.scope.value
                if scope not in analytics["scope_distribution"]:
                    analytics["scope_distribution"][scope] = 0
                analytics["scope_distribution"][scope] += 1
            
            # Cluster health
            for cluster in self.context_graph.clusters.values():
                analytics["cluster_health"][cluster.name] = {
                    "vector_count": len(cluster.vectors),
                    "coherence_score": cluster.coherence_score,
                    "last_updated": cluster.last_updated.isoformat()
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting context analytics: {e}")
            return {}


# Global instance
_persistent_context_manager = None


def get_persistent_context_manager(db: Session) -> PersistentContextManager:
    """Get global persistent context manager instance"""
    global _persistent_context_manager
    if _persistent_context_manager is None:
        _persistent_context_manager = PersistentContextManager(db)
    return _persistent_context_manager