"""Context Compression Strategies

This module implements multiple compression strategies to manage context windows
effectively while preserving the most important information for Claude-Code.

Strategies included:
1. Importance-based pruning
2. Recency weighting
3. Summarization compression
4. Entity consolidation
5. Semantic clustering
6. Hierarchical compression
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from uuid import UUID
from enum import Enum
import json
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from ..models import Memory, MemorySession, MemoryType
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Available compression strategies"""
    IMPORTANCE_BASED = "importance_based"
    RECENCY_WEIGHTED = "recency_weighted"
    SUMMARIZATION = "summarization"
    ENTITY_CONSOLIDATION = "entity_consolidation"
    SEMANTIC_CLUSTERING = "semantic_clustering"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


class CompressionLevel(Enum):
    """Compression intensity levels"""
    LIGHT = "light"      # 10-20% reduction
    MODERATE = "moderate" # 30-50% reduction
    AGGRESSIVE = "aggressive" # 60-80% reduction
    EXTREME = "extreme"   # 80-90% reduction


class CompressedContext:
    """Container for compressed context data"""
    
    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.key_entities: List[str] = []
        self.important_facts: List[str] = []
        self.recent_decisions: List[str] = []
        self.context_stats: Dict[str, Any] = {}
        self.compression_ratio: float = 0.0
        self.strategy_used: str = ""
        self.token_estimate: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "memories": self.memories,
            "summary": self.summary,
            "key_entities": self.key_entities,
            "important_facts": self.important_facts,
            "recent_decisions": self.recent_decisions,
            "context_stats": self.context_stats,
            "compression_ratio": self.compression_ratio,
            "strategy_used": self.strategy_used,
            "token_estimate": self.token_estimate
        }


class ContextCompressionService:
    """Service for compressing memory context using various strategies"""
    
    def __init__(self):
        self.max_context_tokens = 8000  # Conservative limit
        self.token_per_char_ratio = 0.25  # ~4 chars per token
        self.cache_ttl = 1800  # 30 minutes cache
    
    async def compress_context(
        self,
        db: Session,
        session_id: UUID,
        target_tokens: int = 4000,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        level: CompressionLevel = CompressionLevel.MODERATE
    ) -> CompressedContext:
        """Compress session context using specified strategy
        
        Args:
            db: Database session
            session_id: Session to compress
            target_tokens: Target token count
            strategy: Compression strategy to use
            level: Compression intensity
            
        Returns:
            CompressedContext with optimized content
        """
        # Check cache first
        cache_key = self._get_cache_key(session_id, target_tokens, strategy, level)
        cached = await self._get_cached_compression(cache_key)
        if cached:
            return cached
        
        # Get session and memories
        session = db.query(MemorySession).filter_by(id=session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        memories = db.query(Memory).filter_by(
            session_id=session_id
        ).order_by(desc(Memory.created_at)).all()
        
        if not memories:
            return CompressedContext()
        
        # Apply compression strategy
        if strategy == CompressionStrategy.IMPORTANCE_BASED:
            compressed = await self._importance_based_compression(
                memories, target_tokens, level
            )
        elif strategy == CompressionStrategy.RECENCY_WEIGHTED:
            compressed = await self._recency_weighted_compression(
                memories, target_tokens, level
            )
        elif strategy == CompressionStrategy.SUMMARIZATION:
            compressed = await self._summarization_compression(
                memories, target_tokens, level
            )
        elif strategy == CompressionStrategy.ENTITY_CONSOLIDATION:
            compressed = await self._entity_consolidation_compression(
                memories, target_tokens, level
            )
        elif strategy == CompressionStrategy.SEMANTIC_CLUSTERING:
            compressed = await self._semantic_clustering_compression(
                memories, target_tokens, level
            )
        elif strategy == CompressionStrategy.HIERARCHICAL:
            compressed = await self._hierarchical_compression(
                memories, target_tokens, level
            )
        else:  # HYBRID
            compressed = await self._hybrid_compression(
                memories, target_tokens, level
            )
        
        # Add metadata
        compressed.strategy_used = strategy.value
        compressed.compression_ratio = self._calculate_compression_ratio(
            memories, compressed
        )
        compressed.token_estimate = self._estimate_tokens(compressed)
        
        # Cache result
        await self._cache_compression(cache_key, compressed)
        
        logger.info(
            f"Compressed context for session {session_id}: "
            f"{len(memories)} -> {len(compressed.memories)} memories, "
            f"ratio: {compressed.compression_ratio:.2f}, "
            f"tokens: {compressed.token_estimate}"
        )
        
        return compressed
    
    async def _importance_based_compression(
        self,
        memories: List[Memory],
        target_tokens: int,
        level: CompressionLevel
    ) -> CompressedContext:
        """Compress based on memory importance scores"""
        compressed = CompressedContext()
        
        # Sort by importance and recency
        sorted_memories = sorted(
            memories,
            key=lambda m: (m.importance, m.created_at),
            reverse=True
        )
        
        # Determine threshold based on compression level
        thresholds = {
            CompressionLevel.LIGHT: 0.7,
            CompressionLevel.MODERATE: 0.8,
            CompressionLevel.AGGRESSIVE: 0.85,
            CompressionLevel.EXTREME: 0.9
        }
        
        importance_threshold = thresholds[level]
        current_tokens = 0
        
        for memory in sorted_memories:
            if memory.importance >= importance_threshold:
                memory_tokens = self._estimate_memory_tokens(memory)
                if current_tokens + memory_tokens <= target_tokens:
                    compressed.memories.append(self._memory_to_dict(memory))
                    current_tokens += memory_tokens
                else:
                    # Summarize remaining high-importance memories
                    if memory.importance >= 0.9:
                        compressed.important_facts.append(
                            memory.summary or memory.content[:100] + "..."
                        )
        
        # Generate summary from remaining memories
        compressed.summary = self._generate_summary(sorted_memories[len(compressed.memories):])
        
        return compressed
    
    async def _recency_weighted_compression(
        self,
        memories: List[Memory],
        target_tokens: int,
        level: CompressionLevel
    ) -> CompressedContext:
        """Compress with recency weighting"""
        compressed = CompressedContext()
        
        # Calculate recency scores
        now = datetime.now(timezone.utc)
        weighted_memories = []
        
        for memory in memories:
            hours_ago = (now - memory.created_at).total_seconds() / 3600
            # Exponential decay: newer memories have higher scores
            recency_score = memory.importance * (0.95 ** hours_ago)
            weighted_memories.append((memory, recency_score))
        
        # Sort by weighted score
        weighted_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Select memories based on compression level
        selection_ratios = {
            CompressionLevel.LIGHT: 0.8,
            CompressionLevel.MODERATE: 0.6,
            CompressionLevel.AGGRESSIVE: 0.4,
            CompressionLevel.EXTREME: 0.2
        }
        
        selection_count = int(len(weighted_memories) * selection_ratios[level])
        current_tokens = 0
        
        for memory, score in weighted_memories[:selection_count]:
            memory_tokens = self._estimate_memory_tokens(memory)
            if current_tokens + memory_tokens <= target_tokens:
                compressed.memories.append(self._memory_to_dict(memory))
                current_tokens += memory_tokens
            else:
                break
        
        # Extract key information from remaining memories
        remaining = [m for m, s in weighted_memories[selection_count:]]
        compressed.key_entities = self._extract_entities(remaining)
        compressed.summary = self._generate_summary(remaining)
        
        return compressed
    
    async def _summarization_compression(
        self,
        memories: List[Memory],
        target_tokens: int,
        level: CompressionLevel
    ) -> CompressedContext:
        """Compress by creating summaries of memory groups"""
        compressed = CompressedContext()
        
        # Group memories by type and time windows
        memory_groups = self._group_memories_for_summarization(memories)
        
        current_tokens = 0
        preserved_memories = []
        
        for group_type, group_memories in memory_groups.items():
            if group_type in [MemoryType.DECISION.value, MemoryType.ERROR.value]:
                # Preserve important memory types
                for memory in group_memories:
                    if memory.importance >= 0.8:
                        memory_tokens = self._estimate_memory_tokens(memory)
                        if current_tokens + memory_tokens <= target_tokens * 0.6:
                            preserved_memories.append(memory)
                            current_tokens += memory_tokens
            
            # Summarize the rest
            if len(group_memories) > 1:
                summary = self._create_group_summary(group_memories, group_type)
                if group_type == MemoryType.DECISION.value:
                    compressed.recent_decisions.append(summary)
                elif group_type == MemoryType.FACT.value:
                    compressed.important_facts.append(summary)
        
        compressed.memories = [self._memory_to_dict(m) for m in preserved_memories]
        compressed.summary = self._generate_overall_summary(memory_groups)
        
        return compressed
    
    async def _entity_consolidation_compression(
        self,
        memories: List[Memory],
        target_tokens: int,
        level: CompressionLevel
    ) -> CompressedContext:
        """Compress by consolidating entity-related memories"""
        compressed = CompressedContext()
        
        # Group memories by entities
        entity_groups = self._group_memories_by_entities(memories)
        
        # Consolidate information about each entity
        entity_summaries = {}
        current_tokens = 0
        preserved_memories = []
        
        for entity, entity_memories in entity_groups.items():
            # Keep the most important memory about each entity
            best_memory = max(entity_memories, key=lambda m: m.importance)
            
            memory_tokens = self._estimate_memory_tokens(best_memory)
            if current_tokens + memory_tokens <= target_tokens:
                preserved_memories.append(best_memory)
                current_tokens += memory_tokens
            else:
                # Create entity summary from all related memories
                entity_summaries[entity] = self._create_entity_summary(
                    entity, entity_memories
                )
        
        compressed.memories = [self._memory_to_dict(m) for m in preserved_memories]
        compressed.key_entities = list(entity_summaries.keys())
        
        # Add entity summaries to facts
        for entity, summary in entity_summaries.items():
            compressed.important_facts.append(f"{entity}: {summary}")
        
        return compressed
    
    async def _semantic_clustering_compression(
        self,
        memories: List[Memory],
        target_tokens: int,
        level: CompressionLevel
    ) -> CompressedContext:
        """Compress by clustering semantically similar memories"""
        compressed = CompressedContext()
        
        # Simple clustering based on shared entities and keywords
        clusters = self._create_semantic_clusters(memories)
        
        current_tokens = 0
        preserved_memories = []
        
        for cluster in clusters:
            # Keep the most representative memory from each cluster
            representative = self._find_cluster_representative(cluster)
            
            memory_tokens = self._estimate_memory_tokens(representative)
            if current_tokens + memory_tokens <= target_tokens:
                preserved_memories.append(representative)
                current_tokens += memory_tokens
            else:
                # Summarize the cluster
                cluster_summary = self._create_cluster_summary(cluster)
                compressed.important_facts.append(cluster_summary)
        
        compressed.memories = [self._memory_to_dict(m) for m in preserved_memories]
        
        return compressed
    
    async def _hierarchical_compression(
        self,
        memories: List[Memory],
        target_tokens: int,
        level: CompressionLevel
    ) -> CompressedContext:
        """Compress using hierarchical importance levels"""
        compressed = CompressedContext()
        
        # Create importance tiers
        critical_memories = [m for m in memories if m.importance >= 0.9]
        important_memories = [m for m in memories if 0.7 <= m.importance < 0.9]
        regular_memories = [m for m in memories if m.importance < 0.7]
        
        current_tokens = 0
        
        # Always include critical memories
        for memory in critical_memories:
            memory_tokens = self._estimate_memory_tokens(memory)
            if current_tokens + memory_tokens <= target_tokens * 0.5:
                compressed.memories.append(self._memory_to_dict(memory))
                current_tokens += memory_tokens
        
        # Include important memories based on available space
        remaining_tokens = target_tokens - current_tokens
        for memory in important_memories:
            memory_tokens = self._estimate_memory_tokens(memory)
            if memory_tokens <= remaining_tokens * 0.7:
                compressed.memories.append(self._memory_to_dict(memory))
                current_tokens += memory_tokens
                remaining_tokens -= memory_tokens
        
        # Summarize regular memories
        if regular_memories:
            compressed.summary = self._generate_summary(regular_memories)
        
        # Extract key entities from all memories
        compressed.key_entities = self._extract_entities(memories)
        
        return compressed
    
    async def _hybrid_compression(
        self,
        memories: List[Memory],
        target_tokens: int,
        level: CompressionLevel
    ) -> CompressedContext:
        """Combine multiple compression strategies"""
        # Start with importance-based selection
        importance_compressed = await self._importance_based_compression(
            memories, target_tokens, level
        )
        
        # Apply recency weighting to remaining space
        if len(importance_compressed.memories) * 100 < target_tokens:
            recency_compressed = await self._recency_weighted_compression(
                memories, target_tokens - len(importance_compressed.memories) * 100, level
            )
            
            # Merge results
            seen_ids = {m['id'] for m in importance_compressed.memories}
            for memory in recency_compressed.memories:
                if memory['id'] not in seen_ids:
                    importance_compressed.memories.append(memory)
        
        # Apply entity consolidation for summary
        entity_compressed = await self._entity_consolidation_compression(
            memories, target_tokens, level
        )
        
        # Combine summaries and entities
        importance_compressed.key_entities = entity_compressed.key_entities
        importance_compressed.important_facts.extend(entity_compressed.important_facts)
        
        # Generate comprehensive summary
        remaining_memories = [
            m for m in memories 
            if str(m.id) not in {mem['id'] for mem in importance_compressed.memories}
        ]
        
        if remaining_memories:
            importance_compressed.summary = self._generate_summary(remaining_memories)
        
        return importance_compressed
    
    def _group_memories_for_summarization(self, memories: List[Memory]) -> Dict[str, List[Memory]]:
        """Group memories by type for summarization"""
        groups = {}
        for memory in memories:
            memory_type = memory.memory_type
            if memory_type not in groups:
                groups[memory_type] = []
            groups[memory_type].append(memory)
        return groups
    
    def _group_memories_by_entities(self, memories: List[Memory]) -> Dict[str, List[Memory]]:
        """Group memories by shared entities"""
        entity_groups = {}
        
        for memory in memories:
            if memory.entities:
                for entity in memory.entities:
                    if entity not in entity_groups:
                        entity_groups[entity] = []
                    entity_groups[entity].append(memory)
        
        return entity_groups
    
    def _create_semantic_clusters(self, memories: List[Memory]) -> List[List[Memory]]:
        """Create clusters based on semantic similarity"""
        clusters = []
        processed = set()
        
        for memory in memories:
            if memory.id in processed:
                continue
            
            cluster = [memory]
            processed.add(memory.id)
            
            # Find similar memories based on shared entities
            if memory.entities:
                for other_memory in memories:
                    if other_memory.id in processed:
                        continue
                    
                    if other_memory.entities:
                        shared_entities = set(memory.entities) & set(other_memory.entities)
                        if len(shared_entities) >= 2:  # At least 2 shared entities
                            cluster.append(other_memory)
                            processed.add(other_memory.id)
            
            clusters.append(cluster)
        
        return clusters
    
    def _find_cluster_representative(self, cluster: List[Memory]) -> Memory:
        """Find the most representative memory in a cluster"""
        return max(cluster, key=lambda m: m.importance)
    
    def _create_cluster_summary(self, cluster: List[Memory]) -> str:
        """Create a summary for a memory cluster"""
        if len(cluster) == 1:
            return cluster[0].summary or cluster[0].content[:100] + "..."
        
        # Extract common themes
        all_entities = []
        for memory in cluster:
            if memory.entities:
                all_entities.extend(memory.entities)
        
        common_entities = []
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        for entity, count in entity_counts.items():
            if count >= len(cluster) * 0.5:  # At least half the memories mention it
                common_entities.append(entity)
        
        # Create summary
        entity_str = ", ".join(common_entities[:3]) if common_entities else "various topics"
        return f"Discussion about {entity_str} ({len(cluster)} related memories)"
    
    def _create_group_summary(self, group_memories: List[Memory], group_type: str) -> str:
        """Create summary for a group of memories of the same type"""
        if not group_memories:
            return ""
        
        if group_type == MemoryType.DECISION.value:
            decisions = [m.summary or m.content[:50] for m in group_memories]
            return f"Decisions made: {'; '.join(decisions[:3])}"
        elif group_type == MemoryType.ERROR.value:
            errors = [m.summary or m.content[:50] for m in group_memories]
            return f"Errors encountered: {'; '.join(errors[:3])}"
        elif group_type == MemoryType.FACT.value:
            facts = [m.summary or m.content[:50] for m in group_memories]
            return f"Key facts: {'; '.join(facts[:3])}"
        else:
            return f"{len(group_memories)} {group_type} memories"
    
    def _create_entity_summary(self, entity: str, entity_memories: List[Memory]) -> str:
        """Create summary for memories related to a specific entity"""
        important_points = []
        for memory in entity_memories:
            if memory.importance >= 0.8:
                point = memory.summary or memory.content[:50]
                important_points.append(point)
        
        if important_points:
            return "; ".join(important_points[:2])
        else:
            return f"Mentioned in {len(entity_memories)} memories"
    
    def _generate_summary(self, memories: List[Memory]) -> str:
        """Generate a summary from a list of memories"""
        if not memories:
            return ""
        
        # Extract key themes
        memory_types = {}
        all_entities = []
        
        for memory in memories:
            memory_type = memory.memory_type
            memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
            
            if memory.entities:
                all_entities.extend(memory.entities)
        
        # Find most common entities
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create summary
        summary_parts = []
        
        if memory_types:
            type_summary = ", ".join(f"{count} {mtype}" for mtype, count in memory_types.items())
            summary_parts.append(f"Contains {type_summary} memories")
        
        if top_entities:
            entity_list = ", ".join(entity for entity, count in top_entities)
            summary_parts.append(f"Key topics: {entity_list}")
        
        return ". ".join(summary_parts)
    
    def _generate_overall_summary(self, memory_groups: Dict[str, List[Memory]]) -> str:
        """Generate overall summary from memory groups"""
        summary_parts = []
        
        for group_type, group_memories in memory_groups.items():
            if group_memories:
                summary_parts.append(f"{len(group_memories)} {group_type} memories")
        
        return ", ".join(summary_parts)
    
    def _extract_entities(self, memories: List[Memory]) -> List[str]:
        """Extract unique entities from memories"""
        entities = set()
        for memory in memories:
            if memory.entities:
                entities.update(memory.entities)
        return list(entities)[:20]  # Limit to top 20
    
    def _memory_to_dict(self, memory: Memory) -> Dict[str, Any]:
        """Convert memory to dictionary format"""
        return {
            "id": str(memory.id),
            "content": memory.content,
            "summary": memory.summary,
            "memory_type": memory.memory_type,
            "importance": memory.importance,
            "entities": memory.entities or [],
            "created_at": memory.created_at.isoformat()
        }
    
    def _estimate_memory_tokens(self, memory: Memory) -> int:
        """Estimate token count for a memory"""
        content_chars = len(memory.content) + len(memory.summary or "")
        return int(content_chars * self.token_per_char_ratio)
    
    def _estimate_tokens(self, compressed: CompressedContext) -> int:
        """Estimate total tokens in compressed context"""
        total_chars = 0
        
        for memory in compressed.memories:
            total_chars += len(memory["content"]) + len(memory.get("summary", ""))
        
        total_chars += len(compressed.summary)
        total_chars += sum(len(fact) for fact in compressed.important_facts)
        total_chars += sum(len(decision) for decision in compressed.recent_decisions)
        
        return int(total_chars * self.token_per_char_ratio)
    
    def _calculate_compression_ratio(self, original: List[Memory], compressed: CompressedContext) -> float:
        """Calculate compression ratio"""
        if not original:
            return 0.0
        
        original_size = sum(len(m.content) + len(m.summary or "") for m in original)
        compressed_size = self._estimate_tokens(compressed) / self.token_per_char_ratio
        
        if original_size == 0:
            return 0.0
        
        return 1.0 - (compressed_size / original_size)
    
    def _get_cache_key(self, session_id: UUID, target_tokens: int, strategy: CompressionStrategy, level: CompressionLevel) -> str:
        """Generate cache key for compression"""
        key_data = f"{session_id}:{target_tokens}:{strategy.value}:{level.value}"
        return f"compression:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def _get_cached_compression(self, cache_key: str) -> Optional[CompressedContext]:
        """Get cached compression result"""
        try:
            if redis_client.client:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    compressed = CompressedContext()
                    compressed.__dict__.update(cached_data)
                    return compressed
        except Exception as e:
            logger.warning(f"Failed to get cached compression: {e}")
        
        return None
    
    async def _cache_compression(self, cache_key: str, compressed: CompressedContext):
        """Cache compression result"""
        try:
            if redis_client.client:
                await redis_client.set(cache_key, compressed.to_dict(), self.cache_ttl)
        except Exception as e:
            logger.warning(f"Failed to cache compression: {e}")


# Global instance
context_compression_service = ContextCompressionService()
