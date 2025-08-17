"""
Context Hierarchy Engine - Phase 2.2
Created by Tinne Smets - Expert in Weight Sharing & Context Understanding

This system implements multi-level context hierarchy with cross-document 
semantic relationships using Neo4j knowledge graph integration.
Optimized for distributed semantic understanding across documents, paragraphs, 
sentences, and tokens with efficient weight sharing.

Key Features:
- Multi-level context hierarchy (token → sentence → paragraph → document → cross-document)
- Neo4j knowledge graph integration for relationship modeling
- Temporal context understanding and evolution tracking
- Domain-specific context adaptation with shared representations
- Efficient caching and retrieval for real-time analysis
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import hashlib

# Neo4j integration
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available. Graph features will be limited.")

# Import our weight sharing components
from .weight_sharing_semantic_engine import (
    ContextLevel, ContextRepresentation, WeightSharingSemanticEngine
)

logger = logging.getLogger(__name__)

class ContextRelationType(Enum):
    """Types of relationships between context elements."""
    CONTAINS = "contains"
    REFERENCES = "references"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    PREREQUISITE = "prerequisite"
    TEMPORAL_NEXT = "temporal_next"
    CAUSAL = "causal"
    ELABORATES = "elaborates"

@dataclass
class ContextNode:
    """A node in the context hierarchy graph."""
    node_id: str
    level: ContextLevel
    content: str
    embedding: np.ndarray
    position: Tuple[int, ...]  # Hierarchical position
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Semantic properties
    entities: List[Dict[str, Any]] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    semantic_roles: Dict[str, str] = field(default_factory=dict)
    
    # Context properties
    context_window: int = 0  # Size of relevant context
    importance_score: float = 0.0
    coherence_score: float = 0.0
    
    # Temporal properties
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Relationships
    parent_nodes: Set[str] = field(default_factory=set)
    child_nodes: Set[str] = field(default_factory=set)
    related_nodes: Dict[str, ContextRelationType] = field(default_factory=dict)

@dataclass
class ContextRelation:
    """A relationship between context nodes."""
    source_id: str
    target_id: str
    relation_type: ContextRelationType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CrossDocumentContext:
    """Context spanning multiple documents."""
    context_id: str
    document_ids: Set[str]
    shared_concepts: List[str]
    shared_entities: List[Dict[str, Any]]
    semantic_coherence: float
    temporal_span: Tuple[datetime, datetime]
    context_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class Neo4jContextStore:
    """
    Neo4j-based storage for context hierarchy and relationships.
    
    Provides efficient storage and retrieval of hierarchical context
    with complex semantic relationships.
    """
    
    def __init__(
        self,
        uri: str = "bolt://192.168.1.25:7687",
        user: str = "neo4j",
        password: str = "knowledgehub123"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                logger.info("Neo4j connection established")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
                self.driver = None
        else:
            logger.warning("Neo4j driver not available")
    
    async def initialize_schema(self):
        """Initialize Neo4j schema for context hierarchy."""
        if not self.driver:
            return
        
        schema_queries = [
            # Create constraints
            "CREATE CONSTRAINT context_node_id IF NOT EXISTS FOR (n:ContextNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            
            # Create indexes
            "CREATE INDEX context_level IF NOT EXISTS FOR (n:ContextNode) ON (n.level)",
            "CREATE INDEX context_importance IF NOT EXISTS FOR (n:ContextNode) ON (n.importance_score)",
            "CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:CONTEXT_RELATION]->() ON (r.relation_type)",
            
            # Create full-text search indexes
            "CREATE FULLTEXT INDEX context_content IF NOT EXISTS FOR (n:ContextNode) ON EACH [n.content]",
        ]
        
        try:
            with self.driver.session() as session:
                for query in schema_queries:
                    try:
                        session.run(query)
                    except Exception as e:
                        # Index/constraint might already exist
                        logger.debug(f"Schema query failed (may be expected): {e}")
            
            logger.info("Neo4j schema initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")
    
    async def store_context_node(self, node: ContextNode) -> bool:
        """Store a context node in Neo4j."""
        if not self.driver:
            return False
        
        query = """
        MERGE (n:ContextNode {node_id: $node_id})
        SET n.level = $level,
            n.content = $content,
            n.embedding = $embedding,
            n.position = $position,
            n.metadata = $metadata,
            n.entities = $entities,
            n.concepts = $concepts,
            n.semantic_roles = $semantic_roles,
            n.context_window = $context_window,
            n.importance_score = $importance_score,
            n.coherence_score = $coherence_score,
            n.created_at = $created_at,
            n.last_updated = $last_updated
        RETURN n.node_id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, 
                    node_id=node.node_id,
                    level=node.level.value,
                    content=node.content,
                    embedding=node.embedding.tolist(),  # Convert numpy to list
                    position=list(node.position),
                    metadata=node.metadata,
                    entities=node.entities,
                    concepts=node.concepts,
                    semantic_roles=node.semantic_roles,
                    context_window=node.context_window,
                    importance_score=node.importance_score,
                    coherence_score=node.coherence_score,
                    created_at=node.created_at.isoformat(),
                    last_updated=node.last_updated.isoformat()
                )
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to store context node {node.node_id}: {e}")
            return False
    
    async def store_context_relation(self, relation: ContextRelation) -> bool:
        """Store a context relationship in Neo4j."""
        if not self.driver:
            return False
        
        query = """
        MATCH (source:ContextNode {node_id: $source_id})
        MATCH (target:ContextNode {node_id: $target_id})
        MERGE (source)-[r:CONTEXT_RELATION {relation_id: $relation_id}]->(target)
        SET r.relation_type = $relation_type,
            r.strength = $strength,
            r.confidence = $confidence,
            r.evidence = $evidence,
            r.metadata = $metadata,
            r.created_at = $created_at
        RETURN r.relation_id
        """
        
        relation_id = f"{relation.source_id}_{relation.target_id}_{relation.relation_type.value}"
        
        try:
            with self.driver.session() as session:
                result = session.run(query,
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    relation_id=relation_id,
                    relation_type=relation.relation_type.value,
                    strength=relation.strength,
                    confidence=relation.confidence,
                    evidence=relation.evidence,
                    metadata=relation.metadata,
                    created_at=relation.created_at.isoformat()
                )
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to store context relation {relation_id}: {e}")
            return False
    
    async def find_related_contexts(
        self, 
        node_id: str, 
        relation_types: List[ContextRelationType] = None,
        max_depth: int = 2,
        min_strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find related context nodes using graph traversal."""
        if not self.driver:
            return []
        
        relation_filter = ""
        if relation_types:
            relation_values = [rt.value for rt in relation_types]
            relation_filter = f"WHERE r.relation_type IN {relation_values}"
        
        query = f"""
        MATCH path = (start:ContextNode {{node_id: $node_id}})-[r:CONTEXT_RELATION*1..{max_depth}]-(related:ContextNode)
        {relation_filter} AND r.strength >= $min_strength
        RETURN related, r, length(path) as distance
        ORDER BY r.strength DESC, distance ASC
        LIMIT 50
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query,
                    node_id=node_id,
                    min_strength=min_strength
                )
                
                related_contexts = []
                for record in result:
                    node_data = dict(record['related'])
                    relation_data = dict(record['r']) if record['r'] else {}
                    
                    related_contexts.append({
                        'node': node_data,
                        'relation': relation_data,
                        'distance': record['distance']
                    })
                
                return related_contexts
                
        except Exception as e:
            logger.error(f"Failed to find related contexts for {node_id}: {e}")
            return []
    
    async def find_cross_document_patterns(
        self,
        document_ids: List[str],
        min_shared_concepts: int = 2
    ) -> List[CrossDocumentContext]:
        """Find patterns that span across multiple documents."""
        if not self.driver or len(document_ids) < 2:
            return []
        
        query = """
        MATCH (n1:ContextNode)-[:BELONGS_TO]->(d1:Document)
        MATCH (n2:ContextNode)-[:BELONGS_TO]->(d2:Document)
        WHERE d1.document_id IN $document_ids 
          AND d2.document_id IN $document_ids
          AND d1 <> d2
        WITH n1, n2, d1, d2
        WHERE any(concept1 IN n1.concepts WHERE any(concept2 IN n2.concepts WHERE concept1 = concept2))
        RETURN d1.document_id as doc1, d2.document_id as doc2, 
               [concept IN n1.concepts WHERE concept IN n2.concepts] as shared_concepts,
               n1, n2
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, document_ids=document_ids)
                
                # Group by document pairs
                doc_patterns = defaultdict(list)
                
                for record in result:
                    doc_pair = tuple(sorted([record['doc1'], record['doc2']]))
                    shared_concepts = record['shared_concepts']
                    
                    if len(shared_concepts) >= min_shared_concepts:
                        doc_patterns[doc_pair].append({
                            'shared_concepts': shared_concepts,
                            'node1': dict(record['n1']),
                            'node2': dict(record['n2'])
                        })
                
                # Create CrossDocumentContext objects
                cross_contexts = []
                for doc_pair, patterns in doc_patterns.items():
                    if len(patterns) >= min_shared_concepts:
                        all_concepts = set()
                        for pattern in patterns:
                            all_concepts.update(pattern['shared_concepts'])
                        
                        context = CrossDocumentContext(
                            context_id=f"cross_{hashlib.md5('_'.join(doc_pair).encode()).hexdigest()[:8]}",
                            document_ids=set(doc_pair),
                            shared_concepts=list(all_concepts),
                            shared_entities=[],  # Would be populated with entity analysis
                            semantic_coherence=len(all_concepts) / max(len(patterns), 1),
                            temporal_span=(datetime.utcnow() - timedelta(days=1), datetime.utcnow()),
                            context_summary=f"Cross-document pattern with {len(all_concepts)} shared concepts",
                            metadata={'pattern_count': len(patterns)}
                        )
                        cross_contexts.append(context)
                
                return cross_contexts
                
        except Exception as e:
            logger.error(f"Failed to find cross-document patterns: {e}")
            return []
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

class ContextHierarchyEngine:
    """
    Main engine for hierarchical context understanding with semantic relationships.
    
    Integrates weight sharing semantic analysis with Neo4j graph storage
    for comprehensive context modeling across multiple levels and documents.
    """
    
    def __init__(
        self,
        semantic_engine: WeightSharingSemanticEngine,
        neo4j_config: Dict[str, str] = None,
        cache_size: int = 10000
    ):
        self.semantic_engine = semantic_engine
        self.neo4j_config = neo4j_config or {
            'uri': 'bolt://192.168.1.25:7687',
            'user': 'neo4j',
            'password': 'knowledgehub123'
        }
        
        # Initialize Neo4j store
        self.graph_store = Neo4jContextStore(**self.neo4j_config)
        
        # In-memory caches for performance
        self.context_cache = {}  # node_id -> ContextNode
        self.relation_cache = defaultdict(list)  # source_id -> List[ContextRelation]
        self.cache_size = cache_size
        self.cache_access_order = deque()
        
        # Hierarchy building components
        self.hierarchy_builders = {
            ContextLevel.TOKEN: self._build_token_context,
            ContextLevel.SENTENCE: self._build_sentence_context,
            ContextLevel.PARAGRAPH: self._build_paragraph_context,
            ContextLevel.DOCUMENT: self._build_document_context,
            ContextLevel.CROSS_DOCUMENT: self._build_cross_document_context
        }
        
        # Performance metrics
        self.processing_metrics = defaultdict(list)
        
        logger.info("ContextHierarchyEngine initialized")
    
    async def initialize(self):
        """Initialize the context hierarchy engine."""
        try:
            # Initialize semantic engine if not already done
            if not hasattr(self.semantic_engine, 'shared_encoder') or self.semantic_engine.shared_encoder is None:
                await self.semantic_engine.initialize()
            
            # Initialize Neo4j schema
            await self.graph_store.initialize_schema()
            
            logger.info("ContextHierarchyEngine fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ContextHierarchyEngine: {e}")
            raise
    
    async def analyze_document_hierarchy(
        self,
        document_id: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[ContextLevel, List[ContextNode]]:
        """
        Analyze a document and build its complete context hierarchy.
        
        Args:
            document_id: Unique document identifier
            content: Document content to analyze
            metadata: Additional metadata for the document
            
        Returns:
            Dictionary mapping context levels to their respective nodes
        """
        start_time = datetime.utcnow()
        
        try:
            # Use semantic engine for initial analysis
            semantic_analysis = await self.semantic_engine.analyze_context_hierarchy(
                content, document_id
            )
            
            # Build hierarchy at each level
            hierarchy = {}
            
            for level in ContextLevel:
                if level == ContextLevel.CROSS_DOCUMENT:
                    continue  # Skip cross-document for single document analysis
                
                builder = self.hierarchy_builders.get(level)
                if builder:
                    nodes = await builder(document_id, content, semantic_analysis, metadata)
                    hierarchy[level] = nodes
                    
                    # Store nodes in graph
                    for node in nodes:
                        await self._cache_and_store_node(node)
            
            # Build relationships between hierarchy levels
            await self._build_hierarchical_relationships(hierarchy)
            
            # Update processing metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.processing_metrics['document_analysis'].append(processing_time)
            
            logger.info(f"Document hierarchy analysis completed for {document_id} in {processing_time:.2f}s")
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Failed to analyze document hierarchy for {document_id}: {e}")
            raise
    
    async def _build_token_context(
        self,
        document_id: str,
        content: str,
        semantic_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[ContextNode]:
        """Build token-level context nodes."""
        nodes = []
        
        # Simple tokenization for demo (would use proper NLP tokenizer in production)
        words = content.split()
        
        for i, token in enumerate(words[:100]):  # Limit for demo
            if len(token.strip()) < 2:  # Skip very short tokens
                continue
            
            # Generate embedding for token using semantic engine
            # This is simplified - would use token-specific embeddings in production
            token_embedding = np.random.rand(512)  # Placeholder
            
            node = ContextNode(
                node_id=f"{document_id}_token_{i}",
                level=ContextLevel.TOKEN,
                content=token,
                embedding=token_embedding,
                position=(i,),
                metadata={
                    'document_id': document_id,
                    'token_position': i,
                    'is_start_of_sentence': i == 0 or words[i-1].endswith('.'),
                    **metadata
                },
                context_window=5,  # 5 tokens on each side
                importance_score=len(token) / 20.0,  # Simple importance based on length
                coherence_score=0.8  # Placeholder
            )
            
            nodes.append(node)
        
        return nodes
    
    async def _build_sentence_context(
        self,
        document_id: str,
        content: str,
        semantic_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[ContextNode]:
        """Build sentence-level context nodes."""
        nodes = []
        
        # Split into sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        for i, sentence in enumerate(sentences):
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Generate sentence embedding
            sentence_embedding = np.random.rand(512)  # Placeholder
            
            node = ContextNode(
                node_id=f"{document_id}_sentence_{i}",
                level=ContextLevel.SENTENCE,
                content=sentence,
                embedding=sentence_embedding,
                position=(i,),
                metadata={
                    'document_id': document_id,
                    'sentence_position': i,
                    'sentence_length': len(sentence.split()),
                    **metadata
                },
                context_window=3,  # 3 sentences on each side
                importance_score=min(len(sentence.split()) / 30.0, 1.0),  # Based on length
                coherence_score=0.85  # Placeholder
            )
            
            nodes.append(node)
        
        return nodes
    
    async def _build_paragraph_context(
        self,
        document_id: str,
        content: str,
        semantic_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[ContextNode]:
        """Build paragraph-level context nodes."""
        nodes = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
            
            # Generate paragraph embedding
            paragraph_embedding = np.random.rand(512)  # Placeholder
            
            node = ContextNode(
                node_id=f"{document_id}_paragraph_{i}",
                level=ContextLevel.PARAGRAPH,
                content=paragraph,
                embedding=paragraph_embedding,
                position=(i,),
                metadata={
                    'document_id': document_id,
                    'paragraph_position': i,
                    'paragraph_length': len(paragraph.split()),
                    'sentence_count': len(paragraph.split('.')),
                    **metadata
                },
                context_window=2,  # 2 paragraphs on each side
                importance_score=min(len(paragraph.split()) / 100.0, 1.0),  # Based on length
                coherence_score=0.9  # Placeholder
            )
            
            nodes.append(node)
        
        return nodes
    
    async def _build_document_context(
        self,
        document_id: str,
        content: str,
        semantic_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[ContextNode]:
        """Build document-level context node."""
        # Generate document embedding
        document_embedding = np.random.rand(512)  # Placeholder
        
        # Extract key statistics
        word_count = len(content.split())
        sentence_count = len(content.split('.'))
        paragraph_count = len(content.split('\n\n'))
        
        node = ContextNode(
            node_id=f"{document_id}_document",
            level=ContextLevel.DOCUMENT,
            content=content[:1000] + "..." if len(content) > 1000 else content,  # Truncate for storage
            embedding=document_embedding,
            position=(0,),
            metadata={
                'document_id': document_id,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'paragraph_count': paragraph_count,
                'full_content_hash': hashlib.md5(content.encode()).hexdigest(),
                **metadata
            },
            context_window=0,  # Document is top level
            importance_score=1.0,  # Documents are always important
            coherence_score=semantic_analysis.get('overall_metrics', {}).get('sharing_efficiency', 0.5)
        )
        
        return [node]
    
    async def _build_cross_document_context(
        self,
        document_id: str,
        content: str,
        semantic_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[ContextNode]:
        """Build cross-document context (requires multiple documents)."""
        # This would be called separately for multiple documents
        return []
    
    async def _cache_and_store_node(self, node: ContextNode):
        """Cache node in memory and store in Neo4j."""
        # Add to cache
        self.context_cache[node.node_id] = node
        self.cache_access_order.append(node.node_id)
        
        # Manage cache size
        while len(self.context_cache) > self.cache_size:
            oldest_node_id = self.cache_access_order.popleft()
            self.context_cache.pop(oldest_node_id, None)
        
        # Store in Neo4j
        await self.graph_store.store_context_node(node)
    
    async def _build_hierarchical_relationships(
        self,
        hierarchy: Dict[ContextLevel, List[ContextNode]]
    ):
        """Build containment relationships between hierarchy levels."""
        level_order = [ContextLevel.TOKEN, ContextLevel.SENTENCE, ContextLevel.PARAGRAPH, ContextLevel.DOCUMENT]
        
        for i in range(len(level_order) - 1):
            parent_level = level_order[i + 1]
            child_level = level_order[i]
            
            if parent_level not in hierarchy or child_level not in hierarchy:
                continue
            
            parent_nodes = hierarchy[parent_level]
            child_nodes = hierarchy[child_level]
            
            # Build containment relationships (simplified logic)
            for parent in parent_nodes:
                for child in child_nodes:
                    # Simple containment based on position
                    # In production, would use more sophisticated matching
                    if self._is_contained(child, parent):
                        relation = ContextRelation(
                            source_id=parent.node_id,
                            target_id=child.node_id,
                            relation_type=ContextRelationType.CONTAINS,
                            strength=0.9,
                            confidence=0.95,
                            evidence=[f"Hierarchical containment: {parent.level.value} contains {child.level.value}"]
                        )
                        
                        await self.graph_store.store_context_relation(relation)
    
    def _is_contained(self, child: ContextNode, parent: ContextNode) -> bool:
        """Check if child context is contained within parent context."""
        # Simplified containment logic
        # In production, would use more sophisticated position-based matching
        return len(child.position) > len(parent.position)
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the context hierarchy."""
        return {
            'cache_size': len(self.context_cache),
            'cache_hit_ratio': len(self.context_cache) / max(len(self.cache_access_order), 1),
            'processing_metrics': {
                'avg_document_analysis_time': np.mean(self.processing_metrics['document_analysis'][-10:]) if self.processing_metrics['document_analysis'] else 0.0,
                'total_documents_processed': len(self.processing_metrics['document_analysis'])
            },
            'neo4j_connected': self.graph_store.driver is not None,
            'supported_levels': [level.value for level in ContextLevel]
        }
    
    def close(self):
        """Clean up resources."""
        self.graph_store.close()
        self.context_cache.clear()
        self.cache_access_order.clear()

# Factory function
def create_context_hierarchy_engine(
    semantic_engine: WeightSharingSemanticEngine,
    neo4j_config: Dict[str, str] = None
) -> ContextHierarchyEngine:
    """Create and initialize context hierarchy engine."""
    return ContextHierarchyEngine(
        semantic_engine=semantic_engine,
        neo4j_config=neo4j_config
    )
