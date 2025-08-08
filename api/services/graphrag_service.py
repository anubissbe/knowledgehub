"""
GraphRAG Service - Neo4j Enhanced Retrieval-Augmented Generation
Dynamic Parallelism and Memory Bandwidth Optimization for Knowledge Graphs

This service combines traditional vector RAG with graph-based knowledge retrieval,
leveraging dynamic parallelism for efficient graph operations and memory bandwidth
optimization for handling large knowledge graphs.

Author: Charlotte Cools - Dynamic Parallelism Expert
"""

import asyncio
import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import threading

from neo4j import GraphDatabase, Driver, Session as Neo4jSession
from neo4j.exceptions import Neo4jError
import networkx as nx
from sqlalchemy.orm import Session
from sqlalchemy import text

from .knowledge_graph import KnowledgeGraphService, NodeType, RelationType
from .rag_pipeline import RAGPipeline, ChunkingStrategy, RetrievalStrategy
from ..database import get_db
from shared.config import Config

logger = logging.getLogger(__name__)


class GraphRAGStrategy(Enum):
    """GraphRAG retrieval strategies"""
    VECTOR_FIRST = "vector_first"  # Vector search then graph expansion  
    GRAPH_FIRST = "graph_first"    # Graph search then vector refinement
    HYBRID_PARALLEL = "hybrid_parallel"  # Parallel vector and graph search
    ENTITY_CENTRIC = "entity_centric"    # Entity-focused retrieval
    RELATIONSHIP_WEIGHTED = "relationship_weighted"  # Weighted by relationships
    COMMUNITY_AWARE = "community_aware"  # Graph community detection


class EntityType(Enum):
    """Entity types for graph extraction"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION" 
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECHNOLOGY"
    METHOD = "METHOD"
    METRIC = "METRIC"
    EVENT = "EVENT"


@dataclass
class GraphMemoryConfig:
    """Memory bandwidth optimization configuration"""
    # Memory pool settings
    max_memory_mb: int = 1024
    chunk_size_mb: int = 64
    prefetch_factor: float = 0.3
    
    # Parallel processing
    max_workers: int = 8
    batch_size: int = 100
    memory_threshold: float = 0.8
    
    # Graph traversal optimization
    max_depth: int = 3
    node_cache_size: int = 10000
    relationship_cache_size: int = 50000


@dataclass 
class EntityExtraction:
    """Extracted entity with graph context"""
    entity: str
    entity_type: EntityType
    confidence: float
    context: str
    relationships: List[Tuple[str, str, float]] = field(default_factory=list)
    
    
@dataclass
class GraphRAGResult:
    """Combined vector and graph RAG result"""
    content: str
    score: float
    vector_score: float
    graph_score: float
    entities: List[EntityExtraction]
    relationships: List[Dict[str, Any]]
    reasoning_path: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParallelGraphProcessor:
    """Dynamic parallelism processor for graph operations"""
    
    def __init__(self, config: GraphMemoryConfig):
        self.config = config
        self.memory_pool = {}
        self.active_workers = 0
        self.lock = threading.Lock()
        
    async def parallel_entity_extraction(self, 
                                       documents: List[Dict[str, Any]],
                                       batch_size: Optional[int] = None) -> List[EntityExtraction]:
        """Extract entities in parallel with memory optimization"""
        batch_size = batch_size or self.config.batch_size
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Create batches for memory efficiency
            batches = [documents[i:i + batch_size] 
                      for i in range(0, len(documents), batch_size)]
            
            # Submit parallel extraction tasks
            future_to_batch = {
                executor.submit(self._extract_entities_batch, batch): batch
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Entity extraction batch failed: {e}")
                    
        return results
    
    def _extract_entities_batch(self, documents: List[Dict[str, Any]]) -> List[EntityExtraction]:
        """Extract entities from document batch"""
        entities = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # Simple pattern-based entity extraction (can be enhanced with NLP models)
            doc_entities = self._extract_entities_from_text(content)
            entities.extend(doc_entities)
            
        return entities
    
    def _extract_entities_from_text(self, text: str) -> List[EntityExtraction]:
        """Extract entities from text using patterns"""
        entities = []
        
        # Technology patterns (leveraging expertise in dynamic parallelism)
        tech_patterns = ['GPU', 'CPU', 'CUDA', 'TensorRT', 'PyTorch', 'TensorFlow', 
                        'V100', 'A100', 'kernel', 'memory bandwidth', 'parallelism',
                        'dynamic parallelism', 'SIMD', 'SIMT', 'warp', 'thread block',
                        'shared memory', 'global memory', 'register', 'cache']
        
        # Method patterns  
        method_patterns = ['optimization', 'compression', 'quantization', 'pruning',
                          'factorization', 'algorithm', 'implementation', 'kernel fusion',
                          'memory coalescing', 'occupancy', 'bandwidth optimization']
        
        # Hardware patterns
        hardware_patterns = ['Tesla V100', 'A100', 'RTX', 'GeForce', 'Quadro',
                           'streaming multiprocessor', 'tensor core', 'RT core',
                           'GDDR', 'HBM', 'PCIe', 'NVLink']
        
        text_lower = text.lower()
        
        for pattern in tech_patterns:
            if pattern.lower() in text_lower:
                entities.append(EntityExtraction(
                    entity=pattern,
                    entity_type=EntityType.TECHNOLOGY,
                    confidence=0.8,
                    context=text[:200]  # Context snippet
                ))
                
        for pattern in method_patterns:
            if pattern.lower() in text_lower:
                entities.append(EntityExtraction(
                    entity=pattern,
                    entity_type=EntityType.METHOD,
                    confidence=0.7,
                    context=text[:200]
                ))
                
        for pattern in hardware_patterns:
            if pattern.lower() in text_lower:
                entities.append(EntityExtraction(
                    entity=pattern,
                    entity_type=EntityType.TECHNOLOGY,
                    confidence=0.9,
                    context=text[:200]
                ))
                
        return entities


class GraphRAGService:
    """
    GraphRAG Service combining Neo4j graph database with vector RAG
    
    Features:
    - Dynamic parallel graph traversal
    - Memory bandwidth optimized operations
    - Entity extraction and relationship mapping
    - Hybrid vector + graph retrieval
    - Graph-aware chunking strategies
    """
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://192.168.1.25:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "knowledgehub123"):
        self.config = GraphMemoryConfig()
        self.kg_service = KnowledgeGraphService(neo4j_uri, neo4j_user, neo4j_password)
        self.processor = ParallelGraphProcessor(self.config)
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Initialize RAG pipeline for fallback
        self.rag_pipeline = None
        
        # Memory bandwidth optimization 
        self.node_cache = {}
        self.relationship_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def initialize_rag_pipeline(self):
        """Initialize the RAG pipeline for hybrid operations"""
        try:
            self.rag_pipeline = RAGPipeline()
            await self.rag_pipeline.initialize()
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
    
    async def index_documents_with_graph(self,
                                       documents: List[Dict[str, Any]],
                                       extract_entities: bool = True,
                                       build_relationships: bool = True) -> Dict[str, Any]:
        """
        Index documents with both vector embeddings and graph structure
        
        Args:
            documents: List of documents to index
            extract_entities: Whether to extract entities
            build_relationships: Whether to build entity relationships
            
        Returns:
            Indexing results with statistics
        """
        start_time = datetime.now()
        stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'relationships_created': 0,
            'vector_embeddings': 0,
            'graph_nodes': 0,
            'processing_time_ms': 0
        }
        
        try:
            # Step 1: Extract entities in parallel
            entities = []
            if extract_entities:
                entities = await self.processor.parallel_entity_extraction(documents)
                stats['entities_extracted'] = len(entities)
                
            # Step 2: Create graph nodes for entities (parallel batch processing)
            if entities:
                await self._create_entity_nodes_parallel(entities)
                stats['graph_nodes'] = len(entities)
                
            # Step 3: Build relationships between entities 
            if build_relationships and entities:
                relationships = await self._build_entity_relationships(entities, documents)
                stats['relationships_created'] = len(relationships)
                
            # Step 4: Create document nodes and link to entities
            await self._create_document_nodes(documents, entities)
            
            # Step 5: Vector indexing (if RAG pipeline available)
            if self.rag_pipeline:
                vector_results = await self.rag_pipeline.index_documents(documents)
                stats['vector_embeddings'] = vector_results.get('embeddings_created', 0)
                
            stats['documents_processed'] = len(documents)
            stats['processing_time_ms'] = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"GraphRAG indexing completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"GraphRAG indexing failed: {e}")
            raise
    
    async def query_graphrag(self,
                           query: str,
                           strategy: GraphRAGStrategy = GraphRAGStrategy.HYBRID_PARALLEL,
                           max_results: int = 10,
                           include_reasoning: bool = True) -> List[GraphRAGResult]:
        """
        Query using GraphRAG with specified strategy
        
        Args:
            query: User query
            strategy: GraphRAG retrieval strategy
            max_results: Maximum results to return
            include_reasoning: Whether to include reasoning path
            
        Returns:
            List of GraphRAG results
        """
        try:
            if strategy == GraphRAGStrategy.HYBRID_PARALLEL:
                return await self._hybrid_parallel_query(query, max_results, include_reasoning)
            elif strategy == GraphRAGStrategy.VECTOR_FIRST:
                return await self._vector_first_query(query, max_results, include_reasoning) 
            elif strategy == GraphRAGStrategy.GRAPH_FIRST:
                return await self._graph_first_query(query, max_results, include_reasoning)
            elif strategy == GraphRAGStrategy.ENTITY_CENTRIC:
                return await self._entity_centric_query(query, max_results, include_reasoning)
            else:
                # Default to hybrid parallel
                return await self._hybrid_parallel_query(query, max_results, include_reasoning)
                
        except Exception as e:
            logger.error(f"GraphRAG query failed: {e}")
            return []
    
    async def _hybrid_parallel_query(self, query: str, max_results: int, include_reasoning: bool) -> List[GraphRAGResult]:
        """Execute hybrid parallel query with dynamic parallelism"""
        
        # Run vector and graph queries in parallel
        vector_task = asyncio.create_task(self._vector_query(query, max_results))
        graph_task = asyncio.create_task(self._graph_query(query, max_results))
        
        vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
        
        # Combine and rank results
        combined_results = []
        
        # Memory-efficient result merging
        vector_dict = {r['content'][:100]: r for r in vector_results}  # Use content prefix as key
        graph_dict = {r['content'][:100]: r for r in graph_results}
        
        # Process common results first (higher scores)
        for key in vector_dict.keys() & graph_dict.keys():
            v_result = vector_dict[key]
            g_result = graph_dict[key]
            
            combined_score = (v_result['score'] * 0.6) + (g_result['score'] * 0.4)
            
            result = GraphRAGResult(
                content=v_result['content'],
                score=combined_score,
                vector_score=v_result['score'],
                graph_score=g_result['score'],
                entities=g_result.get('entities', []),
                relationships=g_result.get('relationships', []),
                reasoning_path=g_result.get('reasoning_path', []) if include_reasoning else [],
                metadata={
                    'source': 'hybrid_parallel',
                    'vector_match': True,
                    'graph_match': True
                }
            )
            combined_results.append(result)
            
        # Add vector-only results
        for key, v_result in vector_dict.items():
            if key not in graph_dict:
                result = GraphRAGResult(
                    content=v_result['content'],
                    score=v_result['score'] * 0.8,  # Slight penalty for no graph match
                    vector_score=v_result['score'],
                    graph_score=0.0,
                    entities=[],
                    relationships=[],
                    reasoning_path=[],
                    metadata={
                        'source': 'vector_only',
                        'vector_match': True,
                        'graph_match': False
                    }
                )
                combined_results.append(result)
                
        # Add graph-only results  
        for key, g_result in graph_dict.items():
            if key not in vector_dict:
                result = GraphRAGResult(
                    content=g_result['content'],
                    score=g_result['score'] * 0.7,  # Penalty for no vector match
                    vector_score=0.0,
                    graph_score=g_result['score'],
                    entities=g_result.get('entities', []),
                    relationships=g_result.get('relationships', []),
                    reasoning_path=g_result.get('reasoning_path', []) if include_reasoning else [],
                    metadata={
                        'source': 'graph_only',
                        'vector_match': False,
                        'graph_match': True
                    }
                )
                combined_results.append(result)
        
        # Sort by combined score and return top results
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:max_results]
    
    async def _vector_query(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Execute vector similarity query"""
        if not self.rag_pipeline:
            return []
            
        try:
            results = await self.rag_pipeline.query(
                query=query,
                max_results=max_results,
                strategy=RetrievalStrategy.VECTOR
            )
            return results
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return []
    
    async def _graph_query(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Execute graph traversal query with memory optimization"""
        try:
            # Extract entities from query
            query_entities = self._extract_entities_from_text(query)
            
            if not query_entities:
                return []
                
            # Memory-optimized parallel graph traversal
            results = []
            
            with self.driver.session() as session:
                # Find related nodes through optimized graph traversal
                for entity in query_entities[:5]:  # Limit to prevent memory explosion
                    entity_results = await self._traverse_graph_from_entity(
                        session, entity.entity, max_depth=self.config.max_depth
                    )
                    results.extend(entity_results)
                    
            # Remove duplicates and sort
            seen_content = set()
            unique_results = []
            for result in results:
                content_key = result['content'][:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_results.append(result)
                    
            return unique_results[:max_results]
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []
    
    async def _traverse_graph_from_entity(self, session: Neo4jSession, entity: str, max_depth: int) -> List[Dict[str, Any]]:
        """Memory-optimized graph traversal from entity"""
        
        # Check cache first
        cache_key = f"{entity}_{max_depth}"
        if cache_key in self.node_cache:
            self.cache_hits += 1
            return self.node_cache[cache_key]
            
        self.cache_misses += 1
        
        # Cypher query with memory limits
        cypher = """
        MATCH (e:Entity {name: $entity})-[r*1..$max_depth]-(related)
        WHERE related:Document OR related:Entity
        RETURN related.content as content, 
               related.name as name,
               type(last(r)) as relationship_type,
               length(r) as distance
        ORDER BY distance ASC, related.score DESC
        LIMIT 20
        """
        
        try:
            result = session.run(cypher, entity=entity, max_depth=max_depth)
            
            graph_results = []
            for record in result:
                content = record.get('content', record.get('name', ''))
                if content:
                    graph_results.append({
                        'content': content,
                        'score': 1.0 / (record.get('distance', 1) + 1),  # Distance-based scoring
                        'entities': [entity],
                        'relationships': [record.get('relationship_type', 'RELATED')],
                        'reasoning_path': [f"Connected to {entity} via {record.get('relationship_type', 'RELATED')}"]
                    })
                    
            # Cache results (with size limit)
            if len(self.node_cache) < self.config.node_cache_size:
                self.node_cache[cache_key] = graph_results
                
            return graph_results
            
        except Neo4jError as e:
            logger.error(f"Neo4j traversal error: {e}")
            return []
    
    async def _create_entity_nodes_parallel(self, entities: List[EntityExtraction]):
        """Create entity nodes in parallel batches"""
        
        batch_size = self.config.batch_size
        batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._create_entity_batch, batch)
                for batch in batches
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Entity creation batch failed: {e}")
    
    def _create_entity_batch(self, entities: List[EntityExtraction]):
        """Create a batch of entity nodes"""
        with self.driver.session() as session:
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.confidence = $confidence,
                        e.context = $context,
                        e.created_at = datetime(),
                        e.updated_at = datetime()
                    """,
                    name=entity.entity,
                    type=entity.entity_type.value,
                    confidence=entity.confidence,
                    context=entity.context
                )
    
    async def _build_entity_relationships(self, entities: List[EntityExtraction], documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build relationships between entities based on co-occurrence"""
        relationships = []
        
        # Build entity co-occurrence matrix for relationship detection
        entity_docs = {}
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '').lower()
            for entity in entities:
                if entity.entity.lower() in content:
                    if entity.entity not in entity_docs:
                        entity_docs[entity.entity] = []
                    entity_docs[entity.entity].append(doc_idx)
        
        # Create relationships for entities that co-occur
        entity_names = list(entity_docs.keys())
        for i, entity1 in enumerate(entity_names):
            for entity2 in entity_names[i+1:]:
                common_docs = set(entity_docs[entity1]) & set(entity_docs[entity2])
                if len(common_docs) > 0:
                    # Co-occurrence strength
                    strength = len(common_docs) / (len(entity_docs[entity1]) + len(entity_docs[entity2]) - len(common_docs))
                    
                    if strength > 0.1:  # Threshold for creating relationship
                        relationships.append({
                            'source': entity1,
                            'target': entity2,
                            'type': 'CO_OCCURS',
                            'strength': strength,
                            'common_documents': len(common_docs)
                        })
        
        # Create relationships in Neo4j
        await self._create_relationships_parallel(relationships)
        
        return relationships
    
    async def _create_relationships_parallel(self, relationships: List[Dict[str, Any]]):
        """Create relationships in parallel batches"""
        
        batch_size = self.config.batch_size
        batches = [relationships[i:i + batch_size] for i in range(0, len(relationships), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._create_relationship_batch, batch)
                for batch in batches
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Relationship creation batch failed: {e}")
    
    def _create_relationship_batch(self, relationships: List[Dict[str, Any]]):
        """Create a batch of relationships"""
        with self.driver.session() as session:
            for rel in relationships:
                session.run(
                    """
                    MATCH (a:Entity {name: $source})
                    MATCH (b:Entity {name: $target})
                    MERGE (a)-[r:CO_OCCURS]->(b)
                    SET r.strength = $strength,
                        r.common_documents = $common_documents,
                        r.created_at = datetime()
                    """,
                    source=rel['source'],
                    target=rel['target'],
                    strength=rel['strength'],
                    common_documents=rel['common_documents']
                )
    
    async def _create_document_nodes(self, documents: List[Dict[str, Any]], entities: List[EntityExtraction]):
        """Create document nodes and link to entities"""
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._create_document_with_entities, doc, entities)
                for doc in documents
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Document creation failed: {e}")
    
    def _create_document_with_entities(self, document: Dict[str, Any], entities: List[EntityExtraction]):
        """Create document node and link to relevant entities"""
        with self.driver.session() as session:
            doc_id = document.get('id', str(uuid.uuid4()))
            content = document.get('content', '')
            
            # Create document node
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.content = $content,
                    d.title = $title,
                    d.created_at = datetime(),
                    d.updated_at = datetime()
                """,
                id=doc_id,
                content=content,
                title=document.get('title', 'Untitled')
            )
            
            # Link to entities found in document
            content_lower = content.lower()
            for entity in entities:
                if entity.entity.lower() in content_lower:
                    session.run(
                        """
                        MATCH (d:Document {id: $doc_id})
                        MATCH (e:Entity {name: $entity})
                        MERGE (d)-[r:MENTIONS]->(e)
                        SET r.confidence = $confidence,
                            r.context = $context
                        """,
                        doc_id=doc_id,
                        entity=entity.entity,
                        confidence=entity.confidence,
                        context=entity.context[:500]  # Limit context length
                    )
    
    def _extract_entities_from_text(self, text: str) -> List[EntityExtraction]:
        """Extract entities from text (simplified version)"""
        return self.processor._extract_entities_from_text(text)
    
    async def _vector_first_query(self, query: str, max_results: int, include_reasoning: bool) -> List[GraphRAGResult]:
        """Vector-first query strategy"""
        # Get vector results first
        vector_results = await self._vector_query(query, max_results * 2)
        
        # Enhance with graph information
        enhanced_results = []
        for v_result in vector_results[:max_results]:
            # Find entities in result content
            entities = self._extract_entities_from_text(v_result['content'])
            
            # Get graph relationships for entities
            relationships = []
            reasoning_path = []
            
            if entities:
                with self.driver.session() as session:
                    for entity in entities[:3]:  # Limit to prevent slowdown
                        rel_query = """
                        MATCH (e:Entity {name: $entity})-[r]-(other)
                        RETURN other.name as related, type(r) as relationship
                        LIMIT 5
                        """
                        result = session.run(rel_query, entity=entity.entity)
                        for record in result:
                            relationships.append({
                                'entity': entity.entity,
                                'related': record['related'],
                                'relationship': record['relationship']
                            })
                            reasoning_path.append(f"{entity.entity} -> {record['relationship']} -> {record['related']}")
            
            enhanced_results.append(GraphRAGResult(
                content=v_result['content'],
                score=v_result['score'],
                vector_score=v_result['score'],
                graph_score=0.5 if relationships else 0.0,
                entities=entities,
                relationships=relationships,
                reasoning_path=reasoning_path if include_reasoning else [],
                metadata={'source': 'vector_first'}
            ))
        
        return enhanced_results
    
    async def _graph_first_query(self, query: str, max_results: int, include_reasoning: bool) -> List[GraphRAGResult]:
        """Graph-first query strategy"""
        graph_results = await self._graph_query(query, max_results * 2)
        
        # Convert to GraphRAG results
        rag_results = []
        for g_result in graph_results[:max_results]:
            rag_results.append(GraphRAGResult(
                content=g_result['content'],
                score=g_result['score'],
                vector_score=0.0,
                graph_score=g_result['score'],
                entities=g_result.get('entities', []),
                relationships=g_result.get('relationships', []),
                reasoning_path=g_result.get('reasoning_path', []) if include_reasoning else [],
                metadata={'source': 'graph_first'}
            ))
        
        return rag_results
    
    async def _entity_centric_query(self, query: str, max_results: int, include_reasoning: bool) -> List[GraphRAGResult]:
        """Entity-centric query strategy"""
        # Extract entities from query
        query_entities = self._extract_entities_from_text(query)
        
        if not query_entities:
            # Fallback to vector search if no entities found
            return await self._vector_first_query(query, max_results, include_reasoning)
        
        results = []
        
        with self.driver.session() as session:
            for entity in query_entities:
                # Find documents mentioning this entity
                cypher = """
                MATCH (d:Document)-[r:MENTIONS]->(e:Entity {name: $entity})
                RETURN d.content as content, r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT $limit
                """
                
                result = session.run(cypher, entity=entity.entity, limit=max_results // len(query_entities) + 1)
                
                for record in result:
                    results.append(GraphRAGResult(
                        content=record['content'],
                        score=record['confidence'],
                        vector_score=0.0,
                        graph_score=record['confidence'],
                        entities=[entity],
                        relationships=[],
                        reasoning_path=[f"Document mentions {entity.entity}"] if include_reasoning else [],
                        metadata={'source': 'entity_centric', 'target_entity': entity.entity}
                    ))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory and cache statistics"""
        return {
            'node_cache_size': len(self.node_cache),
            'relationship_cache_size': len(self.relationship_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'max_memory_mb': self.config.max_memory_mb,
            'active_workers': self.processor.active_workers
        }
    
    async def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.close()
        if self.rag_pipeline:
            await self.rag_pipeline.close()


# Global GraphRAG service instance
_graphrag_service: Optional[GraphRAGService] = None

async def get_graphrag_service() -> GraphRAGService:
    """Get or create GraphRAG service instance"""
    global _graphrag_service
    
    if _graphrag_service is None:
        _graphrag_service = GraphRAGService()
        await _graphrag_service.initialize_rag_pipeline()
    
    return _graphrag_service
