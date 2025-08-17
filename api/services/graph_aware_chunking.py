"""
Graph-Aware Chunking Strategies for GraphRAG
Dynamic Parallelism and Memory Bandwidth Optimization for Knowledge Extraction

This module implements advanced chunking strategies that leverage knowledge graphs
to create semantically coherent document chunks optimized for parallel processing.

Author: Charlotte Cools - Dynamic Parallelism Expert
"""

import asyncio
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from neo4j import GraphDatabase, Session as Neo4jSession
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)


class GraphChunkingStrategy(Enum):
    """Graph-aware chunking strategies"""
    ENTITY_BOUNDARY = "entity_boundary"      # Split on entity boundaries
    SEMANTIC_GRAPH = "semantic_graph"        # Use graph relationships for coherence
    HIERARCHICAL_GRAPH = "hierarchical_graph"  # Multi-level graph-based chunking
    COMMUNITY_BASED = "community_based"      # Leverage graph communities
    RELATIONSHIP_DENSITY = "relationship_density"  # Based on relationship density


@dataclass
class GraphChunk:
    """Graph-aware chunk with entity and relationship context"""
    content: str
    chunk_id: str
    entities: List[str]
    relationships: List[Dict[str, Any]]
    community_id: Optional[str] = None
    coherence_score: float = 0.0
    entity_density: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ChunkingConfig:
    """Configuration for graph-aware chunking"""
    # Base chunking parameters
    target_chunk_size: int = 512
    max_chunk_size: int = 1024
    min_chunk_size: int = 128
    overlap_size: int = 50
    
    # Graph-specific parameters
    entity_boundary_weight: float = 0.7
    relationship_weight: float = 0.3
    coherence_threshold: float = 0.5
    
    # Performance optimization
    parallel_workers: int = 4
    batch_size: int = 50
    memory_limit_mb: int = 256


class ParallelGraphChunker:
    """Dynamic parallelism processor for graph-aware chunking"""
    
    def __init__(self, 
                 config: ChunkingConfig,
                 neo4j_driver: Optional[Any] = None):
        self.config = config
        self.neo4j_driver = neo4j_driver
        
        # Memory bandwidth optimization
        self.entity_cache = {}
        self.relationship_cache = {}
        self.community_cache = {}
        
    async def chunk_documents_parallel(self,
                                     documents: List[Dict[str, Any]],
                                     strategy: GraphChunkingStrategy = GraphChunkingStrategy.SEMANTIC_GRAPH) -> List[GraphChunk]:
        """Chunk documents in parallel with graph awareness"""
        
        all_chunks = []
        
        # Process documents in parallel batches
        batch_size = self.config.batch_size
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = []
            
            for batch in batches:
                future = executor.submit(self._chunk_document_batch, batch, strategy)
                futures.append(future)
            
            # Collect results with memory management
            for future in as_completed(futures):
                try:
                    batch_chunks = future.result()
                    all_chunks.extend(batch_chunks)
                    
                    # Memory management: clear caches if getting large
                    if len(self.entity_cache) > 10000:
                        self._clear_caches()
                        
                except Exception as e:
                    logger.error(f"Document batch chunking failed: {e}")
        
        # Post-processing: optimize chunk boundaries across documents
        optimized_chunks = await self._optimize_chunk_boundaries(all_chunks)
        
        return optimized_chunks
    
    def _chunk_document_batch(self, 
                            documents: List[Dict[str, Any]],
                            strategy: GraphChunkingStrategy) -> List[GraphChunk]:
        """Process a batch of documents for chunking"""
        
        batch_chunks = []
        
        for document in documents:
            try:
                doc_chunks = self._chunk_single_document(document, strategy)
                batch_chunks.extend(doc_chunks)
            except Exception as e:
                logger.error(f"Document chunking failed: {e}")
                # Fallback to simple chunking
                fallback_chunks = self._fallback_chunking(document)
                batch_chunks.extend(fallback_chunks)
        
        return batch_chunks
    
    def _chunk_single_document(self,
                             document: Dict[str, Any], 
                             strategy: GraphChunkingStrategy) -> List[GraphChunk]:
        """Chunk a single document using graph-aware strategy"""
        
        content = document.get('content', '')
        doc_id = document.get('id', 'unknown')
        
        if strategy == GraphChunkingStrategy.ENTITY_BOUNDARY:
            return self._entity_boundary_chunking(content, doc_id)
        elif strategy == GraphChunkingStrategy.SEMANTIC_GRAPH:
            return self._semantic_graph_chunking(content, doc_id)
        elif strategy == GraphChunkingStrategy.HIERARCHICAL_GRAPH:
            return self._hierarchical_graph_chunking(content, doc_id)
        elif strategy == GraphChunkingStrategy.COMMUNITY_BASED:
            return self._community_based_chunking(content, doc_id)
        elif strategy == GraphChunkingStrategy.RELATIONSHIP_DENSITY:
            return self._relationship_density_chunking(content, doc_id)
        else:
            return self._fallback_chunking(document)
    
    def _entity_boundary_chunking(self, content: str, doc_id: str) -> List[GraphChunk]:
        """Chunk based on entity boundaries"""
        
        # Extract entities and their positions
        entities_with_positions = self._extract_entities_with_positions(content)
        
        if not entities_with_positions:
            return self._fallback_chunking({'content': content, 'id': doc_id})
        
        chunks = []
        sentences = self._split_into_sentences(content)
        current_chunk = ""
        current_entities = []
        chunk_start_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence_entities = [e for e, start, end in entities_with_positions 
                               if start >= len(current_chunk) and end <= len(current_chunk + sentence)]
            
            # Check if adding this sentence would exceed size limits
            if len(current_chunk + sentence) > self.config.max_chunk_size and current_chunk:
                # Create chunk
                chunk = self._create_graph_chunk(
                    content=current_chunk.strip(),
                    doc_id=doc_id,
                    chunk_idx=len(chunks),
                    entities=current_entities,
                    strategy="entity_boundary"
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_content = self._get_overlap_content(current_chunk, self.config.overlap_size)
                current_chunk = overlap_content + sentence
                current_entities = sentence_entities
            else:
                current_chunk += sentence
                current_entities.extend(sentence_entities)
        
        # Handle remaining content
        if current_chunk.strip():
            chunk = self._create_graph_chunk(
                content=current_chunk.strip(),
                doc_id=doc_id,
                chunk_idx=len(chunks),
                entities=current_entities,
                strategy="entity_boundary"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _semantic_graph_chunking(self, content: str, doc_id: str) -> List[GraphChunk]:
        """Chunk based on semantic graph relationships"""
        
        sentences = self._split_into_sentences(content)
        sentence_entities = []
        
        # Extract entities for each sentence
        for sentence in sentences:
            entities = self._extract_entities_from_text(sentence)
            sentence_entities.append(entities)
        
        # Build coherence matrix based on entity relationships
        coherence_matrix = self._build_coherence_matrix(sentence_entities)
        
        # Use dynamic programming to find optimal chunk boundaries
        chunks = self._find_optimal_chunks(sentences, sentence_entities, coherence_matrix, doc_id)
        
        return chunks
    
    def _hierarchical_graph_chunking(self, content: str, doc_id: str) -> List[GraphChunk]:
        """Multi-level hierarchical chunking using graph structure"""
        
        # Level 1: Paragraph-level chunking
        paragraphs = content.split('\n\n')
        paragraph_chunks = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > self.config.min_chunk_size:
                # Level 2: Sentence-level optimization within paragraph
                para_chunks = self._semantic_graph_chunking(paragraph, f"{doc_id}_p{para_idx}")
                paragraph_chunks.extend(para_chunks)
        
        return paragraph_chunks
    
    def _community_based_chunking(self, content: str, doc_id: str) -> List[GraphChunk]:
        """Chunk based on entity community detection"""
        
        if not self.neo4j_driver:
            return self._fallback_chunking({'content': content, 'id': doc_id})
        
        # Extract entities from entire document
        entities = self._extract_entities_from_text(content)
        
        if not entities:
            return self._fallback_chunking({'content': content, 'id': doc_id})
        
        # Get entity communities from Neo4j
        communities = self._get_entity_communities(entities)
        
        # Split content based on community boundaries
        chunks = []
        sentences = self._split_into_sentences(content)
        current_chunk = ""
        current_community = None
        
        for sentence in sentences:
            sentence_entities = self._extract_entities_from_text(sentence)
            
            if sentence_entities:
                # Determine primary community for this sentence
                sentence_community = self._get_primary_community(sentence_entities, communities)
                
                # If community changes or chunk is too large, create new chunk
                if (sentence_community != current_community and current_chunk) or \
                   len(current_chunk + sentence) > self.config.max_chunk_size:
                    
                    if current_chunk.strip():
                        chunk = self._create_graph_chunk(
                            content=current_chunk.strip(),
                            doc_id=doc_id,
                            chunk_idx=len(chunks),
                            entities=self._extract_entities_from_text(current_chunk),
                            strategy="community_based"
                        )
                        chunk.community_id = current_community
                        chunks.append(chunk)
                    
                    current_chunk = sentence
                    current_community = sentence_community
                else:
                    current_chunk += sentence
            else:
                current_chunk += sentence
        
        # Handle remaining content
        if current_chunk.strip():
            chunk = self._create_graph_chunk(
                content=current_chunk.strip(),
                doc_id=doc_id,
                chunk_idx=len(chunks),
                entities=self._extract_entities_from_text(current_chunk),
                strategy="community_based"
            )
            chunk.community_id = current_community
            chunks.append(chunk)
        
        return chunks
    
    def _relationship_density_chunking(self, content: str, doc_id: str) -> List[GraphChunk]:
        """Chunk based on relationship density between entities"""
        
        sentences = self._split_into_sentences(content)
        sentence_relationships = []
        
        # Calculate relationship density for each sentence
        for sentence in sentences:
            entities = self._extract_entities_from_text(sentence)
            relationships = self._get_entity_relationships(entities) if self.neo4j_driver else []
            density = len(relationships) / max(len(entities), 1) if entities else 0
            sentence_relationships.append((sentence, entities, relationships, density))
        
        # Group sentences by relationship density
        chunks = []
        current_group = []
        current_density_avg = 0
        
        for sentence, entities, relationships, density in sentence_relationships:
            # If density change is significant, start new chunk
            if current_group and abs(density - current_density_avg) > 0.3:
                chunk_content = " ".join([s[0] for s in current_group])
                chunk_entities = []
                chunk_relationships = []
                
                for _, ents, rels, _ in current_group:
                    chunk_entities.extend(ents)
                    chunk_relationships.extend(rels)
                
                chunk = self._create_graph_chunk(
                    content=chunk_content,
                    doc_id=doc_id,
                    chunk_idx=len(chunks),
                    entities=list(set(chunk_entities)),  # Remove duplicates
                    strategy="relationship_density"
                )
                chunk.entity_density = current_density_avg
                chunks.append(chunk)
                
                current_group = [(sentence, entities, relationships, density)]
                current_density_avg = density
            else:
                current_group.append((sentence, entities, relationships, density))
                if current_group:
                    current_density_avg = sum([d for _, _, _, d in current_group]) / len(current_group)
        
        # Handle remaining group
        if current_group:
            chunk_content = " ".join([s[0] for s in current_group])
            chunk_entities = []
            chunk_relationships = []
            
            for _, ents, rels, _ in current_group:
                chunk_entities.extend(ents)
                chunk_relationships.extend(rels)
            
            chunk = self._create_graph_chunk(
                content=chunk_content,
                doc_id=doc_id,
                chunk_idx=len(chunks),
                entities=list(set(chunk_entities)),
                strategy="relationship_density"
            )
            chunk.entity_density = current_density_avg
            chunks.append(chunk)
        
        return chunks
    
    def _extract_entities_with_positions(self, content: str) -> List[Tuple[str, int, int]]:
        """Extract entities with their positions in text"""
        entities_with_pos = []
        
        # Technology patterns with positions
        tech_patterns = [
            r'\b(GPU|CPU|CUDA|TensorRT|PyTorch|TensorFlow)\b',
            r'\b(V100|A100|RTX \d+|GeForce)\b',
            r'\b(kernel|memory bandwidth|parallelism)\b',
            r'\b(dynamic parallelism|SIMD|SIMT)\b'
        ]
        
        for pattern in tech_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                entities_with_pos.append((match.group(), match.start(), match.end()))
        
        return entities_with_pos
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entities from text"""
        entities = []
        
        # Simplified entity extraction (can be enhanced with NER models)
        tech_patterns = ['GPU', 'CPU', 'CUDA', 'kernel', 'memory bandwidth', 'parallelism',
                        'V100', 'A100', 'dynamic parallelism', 'optimization']
        
        text_lower = text.lower()
        for pattern in tech_patterns:
            if pattern.lower() in text_lower:
                entities.append(pattern)
        
        return list(set(entities))  # Remove duplicates
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences"""
        # Simple sentence splitting (can be enhanced with NLP libraries)
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() + '.' for s in sentences if s.strip()]
    
    def _build_coherence_matrix(self, sentence_entities: List[List[str]]) -> np.ndarray:
        """Build coherence matrix between sentences based on entity relationships"""
        n_sentences = len(sentence_entities)
        coherence_matrix = np.zeros((n_sentences, n_sentences))
        
        for i in range(n_sentences):
            for j in range(i + 1, n_sentences):
                # Calculate coherence based on shared entities and relationships
                shared_entities = set(sentence_entities[i]) & set(sentence_entities[j])
                coherence = len(shared_entities) / max(len(sentence_entities[i]) + len(sentence_entities[j]) - len(shared_entities), 1)
                
                # Add relationship-based coherence if Neo4j available
                if self.neo4j_driver and shared_entities:
                    relationship_coherence = self._calculate_relationship_coherence(
                        sentence_entities[i], sentence_entities[j]
                    )
                    coherence = (coherence * self.config.entity_boundary_weight + 
                               relationship_coherence * self.config.relationship_weight)
                
                coherence_matrix[i][j] = coherence_matrix[j][i] = coherence
        
        return coherence_matrix
    
    def _find_optimal_chunks(self, 
                           sentences: List[str],
                           sentence_entities: List[List[str]],
                           coherence_matrix: np.ndarray,
                           doc_id: str) -> List[GraphChunk]:
        """Find optimal chunk boundaries using dynamic programming"""
        
        n_sentences = len(sentences)
        if n_sentences == 0:
            return []
        
        # Dynamic programming to find optimal chunking
        dp = [0] * (n_sentences + 1)  # dp[i] = best score for first i sentences
        chunks_info = []  # Store chunk boundary information
        
        for i in range(1, n_sentences + 1):
            best_score = -float('inf')
            best_start = 0
            
            for j in range(max(0, i - 20), i):  # Limit lookback for performance
                chunk_content = " ".join(sentences[j:i])
                
                # Check size constraints
                if len(chunk_content) < self.config.min_chunk_size:
                    continue
                if len(chunk_content) > self.config.max_chunk_size:
                    break
                
                # Calculate chunk coherence score
                chunk_coherence = self._calculate_chunk_coherence(coherence_matrix, j, i)
                score = dp[j] + chunk_coherence
                
                if score > best_score:
                    best_score = score
                    best_start = j
            
            dp[i] = best_score
            chunks_info.append((best_start, i))
        
        # Reconstruct optimal chunks
        chunks = []
        chunk_boundaries = []
        
        # Backtrack to find chunk boundaries
        i = n_sentences
        while i > 0:
            start_idx = chunks_info[i-1][0]
            chunk_boundaries.append((start_idx, i))
            i = start_idx
        
        chunk_boundaries.reverse()
        
        # Create GraphChunk objects
        for chunk_idx, (start, end) in enumerate(chunk_boundaries):
            chunk_content = " ".join(sentences[start:end])
            chunk_entities = []
            
            for sent_idx in range(start, end):
                chunk_entities.extend(sentence_entities[sent_idx])
            
            chunk = self._create_graph_chunk(
                content=chunk_content,
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                entities=list(set(chunk_entities)),
                strategy="semantic_graph"
            )
            
            # Calculate coherence score for this chunk
            chunk.coherence_score = self._calculate_chunk_coherence(coherence_matrix, start, end)
            chunks.append(chunk)
        
        return chunks
    
    def _calculate_chunk_coherence(self, coherence_matrix: np.ndarray, start: int, end: int) -> float:
        """Calculate coherence score for a chunk"""
        if end <= start + 1:
            return 0.0
        
        total_coherence = 0.0
        count = 0
        
        for i in range(start, end):
            for j in range(i + 1, end):
                total_coherence += coherence_matrix[i][j]
                count += 1
        
        return total_coherence / max(count, 1)
    
    def _calculate_relationship_coherence(self, entities1: List[str], entities2: List[str]) -> float:
        """Calculate coherence based on entity relationships in graph"""
        if not self.neo4j_driver:
            return 0.0
        
        total_relationships = 0
        max_possible = len(entities1) * len(entities2)
        
        if max_possible == 0:
            return 0.0
        
        with self.neo4j_driver.session() as session:
            for e1 in entities1:
                for e2 in entities2:
                    if e1 != e2:
                        # Check if entities are related in graph
                        query = """
                        MATCH (a:Entity {name: $e1})-[r]-(b:Entity {name: $e2})
                        RETURN count(r) as relationship_count
                        """
                        result = session.run(query, e1=e1, e2=e2)
                        record = result.single()
                        if record and record['relationship_count'] > 0:
                            total_relationships += 1
        
        return total_relationships / max_possible
    
    def _get_entity_communities(self, entities: List[str]) -> Dict[str, str]:
        """Get entity communities from Neo4j using community detection"""
        if not self.neo4j_driver:
            return {}
        
        community_map = {}
        
        with self.neo4j_driver.session() as session:
            # Simple community detection based on connected components
            for entity in entities:
                if entity not in community_map:
                    # Find connected entities
                    query = """
                    MATCH (e:Entity {name: $entity})-[*1..2]-(connected:Entity)
                    RETURN collect(distinct connected.name) as community
                    """
                    result = session.run(query, entity=entity)
                    record = result.single()
                    
                    if record and record['community']:
                        # Create community ID based on sorted entity names
                        community_entities = sorted(record['community'])
                        community_id = hashlib.md5("_".join(community_entities).encode()).hexdigest()[:8]
                        
                        for ent in community_entities:
                            if ent not in community_map:
                                community_map[ent] = community_id
        
        return community_map
    
    def _get_primary_community(self, entities: List[str], communities: Dict[str, str]) -> Optional[str]:
        """Get primary community for a list of entities"""
        if not entities:
            return None
        
        community_counts = {}
        for entity in entities:
            if entity in communities:
                community_id = communities[entity]
                community_counts[community_id] = community_counts.get(community_id, 0) + 1
        
        if community_counts:
            return max(community_counts, key=community_counts.get)
        return None
    
    def _get_entity_relationships(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Get relationships between entities from Neo4j"""
        if not self.neo4j_driver or len(entities) < 2:
            return []
        
        relationships = []
        
        with self.neo4j_driver.session() as session:
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    query = """
                    MATCH (a:Entity {name: $e1})-[r]-(b:Entity {name: $e2})
                    RETURN type(r) as relationship_type, r.strength as strength
                    """
                    result = session.run(query, e1=e1, e2=e2)
                    
                    for record in result:
                        relationships.append({
                            'source': e1,
                            'target': e2,
                            'type': record['relationship_type'],
                            'strength': record.get('strength', 1.0)
                        })
        
        return relationships
    
    def _create_graph_chunk(self,
                          content: str,
                          doc_id: str,
                          chunk_idx: int,
                          entities: List[str],
                          strategy: str) -> GraphChunk:
        """Create a GraphChunk object"""
        
        chunk_id = f"{doc_id}_chunk_{chunk_idx}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        # Get relationships for entities if Neo4j available
        relationships = self._get_entity_relationships(entities) if self.neo4j_driver else []
        
        return GraphChunk(
            content=content,
            chunk_id=chunk_id,
            entities=entities,
            relationships=relationships,
            coherence_score=0.0,
            entity_density=len(entities) / max(len(content.split()), 1),
            metadata={
                'doc_id': doc_id,
                'chunk_index': chunk_idx,
                'strategy': strategy,
                'content_length': len(content),
                'entity_count': len(entities),
                'relationship_count': len(relationships)
            }
        )
    
    def _get_overlap_content(self, content: str, overlap_size: int) -> str:
        """Get overlap content from end of chunk"""
        words = content.split()
        if len(words) <= overlap_size:
            return content
        return " ".join(words[-overlap_size:]) + " "
    
    def _fallback_chunking(self, document: Dict[str, Any]) -> List[GraphChunk]:
        """Fallback to simple chunking when graph methods fail"""
        content = document.get('content', '')
        doc_id = document.get('id', 'unknown')
        
        chunks = []
        words = content.split()
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            if len(" ".join(current_chunk)) >= self.config.target_chunk_size:
                chunk_content = " ".join(current_chunk)
                entities = self._extract_entities_from_text(chunk_content)
                
                chunk = self._create_graph_chunk(
                    content=chunk_content,
                    doc_id=doc_id,
                    chunk_idx=len(chunks),
                    entities=entities,
                    strategy="fallback"
                )
                chunks.append(chunk)
                
                # Overlap
                overlap_words = min(self.config.overlap_size, len(current_chunk) // 2)
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
        
        # Handle remaining content
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            entities = self._extract_entities_from_text(chunk_content)
            
            chunk = self._create_graph_chunk(
                content=chunk_content,
                doc_id=doc_id,
                chunk_idx=len(chunks),
                entities=entities,
                strategy="fallback"
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _optimize_chunk_boundaries(self, chunks: List[GraphChunk]) -> List[GraphChunk]:
        """Optimize chunk boundaries across documents"""
        
        # Group chunks by document for optimization
        doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get('doc_id', 'unknown')
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk)
        
        optimized_chunks = []
        
        # Optimize each document's chunks
        for doc_id, doc_chunk_list in doc_chunks.items():
            # Sort chunks by index
            doc_chunk_list.sort(key=lambda c: c.metadata.get('chunk_index', 0))
            
            # Merge small adjacent chunks if beneficial
            merged_chunks = self._merge_small_chunks(doc_chunk_list)
            
            # Split large chunks if necessary
            final_chunks = self._split_large_chunks(merged_chunks)
            
            optimized_chunks.extend(final_chunks)
        
        return optimized_chunks
    
    def _merge_small_chunks(self, chunks: List[GraphChunk]) -> List[GraphChunk]:
        """Merge small adjacent chunks"""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check if current chunk is small and can be merged
            if (len(current_chunk.content) < self.config.min_chunk_size and 
                i + 1 < len(chunks) and
                len(current_chunk.content) + len(chunks[i + 1].content) <= self.config.max_chunk_size):
                
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_content = current_chunk.content + " " + next_chunk.content
                merged_entities = list(set(current_chunk.entities + next_chunk.entities))
                merged_relationships = current_chunk.relationships + next_chunk.relationships
                
                merged_chunk = GraphChunk(
                    content=merged_content,
                    chunk_id=f"{current_chunk.chunk_id}_merged",
                    entities=merged_entities,
                    relationships=merged_relationships,
                    coherence_score=(current_chunk.coherence_score + next_chunk.coherence_score) / 2,
                    entity_density=len(merged_entities) / max(len(merged_content.split()), 1),
                    metadata={
                        **current_chunk.metadata,
                        'merged_from': [current_chunk.chunk_id, next_chunk.chunk_id],
                        'content_length': len(merged_content),
                        'entity_count': len(merged_entities)
                    }
                )
                
                merged.append(merged_chunk)
                i += 2  # Skip next chunk as it's been merged
            else:
                merged.append(current_chunk)
                i += 1
        
        return merged
    
    def _split_large_chunks(self, chunks: List[GraphChunk]) -> List[GraphChunk]:
        """Split chunks that are too large"""
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.content) > self.config.max_chunk_size:
                # Split the chunk
                sub_chunks = self._split_single_chunk(chunk)
                split_chunks.extend(sub_chunks)
            else:
                split_chunks.append(chunk)
        
        return split_chunks
    
    def _split_single_chunk(self, chunk: GraphChunk) -> List[GraphChunk]:
        """Split a single large chunk"""
        sentences = self._split_into_sentences(chunk.content)
        
        sub_chunks = []
        current_content = ""
        current_entities = []
        
        for sentence in sentences:
            sentence_entities = self._extract_entities_from_text(sentence)
            
            if len(current_content + sentence) > self.config.max_chunk_size and current_content:
                # Create sub-chunk
                sub_chunk = GraphChunk(
                    content=current_content.strip(),
                    chunk_id=f"{chunk.chunk_id}_split_{len(sub_chunks)}",
                    entities=list(set(current_entities)),
                    relationships=self._get_entity_relationships(list(set(current_entities))),
                    coherence_score=chunk.coherence_score,  # Inherit parent coherence
                    entity_density=len(set(current_entities)) / max(len(current_content.split()), 1),
                    metadata={
                        **chunk.metadata,
                        'split_from': chunk.chunk_id,
                        'split_index': len(sub_chunks),
                        'content_length': len(current_content),
                        'entity_count': len(set(current_entities))
                    }
                )
                sub_chunks.append(sub_chunk)
                
                # Start new sub-chunk with overlap
                overlap_content = self._get_overlap_content(current_content, self.config.overlap_size)
                current_content = overlap_content + sentence
                current_entities = sentence_entities
            else:
                current_content += sentence
                current_entities.extend(sentence_entities)
        
        # Handle remaining content
        if current_content.strip():
            sub_chunk = GraphChunk(
                content=current_content.strip(),
                chunk_id=f"{chunk.chunk_id}_split_{len(sub_chunks)}",
                entities=list(set(current_entities)),
                relationships=self._get_entity_relationships(list(set(current_entities))),
                coherence_score=chunk.coherence_score,
                entity_density=len(set(current_entities)) / max(len(current_content.split()), 1),
                metadata={
                    **chunk.metadata,
                    'split_from': chunk.chunk_id,
                    'split_index': len(sub_chunks),
                    'content_length': len(current_content),
                    'entity_count': len(set(current_entities))
                }
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk]
    
    def _clear_caches(self):
        """Clear caches to manage memory usage"""
        self.entity_cache.clear()
        self.relationship_cache.clear()
        self.community_cache.clear()
        logger.info("Cleared graph chunking caches for memory optimization")
