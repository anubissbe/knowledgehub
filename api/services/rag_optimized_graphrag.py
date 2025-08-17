"""
Optimized GraphRAG Service with Advanced Caching and Performance Enhancements
Combines Neo4j graph operations with intelligent caching for maximum performance

Author: Adrien Stevens - Python Performance Optimization Expert
"""

import asyncio
import logging
import hashlib
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from .graphrag_service import GraphRAGService, GraphRAGStrategy, GraphRAGResult
from .rag_cache_optimizer import get_rag_cache_optimizer

logger = logging.getLogger(__name__)


class OptimizedGraphRAGService(GraphRAGService):
    """
    Performance-optimized GraphRAG service with intelligent caching
    
    Enhancements:
    - Cached Neo4j query results
    - Optimized graph traversal patterns  
    - Batch entity processing
    - Memory-efficient graph operations
    - Intelligent prefetching of related nodes
    """
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://192.168.1.25:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "knowledgehub123"):
        super().__init__(neo4j_uri, neo4j_user, neo4j_password)
        self.cache_optimizer = None
        
        # Performance tracking
        self.graph_query_count = 0
        self.cache_hit_count = 0
        self.total_query_time = 0.0
        
    async def initialize_with_cache_optimizer(self):
        """Initialize with cache optimizer integration"""
        try:
            # Initialize base GraphRAG
            await self.initialize_rag_pipeline()
            
            # Get cache optimizer
            self.cache_optimizer = await get_rag_cache_optimizer()
            
            logger.info("Optimized GraphRAG service initialized with cache optimizer")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized GraphRAG service: {e}")
            raise
    
    async def query_graphrag_optimized(
        self,
        query: str,
        strategy: GraphRAGStrategy = GraphRAGStrategy.HYBRID_PARALLEL,
        max_results: int = 10,
        include_reasoning: bool = True,
        use_cache: bool = True
    ) -> List[GraphRAGResult]:
        """
        Optimized GraphRAG query with intelligent caching and performance enhancements
        """
        start_time = time.time()
        self.graph_query_count += 1
        
        try:
            # Generate cache key for the query
            query_hash = self._generate_graph_query_hash(query, strategy, max_results, include_reasoning)
            
            # Try cache first if enabled
            if use_cache and self.cache_optimizer:
                cached_results = await self._get_cached_graph_results(query_hash)
                if cached_results:
                    self.cache_hit_count += 1
                    
                    # Schedule prefetch of related graph queries
                    asyncio.create_task(self._prefetch_related_graph_queries(query))
                    
                    return cached_results
            
            # Execute optimized graph query
            results = await self._execute_optimized_graph_query(
                query, strategy, max_results, include_reasoning
            )
            
            # Cache successful results
            if use_cache and self.cache_optimizer and results:
                await self._cache_graph_results(query_hash, results)
            
            # Update performance metrics
            query_time = time.time() - start_time
            self.total_query_time += query_time
            
            # Add performance metadata to results
            for result in results:
                result.metadata.update({
                    'cache_hit': False,
                    'query_time_ms': query_time * 1000,
                    'graph_query_count': self.graph_query_count
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Optimized GraphRAG query failed: {e}")
            return []
    
    async def _execute_optimized_graph_query(
        self,
        query: str,
        strategy: GraphRAGStrategy,
        max_results: int,
        include_reasoning: bool
    ) -> List[GraphRAGResult]:
        """Execute graph query with performance optimizations"""
        
        if strategy == GraphRAGStrategy.HYBRID_PARALLEL:
            return await self._optimized_hybrid_parallel_query(query, max_results, include_reasoning)
        elif strategy == GraphRAGStrategy.ENTITY_CENTRIC:
            return await self._optimized_entity_centric_query(query, max_results, include_reasoning)
        else:
            # Fallback to base implementation
            return await super().query_graphrag(query, strategy, max_results, include_reasoning)
    
    async def _optimized_hybrid_parallel_query(
        self, 
        query: str, 
        max_results: int, 
        include_reasoning: bool
    ) -> List[GraphRAGResult]:
        """Optimized hybrid parallel query with caching"""
        
        # Extract entities with caching
        query_entities = await self._get_cached_query_entities(query)
        
        if not query_entities:
            return []
        
        # Run vector and graph queries in parallel with optimization
        vector_task = asyncio.create_task(self._cached_vector_query(query, max_results))
        graph_task = asyncio.create_task(self._optimized_graph_query(query_entities, max_results))
        
        vector_results, graph_results = await asyncio.gather(
            vector_task, graph_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(vector_results, Exception):
            logger.error(f"Vector query failed: {vector_results}")
            vector_results = []
        
        if isinstance(graph_results, Exception):
            logger.error(f"Graph query failed: {graph_results}")
            graph_results = []
        
        # Combine results with improved scoring
        return self._merge_and_rank_results(vector_results, graph_results, max_results, include_reasoning)
    
    async def _get_cached_query_entities(self, query: str) -> List[Any]:
        """Get entities from query with caching"""
        
        # Generate cache key for entity extraction
        entity_cache_key = f"entities_{hashlib.md5(query.encode()).hexdigest()}"
        
        if self.cache_optimizer:
            cached_entities = await self.cache_optimizer.get_cached_query_results(entity_cache_key)
            if cached_entities:
                return cached_entities[0] if cached_entities else []
        
        # Extract entities
        entities = self._extract_entities_from_text(query)
        
        # Cache entities
        if self.cache_optimizer and entities:
            await self.cache_optimizer.cache_query_results(
                entity_cache_key,
                [entities],
                metadata={'operation': 'entity_extraction'},
                ttl=3600  # 1 hour cache for entity extraction
            )
        
        return entities
    
    async def _cached_vector_query(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Vector query with result caching"""
        
        vector_cache_key = f"vector_{hashlib.md5(f'{query}:{max_results}'.encode()).hexdigest()}"
        
        if self.cache_optimizer:
            cached_vector = await self.cache_optimizer.get_cached_query_results(vector_cache_key)
            if cached_vector:
                return cached_vector[0] if cached_vector else []
        
        # Execute vector query
        vector_results = await self._vector_query(query, max_results)
        
        # Cache vector results
        if self.cache_optimizer and vector_results:
            await self.cache_optimizer.cache_query_results(
                vector_cache_key,
                [vector_results],
                metadata={'operation': 'vector_query'},
                ttl=600  # 10 minutes for vector results
            )
        
        return vector_results
    
    async def _optimized_graph_query(
        self, 
        query_entities: List[Any], 
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Optimized graph traversal with batching and caching"""
        
        try:
            # Batch entity processing for better performance
            results = []
            batch_size = 5  # Process entities in batches
            
            for i in range(0, len(query_entities), batch_size):
                batch_entities = query_entities[i:i + batch_size]
                
                # Process batch in parallel
                batch_tasks = [
                    self._cached_traverse_graph_from_entity(entity.entity)
                    for entity in batch_entities
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Collect successful results
                for batch_result in batch_results:
                    if not isinstance(batch_result, Exception):
                        results.extend(batch_result)
            
            # Remove duplicates and sort by relevance
            unique_results = self._deduplicate_graph_results(results)
            return unique_results[:max_results]
            
        except Exception as e:
            logger.error(f"Optimized graph query failed: {e}")
            return []
    
    async def _cached_traverse_graph_from_entity(self, entity: str) -> List[Dict[str, Any]]:
        """Traverse graph from entity with caching"""
        
        # Generate cache key for graph traversal
        traversal_cache_key = f"graph_traversal_{hashlib.md5(entity.encode()).hexdigest()}"
        
        if self.cache_optimizer:
            cached_traversal = await self.cache_optimizer.get_cached_neo4j_results(traversal_cache_key)
            if cached_traversal:
                return cached_traversal
        
        # Execute graph traversal
        try:
            with self.driver.session() as session:
                traversal_results = await self._traverse_graph_from_entity(
                    session, entity, self.config.max_depth
                )
                
            # Cache traversal results
            if self.cache_optimizer and traversal_results:
                await self.cache_optimizer.cache_neo4j_results(
                    traversal_cache_key,
                    traversal_results,
                    ttl=900  # 15 minutes for graph traversal
                )
                
            return traversal_results
            
        except Exception as e:
            logger.error(f"Graph traversal failed for entity {entity}: {e}")
            return []
    
    def _deduplicate_graph_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate graph results efficiently"""
        
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_key = result.get('content', '')[:100]  # Use first 100 chars as key
            
            if content_key and content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        # Sort by score (descending)
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return unique_results
    
    def _merge_and_rank_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        max_results: int,
        include_reasoning: bool
    ) -> List[GraphRAGResult]:
        """Merge and rank vector and graph results with improved scoring"""
        
        combined_results = []
        
        # Create content-based lookup for efficient merging
        vector_dict = {self._get_content_key(r): r for r in vector_results}
        graph_dict = {self._get_content_key(r): r for r in graph_results}
        
        # Process common results (highest priority)
        common_keys = set(vector_dict.keys()) & set(graph_dict.keys())
        for key in common_keys:
            v_result = vector_dict[key]
            g_result = graph_dict[key]
            
            # Weighted scoring: vector similarity + graph connectivity
            combined_score = (v_result.get('score', 0) * 0.6) + (g_result.get('score', 0) * 0.4)
            
            result = GraphRAGResult(
                content=v_result.get('content', ''),
                score=combined_score,
                vector_score=v_result.get('score', 0),
                graph_score=g_result.get('score', 0),
                entities=g_result.get('entities', []),
                relationships=g_result.get('relationships', []),
                reasoning_path=g_result.get('reasoning_path', []) if include_reasoning else [],
                metadata={
                    'source': 'hybrid_optimized',
                    'vector_match': True,
                    'graph_match': True,
                    'cache_optimized': True
                }
            )
            combined_results.append(result)
        
        # Add vector-only results
        for key, v_result in vector_dict.items():
            if key not in common_keys:
                result = GraphRAGResult(
                    content=v_result.get('content', ''),
                    score=v_result.get('score', 0) * 0.8,  # Slight penalty
                    vector_score=v_result.get('score', 0),
                    graph_score=0.0,
                    entities=[],
                    relationships=[],
                    reasoning_path=[],
                    metadata={
                        'source': 'vector_only_optimized',
                        'vector_match': True,
                        'graph_match': False
                    }
                )
                combined_results.append(result)
        
        # Add graph-only results
        for key, g_result in graph_dict.items():
            if key not in common_keys:
                result = GraphRAGResult(
                    content=g_result.get('content', ''),
                    score=g_result.get('score', 0) * 0.7,  # Penalty for no vector match
                    vector_score=0.0,
                    graph_score=g_result.get('score', 0),
                    entities=g_result.get('entities', []),
                    relationships=g_result.get('relationships', []),
                    reasoning_path=g_result.get('reasoning_path', []) if include_reasoning else [],
                    metadata={
                        'source': 'graph_only_optimized',
                        'vector_match': False,
                        'graph_match': True
                    }
                )
                combined_results.append(result)
        
        # Sort by combined score and return top results
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:max_results]
    
    def _get_content_key(self, result: Dict[str, Any]) -> str:
        """Generate consistent content key for result merging"""
        content = result.get('content', '')
        return content[:100] if content else ''
    
    async def _get_cached_graph_results(self, query_hash: str) -> Optional[List[GraphRAGResult]]:
        """Retrieve cached graph results and convert to GraphRAGResult objects"""
        
        if not self.cache_optimizer:
            return None
            
        try:
            cached_data = await self.cache_optimizer.get_cached_neo4j_results(query_hash)
            
            if not cached_data:
                return None
            
            # Convert cached data back to GraphRAGResult objects
            results = []
            for item in cached_data:
                if isinstance(item, dict):
                    result = GraphRAGResult(
                        content=item.get('content', ''),
                        score=item.get('score', 0.0),
                        vector_score=item.get('vector_score', 0.0),
                        graph_score=item.get('graph_score', 0.0),
                        entities=item.get('entities', []),
                        relationships=item.get('relationships', []),
                        reasoning_path=item.get('reasoning_path', []),
                        metadata={**item.get('metadata', {}), 'cache_hit': True}
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached graph results: {e}")
            return None
    
    async def _cache_graph_results(self, query_hash: str, results: List[GraphRAGResult]):
        """Cache graph results for future queries"""
        
        if not self.cache_optimizer:
            return
            
        try:
            # Convert GraphRAGResult objects to dictionary format for caching
            cache_data = []
            for result in results:
                cache_item = {
                    'content': result.content,
                    'score': result.score,
                    'vector_score': result.vector_score,
                    'graph_score': result.graph_score,
                    'entities': result.entities,
                    'relationships': result.relationships,
                    'reasoning_path': result.reasoning_path,
                    'metadata': result.metadata
                }
                cache_data.append(cache_item)
            
            await self.cache_optimizer.cache_neo4j_results(
                query_hash,
                cache_data,
                ttl=900  # 15 minutes for graph query results
            )
            
        except Exception as e:
            logger.error(f"Failed to cache graph results: {e}")
    
    async def _prefetch_related_graph_queries(self, query: str):
        """Prefetch related graph queries based on entity relationships"""
        try:
            # Extract entities from the query
            entities = await self._get_cached_query_entities(query)
            
            # For each entity, prefetch related entities and their connections
            for entity in entities[:2]:  # Limit to prevent overwhelming
                if hasattr(entity, 'entity'):
                    # Find related entities
                    with self.driver.session() as session:
                        related_query = """
                        MATCH (e:Entity {name: $entity})-[r]-(related:Entity)
                        RETURN related.name as related_entity
                        LIMIT 3
                        """
                        result = session.run(related_query, entity=entity.entity)
                        
                        # Prefetch queries for related entities
                        for record in result:
                            related_entity = record['related_entity']
                            prefetch_query = f"{related_entity} {query}"
                            
                            # Execute prefetch in background
                            asyncio.create_task(
                                self._execute_background_prefetch(prefetch_query)
                            )
                            
        except Exception as e:
            logger.debug(f"Graph prefetch failed (non-critical): {e}")
    
    async def _execute_background_prefetch(self, query: str):
        """Execute a prefetch query in the background"""
        try:
            await self.query_graphrag_optimized(
                query=query,
                strategy=GraphRAGStrategy.ENTITY_CENTRIC,
                max_results=3,  # Smaller result set for prefetch
                include_reasoning=False,
                use_cache=True
            )
        except Exception as e:
            logger.debug(f"Background graph prefetch failed: {e}")
    
    def _generate_graph_query_hash(
        self,
        query: str,
        strategy: GraphRAGStrategy,
        max_results: int,
        include_reasoning: bool
    ) -> str:
        """Generate hash for graph query caching"""
        hash_material = f"{query}:{strategy.value}:{max_results}:{include_reasoning}"
        return hashlib.md5(hash_material.encode()).hexdigest()
    
    async def get_graph_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for GraphRAG service"""
        try:
            base_metrics = self.get_memory_stats()
            
            service_metrics = {
                'graph_queries_processed': self.graph_query_count,
                'cache_hits': self.cache_hit_count,
                'cache_hit_ratio': self.cache_hit_count / self.graph_query_count if self.graph_query_count > 0 else 0,
                'total_query_time_ms': self.total_query_time * 1000,
                'avg_query_time_ms': (self.total_query_time / self.graph_query_count) * 1000 if self.graph_query_count > 0 else 0
            }
            
            return {
                'service_metrics': service_metrics,
                'memory_stats': base_metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph performance metrics: {e}")
            return {'error': str(e)}
    
    async def batch_index_documents_optimized(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Batch index documents with optimized performance"""
        
        start_time = time.time()
        total_stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'relationships_created': 0,
            'batches_processed': 0
        }
        
        try:
            # Process documents in batches
            batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
            
            for batch in batches:
                batch_stats = await self.index_documents_with_graph(
                    documents=batch,
                    extract_entities=True,
                    build_relationships=True
                )
                
                # Accumulate stats
                total_stats['documents_processed'] += batch_stats.get('documents_processed', 0)
                total_stats['entities_extracted'] += batch_stats.get('entities_extracted', 0)
                total_stats['relationships_created'] += batch_stats.get('relationships_created', 0)
                total_stats['batches_processed'] += 1
                
                # Brief pause to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            total_stats['processing_time_ms'] = int((time.time() - start_time) * 1000)
            total_stats['avg_time_per_doc_ms'] = total_stats['processing_time_ms'] / len(documents) if documents else 0
            
            logger.info(f"Batch indexing completed: {total_stats}")
            return total_stats
            
        except Exception as e:
            logger.error(f"Batch indexing failed: {e}")
            return {
                'error': str(e),
                'documents_processed': total_stats.get('documents_processed', 0),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }


# Global optimized GraphRAG service instance  
_optimized_graphrag_service: Optional[OptimizedGraphRAGService] = None


async def get_optimized_graphrag_service() -> OptimizedGraphRAGService:
    """Get or create the optimized GraphRAG service instance"""
    global _optimized_graphrag_service
    
    if _optimized_graphrag_service is None:
        _optimized_graphrag_service = OptimizedGraphRAGService()
        await _optimized_graphrag_service.initialize_with_cache_optimizer()
    
    return _optimized_graphrag_service
