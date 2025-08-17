"""
Optimized LlamaIndex RAG Service with Advanced Caching
Integration of LlamaIndex with the RAG Cache Optimizer for maximum performance

Author: Adrien Stevens - Python Performance Optimization Expert
"""

import asyncio
import logging
import hashlib
import time
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime
import numpy as np

from llama_index.core import VectorStoreIndex, Document, Response
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine

from .rag.llamaindex_service import LlamaIndexRAGService
from .rag_cache_optimizer import get_rag_cache_optimizer
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class OptimizedLlamaIndexRAGService(LlamaIndexRAGService):
    """
    Performance-optimized LlamaIndex RAG service with intelligent caching
    
    Enhancements:
    - Multi-layer caching for embeddings and results
    - Batch processing optimizations
    - Intelligent prefetching
    - Memory-efficient operations
    - Performance monitoring
    """
    
    def __init__(self):
        super().__init__()
        self.cache_optimizer = None
        self.embedding_service = EmbeddingService()
        
        # Performance tracking
        self.query_count = 0
        self.cache_hit_count = 0
        self.total_query_time = 0.0
        
    async def initialize_with_cache_optimizer(self):
        """Initialize with cache optimizer integration"""
        try:
            # Initialize base service
            await self.initialize_index()
            
            # Get cache optimizer
            self.cache_optimizer = await get_rag_cache_optimizer()
            
            logger.info("Optimized LlamaIndex RAG service initialized with cache optimizer")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized RAG service: {e}")
            raise
    
    async def ingest_document_optimized(
        self,
        content: str,
        metadata: Dict[str, Any],
        source_type: str = "documentation",
        use_batch_processing: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized document ingestion with caching and batch processing
        """
        start_time = time.time()
        
        try:
            # Generate document hash for caching
            doc_hash = self._generate_document_hash(content, metadata)
            
            # Check if document already processed
            if self.cache_optimizer:
                cached_result = await self.cache_optimizer.get_cached_query_results(
                    f"doc_ingest_{doc_hash}"
                )
                if cached_result:
                    logger.info(f"Document ingestion cache hit for {doc_hash}")
                    return cached_result[0] if cached_result else {}
            
            # Process document chunks in batches for efficiency
            if use_batch_processing:
                result = await self._batch_ingest_document(content, metadata, source_type)
            else:
                result = await self.ingest_document(content, metadata, source_type)
            
            # Cache the ingestion result
            if self.cache_optimizer and result.get('success'):
                await self.cache_optimizer.cache_query_results(
                    f"doc_ingest_{doc_hash}",
                    [result],
                    metadata={'operation': 'document_ingestion'},
                    ttl=3600  # 1 hour cache
                )
            
            # Add performance metrics
            processing_time = time.time() - start_time
            result['performance'] = {
                'processing_time_ms': processing_time * 1000,
                'cache_used': bool(self.cache_optimizer),
                'batch_processing': use_batch_processing
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized document ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    async def _batch_ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        source_type: str
    ) -> Dict[str, Any]:
        """Batch process document ingestion for better performance"""
        
        # Split content into optimal chunks
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        text_chunks = splitter.split_text(content)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(text_chunks):
            doc = Document(
                text=chunk,
                metadata={
                    **metadata,
                    'chunk_id': i,
                    'chunk_count': len(text_chunks),
                    'source_type': source_type
                }
            )
            documents.append(doc)
        
        # Batch process embeddings if cache optimizer available
        if self.cache_optimizer:
            await self._batch_cache_embeddings(documents)
        
        # Insert into index
        nodes = await asyncio.to_thread(
            self.ingestion_pipeline.run,
            documents=documents
        )
        
        self.index.insert_nodes(nodes)
        
        return {
            'success': True,
            'chunks_created': len(nodes),
            'documents_processed': len(documents),
            'metadata': metadata,
            'source_type': source_type
        }
    
    async def _batch_cache_embeddings(self, documents: List[Document]):
        """Batch cache embeddings for documents"""
        try:
            # Extract texts for embedding
            texts = [doc.text for doc in documents]
            
            # Generate embeddings in batch
            embeddings = await asyncio.to_thread(
                self.embedding_service.get_embeddings_batch,
                texts
            )
            
            # Cache embeddings individually
            cache_tasks = []
            for doc, embedding in zip(documents, embeddings):
                doc_hash = self._generate_document_hash(doc.text, doc.metadata)
                
                cache_task = self.cache_optimizer.cache_embeddings(
                    key=doc_hash,
                    embeddings=np.array(embedding),
                    metadata=doc.metadata,
                    ttl=7200  # 2 hours
                )
                cache_tasks.append(cache_task)
            
            # Execute caching in parallel
            await asyncio.gather(*cache_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Batch embedding caching failed: {e}")
    
    async def query_optimized(
        self,
        query_text: str,
        user_id: str,
        project_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_cache: bool = True,
        enable_prefetch: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized RAG query with multi-layer caching and intelligent optimizations
        """
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Generate cache key
            query_hash = self._generate_query_hash(query_text, user_id, project_id, filters, top_k)
            
            # Try cache first if enabled
            if use_cache and self.cache_optimizer:
                cached_result = await self.cache_optimizer.get_cached_query_results(query_hash)
                if cached_result:
                    self.cache_hit_count += 1
                    
                    # Schedule prefetch of related queries if enabled
                    if enable_prefetch:
                        asyncio.create_task(self._prefetch_related_queries(query_text, user_id))
                    
                    # Add performance metadata
                    result = cached_result[0] if cached_result else {}
                    result['performance'] = {
                        'cache_hit': True,
                        'query_time_ms': (time.time() - start_time) * 1000,
                        'total_queries': self.query_count,
                        'cache_hit_ratio': self.cache_hit_count / self.query_count
                    }
                    
                    return result
            
            # Execute query with optimizations
            result = await self._execute_optimized_query(
                query_text, user_id, project_id, filters, top_k
            )
            
            # Cache successful results
            if use_cache and self.cache_optimizer and result.get('response'):
                await self.cache_optimizer.cache_query_results(
                    query_hash,
                    [result],
                    metadata={
                        'user_id': user_id,
                        'project_id': project_id,
                        'query_type': 'rag_query'
                    },
                    ttl=300  # 5 minutes for query results
                )
            
            # Add performance metadata
            query_time = time.time() - start_time
            self.total_query_time += query_time
            
            result['performance'] = {
                'cache_hit': False,
                'query_time_ms': query_time * 1000,
                'total_queries': self.query_count,
                'cache_hit_ratio': self.cache_hit_count / self.query_count if self.query_count > 0 else 0,
                'avg_query_time_ms': (self.total_query_time / self.query_count) * 1000 if self.query_count > 0 else 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized RAG query failed: {e}")
            
            # Return error with performance metadata
            return {
                'response': f"Query failed: {str(e)}",
                'source_nodes': [],
                'metadata': {
                    'error': str(e),
                    'query': query_text
                },
                'performance': {
                    'cache_hit': False,
                    'query_time_ms': (time.time() - start_time) * 1000,
                    'error': True
                }
            }
    
    async def _execute_optimized_query(
        self,
        query_text: str,
        user_id: str,
        project_id: Optional[str],
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> Dict[str, Any]:
        """Execute the actual query with optimizations"""
        
        # Pre-compute query embedding if cache optimizer available
        query_embedding = None
        if self.cache_optimizer:
            query_hash = hashlib.md5(query_text.encode()).hexdigest()
            cached_embedding = await self.cache_optimizer.get_cached_embeddings(f"query_{query_hash}")
            
            if cached_embedding:
                query_embedding = cached_embedding[0]
            else:
                # Generate and cache query embedding
                embedding = await asyncio.to_thread(
                    self.embedding_service.get_embedding,
                    query_text
                )
                query_embedding = np.array(embedding)
                
                await self.cache_optimizer.cache_embeddings(
                    f"query_{query_hash}",
                    query_embedding,
                    metadata={'query_type': 'user_query'},
                    ttl=1800  # 30 minutes
                )
        
        # Execute query using parent class method
        return await self.query(
            query_text=query_text,
            user_id=user_id,
            project_id=project_id,
            filters=filters,
            top_k=top_k
        )
    
    async def _prefetch_related_queries(self, query_text: str, user_id: str):
        """Prefetch related queries based on patterns"""
        try:
            # Simple prefetch logic - could be enhanced with ML
            # For now, prefetch variations of the query
            variations = [
                f"how to {query_text}",
                f"what is {query_text}",
                f"example of {query_text}",
                f"{query_text} tutorial",
                f"{query_text} best practices"
            ]
            
            # Limit prefetch to avoid overwhelming the system
            for variation in variations[:2]:
                variation_hash = self._generate_query_hash(variation, user_id, None, None, 5)
                
                # Check if already cached
                if self.cache_optimizer:
                    cached = await self.cache_optimizer.get_cached_query_results(variation_hash)
                    if not cached:
                        # Execute and cache prefetch query
                        asyncio.create_task(self._execute_prefetch_query(variation, user_id))
                        
        except Exception as e:
            logger.debug(f"Prefetch failed (non-critical): {e}")
    
    async def _execute_prefetch_query(self, query_text: str, user_id: str):
        """Execute a prefetch query in the background"""
        try:
            await self.query_optimized(
                query_text=query_text,
                user_id=user_id,
                top_k=3,  # Smaller result set for prefetch
                use_cache=True,
                enable_prefetch=False  # Avoid recursive prefetching
            )
        except Exception as e:
            logger.debug(f"Background prefetch query failed: {e}")
    
    def _generate_document_hash(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate hash for document caching"""
        hash_material = f"{content[:1000]}{str(sorted(metadata.items()))}"
        return hashlib.md5(hash_material.encode()).hexdigest()
    
    def _generate_query_hash(
        self,
        query: str,
        user_id: str,
        project_id: Optional[str],
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> str:
        """Generate hash for query caching"""
        hash_material = f"{query}:{user_id}:{project_id}:{str(filters)}:{top_k}"
        return hashlib.md5(hash_material.encode()).hexdigest()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this service"""
        try:
            cache_stats = {}
            if self.cache_optimizer:
                cache_stats = await self.cache_optimizer.get_performance_stats()
            
            service_metrics = {
                'queries_processed': self.query_count,
                'cache_hits': self.cache_hit_count,
                'cache_hit_ratio': self.cache_hit_count / self.query_count if self.query_count > 0 else 0,
                'total_query_time_ms': self.total_query_time * 1000,
                'avg_query_time_ms': (self.total_query_time / self.query_count) * 1000 if self.query_count > 0 else 0
            }
            
            return {
                'service_metrics': service_metrics,
                'cache_optimizer_stats': cache_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {'error': str(e)}
    
    async def batch_query(
        self,
        queries: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple queries in optimized batches"""
        try:
            results = []
            
            # Process queries in batches
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                
                # Execute batch concurrently
                batch_tasks = []
                for query_data in batch:
                    task = self.query_optimized(
                        query_text=query_data.get('query', ''),
                        user_id=query_data.get('user_id', 'anonymous'),
                        project_id=query_data.get('project_id'),
                        filters=query_data.get('filters'),
                        top_k=query_data.get('top_k', 5)
                    )
                    batch_tasks.append(task)
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Filter out exceptions and add to results
                for result in batch_results:
                    if not isinstance(result, Exception):
                        results.append(result)
                    else:
                        logger.error(f"Batch query error: {result}")
                        results.append({
                            'response': f"Query failed: {str(result)}",
                            'source_nodes': [],
                            'metadata': {'error': str(result)}
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch query processing failed: {e}")
            return [{'error': str(e)} for _ in queries]


# Global optimized service instance
_optimized_rag_service: Optional[OptimizedLlamaIndexRAGService] = None


async def get_optimized_rag_service() -> OptimizedLlamaIndexRAGService:
    """Get or create the optimized RAG service instance"""
    global _optimized_rag_service
    
    if _optimized_rag_service is None:
        _optimized_rag_service = OptimizedLlamaIndexRAGService()
        await _optimized_rag_service.initialize_with_cache_optimizer()
    
    return _optimized_rag_service
