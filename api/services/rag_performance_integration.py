"""
RAG Performance Integration Service
Integrates all performance optimizations with existing RAG services

Author: Adrien Stevens - Python Performance Optimization Expert
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .rag.llamaindex_service import get_rag_service as get_base_llama_service
from .graphrag_service import get_graphrag_service as get_base_graphrag_service
from .rag_cache_optimizer import get_rag_cache_optimizer
from .async_rag_optimizer import get_async_rag_optimizer
from .rag_optimized_llamaindex import get_optimized_rag_service
from .rag_optimized_graphrag import get_optimized_graphrag_service

logger = logging.getLogger(__name__)


class RAGPerformanceIntegration:
    """
    Central integration service for RAG performance optimizations
    
    Features:
    - Seamless integration with existing RAG services
    - Automatic fallback to base services if optimizations fail
    - Performance monitoring and alerting
    - Dynamic optimization switching
    """
    
    def __init__(self):
        self.cache_optimizer = None
        self.async_optimizer = None
        self.optimized_llama_service = None
        self.optimized_graphrag_service = None
        
        # Base services for fallback
        self.base_llama_service = None
        self.base_graphrag_service = None
        
        # Integration state
        self.initialized = False
        self.optimization_enabled = True
        self.performance_monitoring_enabled = True
        
        # Performance tracking
        self.integration_metrics = {
            'total_queries': 0,
            'optimized_queries': 0,
            'fallback_queries': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize the performance integration system"""
        try:
            logger.info("Initializing RAG Performance Integration")
            
            # Initialize optimization components
            self.cache_optimizer = await get_rag_cache_optimizer()
            self.async_optimizer = await get_async_rag_optimizer()
            
            # Initialize optimized services
            try:
                self.optimized_llama_service = await get_optimized_rag_service()
                logger.info("Optimized LlamaIndex service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize optimized LlamaIndex service: {e}")
            
            try:
                self.optimized_graphrag_service = await get_optimized_graphrag_service()
                logger.info("Optimized GraphRAG service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize optimized GraphRAG service: {e}")
            
            # Initialize base services for fallback
            try:
                self.base_llama_service = get_base_llama_service()
                await self.base_llama_service.initialize_index()
                logger.info("Base LlamaIndex service initialized as fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize base LlamaIndex service: {e}")
            
            try:
                self.base_graphrag_service = await get_base_graphrag_service()
                logger.info("Base GraphRAG service initialized as fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize base GraphRAG service: {e}")
            
            self.initialized = True
            logger.info("RAG Performance Integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Performance Integration: {e}")
            raise
    
    async def query_with_optimization(
        self,
        query_text: str,
        user_id: str,
        service_type: str = "llama",  # "llama" or "graphrag"
        project_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute query with automatic optimization and fallback
        """
        self.integration_metrics['total_queries'] += 1
        
        try:
            # Use optimized service if available and enabled
            if self.optimization_enabled:
                if service_type == "llama" and self.optimized_llama_service:
                    result = await self._query_optimized_llama(
                        query_text, user_id, project_id, filters, top_k, **kwargs
                    )
                    self.integration_metrics['optimized_queries'] += 1
                    return result
                
                elif service_type == "graphrag" and self.optimized_graphrag_service:
                    result = await self._query_optimized_graphrag(
                        query_text, user_id, top_k, **kwargs
                    )
                    self.integration_metrics['optimized_queries'] += 1
                    return result
            
            # Fallback to base service
            logger.info(f"Using fallback service for {service_type} query")
            result = await self._query_fallback_service(
                service_type, query_text, user_id, project_id, filters, top_k, **kwargs
            )
            self.integration_metrics['fallback_queries'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Query with optimization failed: {e}")
            self.integration_metrics['errors'] += 1
            
            # Final fallback attempt
            try:
                result = await self._query_fallback_service(
                    service_type, query_text, user_id, project_id, filters, top_k, **kwargs
                )
                result['metadata'] = result.get('metadata', {})
                result['metadata']['fallback_used'] = True
                result['metadata']['original_error'] = str(e)
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {fallback_error}")
                return {
                    'response': f"Query failed: {str(e)}",
                    'source_nodes': [],
                    'metadata': {
                        'error': str(e),
                        'fallback_error': str(fallback_error),
                        'query': query_text
                    }
                }
    
    async def _query_optimized_llama(
        self, query_text: str, user_id: str, project_id: Optional[str], 
        filters: Optional[Dict[str, Any]], top_k: int, **kwargs
    ) -> Dict[str, Any]:
        """Query using optimized LlamaIndex service"""
        return await self.optimized_llama_service.query_optimized(
            query_text=query_text,
            user_id=user_id,
            project_id=project_id,
            filters=filters,
            top_k=top_k,
            **kwargs
        )
    
    async def _query_optimized_graphrag(
        self, query_text: str, user_id: str, top_k: int, **kwargs
    ) -> Dict[str, Any]:
        """Query using optimized GraphRAG service"""
        from .graphrag_service import GraphRAGStrategy
        
        strategy = kwargs.get('strategy', GraphRAGStrategy.HYBRID_PARALLEL)
        include_reasoning = kwargs.get('include_reasoning', True)
        
        results = await self.optimized_graphrag_service.query_graphrag_optimized(
            query=query_text,
            strategy=strategy,
            max_results=top_k,
            include_reasoning=include_reasoning
        )
        
        # Convert GraphRAG results to standard format
        return {
            'response': self._format_graphrag_response(results),
            'source_nodes': [
                {
                    'text': result.content,
                    'score': result.score,
                    'metadata': result.metadata
                }
                for result in results
            ],
            'metadata': {
                'query': query_text,
                'strategy': strategy.value,
                'result_count': len(results),
                'service_type': 'optimized_graphrag'
            }
        }
    
    async def _query_fallback_service(
        self, service_type: str, query_text: str, user_id: str, 
        project_id: Optional[str], filters: Optional[Dict[str, Any]], 
        top_k: int, **kwargs
    ) -> Dict[str, Any]:
        """Query using fallback base services"""
        
        if service_type == "llama" and self.base_llama_service:
            result = await self.base_llama_service.query(
                query_text=query_text,
                user_id=user_id,
                project_id=project_id,
                filters=filters,
                top_k=top_k
            )
            result['metadata'] = result.get('metadata', {})
            result['metadata']['service_type'] = 'base_llama'
            return result
            
        elif service_type == "graphrag" and self.base_graphrag_service:
            from .graphrag_service import GraphRAGStrategy
            
            strategy = kwargs.get('strategy', GraphRAGStrategy.HYBRID_PARALLEL)
            include_reasoning = kwargs.get('include_reasoning', True)
            
            results = await self.base_graphrag_service.query_graphrag(
                query=query_text,
                strategy=strategy,
                max_results=top_k,
                include_reasoning=include_reasoning
            )
            
            return {
                'response': self._format_graphrag_response(results),
                'source_nodes': [
                    {
                        'text': result.content,
                        'score': result.score,
                        'metadata': result.metadata
                    }
                    for result in results
                ],
                'metadata': {
                    'query': query_text,
                    'strategy': strategy.value,
                    'result_count': len(results),
                    'service_type': 'base_graphrag'
                }
            }
        
        raise RuntimeError(f"No fallback service available for {service_type}")
    
    def _format_graphrag_response(self, results: List) -> str:
        """Format GraphRAG results into a readable response"""
        if not results:
            return "No relevant information found for your query."
        
        response = "Based on the knowledge graph analysis:\n\n"
        for i, result in enumerate(results[:3], 1):
            content = result.content if hasattr(result, 'content') else str(result)
            response += f"{i}. {content[:200]}...\n\n"
        
        return response
    
    async def batch_query(
        self,
        queries: List[Dict[str, Any]],
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple queries with optimization"""
        try:
            # Use async optimizer for batch processing if available
            if self.optimization_enabled and self.async_optimizer:
                return await self._batch_query_optimized(queries, batch_size)
            else:
                return await self._batch_query_fallback(queries, batch_size)
                
        except Exception as e:
            logger.error(f"Batch query failed: {e}")
            # Fallback to individual queries
            results = []
            for query_data in queries:
                try:
                    result = await self.query_with_optimization(**query_data)
                    results.append(result)
                except Exception as query_error:
                    results.append({
                        'error': str(query_error),
                        'query': query_data.get('query_text', '')
                    })
            return results
    
    async def _batch_query_optimized(
        self, queries: List[Dict[str, Any]], batch_size: int
    ) -> List[Dict[str, Any]]:
        """Batch query processing with optimizations"""
        results = []
        
        # Group queries by service type
        llama_queries = [q for q in queries if q.get('service_type', 'llama') == 'llama']
        graphrag_queries = [q for q in queries if q.get('service_type') == 'graphrag']
        
        # Process LlamaIndex queries
        if llama_queries and self.optimized_llama_service:
            llama_results = await self.optimized_llama_service.batch_query(
                llama_queries, batch_size
            )
            results.extend(llama_results)
        
        # Process GraphRAG queries (individual for now)
        for query in graphrag_queries:
            result = await self.query_with_optimization(**query)
            results.append(result)
        
        return results
    
    async def _batch_query_fallback(
        self, queries: List[Dict[str, Any]], batch_size: int
    ) -> List[Dict[str, Any]]:
        """Batch query processing with fallback services"""
        results = []
        
        # Process in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            batch_tasks = [
                self.query_with_optimization(**query_data)
                for query_data in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({'error': str(result)})
                else:
                    results.append(result)
        
        return results
    
    async def ingest_document_optimized(
        self, content: str, metadata: Dict[str, Any], 
        service_type: str = "both", **kwargs
    ) -> Dict[str, Any]:
        """Ingest document with performance optimizations"""
        results = {}
        
        try:
            # Ingest to LlamaIndex if requested
            if service_type in ["llama", "both"] and self.optimized_llama_service:
                llama_result = await self.optimized_llama_service.ingest_document_optimized(
                    content=content,
                    metadata=metadata,
                    **kwargs
                )
                results['llamaindex'] = llama_result
            
            # Ingest to GraphRAG if requested
            if service_type in ["graphrag", "both"] and self.optimized_graphrag_service:
                graph_result = await self.optimized_graphrag_service.batch_index_documents_optimized(
                    documents=[{'content': content, **metadata}]
                )
                results['graphrag'] = graph_result
            
            return {
                'success': True,
                'results': results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Optimized document ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        try:
            metrics = {
                'integration_stats': self.integration_metrics.copy(),
                'optimization_enabled': self.optimization_enabled,
                'initialized': self.initialized,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add component metrics
            if self.cache_optimizer:
                metrics['cache_optimizer'] = await self.cache_optimizer.get_performance_stats()
            
            if self.async_optimizer:
                metrics['async_optimizer'] = await self.async_optimizer.get_performance_stats()
            
            if self.optimized_llama_service:
                metrics['llama_service'] = await self.optimized_llama_service.get_performance_metrics()
            
            if self.optimized_graphrag_service:
                metrics['graphrag_service'] = await self.optimized_graphrag_service.get_graph_performance_metrics()
            
            # Calculate derived metrics
            total_queries = self.integration_metrics['total_queries']
            if total_queries > 0:
                metrics['derived_metrics'] = {
                    'optimization_usage_rate': self.integration_metrics['optimized_queries'] / total_queries,
                    'fallback_rate': self.integration_metrics['fallback_queries'] / total_queries,
                    'error_rate': self.integration_metrics['errors'] / total_queries
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get integration metrics: {e}")
            return {'error': str(e)}
    
    async def enable_optimization(self):
        """Enable performance optimizations"""
        self.optimization_enabled = True
        logger.info("RAG performance optimization enabled")
    
    async def disable_optimization(self):
        """Disable performance optimizations (fallback only)"""
        self.optimization_enabled = False
        logger.info("RAG performance optimization disabled, using fallback services")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the integration system"""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {},
            'issues': []
        }
        
        try:
            # Check initialization
            if not self.initialized:
                health_status['issues'].append("Integration system not initialized")
                health_status['overall_status'] = 'unhealthy'
            
            # Check cache optimizer
            if self.cache_optimizer:
                try:
                    cache_stats = await self.cache_optimizer.get_performance_stats()
                    health_status['components']['cache_optimizer'] = 'healthy'
                except Exception as e:
                    health_status['components']['cache_optimizer'] = f'error: {str(e)}'
                    health_status['issues'].append(f"Cache optimizer error: {str(e)}")
            else:
                health_status['components']['cache_optimizer'] = 'not_available'
            
            # Check async optimizer
            if self.async_optimizer:
                try:
                    async_stats = await self.async_optimizer.get_performance_stats()
                    if async_stats.get('system', {}).get('running', False):
                        health_status['components']['async_optimizer'] = 'healthy'
                    else:
                        health_status['components']['async_optimizer'] = 'not_running'
                        health_status['issues'].append("Async optimizer not running")
                except Exception as e:
                    health_status['components']['async_optimizer'] = f'error: {str(e)}'
                    health_status['issues'].append(f"Async optimizer error: {str(e)}")
            else:
                health_status['components']['async_optimizer'] = 'not_available'
            
            # Check optimized services
            health_status['components']['optimized_llama'] = 'healthy' if self.optimized_llama_service else 'not_available'
            health_status['components']['optimized_graphrag'] = 'healthy' if self.optimized_graphrag_service else 'not_available'
            
            # Check fallback services
            health_status['components']['fallback_llama'] = 'healthy' if self.base_llama_service else 'not_available'
            health_status['components']['fallback_graphrag'] = 'healthy' if self.base_graphrag_service else 'not_available'
            
            # Determine overall status
            if health_status['issues']:
                if len(health_status['issues']) >= 3:
                    health_status['overall_status'] = 'unhealthy'
                else:
                    health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


# Global integration service instance
_integration_service: Optional[RAGPerformanceIntegration] = None


async def get_rag_performance_integration() -> RAGPerformanceIntegration:
    """Get or create the RAG performance integration service instance"""
    global _integration_service
    
    if _integration_service is None:
        _integration_service = RAGPerformanceIntegration()
        await _integration_service.initialize()
    
    return _integration_service
